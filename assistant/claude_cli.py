"""Runs the Claude Code CLI (``claude -p``) as an isolated, tool-less text generator.

Every reply is one ``claude -p`` process:

* no shell is involved (``create_subprocess_exec``);
* the conversation goes in on stdin only, never in argv;
* the system prompt (with the profile) is written to a 0600 temp file, so it never shows up in ``ps``;
* hooks, plugins, MCP servers, tools, slash commands, CLAUDE.md and auto-memory are all disabled;
* the child environment is scrubbed of API keys and of the bot token.

``--bare`` is deliberately not used: it ignores the OAuth login, so the subscription would not work.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .config import AUTH_FROM_ENV_FILE, AUTH_LOCAL_LOGIN
from .store import ensure_private_dir

# Every plugin built into the CLI, as listed in the stream-json init event
# (`claude -p --output-format stream-json --verbose`), is switched off.
BUILTIN_PLUGINS = (
    "cc-plugin-agents-md@builtin",
    "cc-plugin-telemetry@builtin",
    "cc-plugin-plugin-authoring@builtin",
)

ISOLATION_SETTINGS: Dict[str, Any] = {
    "disableAllHooks": True,
    "enabledPlugins": {name: False for name in BUILTIN_PLUGINS},
}

ISOLATION_ENV = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
}

STRIPPED_ENV = (
    "TELEGRAM_BOT_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    # Set by --bare; it turns the OAuth login off.
    "CLAUDE_CODE_SIMPLE",
)

INIT_FIELDS = ("tools", "mcp_servers", "plugins", "skills", "apiKeySource", "apiProvider", "model")

REDACTED = "[REDACTED]"

_LIMIT_RE = re.compile(
    r"usage limit|hit your (usage )?limit|limit reached|rate[ _-]?limit|out of (extra )?usage|quota",
    re.IGNORECASE,
)
_AUTH_RE = re.compile(
    r"not logged in|/login|invalid api key|authentication|unauthori[sz]ed|oauth token",
    re.IGNORECASE,
)
_EPOCH_RE = re.compile(r"\|(\d{9,12})\b")
_RESET_TEXT_RE = re.compile(r"\bresets\s+([^\n|]+)", re.IGNORECASE)
_TOKEN_SHAPE_RE = re.compile(r"sk-ant-[A-Za-z0-9]+-[A-Za-z0-9_\-]{8,}")


class ClaudeError(Exception):
    """A Claude failure whose message is safe to show to the owner."""


class ClaudeTimeout(ClaudeError):
    pass


class ClaudeAuthError(ClaudeError):
    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


class ClaudeLimitError(ClaudeError):
    def __init__(self, detail: str, resets_at: Optional[float] = None, reset_text: Optional[str] = None) -> None:
        if reset_text:
            when = f"resets {reset_text}"
        elif resets_at:
            when = "resets " + time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(resets_at))
        else:
            when = "reset time unknown"
        super().__init__(f"Claude subscription usage limit reached ({when})")
        self.detail = detail
        self.resets_at = resets_at
        self.reset_text = reset_text


def build_env(oauth_token: Optional[str] = None, base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment for a ``claude`` child process."""
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key.upper().startswith("ANTHROPIC_") or key.upper() in STRIPPED_ENV:
            del env[key]
    env.update(ISOLATION_ENV)
    if oauth_token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
    return env


def build_command(
    claude_bin: str,
    system_prompt_file: str,
    *,
    effort: str = "low",
    model: Optional[str] = None,
    output_format: str = "json",
) -> List[str]:
    cmd = [
        claude_bin,
        "-p",
        "--safe-mode",
        "--settings", json.dumps(ISOLATION_SETTINGS),
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt-file", system_prompt_file,
        "--output-format", output_format,
        "--effort", effort,
    ]
    if output_format == "stream-json":
        cmd.append("--verbose")
    if model:
        cmd += ["--model", model]
    return cmd


def classify_failure(message: str, status: Optional[int] = None) -> ClaudeError:
    """Turn a CLI failure message into the matching exception (returned, not raised)."""
    message = (message or "").strip()
    if status == 429 or _LIMIT_RE.search(message):
        epoch = _EPOCH_RE.search(message)
        resets_at = float(epoch.group(1)) if epoch else None
        detail = _EPOCH_RE.sub("", message).strip()
        reset = _RESET_TEXT_RE.search(detail)
        reset_text = reset.group(1).strip().rstrip(".") if reset else None
        return ClaudeLimitError(detail, resets_at=resets_at, reset_text=reset_text)
    if status in (401, 403) or _AUTH_RE.search(message):
        return ClaudeAuthError(
            "Claude CLI is not logged in or its login expired. Run `claude` on this machine and use /login.",
            message[:300],
        )
    return ClaudeError(f"Claude CLI failed: {message[:500]}")


def parse_result(returncode: Optional[int], stdout: str, stderr: str) -> str:
    """Extract the reply text from ``--output-format json`` (or the last stream-json line)."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    data: Any = None
    if lines:
        try:
            data = json.loads(lines[-1])
        except ValueError:
            data = None
    if not isinstance(data, dict):
        raise classify_failure(stderr.strip() or stdout.strip() or f"exit code {returncode}")
    result = data.get("result")
    result_text = result if isinstance(result, str) else ""
    if data.get("is_error") or data.get("subtype") != "success":
        message = " ".join(part for part in (result_text.strip(), stderr.strip()) if part)
        status = data.get("api_error_status")
        raise classify_failure(
            message or f"exit code {returncode} ({data.get('subtype')})",
            status if isinstance(status, int) else None,
        )
    text = result_text.strip()
    if not text:
        raise ClaudeError("Claude returned an empty reply.")
    return text


def _kill_process_tree(pid: int) -> None:
    if os.name == "posix":
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    else:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


async def _reap(proc: "asyncio.subprocess.Process") -> None:
    _kill_process_tree(proc.pid)
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(proc.wait(), 5)
    except (asyncio.TimeoutError, ProcessLookupError):
        pass


class ClaudeCLI:
    def __init__(
        self,
        claude_bin: str,
        *,
        workdir: Path,
        model: Optional[str] = None,
        effort: str = "low",
        timeout: float = 120.0,
        oauth_token: Optional[str] = None,
        auth_label: Optional[str] = None,
    ) -> None:
        self.claude_bin = claude_bin
        self.workdir = Path(workdir)
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self._oauth_token = oauth_token
        self._auth_label = auth_label

    @property
    def auth_mode(self) -> str:
        if self._auth_label:
            return self._auth_label
        return AUTH_FROM_ENV_FILE if self._oauth_token else AUTH_LOCAL_LOGIN

    def __repr__(self) -> str:
        return (
            f"ClaudeCLI(claude_bin={self.claude_bin!r}, model={self.model!r}, "
            f"effort={self.effort!r}, timeout={self.timeout!r}, auth={self.auth_mode!r})"
        )

    def _redact(self, text: str) -> str:
        if self._oauth_token:
            text = text.replace(self._oauth_token, REDACTED)
        return _TOKEN_SHAPE_RE.sub(REDACTED, text)

    def _explain_auth(self, error: ClaudeAuthError) -> ClaudeAuthError:
        if not self._oauth_token:
            return error
        return ClaudeAuthError(
            "The Claude subscription token was rejected or has expired: run `claude setup-token` on a "
            "computer with a browser and update CLAUDE_CODE_OAUTH_TOKEN. "
            f"CLI said: {error.detail or 'nothing'}",
            error.detail,
        )

    async def _run(self, system_prompt: str, stdin_text: str, output_format: str) -> Tuple[Optional[int], str, str]:
        ensure_private_dir(self.workdir)
        fd, prompt_path = tempfile.mkstemp(prefix="system-", suffix=".txt", dir=str(self.workdir))
        try:
            os.chmod(prompt_path, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(system_prompt)
            cmd = build_command(
                self.claude_bin,
                prompt_path,
                effort=self.effort,
                model=self.model,
                output_format=output_format,
            )
            if os.name == "posix":
                group: Dict[str, Any] = {"start_new_session": True}
            else:
                group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.workdir),
                    env=build_env(self._oauth_token),
                    **group,
                )
            except FileNotFoundError:
                raise ClaudeError(
                    f"claude CLI not found at {self.claude_bin}. Install Claude Code or set CLAUDE_BIN."
                ) from None
            except PermissionError:
                raise ClaudeError(f"claude CLI at {self.claude_bin} is not executable.") from None
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(stdin_text.encode("utf-8")), self.timeout
                )
            except asyncio.TimeoutError:
                await _reap(proc)
                raise ClaudeTimeout(f"Claude did not answer within {self.timeout:g} seconds.") from None
            except asyncio.CancelledError:
                await _reap(proc)
                raise
        finally:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
        return (
            proc.returncode,
            self._redact(stdout.decode("utf-8", errors="replace")),
            self._redact(stderr.decode("utf-8", errors="replace")),
        )

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        returncode, stdout, stderr = await self._run(system_prompt, user_prompt, "json")
        try:
            return parse_result(returncode, stdout, stderr)
        except ClaudeAuthError as error:
            raise self._explain_auth(error) from None

    async def inspect(self) -> Dict[str, Any]:
        """A tiny request that reports how the CLI started (auth, tools, plugins...)."""
        returncode, stdout, stderr = await self._run("Reply with exactly: OK", "ping", "stream-json")
        init: Optional[Dict[str, Any]] = None
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "system" and event.get("subtype") == "init":
                init = {name: event.get(name) for name in INIT_FIELDS}
                break
        try:
            reply = parse_result(returncode, stdout, stderr)
        except ClaudeAuthError as error:
            raise self._explain_auth(error) from None
        return {"auth": self.auth_mode, "init": init, "reply": reply}
