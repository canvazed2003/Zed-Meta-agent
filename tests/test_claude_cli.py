"""ClaudeCLI against a fake ``claude`` executable that records what it was given."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from assistant.claude_cli import (
    ISOLATION_ENV,
    ISOLATION_SETTINGS,
    ClaudeAuthError,
    ClaudeCLI,
    ClaudeError,
    ClaudeLimitError,
    ClaudeTimeout,
    classify_failure,
    parse_result,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")

TOKEN = "sk-ant-oat01-" + "T" * 60
PARENT_TOKEN = "sk-ant-oat01-" + "P" * 60

FAKE_CLAUDE = r'''
import json, os, stat, subprocess, sys, time

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
mode = os.environ.get("FAKE_MODE", "ok")
argv = sys.argv[1:]
stdin = sys.stdin.buffer.read().decode("utf-8")
prompt_path = argv[argv.index("--system-prompt-file") + 1]
with open(prompt_path, encoding="utf-8") as handle:
    prompt = handle.read()
record = {
    "argv": argv,
    "stdin": stdin,
    "env": dict(os.environ),
    "cwd": os.getcwd(),
    "cwd_listing": sorted(os.listdir(".")),
    "cwd_mode": stat.S_IMODE(os.stat(".").st_mode),
    "pid": os.getpid(),
    "prompt_path": prompt_path,
    "prompt": prompt,
    "prompt_mode": stat.S_IMODE(os.stat(prompt_path).st_mode),
}

def save():
    with open(os.environ["FAKE_RECORD"], "w", encoding="utf-8") as handle:
        json.dump(record, handle)

def result(text, is_error=False, subtype="success", status=None):
    if "stream-json" in argv:
        print(json.dumps({"type": "system", "subtype": "init", "tools": [], "mcp_servers": [], "plugins": [],
                          "skills": [], "apiKeySource": "none", "apiProvider": "firstParty", "model": "fake-model"}))
    print(json.dumps({"type": "result", "subtype": subtype, "is_error": is_error, "result": text,
                      "api_error_status": status}))

token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
if mode == "sleep":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    record["grandchild"] = child.pid
    save()
    time.sleep(60)
save()
if mode == "ok":
    result("  Hello there!  \n")
elif mode == "limit_epoch":
    result("Claude AI usage limit reached|1760003600", is_error=True)
    sys.exit(1)
elif mode == "limit_text":
    result("You've hit your limit · resets 3pm (Asia/Tashkent)", is_error=True)
    sys.exit(1)
elif mode == "auth":
    result("Invalid API key · Please run /login (token " + token + ")", is_error=True)
    sys.stderr.write("auth failed for " + token + "\n")
    sys.exit(1)
elif mode == "crash":
    sys.stderr.write("Segmentation fault: something broke\n")
    sys.exit(139)
elif mode == "empty":
    result("   ")
elif mode == "echo_token":
    result("token is " + token)
'''


def _make_executable(script: Path, bin_dir: Path) -> Path:
    if os.name == "posix":
        target = bin_dir / "claude"
        target.write_text(f"#!{sys.executable}\n" + script.read_text(encoding="utf-8"), encoding="utf-8")
        target.chmod(0o755)
        return target
    # Windows cannot run a shebang script directly: wrap it in pip's .exe launcher.
    from pip._vendor.distlib.scripts import ScriptMaker

    source = bin_dir / "src"
    source.mkdir()
    (source / "claude.py").write_text("#!python\n" + script.read_text(encoding="utf-8"), encoding="utf-8")
    maker = ScriptMaker(str(source), str(bin_dir), add_launchers=True)
    maker.executable = sys.executable
    maker.variants = {""}
    made = maker.make("claude.py")
    return Path(next(path for path in made if path.endswith(".exe")))


@pytest.fixture(scope="session")
def fake_claude(tmp_path_factory) -> Path:
    bin_dir = tmp_path_factory.mktemp("fakebin")
    script = bin_dir / "fake_claude.py"
    script.write_text(textwrap.dedent(FAKE_CLAUDE), encoding="utf-8")
    return _make_executable(script, bin_dir)


@pytest.fixture
def run(tmp_path, fake_claude, monkeypatch):
    record_path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_RECORD", str(record_path))

    def make(mode: str = "ok", **kwargs) -> ClaudeCLI:
        monkeypatch.setenv("FAKE_MODE", mode)
        kwargs.setdefault("workdir", tmp_path / "data" / "claude-run")
        kwargs.setdefault("timeout", 30)
        return ClaudeCLI(str(fake_claude), **kwargs)

    def record() -> dict:
        return json.loads(record_path.read_text(encoding="utf-8"))

    make.record = record
    return make


def _pid_alive(pid: int) -> bool:
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        try:  # a zombie still answers kill(0); check its state
            with open(f"/proc/{pid}/stat") as handle:
                return handle.read().split()[2] != "Z"
        except OSError:
            return True
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


# ----- happy path and isolation ------------------------------------------------------


async def test_result_is_stripped(run):
    cli = run()
    assert await cli.generate("system", "hello") == "Hello there!"


async def test_message_only_on_stdin_and_profile_only_in_private_file(run):
    cli = run()
    await cli.generate("PROFILE: secret profile text", "USER MESSAGE: are you free?")
    rec = run.record()
    assert rec["stdin"] == "USER MESSAGE: are you free?"
    joined = " ".join(rec["argv"])
    assert "USER MESSAGE" not in joined
    assert "secret profile text" not in joined
    assert rec["prompt"] == "PROFILE: secret profile text"
    assert not Path(rec["prompt_path"]).exists()  # deleted afterwards


@posix_only
async def test_prompt_file_and_workdir_are_private(run):
    cli = run()
    await cli.generate("system", "hi")
    rec = run.record()
    assert rec["prompt_mode"] == 0o600
    assert rec["cwd_mode"] == 0o700


async def test_runs_in_its_own_empty_workdir(run, tmp_path):
    cli = run()
    await cli.generate("system", "hi")
    rec = run.record()
    assert Path(rec["cwd"]).resolve() == (tmp_path / "data" / "claude-run").resolve()
    assert rec["cwd_listing"] == [Path(rec["prompt_path"]).name]


async def test_prompt_file_is_deleted_on_failure(run, tmp_path):
    cli = run("crash")
    with pytest.raises(ClaudeError):
        await cli.generate("system", "hi")
    assert list((tmp_path / "data" / "claude-run").iterdir()) == []


async def test_no_shell_is_involved(run, tmp_path):
    marker = tmp_path / "pwned"
    text = f"$(touch {marker}) `touch {marker}` ; touch {marker} && echo hi | cat"
    cli = run()
    await cli.generate("system", text)
    assert run.record()["stdin"] == text
    assert not marker.exists()


async def test_isolation_flags_in_order(run):
    cli = run(model="sonnet", effort="medium")
    await cli.generate("system", "hi")
    argv = run.record()["argv"]
    prompt_index = argv.index("--system-prompt-file")
    expected = [
        "-p",
        "--safe-mode",
        "--settings", json.dumps(ISOLATION_SETTINGS),
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt-file", argv[prompt_index + 1],
        "--output-format", "json",
        "--effort", "medium",
        "--model", "sonnet",
    ]
    assert argv == expected
    assert "--mcp-config" not in argv
    assert "--bare" not in argv
    assert "--verbose" not in argv
    settings = json.loads(argv[argv.index("--settings") + 1])
    assert settings["disableAllHooks"] is True
    assert settings["enabledPlugins"] and not any(settings["enabledPlugins"].values())


async def test_child_environment_is_scrubbed(run, monkeypatch, tmp_path):
    for key, value in {
        "ANTHROPIC_API_KEY": "sk-ant-api03-x",
        "ANTHROPIC_BASE_URL": "https://evil.example",
        "anthropic_auth_token": "x",
        "TELEGRAM_BOT_TOKEN": "1:abc",
        "CLAUDE_CODE_OAUTH_TOKEN": PARENT_TOKEN,
        "CLAUDE_CODE_SIMPLE": "1",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_USE_VERTEX": "1",
        "CLAUDE_CODE_USE_FOUNDRY": "1",
        "CLAUDECODE": "1",
        "CLAUDE_CODE_ENTRYPOINT": "cli",
        "HOME": str(tmp_path / "home"),
    }.items():
        monkeypatch.setenv(key, value)
    cli = run()
    await cli.generate("system", "hi")
    env = run.record()["env"]
    upper = {key.upper() for key in env}
    assert not any(key.startswith("ANTHROPIC_") for key in upper)
    for key in ("TELEGRAM_BOT_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_SIMPLE", "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
        assert key not in upper
    assert env["HOME"] == str(tmp_path / "home")
    for key, value in ISOLATION_ENV.items():
        assert env[key] == value


async def test_configured_token_only_reaches_child_env(run, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", PARENT_TOKEN)
    cli = run(oauth_token=TOKEN)
    await cli.generate("system prompt", "user text")
    rec = run.record()
    assert rec["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN
    assert TOKEN not in " ".join(rec["argv"])
    assert TOKEN not in rec["stdin"]
    assert TOKEN not in rec["prompt"]
    assert TOKEN not in repr(cli)


async def test_no_token_configured_means_none_in_child(run, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", PARENT_TOKEN)
    cli = run()
    await cli.generate("system", "hi")
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in run.record()["env"]


async def test_token_in_output_is_redacted(run):
    cli = run("echo_token", oauth_token=TOKEN)
    assert await cli.generate("system", "hi") == "token is [REDACTED]"


async def test_rejected_token_message(run):
    cli = run("auth", oauth_token=TOKEN)
    with pytest.raises(ClaudeAuthError) as info:
        await cli.generate("system", "hi")
    message = str(info.value)
    assert "claude setup-token" in message
    assert "CLAUDE_CODE_OAUTH_TOKEN" in message
    assert "CLI said:" in message
    assert "[REDACTED]" in message
    assert TOKEN not in message
    assert TOKEN not in info.value.detail


async def test_auth_error_without_token_suggests_login(run):
    cli = run("auth")
    with pytest.raises(ClaudeAuthError) as info:
        await cli.generate("system", "hi")
    assert "/login" in str(info.value)


async def test_inspect_reports_auth_and_init(run):
    cli = run(oauth_token=TOKEN, auth_label="subscription token from .env")
    info = await cli.inspect()
    assert info["auth"] == "subscription token from .env"
    assert info["reply"] == "Hello there!"
    assert info["init"]["tools"] == [] and info["init"]["apiProvider"] == "firstParty"
    argv = run.record()["argv"]
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv
    assert run.record()["stdin"] == "ping"
    assert run.record()["prompt"] == "Reply with exactly: OK"


async def test_auth_mode_defaults(run):
    assert run().auth_mode == "this machine's claude login"
    assert run(oauth_token=TOKEN).auth_mode == "subscription token from .env"


# ----- failures -------------------------------------------------------------------------


async def test_timeout_kills_the_whole_process_group(run):
    cli = run("sleep", timeout=1.5)
    started = time.monotonic()
    with pytest.raises(ClaudeTimeout):
        await cli.generate("system", "hi")
    assert time.monotonic() - started < 5
    grandchild = run.record()["grandchild"]
    deadline = time.monotonic() + 3
    while _pid_alive(grandchild) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert not _pid_alive(grandchild)


async def test_cancel_kills_the_process(run):
    cli = run("sleep", timeout=30)
    task = asyncio.ensure_future(cli.generate("system", "hi"))
    await asyncio.sleep(1.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    grandchild = run.record()["grandchild"]
    deadline = time.monotonic() + 3
    while _pid_alive(grandchild) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert not _pid_alive(grandchild)


async def test_limit_with_epoch(run):
    with pytest.raises(ClaudeLimitError) as info:
        await run("limit_epoch").generate("system", "hi")
    assert info.value.resets_at == 1760003600
    assert "|" not in info.value.detail
    assert str(info.value).startswith("Claude subscription usage limit reached (resets ")


async def test_limit_with_text(run):
    with pytest.raises(ClaudeLimitError) as info:
        await run("limit_text").generate("system", "hi")
    assert info.value.reset_text == "3pm (Asia/Tashkent)"
    assert str(info.value) == "Claude subscription usage limit reached (resets 3pm (Asia/Tashkent))"


async def test_crash(run):
    with pytest.raises(ClaudeError) as info:
        await run("crash").generate("system", "hi")
    assert "Segmentation fault" in str(info.value)
    assert not isinstance(info.value, (ClaudeAuthError, ClaudeLimitError))


async def test_empty_reply(run):
    with pytest.raises(ClaudeError, match="empty"):
        await run("empty").generate("system", "hi")


async def test_missing_binary(tmp_path):
    cli = ClaudeCLI(str(tmp_path / "nope" / "claude"), workdir=tmp_path / "run")
    with pytest.raises(ClaudeError, match="not found"):
        await cli.generate("system", "hi")
    assert list((tmp_path / "run").iterdir()) == []


# ----- pure helpers ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "message, status",
    [
        ("Claude AI usage limit reached|1760000000", None),
        ("You've hit your limit · resets 5am", None),
        ("You hit your usage limit", None),
        ("rate_limit_error", None),
        ("Rate limit exceeded", None),
        ("You're out of extra usage", None),
        ("quota exceeded", None),
        ("whatever", 429),
    ],
)
def test_classify_limit(message, status):
    assert isinstance(classify_failure(message, status), ClaudeLimitError)


def test_classify_limit_details():
    error = classify_failure("Claude AI usage limit reached|1760000000")
    assert error.resets_at == 1760000000
    assert error.detail == "Claude AI usage limit reached"
    error = classify_failure("You've hit your limit · resets 3pm (Europe/Berlin)")
    assert error.reset_text == "3pm (Europe/Berlin)"
    assert error.resets_at is None


@pytest.mark.parametrize(
    "message, status",
    [
        ("Not logged in · Please run /login", None),
        ("Invalid API key", None),
        ("authentication_error: invalid x-api-key", None),
        ("401 Unauthorized", None),
        ("OAuth token has expired", None),
        ("something", 401),
        ("something", 403),
    ],
)
def test_classify_auth(message, status):
    error = classify_failure(message, status)
    assert isinstance(error, ClaudeAuthError)
    assert "/login" in str(error)
    assert error.detail == message[:300]


def test_classify_other_is_truncated():
    error = classify_failure("x" * 2000)
    assert type(error) is ClaudeError
    assert len(str(error)) < 600


def test_parse_result_success_uses_last_line():
    stdout = '{"type":"system"}\n\n{"type":"result","subtype":"success","is_error":false,"result":" hi "}\n'
    assert parse_result(0, stdout, "") == "hi"


def test_parse_result_errors():
    with pytest.raises(ClaudeLimitError):
        parse_result(1, '{"subtype":"success","is_error":true,"result":"usage limit reached|1760000000"}', "")
    with pytest.raises(ClaudeAuthError):
        parse_result(1, '{"subtype":"success","is_error":true,"result":"x","api_error_status":401}', "")
    with pytest.raises(ClaudeError, match="error_max_turns"):
        parse_result(1, '{"subtype":"error_max_turns","is_error":false}', "")
    with pytest.raises(ClaudeError, match="empty"):
        parse_result(0, '{"subtype":"success","is_error":false,"result":""}', "")
    with pytest.raises(ClaudeAuthError):
        parse_result(1, "not json", "Not logged in")
    with pytest.raises(ClaudeError, match="exit code 7"):
        parse_result(7, "", "")
