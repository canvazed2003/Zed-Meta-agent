"""Settings loading.

Settings come from exactly one source and are never mixed:

* local runs read only the ``.env`` file (``dotenv_values``; the shell environment is ignored);
* container runs (Railway) pass the process environment in as ``source``
  (see ``bot.py --settings-from-env``), and ``.env`` is not read at all.

Nothing here writes to ``os.environ``, so secrets never leak into other processes by accident.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

from dotenv import dotenv_values

PathLike = Union[str, "os.PathLike[str]"]

PROJECT_DIR = Path(__file__).resolve().parent.parent

SETTING_KEYS = (
    "TELEGRAM_BOT_TOKEN",
    "OWNER_ID",
    "AUTO_SEND",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_BIN",
    "CLAUDE_MODEL",
    "CLAUDE_EFFORT",
    "CLAUDE_TIMEOUT_SECONDS",
    "LIMIT_COOLDOWN_MINUTES",
    "OWNER_TAKEOVER_MINUTES",
    "HISTORY_LIMIT",
    "DATA_DIR",
    "PROFILE_PATH",
)

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

AUTH_LOCAL_LOGIN = "this machine's claude login"
AUTH_FROM_ENV_FILE = "subscription token from .env"
AUTH_FROM_ENVIRONMENT = "subscription token from environment variables"

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}
_BOT_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]+$")


class ConfigError(Exception):
    """Invalid or missing settings. Messages never contain secret values."""


@dataclass(frozen=True)
class Settings:
    bot_token: Optional[str] = field(repr=False)
    owner_id: Optional[int]
    auto_send: bool
    claude_oauth_token: Optional[str] = field(repr=False)
    claude_bin: str
    claude_model: Optional[str]
    claude_effort: str
    claude_timeout: float
    limit_cooldown_minutes: int
    owner_takeover_minutes: int
    history_limit: int
    data_dir: Path
    profile_path: Path
    from_environment: bool = False

    @property
    def claude_auth_mode(self) -> str:
        if not self.claude_oauth_token:
            return AUTH_LOCAL_LOGIN
        return AUTH_FROM_ENVIRONMENT if self.from_environment else AUTH_FROM_ENV_FILE


def default_claude_bin() -> str:
    found = shutil.which("claude")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "claude"
    if os.name == "nt" and not local.exists():
        local = local.with_suffix(".exe")
    return str(local)


def load_settings(
    env_file: PathLike,
    *,
    require_telegram: bool = True,
    overrides: Optional[Mapping[str, Optional[str]]] = None,
    source: Optional[Mapping[str, Optional[str]]] = None,
) -> Settings:
    """Load settings from ``env_file``, or from ``source`` when it is given.

    ``env_file``'s folder is always the base for relative paths. ``overrides`` is for tests.
    """
    env_path = Path(env_file)
    base_dir = env_path.resolve().parent
    if source is None:
        if not env_path.is_file():
            raise ConfigError(
                f"{env_path} not found. Create it with: cp .env.example .env && chmod 600 .env"
            )
        raw: Dict[str, Optional[str]] = dict(dotenv_values(env_path))
        where = str(env_path)
    else:
        raw = dict(source)
        where = "the environment variables"
    if overrides:
        raw.update(overrides)
    values = {key: (raw.get(key) or "").strip() for key in SETTING_KEYS}

    def fail(key: str, problem: str) -> ConfigError:
        return ConfigError(f"{key} in {where} {problem}")

    bot_token = values["TELEGRAM_BOT_TOKEN"] or None
    if bot_token and not _BOT_TOKEN_RE.match(bot_token):
        raise fail("TELEGRAM_BOT_TOKEN", "does not look like a bot token from @BotFather (<digits>:<text>).")
    if require_telegram and not bot_token:
        raise fail("TELEGRAM_BOT_TOKEN", "is empty. Get a token from @BotFather.")

    owner_raw = values["OWNER_ID"]
    owner_id: Optional[int] = None
    if owner_raw:
        if not owner_raw.isdigit():
            raise fail("OWNER_ID", "must be your numeric Telegram user id (ask @userinfobot).")
        owner_id = int(owner_raw)
    elif require_telegram:
        raise fail("OWNER_ID", "is empty. Put your numeric Telegram user id there (ask @userinfobot).")

    auto_raw = values["AUTO_SEND"].lower() or "false"
    if auto_raw in _TRUE:
        auto_send = True
    elif auto_raw in _FALSE:
        auto_send = False
    else:
        raise fail("AUTO_SEND", "must be one of true/false/1/0/yes/no/on/off.")

    token = values["CLAUDE_CODE_OAUTH_TOKEN"] or None
    if token:
        if token.startswith("sk-ant-api"):
            raise fail(
                "CLAUDE_CODE_OAUTH_TOKEN",
                "is an Anthropic API key, not a subscription token. "
                "Run `claude setup-token` and use the sk-ant-oat... token it prints.",
            )
        if not token.startswith("sk-ant-oat") or any(ch.isspace() for ch in token):
            raise fail(
                "CLAUDE_CODE_OAUTH_TOKEN",
                "is not a valid `claude setup-token` token. It must start with sk-ant-oat "
                "and be on a single line without spaces.",
            )

    effort = values["CLAUDE_EFFORT"].lower() or "low"
    if effort not in EFFORT_LEVELS:
        raise fail("CLAUDE_EFFORT", f"must be one of: {', '.join(EFFORT_LEVELS)}.")

    def number(key: str, default: int, minimum: int) -> int:
        text = values[key]
        if not text:
            return default
        try:
            value = int(text)
        except ValueError:
            raise fail(key, "must be a whole number.") from None
        if value < minimum:
            raise fail(key, f"must be at least {minimum}.")
        return value

    def path(key: str, default: str) -> Path:
        candidate = Path(values[key] or default).expanduser()
        return candidate if candidate.is_absolute() else base_dir / candidate

    return Settings(
        bot_token=bot_token,
        owner_id=owner_id,
        auto_send=auto_send,
        claude_oauth_token=token,
        claude_bin=values["CLAUDE_BIN"] or default_claude_bin(),
        claude_model=values["CLAUDE_MODEL"] or None,
        claude_effort=effort,
        claude_timeout=float(number("CLAUDE_TIMEOUT_SECONDS", 120, 1)),
        limit_cooldown_minutes=number("LIMIT_COOLDOWN_MINUTES", 30, 1),
        owner_takeover_minutes=number("OWNER_TAKEOVER_MINUTES", 15, 0),
        history_limit=number("HISTORY_LIMIT", 30, 1),
        data_dir=path("DATA_DIR", "data"),
        profile_path=path("PROFILE_PATH", "profile.md"),
        from_environment=source is not None,
    )


def env_file_warning(path: PathLike) -> Optional[str]:
    """Return a warning when ``.env`` is readable by the group or other users."""
    env_path = Path(path)
    if os.name != "posix" or not env_path.exists():
        return None
    mode = stat.S_IMODE(env_path.stat().st_mode)
    if mode & 0o077:
        return (
            f"{env_path} is readable by other users (mode {mode:o}); it holds secrets. "
            f"Run: chmod 600 {env_path}"
        )
    return None
