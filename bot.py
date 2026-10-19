"""Entry point.

    python bot.py                       run the bot (settings from .env)
    python bot.py --settings-from-env   run in a container (settings from environment variables)
    python bot.py --check-claude        check the Claude CLI setup, no Telegram
    python bot.py --try "text"          draft a reply to a made-up message, no Telegram
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from assistant.claude_cli import ClaudeError
from assistant.config import PROJECT_DIR, SETTING_KEYS, ConfigError, Settings, env_file_warning, load_settings
from assistant.language import detect_language
from assistant.prompts import build_system_prompt, build_user_prompt
from assistant.service import build_assistant, make_claude_cli, profile_looks_empty, read_profile
from assistant.store import ensure_private_dir

ENV_FILE = PROJECT_DIR / ".env"

log = logging.getLogger("bot")

FORBIDDEN_KEY_SOURCES = ("ANTHROPIC_API_KEY", "apiKeyHelper", "/login managed key")


def settings_from_process_env(*, require_telegram: bool = True) -> Settings:
    """Read settings from the process environment, then remove them from it (even on error).

    This way the ``claude`` child process never inherits the bot token.
    """
    source = {key: os.environ[key] for key in SETTING_KEYS if key in os.environ}
    try:
        return load_settings(ENV_FILE, require_telegram=require_telegram, source=source)
    finally:
        for key in SETTING_KEYS:
            os.environ.pop(key, None)


def check_inspection(info: Dict[str, Any]) -> List[str]:
    """Problems with an ``inspect()`` result; an empty list means the setup is safe."""
    init = info.get("init")
    if not init:
        return ["the CLI did not report its init event"]
    problems = []
    for name in ("tools", "mcp_servers", "plugins", "skills"):
        if init.get(name):
            problems.append(f"{name} should be empty but is {init.get(name)!r}")
    if init.get("apiKeySource") in FORBIDDEN_KEY_SOURCES:
        problems.append(f"apiKeySource is {init.get('apiKeySource')!r}: an API key is used instead of the subscription")
    if init.get("apiProvider") not in ("firstParty", None):
        problems.append(f"apiProvider is {init.get('apiProvider')!r}, expected firstParty")
    return problems


async def check_claude(settings: Settings) -> int:
    cli = make_claude_cli(settings)
    try:
        info = await cli.inspect()
    except ClaudeError as error:
        print(json.dumps({"auth": cli.auth_mode, "error": str(error)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(info, indent=2, ensure_ascii=False))
    problems = check_inspection(info)
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print(f"OK: Claude answered using {info['auth']} ({cli.claude_bin}); no tools, MCP servers, plugins or skills.")
    return 0


async def try_reply(settings: Settings, text: str) -> int:
    from datetime import datetime

    cli = make_claude_cli(settings)
    language = detect_language(text)
    system_prompt = build_system_prompt(read_profile(settings.profile_path), datetime.now().astimezone())
    user_prompt = build_user_prompt([{"role": "other", "name": "Test contact", "text": text}], language)
    print(f"Claude auth: {cli.auth_mode}")
    print(f"Language:    {language or 'unknown'}")
    print(f"Message:     {text}")
    try:
        reply = await cli.generate(system_prompt, user_prompt)
    except ClaudeError as error:
        print(f"Error:       {error}")
        return 1
    print(f"Reply:       {reply}")
    return 0


async def run_bot(settings: Settings) -> int:
    from aiogram import Bot, Dispatcher
    from aiogram.exceptions import TelegramUnauthorizedError

    from assistant.handlers import ALLOWED_UPDATES, build_router

    bot = Bot(settings.bot_token)
    try:
        try:
            me = await bot.get_me()
        except TelegramUnauthorizedError:
            log.error("Telegram rejected TELEGRAM_BOT_TOKEN. Check it in @BotFather (or /revoke and use a new one).")
            return 2
        assistant = build_assistant(settings)
        dispatcher = Dispatcher()
        dispatcher.include_router(build_router(assistant))

        log.info("Bot @%s is running", me.username)
        if settings.auto_send:
            log.warning("Mode: AUTO-SEND - replies are sent to chats on your behalf")
        else:
            log.info("Mode: PREVIEW - replies are only shown in the logs (set AUTO_SEND=true to send them)")
        log.info("Claude auth: %s", settings.claude_auth_mode)
        log.info("Claude CLI: %s", settings.claude_bin)
        if assistant.state.paused:
            log.warning("The bot starts PAUSED; send /resume to the bot to continue.")
        if profile_looks_empty(read_profile(settings.profile_path)):
            log.warning("%s looks empty; fill it in so replies sound like you.", settings.profile_path)
        await dispatcher.start_polling(bot, allowed_updates=ALLOWED_UPDATES)
        return 0
    finally:
        await bot.session.close()


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram Business personal assistant (Claude Code CLI).")
    parser.add_argument("--settings-from-env", action="store_true", help="read settings from environment variables")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-claude", action="store_true", help="check the Claude CLI setup and exit")
    group.add_argument("--try", dest="try_text", metavar="TEXT", help="draft a reply to TEXT and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)

    require_telegram = not (args.check_claude or args.try_text is not None)
    try:
        if args.settings_from_env:
            settings = settings_from_process_env(require_telegram=require_telegram)
        else:
            settings = load_settings(ENV_FILE, require_telegram=require_telegram)
            warning = env_file_warning(ENV_FILE)
            if warning:
                log.warning(warning)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    ensure_private_dir(settings.data_dir)
    ensure_private_dir(settings.data_dir / "claude-run")

    if args.check_claude:
        return asyncio.run(check_claude(settings))
    if args.try_text is not None:
        return asyncio.run(try_reply(settings, args.try_text))
    try:
        return asyncio.run(run_bot(settings))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
