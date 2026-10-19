"""Core logic: decide whether to answer a business message, draft a reply, preview or send it."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BusinessConnection, Message

from .claude_cli import ClaudeCLI, ClaudeError, ClaudeLimitError
from .config import Settings
from .language import detect_language
from .prompts import SKIP_TOKEN, build_system_prompt, build_user_prompt
from .store import HistoryStore, RuntimeState, SeenMessages, append_private_line, ensure_private_dir

log = logging.getLogger("assistant")

TELEGRAM_TEXT_LIMIT = 4000
ERROR_ALERT_INTERVAL = 600.0


class Outcome(str, Enum):
    PREVIEWED = "previewed"
    SENT = "sent"
    DUPLICATE = "duplicate"
    IGNORED_BOT = "ignored_bot"
    NO_CONNECTION = "no_connection"
    FOREIGN_CONNECTION = "foreign_connection"
    CONNECTION_DISABLED = "connection_disabled"
    UNSUPPORTED = "unsupported"
    OWNER_MESSAGE = "owner_message"
    OWNER_ACTIVE = "owner_active"
    PAUSED = "paused"
    LIMITED = "limited"
    SKIPPED = "skipped"
    PERMISSION_DENIED = "permission_denied"
    ERROR = "error"


@dataclass
class Preview:
    connection_id: str
    chat_id: int
    chat_name: str
    language: Optional[str]
    incoming: str
    reply: str
    note: str = ""


def print_preview(preview: Preview) -> None:
    lines = [
        "",
        "=" * 20 + " PREVIEW, not sent " + "=" * 20,
        f"Chat:     {preview.chat_name} (chat {preview.chat_id}, connection {preview.connection_id})",
        f"Language: {preview.language or 'unknown'}",
        f"Incoming: {preview.incoming}",
        f"Reply:    {preview.reply}",
    ]
    if preview.note:
        lines.append(f"Note:     {preview.note}")
    lines.append("=" * 59)
    log.info("\n".join(lines))


def describe(message: Message) -> Optional[str]:
    """Readable text for a message; ``None`` for service messages."""
    if message.text:
        return message.text
    caption = (message.caption or "").strip()

    def tagged(tag: str, extra: str = "") -> str:
        return " ".join(part for part in (f"[{tag}]", extra or caption) if part)

    if message.sticker:
        return tagged("sticker", message.sticker.emoji or "")
    if message.voice:
        return tagged("voice message")
    if message.video_note:
        return tagged("video message")
    if message.photo:
        return tagged("photo")
    if message.video:
        return tagged("video")
    if message.animation:
        return tagged("GIF")
    if message.audio:
        title = " - ".join(x for x in (message.audio.performer, message.audio.title) if x)
        return tagged("audio", caption or title)
    if message.document:
        return tagged("file", caption or (message.document.file_name or ""))
    if message.contact:
        return tagged("contact", message.contact.first_name)
    if message.venue:
        return tagged("location", message.venue.title)
    if message.location:
        return tagged("location")
    if message.poll:
        return tagged("poll", message.poll.question)
    if message.dice:
        return tagged("dice", message.dice.emoji)
    return None


def can_reply(connection: BusinessConnection) -> bool:
    rights = getattr(connection, "rights", None)
    if rights is not None:
        return bool(getattr(rights, "can_reply", False))
    return bool(getattr(connection, "can_reply", False))


_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_EMPTY_FIELD_RE = re.compile(r"[-*]\s*[^:]*:\s*")


def read_profile(path: Path) -> str:
    """Read the profile on every reply, without ``<!-- -->`` comments."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""
    return _COMMENT_RE.sub("", text).strip()


def profile_looks_empty(profile: str) -> bool:
    for line in profile.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or _EMPTY_FIELD_RE.fullmatch(line):
            continue
        return False
    return True


def split_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> List[str]:
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [""]


def _chat_name(message: Message) -> str:
    chat = message.chat
    return chat.full_name or chat.title or chat.username or str(chat.id)


def _format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class Assistant:
    def __init__(
        self,
        settings: Settings,
        generator: Any,
        *,
        clock: Callable[[], float] = time.time,
        preview_sink: Callable[[Preview], None] = print_preview,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.clock = clock
        self.preview_sink = preview_sink
        data_dir = ensure_private_dir(settings.data_dir)
        self.history = HistoryStore(data_dir / "history.json", settings.history_limit)
        self.seen = SeenMessages(data_dir / "seen.json")
        self.state = RuntimeState(data_dir / "state.json")
        self.previews_path = data_dir / "previews.jsonl"
        self.connections: Dict[str, BusinessConnection] = {}
        self.outcomes: Deque[Outcome] = deque(maxlen=100)
        self._takeover_until: Dict[Tuple[str, int], float] = {}
        self._locks: Dict[Tuple[str, int], asyncio.Lock] = {}
        self._last_error_alert: Optional[float] = None

    # ----- Telegram updates -------------------------------------------------

    def on_business_connection(self, connection: BusinessConnection) -> None:
        self.connections[connection.id] = connection
        if connection.user.id != self.settings.owner_id:
            log.warning(
                "Business connection %s belongs to another account (user=%s); it will be ignored.",
                connection.id,
                connection.user.id,
            )
            return
        log.info(
            "Business connection %s: user=%s enabled=%s can_reply=%s",
            connection.id,
            connection.user.id,
            connection.is_enabled,
            can_reply(connection),
        )

    async def on_business_message(self, bot: Bot, message: Message) -> Outcome:
        outcome = await self._handle(bot, message)
        self.outcomes.append(outcome)
        return outcome

    async def _handle(self, bot: Bot, message: Message) -> Outcome:
        conn_id = message.business_connection_id
        if not conn_id:
            return Outcome.NO_CONNECTION
        chat_id = message.chat.id
        # Marked before any await, so parallel deliveries of the same update are handled once.
        if not self.seen.first_time(f"{conn_id}:{chat_id}:{message.message_id}"):
            return Outcome.DUPLICATE
        if message.sender_business_bot or (message.from_user and message.from_user.is_bot):
            return Outcome.IGNORED_BOT

        connection = await self._connection(bot, conn_id)
        if connection is None:
            return Outcome.NO_CONNECTION
        if connection.user.id != self.settings.owner_id:
            return Outcome.FOREIGN_CONNECTION
        if not connection.is_enabled:
            return Outcome.CONNECTION_DISABLED

        text = describe(message)
        if text is None:
            return Outcome.UNSUPPORTED

        now = self.clock()
        sender = message.from_user
        if sender and sender.id == self.settings.owner_id:
            self.history.add(conn_id, chat_id, role="owner", name="You", text=text, ts=now)
            if self.settings.owner_takeover_minutes > 0:
                self._takeover_until[(conn_id, chat_id)] = now + self.settings.owner_takeover_minutes * 60
            return Outcome.OWNER_MESSAGE

        language = detect_language(text) or self.history.last_language(conn_id, chat_id)
        name = sender.full_name if sender else _chat_name(message)
        self.history.add(conn_id, chat_id, role="other", name=name, text=text, lang=language, ts=now)

        lock = self._locks.setdefault((conn_id, chat_id), asyncio.Lock())
        async with lock:
            blocked = self._blocked(conn_id, chat_id)
            if blocked:
                return blocked
            return await self._reply(bot, message, conn_id, chat_id, text, language)

    async def _reply(
        self, bot: Bot, message: Message, conn_id: str, chat_id: int, text: str, language: Optional[str]
    ) -> Outcome:
        now = self.clock()
        system_prompt = build_system_prompt(
            read_profile(self.settings.profile_path), datetime.fromtimestamp(now).astimezone()
        )
        user_prompt = build_user_prompt(self.history.get(conn_id, chat_id), language)
        try:
            reply = await self.generator.generate(system_prompt, user_prompt)
        except ClaudeLimitError as error:
            now = self.clock()
            if error.resets_at and error.resets_at > now:
                until = error.resets_at
            else:
                until = now + self.settings.limit_cooldown_minutes * 60
            self.state.limit_until = until
            log.warning("%s. Not calling Claude again until %s. CLI said: %s", error, _format_ts(until), error.detail)
            if self.settings.auto_send:
                await self._alert_owner(
                    bot, f"{error}. I will stay quiet until {_format_ts(until)} (or until /resume)."
                )
            return Outcome.LIMITED
        except ClaudeError as error:
            log.error("Claude failed for chat %s: %s", chat_id, error)
            if self.settings.auto_send:
                now = self.clock()
                if self._last_error_alert is None or now - self._last_error_alert >= ERROR_ALERT_INTERVAL:
                    self._last_error_alert = now
                    await self._alert_owner(bot, f"Could not draft a reply for {_chat_name(message)}: {error}")
            return Outcome.ERROR

        reply = (reply or "").strip()
        if not reply or SKIP_TOKEN in reply:
            log.info("Claude decided no reply is needed in chat %s.", chat_id)
            return Outcome.SKIPPED

        preview = Preview(conn_id, chat_id, _chat_name(message), language, text, reply)
        if not self.settings.auto_send:
            self._show(preview)
            return Outcome.PREVIEWED

        blocked = self._blocked(conn_id, chat_id)
        if blocked:
            reasons = {
                Outcome.OWNER_ACTIVE: "you wrote in this chat while the reply was being drafted",
                Outcome.PAUSED: "the bot was paused while the reply was being drafted",
                Outcome.LIMITED: "the Claude usage limit was hit while the reply was being drafted",
            }
            preview.note = f"not sent: {reasons[blocked]}"
            self._show(preview)
            return blocked

        problem = await self._send_problem(bot, conn_id)
        if problem:
            preview.note = f"not sent: {problem}"
            self._show(preview)
            return Outcome.PERMISSION_DENIED

        try:
            for chunk in split_text(reply):
                await bot.send_message(chat_id, chunk, business_connection_id=conn_id)
        except TelegramAPIError as error:
            log.error("Sending the reply to chat %s failed: %s", chat_id, error)
            return Outcome.ERROR
        self.history.add(conn_id, chat_id, role="owner", name="You", text=reply, ts=self.clock())
        log.info("Sent a reply to %s (chat %s).", preview.chat_name, chat_id)
        return Outcome.SENT

    # ----- helpers -----------------------------------------------------------

    async def _connection(self, bot: Bot, conn_id: str) -> Optional[BusinessConnection]:
        cached = self.connections.get(conn_id)
        if cached is not None:
            return cached
        try:
            connection = await bot.get_business_connection(business_connection_id=conn_id)
        except Exception as error:  # noqa: BLE001 - any failure means we cannot trust this connection
            log.warning("Could not look up business connection %s: %s", conn_id, error)
            return None
        self.connections[conn_id] = connection
        return connection

    async def _send_problem(self, bot: Bot, conn_id: str) -> Optional[str]:
        """Re-check the connection with Telegram (not the cache) right before sending."""
        try:
            connection = await bot.get_business_connection(business_connection_id=conn_id)
        except Exception as error:  # noqa: BLE001
            return f"could not re-check the business connection ({error})"
        self.connections[conn_id] = connection
        if connection.user.id != self.settings.owner_id:
            return "the business connection belongs to another account"
        if not connection.is_enabled:
            return "the business connection is disabled"
        if not can_reply(connection):
            return "the bot is not allowed to reply (Telegram Business -> Chatbots -> reply permission)"
        return None

    def _blocked(self, conn_id: str, chat_id: int) -> Optional[Outcome]:
        now = self.clock()
        if self.state.paused:
            return Outcome.PAUSED
        limit_until = self.state.limit_until
        if limit_until and limit_until > now:
            return Outcome.LIMITED
        if self._takeover_until.get((conn_id, chat_id), 0) > now:
            return Outcome.OWNER_ACTIVE
        return None

    def _show(self, preview: Preview) -> None:
        self.preview_sink(preview)
        record = dict(asdict(preview), ts=self.clock(), sent=False)
        try:
            append_private_line(self.previews_path, json.dumps(record, ensure_ascii=False))
        except OSError as error:
            log.warning("Could not write %s: %s", self.previews_path, error)

    async def _alert_owner(self, bot: Bot, text: str) -> None:
        try:
            await bot.send_message(self.settings.owner_id, text)
        except Exception as error:  # noqa: BLE001
            log.warning("Could not notify the owner: %s", error)

    # ----- owner controls ------------------------------------------------------

    def pause(self) -> None:
        self.state.paused = True
        log.info("Paused by the owner.")

    def resume(self) -> None:
        self.state.paused = False
        self.state.limit_until = None
        self._takeover_until.clear()
        log.info("Resumed by the owner.")

    def _limit_active(self) -> Optional[float]:
        until = self.state.limit_until
        return until if until and until > self.clock() else None

    def status_text(self) -> str:
        mode = (
            "AUTO-SEND (replies are sent to chats)"
            if self.settings.auto_send
            else "PREVIEW (replies are only shown in the logs)"
        )
        limit = self._limit_active()
        return "\n".join(
            [
                f"Mode: {mode}",
                f"State: {'paused' if self.state.paused else 'running'}",
                f"Claude limit: {'until ' + _format_ts(limit) if limit else 'none'}",
                f"Chats in history: {self.history.chat_count()}",
            ]
        )

    def start_text(self) -> str:
        if self.settings.auto_send:
            mode = "AUTO-SEND — javoblar chatlarga avtomatik yuboriladi"
        else:
            mode = "PREVIEW — javoblar yuborilmaydi, faqat logda ko'rinadi"
        state = "pauzada (/resume bilan davom ettiring)" if self.state.paused else "ishlayapti"
        minutes = self.settings.owner_takeover_minutes
        if minutes > 0:
            takeover = f"• Biror chatda o'zingiz yozsangiz, men u chatda {minutes} daqiqa jim turaman."
        else:
            takeover = "• O'zingiz yozganingizda jim turish o'chirilgan (OWNER_TAKEOVER_MINUTES=0)."
        lines = [
            "Salom! Men sizning shaxsiy yordamchingizman.",
            "Kimdir sizga Telegram'da yozsa, profile.md dagi ma'lumotlar asosida sizning uslubingizda "
            "javob tayyorlayman.",
            "",
            "Hozirgi holat:",
            f"• Rejim: {mode}",
            f"• Holat: {state}",
        ]
        limit = self._limit_active()
        if limit:
            lines.append(f"• Claude limiti tugagan: {_format_ts(limit)} gacha kutaman")
        lines += [
            "",
            "Qanday ulanadi:",
            "1. Telegram → Settings → Telegram Business → Chatbots",
            "2. Shu botning username'ini kiriting",
            "3. Bot qaysi chatlarni ko'rishini tanlang",
            "4. Javob berish (reply) ruxsatini yoqing",
            "Buning uchun Telegram Premium kerak.",
            "",
            "Buyruqlar:",
            "/pause — javob berishni to'xtatish",
            "/resume — davom ettirish (limit va jim turish taymerlarini ham tozalaydi)",
            "/status — holat",
            "",
            "Bilib qo'ying:",
            takeover,
            "• Pul, qarz, uchrashuv va rejalar bo'yicha hech narsaga rozilik bermayman va va'da qilmayman.",
            "• profile.md ni to'ldiring: men faqat undagi ma'lumotlarni bilaman.",
        ]
        if not self.settings.auto_send:
            lines.append("• Javoblar chatga yuborilishi uchun AUTO_SEND=true qiling.")
        return "\n".join(lines)


def make_claude_cli(settings: Settings) -> ClaudeCLI:
    return ClaudeCLI(
        settings.claude_bin,
        workdir=settings.data_dir / "claude-run",
        model=settings.claude_model,
        effort=settings.claude_effort,
        timeout=settings.claude_timeout,
        oauth_token=settings.claude_oauth_token,
        auth_label=settings.claude_auth_mode,
    )


def build_assistant(settings: Settings, **kwargs: Any) -> Assistant:
    return Assistant(settings, make_claude_cli(settings), **kwargs)
