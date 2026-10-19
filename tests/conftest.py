"""Offline test harness: a real aiogram Dispatcher fed with fake updates, a recording session, a fake Claude."""

from __future__ import annotations

import asyncio
import itertools
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import GetBusinessConnection, SendMessage
from aiogram.types import BusinessConnection, Chat, Message, Update

from assistant.config import load_settings
from assistant.handlers import build_router
from assistant.service import Assistant, Outcome, Preview

OWNER_ID = 1001
OTHER_ID = 2002
CONN_ID = "conn-1"
BOT_TOKEN = "123456:TEST-token"


class RecordingSession(BaseSession):
    """Records every Bot API call; never touches the network."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: List[Any] = []
        self.connections: Dict[str, BusinessConnection] = {}
        self._message_ids = itertools.count(10_000)

    async def close(self) -> None:
        pass

    async def make_request(self, bot: Bot, method: Any, timeout: Optional[int] = None) -> Any:
        self.requests.append(method)
        if isinstance(method, GetBusinessConnection):
            connection = self.connections.get(method.business_connection_id)
            if connection is None:
                raise TelegramBadRequest(method=method, message="Bad Request: business connection not found")
            return connection
        if isinstance(method, SendMessage):
            return Message(
                message_id=next(self._message_ids),
                date=datetime.now(tz=timezone.utc),
                chat=Chat(id=method.chat_id, type="private"),
                text=method.text,
                business_connection_id=method.business_connection_id,
            )
        return True

    async def stream_content(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("tests must not download files")

    def sent(self) -> List[SendMessage]:
        return [r for r in self.requests if isinstance(r, SendMessage)]

    def lookups(self) -> List[GetBusinessConnection]:
        return [r for r in self.requests if isinstance(r, GetBusinessConnection)]


class FakeGenerator:
    """Returns (or raises) the queued responses in order and records every call."""

    def __init__(self, *responses: Any) -> None:
        self.responses = deque(responses)
        self.calls: List[Dict[str, str]] = []
        self.hook: Optional[Callable[[], Awaitable[None]]] = None

    def queue(self, *responses: Any) -> None:
        self.responses.extend(responses)

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        await asyncio.sleep(0)
        if self.hook is not None:
            await self.hook()
        item = self.responses.popleft() if self.responses else "Default reply"
        if isinstance(item, BaseException):
            raise item
        return item


class Clock:
    def __init__(self, now: float = 1_760_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Harness:
    def __init__(self, tmp_path: Path, **overrides: Any) -> None:
        self.tmp_path = tmp_path
        self.data_dir = tmp_path / "data"
        self.profile_path = tmp_path / "profile.md"
        self.profile_path.write_text(
            "<!-- comment that must not reach Claude -->\n# Men haqimda\n- Ism: Test Owner\n", encoding="utf-8"
        )
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")
        values = {
            "TELEGRAM_BOT_TOKEN": BOT_TOKEN,
            "OWNER_ID": str(OWNER_ID),
            "AUTO_SEND": "false",
            "CLAUDE_BIN": str(tmp_path / "no-claude-here"),
            "DATA_DIR": str(self.data_dir),
            "PROFILE_PATH": str(self.profile_path),
        }
        values.update({key: str(value) for key, value in overrides.items()})
        self.settings = load_settings(env_file, overrides=values)
        self.clock = Clock()
        self.generator = FakeGenerator()
        self.session = RecordingSession()
        self.bot = Bot(BOT_TOKEN, session=self.session)
        self.previews: List[Preview] = []
        self._update_ids = itertools.count(1)
        self._message_ids = itertools.count(1)
        self._build()

    def _build(self) -> None:
        self.assistant = Assistant(
            self.settings, self.generator, clock=self.clock, preview_sink=self.previews.append
        )
        self.dp = Dispatcher()
        self.dp.include_router(build_router(self.assistant))

    def restart(self) -> None:
        """A fresh Assistant on the same data directory (simulates a process restart)."""
        self._build()

    # ----- feeding updates ---------------------------------------------------

    async def feed(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload, update_id=next(self._update_ids))
        update = Update.model_validate(payload, context={"bot": self.bot})
        await self.dp.feed_update(self.bot, update)

    @property
    def last_outcome(self) -> Optional[Outcome]:
        return self.assistant.outcomes[-1] if self.assistant.outcomes else None

    def connection_payload(
        self, *, user_id: int = OWNER_ID, enabled: bool = True, can_reply: bool = True, connection_id: str = CONN_ID
    ) -> Dict[str, Any]:
        return {
            "id": connection_id,
            "user": {"id": user_id, "is_bot": False, "first_name": "Owner"},
            "user_chat_id": user_id,
            "date": int(self.clock()),
            "is_enabled": enabled,
            # Both the current `rights` object and the legacy flag, for older aiogram versions.
            "rights": {"can_reply": can_reply},
            "can_reply": can_reply,
        }

    def set_connection(self, **kwargs: Any) -> None:
        """Change what Telegram answers to getBusinessConnection, without sending an update."""
        payload = self.connection_payload(**kwargs)
        self.session.connections[payload["id"]] = BusinessConnection.model_validate(payload)

    async def connect(self, **kwargs: Any) -> None:
        self.set_connection(**kwargs)
        await self.feed({"business_connection": self.connection_payload(**kwargs)})

    async def business_message(
        self,
        text: Optional[str] = "Salom, qalesan?",
        *,
        from_id: int = OTHER_ID,
        chat_id: Optional[int] = None,
        message_id: Optional[int] = None,
        connection_id: Optional[str] = CONN_ID,
        is_bot: bool = False,
        via_business_bot: bool = False,
        first_name: str = "Ali",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Outcome]:
        payload = self.business_message_payload(
            text,
            from_id=from_id,
            chat_id=chat_id,
            message_id=message_id,
            connection_id=connection_id,
            is_bot=is_bot,
            via_business_bot=via_business_bot,
            first_name=first_name,
            extra=extra,
        )
        await self.feed({"business_message": payload})
        return self.last_outcome

    def business_message_payload(
        self,
        text: Optional[str] = "Salom",
        *,
        from_id: int = OTHER_ID,
        chat_id: Optional[int] = None,
        message_id: Optional[int] = None,
        connection_id: Optional[str] = CONN_ID,
        is_bot: bool = False,
        via_business_bot: bool = False,
        first_name: str = "Ali",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chat = chat_id if chat_id is not None else (OTHER_ID if from_id == OWNER_ID else from_id)
        message: Dict[str, Any] = {
            "message_id": message_id if message_id is not None else next(self._message_ids),
            "date": int(self.clock()),
            "chat": {"id": chat, "type": "private", "first_name": first_name},
            "from": {"id": from_id, "is_bot": is_bot, "first_name": "Owner" if from_id == OWNER_ID else first_name},
        }
        if connection_id is not None:
            message["business_connection_id"] = connection_id
        if text is not None:
            message["text"] = text
        if via_business_bot:
            message["sender_business_bot"] = {"id": 999, "is_bot": True, "first_name": "Helper", "username": "helper_bot"}
        if extra:
            message.update(extra)
        return message

    async def direct_message(self, text: str, *, from_id: int = OWNER_ID) -> None:
        message: Dict[str, Any] = {
            "message_id": next(self._message_ids),
            "date": int(self.clock()),
            "chat": {"id": from_id, "type": "private", "first_name": "Someone"},
            "from": {"id": from_id, "is_bot": False, "first_name": "Someone"},
            "text": text,
        }
        if text.startswith("/"):
            command = text.split()[0]
            message["entities"] = [{"type": "bot_command", "offset": 0, "length": len(command)}]
        await self.feed({"message": message})

    # ----- reading results ---------------------------------------------------

    def history(self, chat_id: int = OTHER_ID, connection_id: str = CONN_ID) -> List[Dict[str, Any]]:
        return self.assistant.history.get(connection_id, chat_id)


@pytest.fixture
def make_harness(tmp_path: Path) -> Callable[..., Harness]:
    def factory(**overrides: Any) -> Harness:
        return Harness(tmp_path, **overrides)

    return factory
