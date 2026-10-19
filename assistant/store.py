"""Small JSON-file stores kept in the private data directory."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) and restrict it to the current user (0700)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def append_private_line(path: Path, line: str) -> None:
    """Append one line to a file that is created with mode 0600."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")


class JsonFile:
    """A JSON document written atomically (0600 temp file + ``os.replace``)."""

    def __init__(self, path: Path, default: Any) -> None:
        self.path = Path(path)
        self.default = default

    def read(self) -> Any:
        try:
            with open(self.path, encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return copy.deepcopy(self.default)

    def write(self, data: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            os.chmod(tmp, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


class HistoryStore:
    """Recent messages, kept separately for every business connection and chat."""

    def __init__(self, path: Path, limit: int) -> None:
        self._file = JsonFile(path, {})
        self.limit = limit
        data = self._file.read()
        self._data: Dict[str, List[Dict[str, Any]]] = data if isinstance(data, dict) else {}

    @staticmethod
    def key(connection_id: str, chat_id: int) -> str:
        return f"{connection_id}:{chat_id}"

    def get(self, connection_id: str, chat_id: int) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._data.get(self.key(connection_id, chat_id), [])]

    def add(
        self,
        connection_id: str,
        chat_id: int,
        *,
        role: str,
        name: str,
        text: str,
        lang: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> None:
        items = self._data.setdefault(self.key(connection_id, chat_id), [])
        items.append(
            {"role": role, "name": name, "text": text, "lang": lang, "ts": time.time() if ts is None else ts}
        )
        del items[: max(0, len(items) - self.limit)]
        self._file.write(self._data)

    def last_language(self, connection_id: str, chat_id: int) -> Optional[str]:
        items = self._data.get(self.key(connection_id, chat_id), [])
        for item in reversed(items):
            if item.get("role") == "other" and item.get("lang"):
                return item["lang"]
        return None

    def chat_count(self) -> int:
        return len(self._data)


class SeenMessages:
    """Remembers recently handled message keys, across restarts."""

    def __init__(self, path: Path, capacity: int = 5000) -> None:
        self._file = JsonFile(path, [])
        self.capacity = capacity
        data = self._file.read()
        keys = data if isinstance(data, list) else []
        self._keys: "OrderedDict[str, None]" = OrderedDict((str(key), None) for key in keys[-capacity:])

    def first_time(self, key: str) -> bool:
        if key in self._keys:
            return False
        self._keys[key] = None
        while len(self._keys) > self.capacity:
            self._keys.popitem(last=False)
        self._file.write(list(self._keys))
        return True


class RuntimeState:
    """``paused`` and ``limit_until``; both survive restarts."""

    def __init__(self, path: Path) -> None:
        self._file = JsonFile(path, {})
        data = self._file.read()
        self._data: Dict[str, Any] = data if isinstance(data, dict) else {}

    @property
    def paused(self) -> bool:
        return bool(self._data.get("paused", False))

    @paused.setter
    def paused(self, value: bool) -> None:
        self._data["paused"] = bool(value)
        self._file.write(self._data)

    @property
    def limit_until(self) -> Optional[float]:
        value = self._data.get("limit_until")
        return float(value) if isinstance(value, (int, float)) else None

    @limit_until.setter
    def limit_until(self, value: Optional[float]) -> None:
        self._data["limit_until"] = value
        self._file.write(self._data)
