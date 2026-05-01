"""Per-chat conversation history with day-boundary reset.

Two backends:

- ``InMemoryConversationStore`` — for local development (the long-polling
  runner keeps the process alive, so a process-local dict is fine).
- ``DynamoDBConversationStore`` — for Lambda, where containers come and go
  between invocations and we need durable state.

``get_store()`` picks the backend based on the environment: if the
``CONVERSATION_TABLE_NAME`` env var is set (Lambda), DynamoDB is used; else
in-memory.

Reset rules (both backends):
- Day boundary in the user's configured timezone — when the first message of
  a new day comes in, all prior history is dropped.
- Cap at ``CONVERSATION_HISTORY_LIMIT`` messages. When exceeded, oldest
  messages are evicted first.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from agentic_tasks.config import get_settings

log = logging.getLogger(__name__)


class ConversationStore(Protocol):
    def get_history(self, chat_id: int) -> list[dict[str, Any]]: ...
    def append(self, chat_id: int, message: dict[str, Any]) -> None: ...
    def reset(self, chat_id: int) -> None: ...


@dataclass
class _StoredTurn:
    timestamp: datetime
    message: dict[str, Any]


class InMemoryConversationStore:
    """In-process conversation history per chat id."""

    def __init__(
        self,
        *,
        max_messages: int,
        tz: ZoneInfo,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._max_messages = max_messages
        self._tz = tz
        self._clock = clock or (lambda: datetime.now(tz))
        self._data: dict[int, list[_StoredTurn]] = defaultdict(list)

    def get_history(self, chat_id: int) -> list[dict[str, Any]]:
        self._reset_if_new_day(chat_id)
        return [t.message for t in self._data[chat_id]]

    def append(self, chat_id: int, message: dict[str, Any]) -> None:
        self._reset_if_new_day(chat_id)
        self._data[chat_id].append(_StoredTurn(timestamp=self._clock(), message=message))
        if len(self._data[chat_id]) > self._max_messages:
            self._data[chat_id] = self._data[chat_id][-self._max_messages :]

    def reset(self, chat_id: int) -> None:
        self._data[chat_id] = []

    def _reset_if_new_day(self, chat_id: int) -> None:
        turns = self._data.get(chat_id)
        if not turns:
            return
        today = self._clock().date()
        if turns[-1].timestamp.date() != today:
            self._data[chat_id] = []


class DynamoDBConversationStore:
    """DynamoDB-backed history. One item per chat_id (PK = chat_id, type N).

    Each item carries the full message list, the date it was last touched
    (for day-boundary reset), and a TTL so old chats are auto-pruned.
    """

    def __init__(
        self,
        *,
        table_name: str,
        max_messages: int,
        tz: ZoneInfo,
    ) -> None:
        import boto3  # lazy: only required when this backend is in use

        self._max_messages = max_messages
        self._tz = tz
        self._table = boto3.resource("dynamodb").Table(table_name)

    def _today_iso(self) -> str:
        return datetime.now(self._tz).date().isoformat()

    def get_history(self, chat_id: int) -> list[dict[str, Any]]:
        response = self._table.get_item(Key={"chat_id": chat_id})
        item = response.get("Item")
        if not item:
            return []
        if item.get("last_updated_date") != self._today_iso():
            # Stale — return empty; the next append will overwrite with today's data.
            return []
        return list(item.get("messages") or [])

    def append(self, chat_id: int, message: dict[str, Any]) -> None:
        history = self.get_history(chat_id)
        history.append(message)
        if len(history) > self._max_messages:
            history = history[-self._max_messages :]
        ttl = int((datetime.now(self._tz) + timedelta(days=2)).timestamp())
        self._table.put_item(
            Item={
                "chat_id": chat_id,
                "messages": history,
                "last_updated_date": self._today_iso(),
                "ttl": ttl,
            }
        )

    def reset(self, chat_id: int) -> None:
        self._table.delete_item(Key={"chat_id": chat_id})


@lru_cache(maxsize=1)
def get_store() -> ConversationStore:
    s = get_settings()
    table_name = os.environ.get("CONVERSATION_TABLE_NAME")
    if table_name:
        log.info("using DynamoDB conversation store: %s", table_name)
        return DynamoDBConversationStore(
            table_name=table_name,
            max_messages=s.conversation_history_limit,
            tz=s.timezone,
        )
    return InMemoryConversationStore(
        max_messages=s.conversation_history_limit,
        tz=s.timezone,
    )
