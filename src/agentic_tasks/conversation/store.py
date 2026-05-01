"""Per-chat conversation history with day-boundary reset.

The Telegram dispatcher (and the local REPL) stores user/assistant/tool
messages here so the agent sees the recent conversation context. When you
refer to "that task" or "the second one", the LLM can resolve the page_id from
earlier tool results in this history instead of re-querying.

Reset rules:
- Day boundary in the user's configured timezone — when the first message of a
  new day comes in, all prior history is dropped.
- Cap at ``CONVERSATION_HISTORY_LIMIT`` messages (default 40, configurable via
  env). When exceeded, oldest messages are evicted first.

The default backend is in-process. That works for the local long-polling
runner (the process stays alive). A DynamoDB-backed implementation is added
in the AWS step, where Lambda containers don't persist between invocations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo

from agentic_tasks.config import get_settings


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


@lru_cache(maxsize=1)
def get_store() -> InMemoryConversationStore:
    s = get_settings()
    return InMemoryConversationStore(
        max_messages=s.conversation_history_limit,
        tz=s.timezone,
    )
