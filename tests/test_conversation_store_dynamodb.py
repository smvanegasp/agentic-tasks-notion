"""DynamoDB ConversationStore — covered with a hand-rolled fake table."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/New_York")


class _FakeTable:
    """In-memory stand-in for boto3's DynamoDB Table resource."""

    def __init__(self) -> None:
        self.items: dict[Any, dict] = {}

    def get_item(self, *, Key):
        item = self.items.get(Key["chat_id"])
        return {"Item": item} if item else {}

    def put_item(self, *, Item):
        self.items[Item["chat_id"]] = Item

    def delete_item(self, *, Key):
        self.items.pop(Key["chat_id"], None)


def _build_store(today_iso: str = "2026-04-30"):
    """Construct a DynamoDBConversationStore with a fake boto3 client."""
    fake_table = _FakeTable()
    fake_resource = MagicMock()
    fake_resource.Table.return_value = fake_table

    with patch("boto3.resource", return_value=fake_resource):
        from agentic_tasks.conversation.store import DynamoDBConversationStore

        store = DynamoDBConversationStore(
            table_name="test-table",
            max_messages=5,
            tz=TZ,
        )

    # Pin the date the store sees as "today" so we can test boundaries
    # deterministically.
    store._today_iso = lambda: today_iso  # type: ignore[method-assign]
    return store, fake_table


def test_append_and_get():
    store, _ = _build_store()
    store.append(1, {"role": "user", "content": "hi"})
    store.append(1, {"role": "assistant", "content": "hello"})

    history = store.get_history(1)
    assert [m["content"] for m in history] == ["hi", "hello"]


def test_isolated_per_chat():
    store, _ = _build_store()
    store.append(1, {"role": "user", "content": "a"})
    store.append(2, {"role": "user", "content": "b"})

    assert [m["content"] for m in store.get_history(1)] == ["a"]
    assert [m["content"] for m in store.get_history(2)] == ["b"]


def test_caps_at_max_messages():
    store, _ = _build_store()
    for i in range(8):
        store.append(1, {"role": "user", "content": str(i)})

    history = store.get_history(1)
    assert [m["content"] for m in history] == ["3", "4", "5", "6", "7"]


def test_returns_empty_when_history_is_from_different_day():
    store, fake_table = _build_store(today_iso="2026-05-01")
    fake_table.items[1] = {
        "chat_id": 1,
        "messages": [{"role": "user", "content": "yesterday"}],
        "last_updated_date": "2026-04-30",
        "ttl": 99999999999,
    }

    assert store.get_history(1) == []


def test_append_after_new_day_overwrites():
    store, fake_table = _build_store(today_iso="2026-05-01")
    fake_table.items[1] = {
        "chat_id": 1,
        "messages": [{"role": "user", "content": "yesterday"}],
        "last_updated_date": "2026-04-30",
        "ttl": 99999999999,
    }

    store.append(1, {"role": "user", "content": "today"})

    history = store.get_history(1)
    assert [m["content"] for m in history] == ["today"]


def test_reset_deletes_item():
    store, fake_table = _build_store()
    store.append(1, {"role": "user", "content": "x"})
    assert 1 in fake_table.items

    store.reset(1)
    assert 1 not in fake_table.items


def test_appended_item_has_ttl_in_the_future():
    store, fake_table = _build_store()
    store.append(1, {"role": "user", "content": "x"})

    item = fake_table.items[1]
    assert "ttl" in item
    now_ts = int(datetime.now(TZ).timestamp())
    assert item["ttl"] > now_ts


def test_factory_picks_dynamodb_when_table_env_set(monkeypatch):
    """``get_store`` should use the DynamoDB backend when the table-name env
    var is present (i.e. running in Lambda)."""
    monkeypatch.setenv("CONVERSATION_TABLE_NAME", "test-table")

    from agentic_tasks.conversation.store import (
        DynamoDBConversationStore,
        get_store,
    )

    get_store.cache_clear()

    fake_resource = MagicMock()
    fake_resource.Table.return_value = _FakeTable()
    with patch("boto3.resource", return_value=fake_resource):
        store = get_store()

    assert isinstance(store, DynamoDBConversationStore)


def test_factory_uses_in_memory_when_no_table_env(monkeypatch):
    monkeypatch.delenv("CONVERSATION_TABLE_NAME", raising=False)

    from agentic_tasks.conversation.store import (
        InMemoryConversationStore,
        get_store,
    )

    get_store.cache_clear()
    store = get_store()
    assert isinstance(store, InMemoryConversationStore)
