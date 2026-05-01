from datetime import datetime
from zoneinfo import ZoneInfo

from agentic_tasks.conversation.store import InMemoryConversationStore

TZ = ZoneInfo("America/New_York")


def _msg(text: str) -> dict:
    return {"role": "user", "content": text}


def _store(*, max_messages: int = 10, clock_times: list[datetime] | None = None):
    """Build a store with a deterministic clock that walks through ``clock_times``."""
    if clock_times is None:
        clock_times = [datetime(2026, 4, 30, 10, tzinfo=TZ)]
    state = {"i": 0}

    def clock() -> datetime:
        i = min(state["i"], len(clock_times) - 1)
        state["i"] += 1
        return clock_times[i]

    return InMemoryConversationStore(max_messages=max_messages, tz=TZ, clock=clock)


def test_append_and_get():
    store = _store()
    store.append(123, _msg("a"))
    store.append(123, _msg("b"))

    assert [m["content"] for m in store.get_history(123)] == ["a", "b"]


def test_isolated_per_chat_id():
    store = _store()
    store.append(1, _msg("a"))
    store.append(2, _msg("b"))

    assert [m["content"] for m in store.get_history(1)] == ["a"]
    assert [m["content"] for m in store.get_history(2)] == ["b"]


def test_caps_at_max_messages_keeping_most_recent():
    times = [datetime(2026, 4, 30, 10, tzinfo=TZ)] * 10
    store = _store(max_messages=3, clock_times=times)
    for i in range(5):
        store.append(1, _msg(str(i)))

    assert [m["content"] for m in store.get_history(1)] == ["2", "3", "4"]


def test_get_history_resets_on_new_day():
    times = [
        datetime(2026, 4, 30, 23, 50, tzinfo=TZ),  # append
        datetime(2026, 5, 1, 9, 0, tzinfo=TZ),  # get_history (different day)
    ]
    store = _store(clock_times=times)
    store.append(1, _msg("yesterday"))

    assert store.get_history(1) == []


def test_append_after_new_day_clears_old_history():
    times = [
        datetime(2026, 4, 30, 23, 50, tzinfo=TZ),  # first append
        datetime(2026, 5, 1, 9, 0, tzinfo=TZ),  # second append
        datetime(2026, 5, 1, 9, 0, tzinfo=TZ),  # get_history
    ]
    store = _store(clock_times=times)
    store.append(1, _msg("yesterday"))
    store.append(1, _msg("today"))

    assert [m["content"] for m in store.get_history(1)] == ["today"]


def test_reset_clears_history():
    store = _store()
    store.append(1, _msg("a"))
    store.reset(1)

    assert store.get_history(1) == []


def test_empty_chat_returns_empty_list():
    store = _store()
    assert store.get_history(999) == []
