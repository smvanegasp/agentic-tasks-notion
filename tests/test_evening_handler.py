"""Evening digest Lambda handler."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_tasks.notion_io.tasks import Task


def _task(name: str, due=None) -> Task:
    return Task(
        page_id=f"id-{name}",
        name=name,
        status="To Do",
        priority=None,
        due=due,
    )


@patch("agentic_tasks.handlers.evening.query_today")
def test_build_evening_message_returns_text(mock_query):
    from agentic_tasks.handlers.evening import build_evening_message

    mock_query.return_value = [_task("Buy markers", due=date(2026, 5, 1))]
    text = build_evening_message()
    assert "Still open (1)" in text


@patch("agentic_tasks.handlers.evening.query_today")
@patch("agentic_tasks.handlers.evening.Bot")
def test_handler_sends_plain_message_without_keyboard(mock_bot_cls, mock_query):
    from agentic_tasks.handlers.evening import handler

    mock_query.return_value = [_task("Buy markers", due=date(2026, 5, 1))]
    bot = MagicMock()
    bot.send_message = AsyncMock()
    bot.__aenter__ = AsyncMock(return_value=bot)
    bot.__aexit__ = AsyncMock(return_value=None)
    mock_bot_cls.return_value = bot

    result = handler({}, None)

    assert result["statusCode"] == 200
    bot.send_message.assert_awaited_once()
    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["parse_mode"] == "HTML"
    assert "reply_markup" not in kwargs
    assert "Buy markers" in kwargs["text"]


@patch("agentic_tasks.handlers.evening.query_today")
@patch("agentic_tasks.handlers.evening.Bot")
def test_handler_returns_error_status_on_exception(mock_bot_cls, mock_query):
    from agentic_tasks.handlers.evening import handler

    mock_query.side_effect = RuntimeError("notion down")
    result = handler({}, None)
    assert result["statusCode"] == 500
