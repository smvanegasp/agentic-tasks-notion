"""Telegram send-side helpers: message splitting + plain-text fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def test_split_message_short_passes_through():
    from agentic_tasks.telegram_io.dispatcher import _split_message

    text = "Hello, world."
    assert _split_message(text) == [text]


def test_split_message_chunks_at_paragraph_boundaries():
    from agentic_tasks.telegram_io.dispatcher import _split_message

    para = "x" * 1500
    text = "\n\n".join([para, para, para, para])  # ~6000+ chars
    chunks = _split_message(text, max_len=3500)

    assert len(chunks) >= 2
    assert all(len(c) <= 3500 for c in chunks)
    # Content preserved when re-joined.
    assert "\n\n".join(chunks) == text


def test_split_message_keeps_paragraphs_intact():
    """A paragraph that fits should not be sliced mid-line."""
    from agentic_tasks.telegram_io.dispatcher import _split_message

    text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
    chunks = _split_message(text, max_len=150)
    # First chunk should be exactly the first paragraph (100 chars), since
    # adding the second paragraph would exceed 150.
    assert chunks[0] == "A" * 100


def test_split_message_hard_splits_oversize_paragraph():
    """A single paragraph longer than max_len falls back to a hard slice."""
    from agentic_tasks.telegram_io.dispatcher import _split_message

    text = "x" * 5000
    chunks = _split_message(text, max_len=2000)
    assert len(chunks) == 3
    assert all(len(c) <= 2000 for c in chunks)
    assert "".join(chunks) == text


def test_strip_html_removes_tags_and_unescapes_entities():
    from agentic_tasks.telegram_io.dispatcher import _strip_html

    assert _strip_html("<b>hello</b> &amp; goodbye") == "hello & goodbye"
    assert _strip_html("<a href=\"x\">click</a>") == "click"
    assert _strip_html("plain text") == "plain text"


@pytest.mark.asyncio
async def test_send_splits_long_message_into_multiple_telegram_calls():
    from agentic_tasks.telegram_io.dispatcher import _TELEGRAM_MAX_CHARS, _send

    bot = MagicMock()
    bot.send_message = AsyncMock()

    big = ("x" * 1000 + "\n\n") * 6  # ~6000 chars across 6 paragraphs
    await _send(bot, 1, big)

    assert bot.send_message.await_count >= 2
    # Each chunk under the cap.
    for call in bot.send_message.await_args_list:
        assert len(call.kwargs["text"]) <= _TELEGRAM_MAX_CHARS


@pytest.mark.asyncio
async def test_send_short_message_uses_one_call():
    from agentic_tasks.telegram_io.dispatcher import _send

    bot = MagicMock()
    bot.send_message = AsyncMock()

    await _send(bot, 1, "<b>hi</b>")

    assert bot.send_message.await_count == 1
    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["text"] == "<b>hi</b>"
    assert kwargs["parse_mode"] == "HTML"
