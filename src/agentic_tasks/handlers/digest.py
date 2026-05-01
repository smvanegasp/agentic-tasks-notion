"""AWS Lambda entry point for the daily morning digest.

EventBridge Scheduler invokes this once a day at the configured time.
The handler queries Notion (overdue, due today, My Day), renders a digest
message, and sends it to the configured Telegram user.

No LLM call, no agent loop — the digest is deterministic so it's predictable
and free to run.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from telegram import Bot

from agentic_tasks.config import get_settings
from agentic_tasks.digest.render import render_digest
from agentic_tasks.notion_io.tasks import (
    query_due_today,
    query_my_day,
    query_overdue,
)

log = logging.getLogger()
log.setLevel(logging.INFO)


def build_digest_message() -> str:
    """Pull tasks from Notion and produce the rendered digest text."""
    s = get_settings()
    today = datetime.now(s.timezone).date()
    return render_digest(
        today,
        today_tasks=query_due_today(today=today),
        my_day_tasks=query_my_day(),
        overdue_tasks=query_overdue(today=today),
    )


async def _send_digest_message(text: str) -> None:
    s = get_settings()
    bot = Bot(token=s.telegram_bot_token)
    async with bot:
        await bot.send_message(
            chat_id=s.allowed_telegram_user_id,
            text=text,
            parse_mode="HTML",
        )


async def _run_digest() -> None:
    text = build_digest_message()
    log.info("digest length=%d", len(text))
    await _send_digest_message(text)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda handler invoked by EventBridge Scheduler."""
    log.info("digest triggered")
    try:
        asyncio.run(_run_digest())
    except Exception:
        log.exception("digest failed")
        return {"statusCode": 500, "body": "error"}
    return {"statusCode": 200, "body": "ok"}
