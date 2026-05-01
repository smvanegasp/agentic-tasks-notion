"""AWS Lambda entry point for the evening reflection digest.

EventBridge Scheduler invokes this once a day at the configured evening
time. The handler queries Notion for today's still-open tasks (due today
or flagged My Day, status != Done) and sends a plain reflection message
listing what's left.

No LLM call — the rendering is deterministic. The user can text the bot
to act on anything (mark done, reschedule, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from telegram import Bot

from agentic_tasks._aws import bootstrap_secrets, setup_lambda_logging
from agentic_tasks.config import get_settings
from agentic_tasks.digest.evening import render_evening_digest
from agentic_tasks.notion_io.tasks import query_today

log = logging.getLogger()
log.setLevel(logging.INFO)

setup_lambda_logging()
bootstrap_secrets()


def build_evening_message() -> str:
    """Pull today's open tasks and render them as Telegram-HTML."""
    s = get_settings()
    today = datetime.now(s.timezone).date()
    tasks = query_today(today=today)
    return render_evening_digest(today, tasks)


async def _send_evening_message(text: str) -> None:
    s = get_settings()
    bot = Bot(token=s.telegram_bot_token)
    async with bot:
        await bot.send_message(
            chat_id=s.allowed_telegram_user_id,
            text=text,
            parse_mode="HTML",
        )


async def _run_evening() -> None:
    text = build_evening_message()
    log.info("evening digest length=%d", len(text))
    await _send_evening_message(text)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda handler invoked by EventBridge Scheduler."""
    log.info("evening digest triggered")
    try:
        asyncio.run(_run_evening())
    except Exception:
        log.exception("evening digest failed")
        return {"statusCode": 500, "body": "error"}
    return {"statusCode": 200, "body": "ok"}
