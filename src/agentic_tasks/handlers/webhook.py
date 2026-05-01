"""AWS Lambda entry point for the Telegram webhook.

API Gateway → Lambda. Body is a Telegram Update JSON. We parse it, run the
agent, and reply via the Telegram Bot API. Always returns 200 so Telegram does
not retry — failures are logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from telegram import Bot, Update

from agentic_tasks._aws import bootstrap_secrets, setup_lambda_logging
from agentic_tasks.config import get_settings
from agentic_tasks.telegram_io.dispatcher import process_update

log = logging.getLogger()
log.setLevel(logging.INFO)

# Cold-start setup: switch logs to JSON in Lambda, then pull secrets.
setup_lambda_logging()
bootstrap_secrets()


def _verify_secret(headers: dict[str, str] | None) -> bool:
    """If a webhook secret is configured, require it as a header."""
    expected = get_settings().telegram_webhook_secret
    if not expected:
        return True
    lower = {k.lower(): v for k, v in (headers or {}).items()}
    return lower.get("x-telegram-bot-api-secret-token") == expected


async def _process(body: str) -> None:
    s = get_settings()
    bot = Bot(token=s.telegram_bot_token)
    payload = json.loads(body)
    update = Update.de_json(payload, bot)
    if update is None:
        return
    async with bot:
        await process_update(update, bot)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda handler invoked by API Gateway on each Telegram webhook POST."""
    if not _verify_secret(event.get("headers")):
        log.warning("webhook secret mismatch")
        return {"statusCode": 401, "body": "unauthorized"}

    body = event.get("body") or ""
    try:
        asyncio.run(_process(body))
    except Exception:
        log.exception("webhook processing failed")
        return {"statusCode": 200, "body": "error logged"}

    return {"statusCode": 200, "body": "ok"}
