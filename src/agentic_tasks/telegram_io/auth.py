"""Single-user authorization for the Telegram bot."""

from __future__ import annotations

from agentic_tasks.config import get_settings


def is_authorized(chat_id: int) -> bool:
    """The bot only serves the configured user. Everyone else is rejected."""
    return chat_id == get_settings().allowed_telegram_user_id
