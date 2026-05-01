import json
from unittest.mock import AsyncMock, patch


def test_handler_returns_200_on_valid_event():
    from agentic_tasks.handlers.webhook import handler

    event = {"body": json.dumps({"update_id": 1}), "headers": {}}

    with patch("agentic_tasks.handlers.webhook._process", new_callable=AsyncMock):
        result = handler(event, None)

    assert result["statusCode"] == 200


def test_handler_returns_200_on_processing_error():
    """Returning non-200 would make Telegram retry — we'd rather log and move on."""
    from agentic_tasks.handlers.webhook import handler

    event = {"body": json.dumps({"update_id": 1}), "headers": {}}

    with patch(
        "agentic_tasks.handlers.webhook._process",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = handler(event, None)

    assert result["statusCode"] == 200


def test_handler_rejects_bad_secret(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "real-secret")
    from agentic_tasks.config import get_settings

    get_settings.cache_clear()

    from agentic_tasks.handlers.webhook import handler

    event = {
        "body": "{}",
        "headers": {"X-Telegram-Bot-Api-Secret-Token": "wrong"},
    }
    result = handler(event, None)

    assert result["statusCode"] == 401


def test_handler_accepts_correct_secret_case_insensitive(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "real-secret")
    from agentic_tasks.config import get_settings

    get_settings.cache_clear()

    from agentic_tasks.handlers.webhook import handler

    event = {
        "body": json.dumps({"update_id": 1}),
        "headers": {"x-telegram-bot-api-secret-token": "real-secret"},
    }

    with patch("agentic_tasks.handlers.webhook._process", new_callable=AsyncMock):
        result = handler(event, None)

    assert result["statusCode"] == 200


def test_handler_passes_through_when_no_secret_configured():
    """If TELEGRAM_WEBHOOK_SECRET is unset, we accept any headers."""
    from agentic_tasks.handlers.webhook import handler

    event = {"body": json.dumps({"update_id": 1}), "headers": {}}

    with patch("agentic_tasks.handlers.webhook._process", new_callable=AsyncMock):
        result = handler(event, None)

    assert result["statusCode"] == 200
