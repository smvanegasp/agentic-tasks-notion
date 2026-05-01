"""JSON log formatter for Lambda."""

from __future__ import annotations

import json
import logging

import pytest


def test_json_formatter_promotes_extra_fields_to_top_level():
    from agentic_tasks._aws import _JSONFormatter

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg="turn done",
        args=None,
        exc_info=None,
    )
    record.chat_id = 12345
    record.duration_ms = 1234
    payload = json.loads(_JSONFormatter().format(record))
    assert payload["message"] == "turn done"
    assert payload["chat_id"] == 12345
    assert payload["duration_ms"] == 1234
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert "ts" in payload


def test_json_formatter_serializes_unsupported_objects_as_repr():
    from agentic_tasks._aws import _JSONFormatter

    class Weird:
        def __repr__(self) -> str:
            return "<Weird>"

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg="x",
        args=None,
        exc_info=None,
    )
    record.weird = Weird()
    payload = json.loads(_JSONFormatter().format(record))
    assert payload["weird"] == "<Weird>"


def test_setup_lambda_logging_no_op_outside_lambda(monkeypatch):
    """Without AWS_LAMBDA_FUNCTION_NAME, leave logging alone — pytest's
    capture and local stdout depend on the default formatter."""
    from agentic_tasks._aws import setup_lambda_logging

    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    monkeypatch.delenv("_LOGGING_CONFIGURED", raising=False)
    before = list(logging.getLogger().handlers)
    setup_lambda_logging()
    assert logging.getLogger().handlers == before


def test_setup_lambda_logging_installs_json_formatter_in_lambda(monkeypatch):
    from agentic_tasks._aws import _JSONFormatter, setup_lambda_logging

    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "test-fn")
    monkeypatch.delenv("_LOGGING_CONFIGURED", raising=False)
    root = logging.getLogger()
    original = [h.formatter for h in root.handlers]
    try:
        setup_lambda_logging()
        for h in root.handlers:
            assert isinstance(h.formatter, _JSONFormatter)
    finally:
        # Restore, otherwise later tests inherit JSON formatting and pytest's
        # human-readable log capture breaks.
        for h, fmt in zip(root.handlers, original, strict=False):
            h.setFormatter(fmt)
        monkeypatch.delenv("_LOGGING_CONFIGURED", raising=False)


@pytest.fixture(autouse=True)
def _clear_logging_flag(monkeypatch):
    """The setup function uses an env-var sentinel for warm-container
    idempotence. Clear it between tests so each test sees a clean state."""
    monkeypatch.delenv("_LOGGING_CONFIGURED", raising=False)
    yield
