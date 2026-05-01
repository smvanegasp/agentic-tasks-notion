"""AWS-specific bootstrap utilities.

Two cold-start helpers:

- :func:`bootstrap_secrets` pulls the configured SecretsManager secret into
  the process environment so ``config.from_env()`` works the same way as it
  does locally with ``.env``.
- :func:`setup_lambda_logging` installs a JSON formatter on the root logger
  so CloudWatch Logs Insights can index custom fields passed via ``extra=``.

Both are idempotent: subsequent calls in the same warm container are no-ops.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os

log = logging.getLogger(__name__)

# Standard ``LogRecord`` attributes — anything else on the record is treated
# as a custom field and promoted to a top-level key in the JSON output.
_RESERVED_LOG_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class _JSONFormatter(logging.Formatter):
    """Render each LogRecord as one JSON line. Custom fields passed via
    ``extra={...}`` are promoted to top-level keys, which is what makes
    CloudWatch Logs Insights queries like ``fields chat_id, duration_ms``
    work without manual parsing.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, object] = {
            "ts": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.UTC
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_FIELDS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        return json.dumps(payload, default=repr)


def setup_lambda_logging() -> None:
    """Switch the root logger to JSON output when running in Lambda.

    Detected via ``AWS_LAMBDA_FUNCTION_NAME``, which AWS sets automatically.
    Locally and in tests this is a no-op so pytest's log capture and the
    plain stdout from ``scripts/`` keep their human-readable format.
    """
    if not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return
    if os.environ.get("_LOGGING_CONFIGURED") == "1":
        return

    root = logging.getLogger()
    formatter = _JSONFormatter()
    for handler in root.handlers:
        handler.setFormatter(formatter)
    if not root.handlers:
        # Should not happen under Lambda (the runtime installs one), but
        # be defensive so we never silently drop logs.
        h = logging.StreamHandler()
        h.setFormatter(formatter)
        root.addHandler(h)
    root.setLevel(logging.INFO)
    os.environ["_LOGGING_CONFIGURED"] = "1"


def bootstrap_secrets() -> None:
    """If ``SECRETS_NAME`` is set and we haven't loaded yet, fetch and merge."""
    secret_id = os.environ.get("SECRETS_NAME")
    if not secret_id:
        return  # local dev — secrets come from .env
    if os.environ.get("_SECRETS_LOADED") == "1":
        return  # warm container

    import boto3  # imported lazily so non-AWS environments don't need it

    log.info("loading secrets from SecretsManager: %s", secret_id)
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_id)
    payload = json.loads(response["SecretString"])

    for k, v in payload.items():
        # setdefault so explicit Lambda env vars (or test stubs) win.
        os.environ.setdefault(k, str(v))
    os.environ["_SECRETS_LOADED"] = "1"
