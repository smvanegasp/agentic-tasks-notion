"""AWS-specific bootstrap utilities.

When running in Lambda, fetches the configured SecretsManager secret on cold
start and dumps its key/value pairs into the process environment so that
``config.from_env()`` works the same way as it does locally with ``.env``.

Idempotent: subsequent calls in the same container are no-ops.
"""

from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)


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
