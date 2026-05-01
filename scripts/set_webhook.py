"""Tell Telegram where to POST your bot's updates.

Run once after ``sam deploy``, using the ``WebhookUrl`` from the stack outputs.

    uv run python scripts/set_webhook.py --url <WebhookUrl>

The script reads ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_WEBHOOK_SECRET`` from
``.env``. If a secret is set, Telegram will include it as a header on every
webhook POST so the Lambda can verify the request really came from Telegram.
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx
from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--url", required=True, help="API Gateway webhook URL.")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Missing TELEGRAM_BOT_TOKEN in .env", file=sys.stderr)
        sys.exit(1)

    payload: dict[str, str] = {"url": args.url}
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    if secret:
        payload["secret_token"] = secret

    response = httpx.post(
        f"https://api.telegram.org/bot{token}/setWebhook",
        data=payload,
        timeout=15.0,
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        print(f"Failed to set webhook: {result}", file=sys.stderr)
        sys.exit(1)

    print(f"Webhook set: {args.url}")
    if secret:
        print("(secret-token verification enabled)")

    info = httpx.get(
        f"https://api.telegram.org/bot{token}/getWebhookInfo", timeout=15.0
    )
    info.raise_for_status()
    print()
    print("Current webhook info:")
    print(info.json())


if __name__ == "__main__":
    main()
