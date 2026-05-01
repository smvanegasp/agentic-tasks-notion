"""Push secret values from local ``.env`` into AWS Secrets Manager.

Run once after the first ``sam deploy``. Uses the secret name printed in the
stack's outputs (``SecretsName``).

    uv run python scripts/seed_secrets.py --secret-name <secret-name> [--region us-east-2]

Reads the local ``.env`` (gitignored), picks the keys the Lambdas need at
runtime, JSON-encodes them, and writes a new version of the SecretsManager
secret. Re-run any time you rotate a token.
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import dotenv_values

# Keys the Lambda runtime needs from SecretsManager. Non-secret config (model,
# timezone, digest time, etc.) lives in the SAM template parameters instead.
SECRET_KEYS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_WEBHOOK_SECRET",
    "NOTION_TOKEN",
    "LLM_API_KEY",
    "NOTION_TASKS_DB_ID",
    "NOTION_PROJECTS_DB_ID",
    "ALLOWED_TELEGRAM_USER_ID",
]
OPTIONAL_KEYS = {"TELEGRAM_WEBHOOK_SECRET"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "--secret-name",
        required=True,
        help="SecretsManager secret name from the SAM stack output 'SecretsName'.",
    )
    parser.add_argument("--region", default="us-east-2")
    args = parser.parse_args()

    env = dotenv_values(".env")
    payload: dict[str, str] = {}
    missing: list[str] = []
    for key in SECRET_KEYS:
        value = env.get(key)
        if value:
            payload[key] = value
        elif key not in OPTIONAL_KEYS:
            missing.append(key)

    if missing:
        print(f"Missing required keys in .env: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    import boto3

    client = boto3.client("secretsmanager", region_name=args.region)
    client.put_secret_value(
        SecretId=args.secret_name,
        SecretString=json.dumps(payload),
    )
    print(
        f"Wrote {len(payload)} secrets to {args.secret_name} "
        f"({', '.join(sorted(payload))})"
    )


if __name__ == "__main__":
    main()
