"""Run the daily digest locally — same code path as the Lambda handler.

Usage:

    # Real run: query Notion, send to your Telegram.
    uv run python scripts/digest_local.py

    # Print to terminal instead of sending (HTML stripped for readability).
    uv run python scripts/digest_local.py --dry-run

    # Preview a future date's digest (without time-travelling the system clock).
    uv run python scripts/digest_local.py --date 2026-05-05 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from datetime import date, datetime

from dotenv import load_dotenv


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the digest to stdout instead of sending to Telegram.",
    )
    parser.add_argument(
        "--date",
        help="Override 'today' for previewing (YYYY-MM-DD). Notion is still queried live.",
    )
    args = parser.parse_args()

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from agentic_tasks.config import get_settings
    from agentic_tasks.digest.render import render_digest
    from agentic_tasks.handlers.digest import _send_digest_message
    from agentic_tasks.notion_io.tasks import (
        query_due_today,
        query_my_day,
        query_overdue,
    )

    s = get_settings()
    if args.date:
        today = date.fromisoformat(args.date)
    else:
        today = datetime.now(s.timezone).date()

    text = render_digest(
        today,
        today_tasks=query_due_today(today=today),
        my_day_tasks=query_my_day(),
        overdue_tasks=query_overdue(today=today),
    )

    if args.dry_run:
        print()
        print("=" * 60)
        print(f"Digest preview for {today.isoformat()} (Notion state: live)")
        print("=" * 60)
        print(_strip_html(text))
        print("=" * 60)
        return

    asyncio.run(_send_digest_message(text))
    print(f"Sent digest for {today.isoformat()} to chat_id={s.allowed_telegram_user_id}")


if __name__ == "__main__":
    main()
