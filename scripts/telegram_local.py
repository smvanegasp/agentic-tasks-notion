"""Run the bot locally via Telegram long-polling. No public URL needed.

Reads `.env`, registers a single text-message handler that delegates to the
shared dispatcher, then blocks until Ctrl+C.

    uv run python scripts/telegram_local.py
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from telegram import Update
    from telegram.ext import Application, ContextTypes, MessageHandler, filters

    from agentic_tasks.config import get_settings
    from agentic_tasks.telegram_io.dispatcher import process_update

    s = get_settings()
    app = Application.builder().token(s.telegram_bot_token).build()

    async def _on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await process_update(update, context.bot)

    app.add_handler(MessageHandler(filters.ALL, _on_message))

    print(f"Bot running. Allowed chat_id: {s.allowed_telegram_user_id}. Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
