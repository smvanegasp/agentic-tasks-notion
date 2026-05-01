"""Interactive REPL for the task agent.

Reads `.env`, then loops on stdin. Conversation history is kept across messages
within the session (same store the Telegram dispatcher uses), so multi-turn
references like "reschedule it" work.

The agent's replies are formatted as Telegram HTML for production. The REPL
strips the tags so terminal output stays readable.

    uv run python scripts/agent_repl.py

Type a message and press Enter. Ctrl-C or Ctrl-D to exit.
"""

from __future__ import annotations

import logging
import os
import re
import sys

from dotenv import load_dotenv

# Sentinel chat_id for the REPL's slot in the conversation store.
_REPL_CHAT_ID = 0


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def _stub_telegram_env() -> None:
    placeholders = {
        "TELEGRAM_BOT_TOKEN": "repl-placeholder",
        "ALLOWED_TELEGRAM_USER_ID": "0",
    }
    for k, v in placeholders.items():
        if not os.environ.get(k):
            os.environ[k] = v


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    load_dotenv()
    _stub_telegram_env()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from agentic_tasks.agent.loop import run_agent
    from agentic_tasks.conversation.store import get_store

    store = get_store()

    print("Agent REPL. Type a message and press Enter. Ctrl-C or Ctrl-D to exit.")
    print()
    while True:
        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_message:
            continue

        history = store.get_history(_REPL_CHAT_ID)
        store.append(_REPL_CHAT_ID, {"role": "user", "content": user_message})

        try:
            reply, agent_messages, _pending = run_agent(user_message, history)
        except Exception as e:  # noqa: BLE001
            print(f"error: {type(e).__name__}: {e}")
            store.append(
                _REPL_CHAT_ID,
                {"role": "assistant", "content": f"(error: {type(e).__name__})"},
            )
            continue

        for msg in agent_messages:
            store.append(_REPL_CHAT_ID, msg)

        print(f"bot> {_strip_html(reply)}")
        print()


if __name__ == "__main__":
    main()
