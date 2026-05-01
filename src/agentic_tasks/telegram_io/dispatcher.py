"""Core dispatcher: given a Telegram update, run the agent and reply.

Loads conversation history from the store before each agent run, persists the
user message and every assistant/tool message produced during the run. That
gives the agent context across messages (so "reschedule it" resolves to the
task it just listed) without requiring sticky sessions.
"""

from __future__ import annotations

import asyncio
import html
import logging
from time import perf_counter

from telegram import Bot, Update
from telegram.error import BadRequest

from agentic_tasks.agent.loop import run_agent
from agentic_tasks.agent.transcribe import transcribe_audio
from agentic_tasks.config import get_settings
from agentic_tasks.conversation.store import get_store
from agentic_tasks.telegram_io.auth import is_authorized

log = logging.getLogger(__name__)

_REJECT_MESSAGE = "Sorry, this bot is private."
_UNSUPPORTED_MESSAGE = (
    "I can handle text messages and voice notes. "
    "Send me one of those about your tasks."
)
_TIMEOUT_MESSAGE = "Sorry for the delayed response, try again."
_EMPTY_TRANSCRIPT_MESSAGE = (
    "Sorry, I couldn't hear anything in that voice note. Try again?"
)


async def _send(bot: Bot, chat_id: int, text: str) -> None:
    """Send with Telegram HTML parse mode, falling back to plain on bad HTML."""
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    except BadRequest as e:
        log.warning("HTML parse failed (%s); resending as plain text", e)
        await bot.send_message(chat_id=chat_id, text=text)


async def process_update(update: Update, bot: Bot) -> None:
    """Process one Telegram update end-to-end."""
    message = update.message
    if message is None:
        return

    chat_id = message.chat_id

    if not is_authorized(chat_id):
        log.warning("rejected unauthorized chat_id=%s", chat_id)
        await _send(bot, chat_id, _REJECT_MESSAGE)
        return

    voice_prefix = ""
    if message.text:
        user_text = message.text
    elif message.voice:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
        try:
            voice_file = await bot.get_file(message.voice.file_id)
            audio_bytes = bytes(await voice_file.download_as_bytearray())
            user_text = await asyncio.to_thread(transcribe_audio, audio_bytes)
        except Exception as e:  # noqa: BLE001
            log.exception("transcription failed")
            await _send(
                bot,
                chat_id,
                f"Sorry, I couldn't transcribe that: <code>{html.escape(str(e))}</code>",
            )
            return
        if not user_text.strip():
            await _send(bot, chat_id, _EMPTY_TRANSCRIPT_MESSAGE)
            return
        voice_prefix = f"<i>\"{html.escape(user_text)}\"</i>\n\n"
    else:
        await _send(bot, chat_id, _UNSUPPORTED_MESSAGE)
        return

    log.info("processing chat_id=%s text=%r", chat_id, user_text[:80])

    await bot.send_chat_action(chat_id=chat_id, action="typing")

    store = get_store()
    history = store.get_history(chat_id)
    user_msg = {"role": "user", "content": user_text}
    store.append(chat_id, user_msg)

    timeout = get_settings().agent_timeout_seconds
    turn_start = perf_counter()
    try:
        reply, agent_messages = await asyncio.wait_for(
            asyncio.to_thread(run_agent, user_text, history),
            timeout=timeout,
        )
    except TimeoutError:
        log.warning("agent timed out after %ss", timeout)
        reply = _TIMEOUT_MESSAGE
        store.append(chat_id, {"role": "assistant", "content": reply})
    except Exception as e:  # noqa: BLE001
        log.exception("agent error")
        # Escape because the exception text may contain '<' or '>' that would
        # break the HTML parse.
        reply = (
            f"Sorry, something went wrong: <code>{html.escape(type(e).__name__)}"
            f"</code>: {html.escape(str(e))}"
        )
        store.append(chat_id, {"role": "assistant", "content": reply})
    else:
        for msg in agent_messages:
            store.append(chat_id, msg)
    log.info("turn done in %.0fms", (perf_counter() - turn_start) * 1000)

    await _send(bot, chat_id, voice_prefix + reply)
