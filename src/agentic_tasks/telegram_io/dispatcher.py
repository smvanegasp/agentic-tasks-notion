"""Core dispatcher: given a Telegram update, run the agent and reply.

Loads conversation history from the store before each agent run, persists the
user message and every assistant/tool message produced during the run. That
gives the agent context across messages (so "reschedule it" resolves to the
task it just listed) without requiring sticky sessions.

Two routes intercept BEFORE the LLM runs:

* ``/reset`` — clears the chat's history and any pending plan, replies
  "Conversation cleared."
* Pending plan present — the previous turn deferred a write. y/yes/sí
  executes the plan; n/no clears it and asks for feedback; anything else
  clears it and runs the agent normally (the user's message becomes the new
  request, with full prior context).
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from time import perf_counter

from telegram import Bot, Update
from telegram.error import BadRequest

from agentic_tasks.agent.loop import run_agent
from agentic_tasks.agent.preview import (
    execute_plan,
    is_confirmation,
    is_rejection,
)
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
_RESET_REPLY = "Conversation cleared."
_REJECTION_REPLY = "What would you like to change?"

# Telegram's per-message hard cap is 4096 chars; we split below that with a
# margin so HTML entities and emoji don't tip us over.
_TELEGRAM_MAX_CHARS = 4000

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities — used for the plain-text
    fallback when Telegram rejects our HTML, so the user sees readable text
    instead of raw ``<b>`` markers."""
    return html.unescape(_HTML_TAG_RE.sub("", text))


def _split_message(text: str, max_len: int = _TELEGRAM_MAX_CHARS) -> list[str]:
    """Chunk a long message at paragraph boundaries (``\\n\\n``).

    Each chunk fits within ``max_len`` chars. If a single paragraph still
    exceeds the limit (rare), it falls back to a hard slice — accepting that
    HTML tags may break across the boundary, since losing the message is
    worse.
    """
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        if not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= max_len:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)

    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_len:
            final_chunks.append(chunk)
            continue
        for i in range(0, len(chunk), max_len):
            final_chunks.append(chunk[i : i + max_len])
    return final_chunks


async def _send(bot: Bot, chat_id: int, text: str) -> None:
    """Send a Telegram message, splitting on length and stripping tags on
    HTML-parse error so the user always gets a readable reply."""
    for chunk in _split_message(text):
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode="HTML")
        except BadRequest as e:
            log.warning("HTML parse failed (%s); resending as plain text", e)
            await bot.send_message(chat_id=chat_id, text=_strip_html(chunk))


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

    store = get_store()

    # /reset short-circuits everything.
    if user_text.strip().lower() == "/reset":
        store.reset(chat_id)
        log.info("conversation reset for chat_id=%s", chat_id)
        await _send(bot, chat_id, voice_prefix + _RESET_REPLY)
        return

    # Confirmation gate: if a plan is pending, route the message before the LLM.
    correction_note: str | None = None
    pending_plan = store.get_pending_plan(chat_id)
    if pending_plan:
        if is_confirmation(user_text):
            log.info(
                "confirmation gate: executing pending plan (%d entries)",
                len(pending_plan),
            )
            await bot.send_chat_action(chat_id=chat_id, action="typing")
            store.append(chat_id, {"role": "user", "content": user_text})
            try:
                summary = await asyncio.to_thread(execute_plan, pending_plan)
            except Exception as e:  # noqa: BLE001
                log.exception("execute_plan failed")
                summary = (
                    f"Sorry, something went wrong: <code>{html.escape(type(e).__name__)}"
                    f"</code>: {html.escape(str(e))}"
                )
            store.clear_pending_plan(chat_id)
            store.append(chat_id, {"role": "assistant", "content": summary})
            await _send(bot, chat_id, voice_prefix + summary)
            return

        if is_rejection(user_text):
            log.info("confirmation gate: plan rejected; awaiting feedback")
            store.append(chat_id, {"role": "user", "content": user_text})
            store.clear_pending_plan(chat_id)
            store.append(chat_id, {"role": "assistant", "content": _REJECTION_REPLY})
            await _send(bot, chat_id, voice_prefix + _REJECTION_REPLY)
            return

        # Anything else: treat as a clarification or a fresh request. Drop
        # the pending plan; fall through to normal agent flow with the
        # already-persisted preview as part of history. We also pass a
        # transient correction note so the model knows the previous
        # proposal was rejected and shouldn't re-propose the same thing.
        log.info("confirmation gate: clarification — clearing pending plan")
        store.clear_pending_plan(chat_id)
        correction_note = (
            "The user just REJECTED your previous pending plan with the"
            " message that follows. Treat their message as a correction or a"
            " new request — DO NOT re-propose the same task or page_id you"
            " just proposed. If they referenced a task by partial name or"
            " acronym, call find_tasks to look it up rather than guessing"
            " from earlier conversation history."
        )

    # Normal agent flow.
    await bot.send_chat_action(chat_id=chat_id, action="typing")
    history = store.get_history(chat_id)
    user_msg = {"role": "user", "content": user_text}
    store.append(chat_id, user_msg)

    timeout = get_settings().agent_timeout_seconds
    turn_start = perf_counter()
    new_pending_plan: list[dict] | None = None
    try:
        reply, agent_messages, new_pending_plan = await asyncio.wait_for(
            asyncio.to_thread(
                run_agent, user_text, history, correction_note=correction_note
            ),
            timeout=timeout,
        )
    except TimeoutError:
        log.warning("agent timed out after %ss", timeout)
        reply = _TIMEOUT_MESSAGE
        store.append(chat_id, {"role": "assistant", "content": reply})
    except Exception as e:  # noqa: BLE001
        log.exception("agent error")
        reply = (
            f"Sorry, something went wrong: <code>{html.escape(type(e).__name__)}"
            f"</code>: {html.escape(str(e))}"
        )
        store.append(chat_id, {"role": "assistant", "content": reply})
    else:
        for msg in agent_messages:
            store.append(chat_id, msg)
        if new_pending_plan:
            store.set_pending_plan(chat_id, new_pending_plan)
    log.info("turn done in %.0fms", (perf_counter() - turn_start) * 1000)

    await _send(bot, chat_id, voice_prefix + reply)
