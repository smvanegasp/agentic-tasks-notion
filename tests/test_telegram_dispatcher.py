from unittest.mock import AsyncMock, MagicMock, patch

# ALLOWED_TELEGRAM_USER_ID in conftest is "12345"
AUTHORIZED_CHAT_ID = 12345


def _update(chat_id: int, text: str | None = "hello", voice=None):
    """Build a fake Update. ``voice`` is None by default so MagicMock's auto-
    attribute behavior doesn't accidentally make every text update look like a
    voice update too."""
    update = MagicMock()
    update.message = MagicMock()
    update.message.chat_id = chat_id
    update.message.text = text
    update.message.voice = voice
    return update


def _bot():
    bot = MagicMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()
    return bot


async def test_runs_agent_and_replies_with_html_parse_mode():
    from agentic_tasks.telegram_io.dispatcher import process_update

    update = _update(AUTHORIZED_CHAT_ID, "what is on my plate?")
    bot = _bot()

    reply_text = "<b>Nothing for today.</b>"
    with patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent",
        return_value=(reply_text, [{"role": "assistant", "content": reply_text}], None),
    ):
        await process_update(update, bot)

    bot.send_chat_action.assert_awaited_once()
    bot.send_message.assert_awaited_once()
    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["chat_id"] == AUTHORIZED_CHAT_ID
    assert kwargs["text"] == reply_text
    assert kwargs["parse_mode"] == "HTML"


async def test_falls_back_to_plain_text_when_html_is_invalid():
    """If the LLM emits malformed HTML, Telegram raises BadRequest. We retry
    once without parse_mode AND with HTML tags stripped, so the user sees
    readable plain text instead of raw markup."""
    from telegram.error import BadRequest

    from agentic_tasks.telegram_io.dispatcher import process_update

    update = _update(AUTHORIZED_CHAT_ID, "go")
    bot = _bot()
    bot.send_message = AsyncMock(
        side_effect=[BadRequest("can't parse"), None]
    )

    reply_text = "<b>broken<i></b>"
    with patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent",
        return_value=(reply_text, [{"role": "assistant", "content": reply_text}], None),
    ):
        await process_update(update, bot)

    assert bot.send_message.await_count == 2
    second_call = bot.send_message.await_args_list[1].kwargs
    assert second_call["text"] == "broken"  # HTML stripped
    assert "<" not in second_call["text"]
    assert "parse_mode" not in second_call


async def test_persists_history_across_calls():
    """User message + agent messages should be appended to the store so the
    next call has context."""
    from agentic_tasks.conversation.store import get_store
    from agentic_tasks.telegram_io.dispatcher import process_update

    update = _update(AUTHORIZED_CHAT_ID, "hello")
    bot = _bot()
    agent_msgs = [{"role": "assistant", "content": "hi"}]

    with patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent",
        return_value=("hi", agent_msgs, None),
    ):
        await process_update(update, bot)

    history = get_store().get_history(AUTHORIZED_CHAT_ID)
    assert [m["content"] for m in history] == ["hello", "hi"]


async def test_history_is_passed_to_agent_on_subsequent_call():
    from agentic_tasks.conversation.store import get_store
    from agentic_tasks.telegram_io.dispatcher import process_update

    # Pre-populate prior turn in the store
    store = get_store()
    store.append(AUTHORIZED_CHAT_ID, {"role": "user", "content": "earlier"})
    store.append(AUTHORIZED_CHAT_ID, {"role": "assistant", "content": "earlier reply"})

    update = _update(AUTHORIZED_CHAT_ID, "follow-up")
    bot = _bot()

    with patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent",
        return_value=("ok", [{"role": "assistant", "content": "ok"}], None),
    ) as mock_agent:
        await process_update(update, bot)

    history_arg = mock_agent.call_args.args[1]
    assert [m["content"] for m in history_arg] == ["earlier", "earlier reply"]


async def test_timeout_replies_with_apology_and_persists_to_history():
    """If the agent overruns AGENT_TIMEOUT_SECONDS, the user gets a polite
    bail-out message, and the user's prompt + the apology are persisted."""
    from agentic_tasks.conversation.store import get_store
    from agentic_tasks.telegram_io.dispatcher import process_update

    update = _update(AUTHORIZED_CHAT_ID, "do something slow")
    bot = _bot()

    async def _timeout_then_cleanup(coro, timeout):
        # Close the wrapped coroutine so Python doesn't warn about it
        # never being awaited, then raise as if the deadline elapsed.
        coro.close()
        raise TimeoutError

    with patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent",
        return_value=("ignored", [], None),
    ), patch(
        "agentic_tasks.telegram_io.dispatcher.asyncio.wait_for",
        _timeout_then_cleanup,
    ):
        await process_update(update, bot)

    text = bot.send_message.await_args.kwargs["text"]
    assert "delayed" in text.lower()

    history = get_store().get_history(AUTHORIZED_CHAT_ID)
    assert history[0] == {"role": "user", "content": "do something slow"}
    assert "delayed" in history[1]["content"].lower()


async def test_failed_agent_run_still_persists_user_and_error():
    """If the agent raises, history should record the user message and a
    placeholder so the next turn isn't confused by an unanswered user msg."""
    from agentic_tasks.conversation.store import get_store
    from agentic_tasks.telegram_io.dispatcher import process_update

    update = _update(AUTHORIZED_CHAT_ID, "do thing")
    bot = _bot()

    with patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent",
        side_effect=RuntimeError("boom"),
    ):
        await process_update(update, bot)

    history = get_store().get_history(AUTHORIZED_CHAT_ID)
    assert history[0] == {"role": "user", "content": "do thing"}
    assert history[1]["role"] == "assistant"
    assert "boom" in history[1]["content"] or "wrong" in history[1]["content"].lower()


async def test_rejects_unauthorized_chat_with_polite_reply():
    from agentic_tasks.telegram_io.dispatcher import process_update

    update = _update(99999, "hello")
    bot = _bot()

    with patch("agentic_tasks.telegram_io.dispatcher.run_agent") as mock_agent:
        await process_update(update, bot)

    mock_agent.assert_not_called()
    bot.send_chat_action.assert_not_called()
    bot.send_message.assert_awaited_once()
    assert "private" in bot.send_message.await_args.kwargs["text"].lower()


async def test_no_message_object_is_ignored():
    from agentic_tasks.telegram_io.dispatcher import process_update

    update = MagicMock()
    update.message = None
    bot = _bot()

    with patch("agentic_tasks.telegram_io.dispatcher.run_agent") as mock_agent:
        await process_update(update, bot)

    mock_agent.assert_not_called()
    bot.send_message.assert_not_called()


async def test_unsupported_message_type_gets_explanatory_reply():
    """Photos, stickers, etc. — not text and not voice."""
    from agentic_tasks.telegram_io.dispatcher import process_update

    update = _update(AUTHORIZED_CHAT_ID, text=None, voice=None)
    bot = _bot()

    with patch("agentic_tasks.telegram_io.dispatcher.run_agent") as mock_agent:
        await process_update(update, bot)

    mock_agent.assert_not_called()
    bot.send_message.assert_awaited_once()
    text_sent = bot.send_message.await_args.kwargs["text"].lower()
    assert "text" in text_sent or "voice" in text_sent


async def test_voice_note_is_transcribed_and_processed():
    """Voice flow: download → transcribe → run agent → reply with transcript prefix."""
    from agentic_tasks.telegram_io.dispatcher import process_update

    voice = MagicMock()
    voice.file_id = "voice-file-1"
    update = _update(AUTHORIZED_CHAT_ID, text=None, voice=voice)

    voice_file = MagicMock()
    voice_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake-audio"))

    bot = _bot()
    bot.get_file = AsyncMock(return_value=voice_file)

    with patch(
        "agentic_tasks.telegram_io.dispatcher.transcribe_audio",
        return_value="agendar gimnasio mañana a las 8 PM",
    ), patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent",
        return_value=("Hecho.", [{"role": "assistant", "content": "Hecho."}], None),
    ):
        await process_update(update, bot)

    bot.get_file.assert_awaited_once_with("voice-file-1")
    voice_file.download_as_bytearray.assert_awaited_once()

    # Final reply should include the transcript (italics) AND the agent reply
    text = bot.send_message.await_args.kwargs["text"]
    assert "agendar gimnasio" in text  # transcript appears
    assert "Hecho." in text  # agent reply appears
    assert "<i>" in text  # transcript shown as italics


async def test_voice_note_with_empty_transcript_replies_with_apology():
    from agentic_tasks.telegram_io.dispatcher import process_update

    voice = MagicMock()
    voice.file_id = "voice-empty"
    update = _update(AUTHORIZED_CHAT_ID, text=None, voice=voice)

    voice_file = MagicMock()
    voice_file.download_as_bytearray = AsyncMock(return_value=bytearray(b""))

    bot = _bot()
    bot.get_file = AsyncMock(return_value=voice_file)

    with patch(
        "agentic_tasks.telegram_io.dispatcher.transcribe_audio",
        return_value="   ",  # whitespace only
    ), patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent"
    ) as mock_agent:
        await process_update(update, bot)

    mock_agent.assert_not_called()
    text = bot.send_message.await_args.kwargs["text"].lower()
    assert "couldn't hear" in text or "voice note" in text


async def test_voice_note_transcription_failure_replies_politely():
    from agentic_tasks.telegram_io.dispatcher import process_update

    voice = MagicMock()
    voice.file_id = "voice-broken"
    update = _update(AUTHORIZED_CHAT_ID, text=None, voice=voice)

    voice_file = MagicMock()
    voice_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"x"))

    bot = _bot()
    bot.get_file = AsyncMock(return_value=voice_file)

    with patch(
        "agentic_tasks.telegram_io.dispatcher.transcribe_audio",
        side_effect=RuntimeError("api down"),
    ), patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent"
    ) as mock_agent:
        await process_update(update, bot)

    mock_agent.assert_not_called()
    text = bot.send_message.await_args.kwargs["text"].lower()
    assert "transcribe" in text or "api down" in text


async def test_agent_error_is_caught_and_reported_to_user():
    from agentic_tasks.telegram_io.dispatcher import process_update

    update = _update(AUTHORIZED_CHAT_ID, "do thing")
    bot = _bot()

    with patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent",
        side_effect=RuntimeError("boom"),
    ):
        await process_update(update, bot)

    bot.send_message.assert_awaited_once()
    text = bot.send_message.await_args.kwargs["text"]
    assert "boom" in text or "wrong" in text.lower()
