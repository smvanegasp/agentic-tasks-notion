"""Dispatcher routing for the /reset command and the confirmation gate."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

AUTHORIZED_CHAT_ID = 12345


def _update(text: str | None = "hi"):
    update = MagicMock()
    update.message = MagicMock()
    update.message.chat_id = AUTHORIZED_CHAT_ID
    update.message.text = text
    update.message.voice = None
    return update


def _bot():
    bot = MagicMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()
    return bot


async def test_slash_reset_clears_store_and_replies():
    from agentic_tasks.conversation.store import get_store
    from agentic_tasks.telegram_io.dispatcher import process_update

    store = get_store()
    store.append(AUTHORIZED_CHAT_ID, {"role": "user", "content": "earlier"})
    store.set_pending_plan(
        AUTHORIZED_CHAT_ID, [{"tool": "create_task", "arguments": {"name": "X"}}]
    )

    update = _update("/reset")
    bot = _bot()

    with patch("agentic_tasks.telegram_io.dispatcher.run_agent") as mock_agent:
        await process_update(update, bot)

    mock_agent.assert_not_called()
    text = bot.send_message.await_args.kwargs["text"]
    assert "cleared" in text.lower()
    assert store.get_history(AUTHORIZED_CHAT_ID) == []
    assert store.get_pending_plan(AUTHORIZED_CHAT_ID) is None


async def test_yes_executes_pending_plan_and_clears_it():
    from agentic_tasks.conversation.store import get_store
    from agentic_tasks.telegram_io.dispatcher import process_update

    store = get_store()
    plan = [{"tool": "create_task", "arguments": {"name": "Buy markers"}}]
    store.set_pending_plan(AUTHORIZED_CHAT_ID, plan)

    update = _update("y")
    bot = _bot()

    with patch(
        "agentic_tasks.telegram_io.dispatcher.execute_plan",
        return_value="Done.\n• Created Buy markers",
    ) as mock_exec, patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent"
    ) as mock_agent:
        await process_update(update, bot)

    mock_agent.assert_not_called()
    mock_exec.assert_called_once_with(plan)
    assert store.get_pending_plan(AUTHORIZED_CHAT_ID) is None
    text = bot.send_message.await_args.kwargs["text"]
    assert "Done" in text


async def test_no_clears_pending_and_asks_for_feedback():
    from agentic_tasks.conversation.store import get_store
    from agentic_tasks.telegram_io.dispatcher import process_update

    store = get_store()
    store.set_pending_plan(
        AUTHORIZED_CHAT_ID, [{"tool": "create_task", "arguments": {"name": "X"}}]
    )

    update = _update("n")
    bot = _bot()

    with patch(
        "agentic_tasks.telegram_io.dispatcher.execute_plan"
    ) as mock_exec, patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent"
    ) as mock_agent:
        await process_update(update, bot)

    mock_exec.assert_not_called()
    mock_agent.assert_not_called()
    assert store.get_pending_plan(AUTHORIZED_CHAT_ID) is None
    text = bot.send_message.await_args.kwargs["text"]
    assert "change" in text.lower()


async def test_clarification_clears_pending_and_runs_agent():
    """If the user replies with anything other than y/n while a plan is
    pending, treat it as a fresh request: drop the pending plan and run the
    agent with full history (so the model can re-propose)."""
    from agentic_tasks.conversation.store import get_store
    from agentic_tasks.telegram_io.dispatcher import process_update

    store = get_store()
    store.set_pending_plan(
        AUTHORIZED_CHAT_ID, [{"tool": "create_task", "arguments": {"name": "X"}}]
    )

    update = _update("actually make it Friday")
    bot = _bot()

    with patch(
        "agentic_tasks.telegram_io.dispatcher.execute_plan"
    ) as mock_exec, patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent",
        return_value=("ok", [{"role": "assistant", "content": "ok"}], None),
    ) as mock_agent:
        await process_update(update, bot)

    mock_exec.assert_not_called()
    mock_agent.assert_called_once()
    assert store.get_pending_plan(AUTHORIZED_CHAT_ID) is None
    # When the rejection-with-feedback path fires, the dispatcher should
    # pass a correction_note so the model knows to re-evaluate rather than
    # re-propose the same task.
    correction_note = mock_agent.call_args.kwargs.get("correction_note")
    assert correction_note is not None
    assert "REJECTED" in correction_note


async def test_normal_message_passes_no_correction_note():
    """Without a pending plan, no correction note should be passed."""
    from agentic_tasks.telegram_io.dispatcher import process_update

    update = _update("what's today?")
    bot = _bot()

    with patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent",
        return_value=("ok", [{"role": "assistant", "content": "ok"}], None),
    ) as mock_agent:
        await process_update(update, bot)

    correction_note = mock_agent.call_args.kwargs.get("correction_note")
    assert correction_note is None


async def test_cancel_word_clears_pending_via_rejection_path():
    """'Cancel' should be treated as rejection (not as a clarification),
    so the bot asks 'What would you like to change?' rather than running
    the agent."""
    from agentic_tasks.conversation.store import get_store
    from agentic_tasks.telegram_io.dispatcher import process_update

    store = get_store()
    store.set_pending_plan(
        AUTHORIZED_CHAT_ID, [{"tool": "create_task", "arguments": {"name": "X"}}]
    )

    update = _update("Cancel")
    bot = _bot()

    with patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent"
    ) as mock_agent, patch(
        "agentic_tasks.telegram_io.dispatcher.execute_plan"
    ) as mock_exec:
        await process_update(update, bot)

    mock_agent.assert_not_called()
    mock_exec.assert_not_called()
    assert store.get_pending_plan(AUTHORIZED_CHAT_ID) is None
    text = bot.send_message.await_args.kwargs["text"]
    assert "change" in text.lower()


async def test_agent_returning_pending_plan_persists_it_in_store():
    from agentic_tasks.conversation.store import get_store
    from agentic_tasks.telegram_io.dispatcher import process_update

    update = _update("create buy markers")
    bot = _bot()

    plan = [{"tool": "create_task", "arguments": {"name": "Buy markers"}}]
    preview = "About to:\n• Create Buy markers\n\nReply y to proceed or n to cancel."

    with patch(
        "agentic_tasks.telegram_io.dispatcher.run_agent",
        return_value=(preview, [{"role": "assistant", "content": preview}], plan),
    ):
        await process_update(update, bot)

    store = get_store()
    assert store.get_pending_plan(AUTHORIZED_CHAT_ID) == plan
    text = bot.send_message.await_args.kwargs["text"]
    assert "About to" in text
