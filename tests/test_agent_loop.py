from datetime import date
from unittest.mock import MagicMock, patch


def _response(content=None, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    return response


def _tool_call(name: str, arguments: str, tc_id: str = "tc-1"):
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_returns_text_when_no_tool_calls(mock_openai_cls, mock_call_tool):
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(content="Hello.")
    mock_openai_cls.return_value = client

    reply, agent_messages, pending = run_agent("hi")

    assert reply == "Hello."
    assert agent_messages == [{"role": "assistant", "content": "Hello."}]
    assert pending is None
    mock_call_tool.assert_not_called()


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_runs_tool_then_returns(mock_openai_cls, mock_call_tool):
    from agentic_tasks.agent.loop import run_agent

    mock_call_tool.return_value = '{"tasks": []}'
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("query_tasks", '{"due_filter": "today"}')]),
        _response(content="No tasks today."),
    ]
    mock_openai_cls.return_value = client

    reply, agent_messages, pending = run_agent("any tasks today?")

    assert reply == "No tasks today."
    assert pending is None
    mock_call_tool.assert_called_once_with("query_tasks", {"due_filter": "today"})

    # agent_messages should contain: assistant-with-tool-call, tool, assistant-final
    assert len(agent_messages) == 3
    assert agent_messages[0]["role"] == "assistant"
    assert agent_messages[0]["tool_calls"][0]["function"]["name"] == "query_tasks"
    assert agent_messages[1]["role"] == "tool"
    assert agent_messages[1]["tool_call_id"] == "tc-1"
    # Harmony (gpt-oss tokenizer) requires `name` on tool messages — ensure
    # we always include it so persisted history can be replayed safely.
    assert agent_messages[1]["name"] == "query_tasks"
    assert agent_messages[2] == {"role": "assistant", "content": "No tasks today."}


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_history_is_passed_into_messages(mock_openai_cls, mock_call_tool):
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(content="ok")
    mock_openai_cls.return_value = client

    history = [
        {"role": "user", "content": "earlier message"},
        {"role": "assistant", "content": "earlier reply"},
    ]
    run_agent("now", history=history)

    sent = client.chat.completions.create.call_args.kwargs["messages"]
    # system + history (2) + new user
    assert len(sent) == 4
    assert sent[0]["role"] == "system"
    assert sent[1]["content"] == "earlier message"
    assert sent[2]["content"] == "earlier reply"
    assert sent[3] == {"role": "user", "content": "now"}


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_caps_at_max_iterations(mock_openai_cls, mock_call_tool):
    from agentic_tasks.agent.loop import run_agent

    mock_call_tool.return_value = "{}"
    client = MagicMock()
    client.chat.completions.create.return_value = _response(
        tool_calls=[_tool_call("list_projects", "{}")]
    )
    mock_openai_cls.return_value = client

    reply, _, pending = run_agent("loop forever", max_iterations=3)

    assert "too many steps" in reply.lower()
    assert pending is None
    assert client.chat.completions.create.call_count == 3


def test_extract_date_hints_for_upcoming_weekday():
    from agentic_tasks.agent.loop import _extract_date_hints

    # Thursday 2026-04-30 → upcoming Saturday is 2026-05-02
    hints = _extract_date_hints("Schedule for Saturday", date(2026, 4, 30))

    joined = "\n".join(hints)
    assert "saturday" in joined.lower()
    assert "2026-05-02" in joined


def test_extract_date_hints_tomorrow_yesterday():
    from agentic_tasks.agent.loop import _extract_date_hints

    today = date(2026, 4, 30)
    assert any(
        "2026-05-01" in h
        for h in _extract_date_hints("Add a task for tomorrow", today)
    )
    assert any(
        "2026-04-29" in h
        for h in _extract_date_hints("yesterday I went to the gym", today)
    )


def test_extract_date_hints_spanish_weekday_with_accent():
    from agentic_tasks.agent.loop import _extract_date_hints

    # Thursday 2026-04-30 → sábado is 2026-05-02
    hints = _extract_date_hints("Agendar para el sábado", date(2026, 4, 30))
    assert any("2026-05-02" in h for h in hints)


def test_extract_date_hints_spanish_manana():
    from agentic_tasks.agent.loop import _extract_date_hints

    hints = _extract_date_hints(
        "Mañana voy al gimnasio a las 8 PM", date(2026, 4, 30)
    )
    assert any("2026-05-01" in h for h in hints)


def test_extract_date_hints_next_weekday_includes_both_dates():
    from agentic_tasks.agent.loop import _extract_date_hints

    # Thursday 2026-04-30 → Saturday=2026-05-02, "next Saturday"=2026-05-09
    hints = _extract_date_hints("Schedule for next Saturday", date(2026, 4, 30))
    joined = "\n".join(hints)
    assert "2026-05-02" in joined  # this Saturday
    assert "2026-05-09" in joined  # next Saturday


def test_extract_date_hints_when_today_is_that_weekday():
    from agentic_tasks.agent.loop import _extract_date_hints

    # Thursday 2026-04-30 → "Thursday" should resolve to today, not next week
    hints = _extract_date_hints("Add for Thursday", date(2026, 4, 30))
    assert any("2026-04-30" in h for h in hints)


def test_extract_date_hints_no_dates_returns_empty():
    from agentic_tasks.agent.loop import _extract_date_hints

    assert _extract_date_hints("hello there", date(2026, 4, 30)) == []


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_injects_date_hints_when_message_has_weekday(
    mock_openai_cls, mock_call_tool
):
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(content="ok")
    mock_openai_cls.return_value = client

    run_agent("schedule something for Saturday")

    sent = client.chat.completions.create.call_args.kwargs["messages"]
    hint_msgs = [
        m for m in sent
        if m["role"] == "system"
        and "Date references detected" in m["content"]
    ]
    assert len(hint_msgs) == 1
    assert "saturday" in hint_msgs[0]["content"].lower()


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_skips_hint_when_no_date_references(
    mock_openai_cls, mock_call_tool
):
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(content="ok")
    mock_openai_cls.return_value = client

    run_agent("hello there")

    sent = client.chat.completions.create.call_args.kwargs["messages"]
    hint_msgs = [
        m for m in sent
        if m["role"] == "system"
        and "Date references detected" in m["content"]
    ]
    assert hint_msgs == []


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_short_circuits_on_write_tool_call(mock_openai_cls, mock_call_tool):
    """When the model emits create_tasks / update_tasks / complete_tasks, the
    loop must NOT execute it. It returns a preview + pending plan instead."""
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call(
                "create_tasks",
                '{"tasks": [{"name": "Buy markers"}]}',
                tc_id="tc-w1",
            ),
        ]
    )
    mock_openai_cls.return_value = client

    reply, agent_messages, pending = run_agent("create buy markers")

    # The write must NOT have been executed.
    mock_call_tool.assert_not_called()
    # The loop should make exactly one LLM call before exiting.
    assert client.chat.completions.create.call_count == 1
    assert pending == [
        {"tool": "create_tasks", "arguments": {"tasks": [{"name": "Buy markers"}]}}
    ]
    assert "About to" in reply
    assert "Buy markers" in reply
    # Persisted history should contain only a clean preview message — no
    # orphan tool_calls without responses.
    assert len(agent_messages) == 1
    assert agent_messages[0]["role"] == "assistant"
    assert "tool_calls" not in agent_messages[0]


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_short_circuits_on_batch_create(mock_openai_cls, mock_call_tool):
    """Three tasks in one batch → one preview, one pending plan."""
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call(
                "create_tasks",
                '{"tasks": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}',
                tc_id="tc-w",
            ),
        ]
    )
    mock_openai_cls.return_value = client

    reply, _agent_messages, pending = run_agent("create A, B, C")

    mock_call_tool.assert_not_called()
    assert pending is not None and len(pending) == 1
    assert pending[0]["tool"] == "create_tasks"
    assert len(pending[0]["arguments"]["tasks"]) == 3
    assert reply.count("About to:") == 1
    assert reply.count("<b>y</b>") == 1


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_short_circuits_on_combined_complete_and_update(
    mock_openai_cls, mock_call_tool
):
    """The user's golden path: 'mark X done AND reschedule Y'. Both write
    tools in one round → one combined preview, one pending plan."""
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call(
                "complete_tasks",
                '{"page_ids": ["page-x"]}',
                tc_id="tc-c",
            ),
            _tool_call(
                "update_tasks",
                '{"updates": [{"page_id": "page-y", "due": "2026-05-09"}]}',
                tc_id="tc-u",
            ),
        ]
    )
    mock_openai_cls.return_value = client

    # get_task gets called by the preview formatter for both page_ids.
    from agentic_tasks.notion_io.tasks import Task

    def fake_get(page_id):
        return Task(
            page_id=page_id,
            name=f"task-{page_id}",
            status="To Do",
            priority=None,
            due=None,
        )

    with patch("agentic_tasks.agent.preview.get_task", side_effect=fake_get):
        reply, _agent, pending = run_agent("mark X done and reschedule Y")

    mock_call_tool.assert_not_called()
    assert pending is not None
    assert [e["tool"] for e in pending] == ["complete_tasks", "update_tasks"]
    # ONE preview header for the combined batch.
    assert reply.count("About to:") == 1
    assert reply.count("<b>y</b>") == 1


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_defers_entire_round_when_writes_are_present(
    mock_openai_cls, mock_call_tool
):
    """If a round has BOTH read and write tool calls, drop everything in the
    round (the model can re-issue reads next turn). Don't execute the read
    and leave the write deferred."""
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call("query_tasks", '{"limit": 5}', tc_id="tc-r"),
            _tool_call(
                "create_tasks", '{"tasks": [{"name": "X"}]}', tc_id="tc-w"
            ),
        ]
    )
    mock_openai_cls.return_value = client

    _, _, pending = run_agent("...")

    mock_call_tool.assert_not_called()
    assert pending == [
        {"tool": "create_tasks", "arguments": {"tasks": [{"name": "X"}]}}
    ]


def test_looks_degenerate_detects_ellipsis_repetition():
    from agentic_tasks.agent.loop import _looks_degenerate

    bad = "Saturday, May 2:\n• Prepare ... ... … … … … … … … …"
    assert _looks_degenerate(bad)

    assert _looks_degenerate("foo ... ... ... ... ...")
    assert _looks_degenerate("a … … … … … b")


def test_looks_degenerate_allows_normal_ellipsis_use():
    from agentic_tasks.agent.loop import _looks_degenerate

    assert not _looks_degenerate("Done... I'll call you back.")
    assert not _looks_degenerate("Wait — and then it happened…")
    assert not _looks_degenerate(None)
    assert not _looks_degenerate("")
    # Single ellipsis in a real-shaped reply must not trip detection.
    assert not _looks_degenerate("Tomorrow:\n• Buy markers... eventually")
    # Normal "I need to know which" — not a self-coaching pattern.
    assert not _looks_degenerate(
        "I need to know which task you mean — APD or APR?"
    )


def test_looks_degenerate_detects_scattered_ellipses_with_question_marks():
    """Real production failure: ellipsis tokens spread across the reply,
    interleaved with `?` and other punctuation. Old regex required them to
    be consecutive — these pass it but the new count-based check catches
    them."""
    from agentic_tasks.agent.loop import _looks_degenerate

    bad = (
        "Tomorrow:\n• HEA: Get           ...  … ? ? ... ? ? … ..."
    )
    assert _looks_degenerate(bad)

    # Three scattered ellipses across multiple bullets.
    assert _looks_degenerate(
        "Today:\n• A...\n• B…\n• C..."
    )


def test_looks_degenerate_detects_thinking_channel_leakage():
    """gpt-oss reasoning channel sometimes leaks into the user-facing
    output. The phrases below have no place in a normal Telegram reply."""
    from agentic_tasks.agent.loop import _looks_degenerate

    assert _looks_degenerate(
        "The previous answer appears garbled. Need to correctly format..."
    )
    assert _looks_degenerate(
        "Provide bullet list with proper names, no extra formatting."
    )
    assert _looks_degenerate('Use "Tomorrow:" heading.')
    assert _looks_degenerate("Must not include ellipsis.")
    assert _looks_degenerate("The previous reply got truncated.")


def test_looks_degenerate_does_not_match_normal_phrasing():
    """Normal user-facing replies that happen to contain "need to" or
    similar must NOT trip the leak detector."""
    from agentic_tasks.agent.loop import _looks_degenerate

    assert not _looks_degenerate(
        "You'll need to confirm whether to mark it Done or shift it."
    )
    assert not _looks_degenerate(
        "Please provide the project name so I can match it."
    )
    # The bot writes day subheaders but should never SAY "use ... heading".
    assert not _looks_degenerate("<b>Tomorrow, May 3:</b>\n• Task A")


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_retries_degenerate_reply_once(mock_openai_cls, mock_call_tool):
    """When the model emits ellipsis-spam gibberish, the loop retries the
    same LLM call once and returns the clean response instead."""
    from agentic_tasks.agent.loop import run_agent

    bad = "Friday:\n• task ... … … … … … … …"
    good = "Friday:\n• task — at 5 PM"

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(content=bad),
        _response(content=good),
    ]
    mock_openai_cls.return_value = client

    reply, _, _ = run_agent("plan my week")

    assert reply == good
    assert client.chat.completions.create.call_count == 2


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_correction_note_is_injected_as_transient_system_message(
    mock_openai_cls, mock_call_tool
):
    """When run_agent gets a correction_note, it must appear in the
    messages sent to the LLM but NOT in agent_messages (so it's not
    persisted to the user's conversation history)."""
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(content="ok")
    mock_openai_cls.return_value = client

    note = "INTERNAL: User rejected previous proposal."
    _, agent_messages, _ = run_agent("the right one", correction_note=note)

    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assert any(
        m["role"] == "system" and m["content"] == note for m in sent
    )
    assert all(m.get("content") != note for m in agent_messages)


def test_check_guardrails_passes_on_complete_multi_day_reply():
    from agentic_tasks.agent.loop import _check_guardrails

    args = {"due_on_or_after": "2026-05-01", "due_on_or_before": "2026-05-03"}
    tasks = [{"name": "A"}, {"name": "B"}]
    reply = (
        "<b>Friday, May 1:</b>\n• A\n\n"
        "<b>Saturday, May 2:</b>\nNothing.\n\n"
        "<b>Sunday, May 3:</b>\n• B"
    )
    assert _check_guardrails(reply, args, tasks) is None


def test_check_guardrails_flags_missing_days():
    from agentic_tasks.agent.loop import _check_guardrails

    args = {"due_on_or_after": "2026-05-01", "due_on_or_before": "2026-05-07"}
    reply = "<b>Friday, May 1:</b>\n• A\n\n<b>Saturday, May 2:</b>\n• B"
    feedback = _check_guardrails(reply, args, [{"name": "A"}, {"name": "B"}])
    assert feedback is not None
    assert "2 of the 7" in feedback


def test_check_guardrails_flags_missing_task_names():
    from agentic_tasks.agent.loop import _check_guardrails

    args = {"due_on": "2026-05-01"}
    tasks = [{"name": "Buy markers"}, {"name": "Print insurance docs"}]
    reply = "Today:\n• Buy markers — at 5 PM"
    feedback = _check_guardrails(reply, args, tasks)
    assert feedback is not None
    assert "Print insurance docs" in feedback


def test_check_guardrails_tolerates_html_escaped_task_names():
    from agentic_tasks.agent.loop import _check_guardrails

    args = {"due_on": "2026-05-01"}
    tasks = [{"name": "Joe & Alberto"}]
    reply = "Today:\n• Joe &amp; Alberto"
    assert _check_guardrails(reply, args, tasks) is None


def test_check_guardrails_no_op_when_no_query_args():
    from agentic_tasks.agent.loop import _check_guardrails

    assert _check_guardrails("anything", None, None) is None


def test_check_guardrails_skips_day_check_for_single_day_query():
    from agentic_tasks.agent.loop import _check_guardrails

    args = {"due_on": "2026-05-01"}
    reply = "Today:\n• A"
    assert _check_guardrails(reply, args, [{"name": "A"}]) is None


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_retries_with_feedback_when_days_are_missing(
    mock_openai_cls, mock_call_tool
):
    """Range query asked for 7 days, model returns 4 → loop injects system
    feedback and runs another LLM round which produces the corrected reply."""
    from agentic_tasks.agent.loop import run_agent

    mock_call_tool.return_value = (
        '{"tasks": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}'
    )

    incomplete_reply = "<b>Friday, May 1:</b>\n• A"
    complete_reply = (
        "<b>Friday, May 1:</b>\n• A\n\n"
        "<b>Saturday, May 2:</b>\nNothing.\n\n"
        "<b>Sunday, May 3:</b>\nNothing.\n\n"
        "<b>Monday, May 4:</b>\nNothing.\n\n"
        "<b>Tuesday, May 5:</b>\nNothing.\n\n"
        "<b>Wednesday, May 6:</b>\nNothing.\n\n"
        "<b>Thursday, May 7:</b>\nNothing."
    )

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call(
                    "query_tasks",
                    '{"due_on_or_after": "2026-05-01", "due_on_or_before": "2026-05-07"}',
                    tc_id="tc-1",
                )
            ]
        ),
        _response(content=incomplete_reply),
        _response(content=complete_reply),
    ]
    mock_openai_cls.return_value = client

    reply, agent_messages, _ = run_agent("what's coming up?")

    assert reply == complete_reply
    assert client.chat.completions.create.call_count == 3
    final_assistant_msgs = [
        m for m in agent_messages
        if m["role"] == "assistant" and m.get("content") == complete_reply
    ]
    assert len(final_assistant_msgs) == 1
    assert not any(
        m.get("content") == incomplete_reply for m in agent_messages
    )


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_retries_with_feedback_when_tasks_are_dropped(
    mock_openai_cls, mock_call_tool
):
    from agentic_tasks.agent.loop import run_agent

    mock_call_tool.return_value = (
        '{"tasks": [{"name": "Buy markers"}, {"name": "Print insurance docs"}]}'
    )
    bad = "Today:\n• Buy markers"
    good = "Today:\n• Buy markers\n• Print insurance docs"

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("query_tasks", '{"due_on": "2026-05-01"}')]),
        _response(content=bad),
        _response(content=good),
    ]
    mock_openai_cls.return_value = client

    reply, _, _ = run_agent("what's today?")

    assert reply == good


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_does_not_retry_more_than_once_for_guardrails(
    mock_openai_cls, mock_call_tool
):
    """If the second attempt is also incomplete, accept it rather than
    looping forever."""
    from agentic_tasks.agent.loop import run_agent

    mock_call_tool.return_value = (
        '{"tasks": [{"name": "A"}, {"name": "B"}]}'
    )
    bad = "Today:\n• A"

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("query_tasks", '{"due_on": "2026-05-01"}')]),
        _response(content=bad),
        _response(content=bad),
    ]
    mock_openai_cls.return_value = client

    reply, _, _ = run_agent("what's today?")

    assert reply == bad
    assert client.chat.completions.create.call_count == 3


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_returns_degenerate_reply_if_retry_also_bad(
    mock_openai_cls, mock_call_tool
):
    """If the retry is also degenerate, return what we have rather than
    looping forever."""
    from agentic_tasks.agent.loop import run_agent

    bad = "… … … … … … …"
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(content=bad),
        _response(content=bad),
    ]
    mock_openai_cls.return_value = client

    reply, _, _ = run_agent("plan my week")

    assert reply == bad
    assert client.chat.completions.create.call_count == 2


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_handles_invalid_tool_arguments_json(mock_openai_cls, mock_call_tool):
    from agentic_tasks.agent.loop import run_agent

    mock_call_tool.return_value = "{}"
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("list_projects", "not-json")]),
        _response(content="ok"),
    ]
    mock_openai_cls.return_value = client

    reply, _, _ = run_agent("...")

    assert reply == "ok"
    mock_call_tool.assert_called_once_with("list_projects", {})


# ---- ghost-preview defense -------------------------------------------------


def test_looks_like_ghost_preview_matches_system_footer():
    from agentic_tasks.agent.loop import _looks_like_ghost_preview

    assert _looks_like_ghost_preview(
        'About to:\n• Update "X" — due tomorrow\n\nReply y to proceed or n to cancel.'
    )
    # Telegram-rendered HTML form (matches when <b> tags are present too).
    assert _looks_like_ghost_preview(
        'About to:\n• Update "X"\n\nReply <b>y</b> to proceed or <b>n</b> to cancel.'
    )


def test_looks_like_ghost_preview_ignores_normal_replies():
    from agentic_tasks.agent.loop import _looks_like_ghost_preview

    assert not _looks_like_ghost_preview("Done. Updated X.")
    assert not _looks_like_ghost_preview("I'll proceed once you reply.")
    assert not _looks_like_ghost_preview(None)
    assert not _looks_like_ghost_preview("")


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_retries_when_model_emits_ghost_preview(mock_openai_cls, mock_call_tool):
    """The model wrote a fake preview in chat instead of emitting a write
    tool. The loop must inject feedback and retry — and on the retry, the
    model emits the real tool call, the system shows the real preview."""
    from agentic_tasks.agent.loop import run_agent

    fake_preview = (
        'About to:\n• Update "Vietnam docs" — due tomorrow\n\n'
        "Reply y to proceed or n to cancel."
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(content=fake_preview),
        _response(
            tool_calls=[
                _tool_call(
                    "update_tasks",
                    '{"updates": [{"page_id": "p1", "due": "2026-05-03"}]}',
                    tc_id="tc-r",
                ),
            ]
        ),
    ]
    mock_openai_cls.return_value = client

    from unittest.mock import patch as _patch

    from agentic_tasks.notion_io.tasks import Task

    fake_task = Task(
        page_id="p1", name="Vietnam docs", status="To Do", priority=None, due=None
    )
    with _patch("agentic_tasks.agent.preview.get_task", return_value=fake_task):
        reply, agent_messages, pending = run_agent("reschedule Vietnam to tomorrow")

    # Two LLM calls: original (ghost) + retry (real tool).
    assert client.chat.completions.create.call_count == 2

    # Final user-facing reply is the system-generated preview, not the ghost.
    assert "About to" in reply
    assert "Vietnam docs" in reply
    assert pending is not None and pending[0]["tool"] == "update_tasks"

    # Persisted history must NOT contain the ghost — only the clean
    # system-generated preview the user actually sees.
    assert len(agent_messages) == 1
    assert agent_messages[0]["role"] == "assistant"
    assert agent_messages[0]["content"] == reply
    assert "Reply y to proceed or n to cancel" not in agent_messages[0]["content"] \
        or fake_preview not in agent_messages[0]["content"]

    # The retry round must have included a system message telling the model
    # what it did wrong.
    second_call_messages = client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    feedback_msgs = [
        m for m in second_call_messages
        if m["role"] == "system" and "imitated" in m["content"]
    ]
    assert len(feedback_msgs) == 1


def _format_check_response(verdict: str):
    """Build a chat-completions response for the format-check model."""
    msg = MagicMock()
    msg.content = verdict
    msg.tool_calls = []
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    return response


def test_check_format_returns_none_on_ok_verdict():
    from agentic_tasks.agent.loop import _check_format

    client = MagicMock()
    client.chat.completions.create.return_value = _format_check_response("OK")

    feedback = _check_format(client, "openai/gpt-oss-20b", "what's today?", "Today:\n• A")

    assert feedback is None
    client.chat.completions.create.assert_called_once()
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-oss-20b"


def test_check_format_returns_reason_on_bad_verdict():
    from agentic_tasks.agent.loop import _check_format

    client = MagicMock()
    client.chat.completions.create.return_value = _format_check_response(
        "BAD: contains repeated ellipsis tokens"
    )

    feedback = _check_format(
        client, "openai/gpt-oss-20b", "what's today?", "Today:\n• A ... ..."
    )

    assert feedback == "contains repeated ellipsis tokens"


def test_check_format_fails_open_on_exception():
    """Validator API errors must NOT block legitimate replies — fail open."""
    from agentic_tasks.agent.loop import _check_format

    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("validator down")

    feedback = _check_format(client, "openai/gpt-oss-20b", "q", "r")

    assert feedback is None


def test_check_format_fails_open_on_unparseable_verdict():
    from agentic_tasks.agent.loop import _check_format

    client = MagicMock()
    client.chat.completions.create.return_value = _format_check_response(
        "I think this looks fine to me actually"
    )

    feedback = _check_format(client, "openai/gpt-oss-20b", "q", "r")

    assert feedback is None


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_runs_format_check_when_enabled(
    mock_openai_cls, mock_call_tool, monkeypatch
):
    """When LLM_FORMAT_CHECK_MODEL is set, the loop calls the format checker
    after producing a final reply."""
    monkeypatch.setenv("LLM_FORMAT_CHECK_MODEL", "openai/gpt-oss-20b")
    from agentic_tasks.config import get_settings

    get_settings.cache_clear()

    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    # First call: main model produces reply. Second call: format check returns OK.
    client.chat.completions.create.side_effect = [
        _response(content="Today:\n• Buy markers"),
        _format_check_response("OK"),
    ]
    mock_openai_cls.return_value = client

    reply, _, pending = run_agent("what's today?")

    assert reply == "Today:\n• Buy markers"
    assert pending is None
    # Two calls: main agent + format check. Format check used the small model.
    assert client.chat.completions.create.call_count == 2
    second_call_kwargs = client.chat.completions.create.call_args_list[1].kwargs
    assert second_call_kwargs["model"] == "openai/gpt-oss-20b"


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_skips_format_check_when_disabled(mock_openai_cls, mock_call_tool):
    """Empty LLM_FORMAT_CHECK_MODEL disables the check — only one LLM call."""
    # conftest already sets LLM_FORMAT_CHECK_MODEL="" by default.
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(content="Hello.")
    mock_openai_cls.return_value = client

    run_agent("hi")

    assert client.chat.completions.create.call_count == 1


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_retries_when_format_check_says_bad(
    mock_openai_cls, mock_call_tool, monkeypatch
):
    """BAD verdict from the format checker triggers a one-shot retry of the
    main model with feedback. The corrected reply is returned to the user."""
    monkeypatch.setenv("LLM_FORMAT_CHECK_MODEL", "openai/gpt-oss-20b")
    from agentic_tasks.config import get_settings

    get_settings.cache_clear()

    from agentic_tasks.agent.loop import run_agent

    # Use replies that don't trip the cheap degenerate regex, so the BAD
    # verdict comes from the LLM checker (not the inline regex retry).
    bad = "Today:\n• Buy markers (something off here)"
    good = "Today:\n• Buy markers"
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(content=bad),
        _format_check_response("BAD: contains 3 ellipsis tokens"),
        _response(content=good),
        # No second format check — the retry-done flag prevents a re-validate
        # of the corrected reply, which is fine: at worst the user sees one
        # extra slightly-imperfect reply rather than an infinite loop.
    ]
    mock_openai_cls.return_value = client

    reply, agent_messages, _ = run_agent("what's today?")

    assert reply == good
    # 3 LLM calls: main, format check (BAD), main retry.
    assert client.chat.completions.create.call_count == 3
    # Persisted history contains exactly the corrected reply, not the bad one.
    assert len(agent_messages) == 1
    assert agent_messages[0] == {"role": "assistant", "content": good}

    # Retry round must have included a system message with the validator's
    # rejection reason.
    third_call_messages = client.chat.completions.create.call_args_list[2].kwargs[
        "messages"
    ]
    feedback_msgs = [
        m for m in third_call_messages
        if m["role"] == "system" and "format validator rejected" in m["content"]
    ]
    assert len(feedback_msgs) == 1
    assert "ellipsis tokens" in feedback_msgs[0]["content"]


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_accepts_second_bad_format_to_avoid_loop(
    mock_openai_cls, mock_call_tool, monkeypatch
):
    """If the retry is ALSO judged BAD, the loop accepts it rather than
    retrying forever."""
    monkeypatch.setenv("LLM_FORMAT_CHECK_MODEL", "openai/gpt-oss-20b")
    from agentic_tasks.config import get_settings

    get_settings.cache_clear()

    from agentic_tasks.agent.loop import run_agent

    # Plain replies that don't trip the cheap regex — the BAD verdicts come
    # from the format checker mock, not from the inline degenerate path.
    bad1 = "first attempt looks weird"
    bad2 = "second attempt also weird"
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(content=bad1),
        _format_check_response("BAD: weird"),
        _response(content=bad2),
        # No more format-check calls — flag is set after first BAD.
    ]
    mock_openai_cls.return_value = client

    reply, _, _ = run_agent("hi")

    assert reply == bad2
    # Three calls: main, check, main retry. Second check is skipped.
    assert client.chat.completions.create.call_count == 3


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_does_not_run_format_check_on_tool_call_rounds(
    mock_openai_cls, mock_call_tool, monkeypatch
):
    """Format check only fires on FINAL replies — never on rounds where the
    model emits a tool call (those go straight to dispatch)."""
    monkeypatch.setenv("LLM_FORMAT_CHECK_MODEL", "openai/gpt-oss-20b")
    from agentic_tasks.config import get_settings

    get_settings.cache_clear()

    from agentic_tasks.agent.loop import run_agent

    mock_call_tool.return_value = '{"tasks": []}'
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("query_tasks", '{"due_on": "2026-05-03"}')]),
        _response(content="Nothing for tomorrow."),
        _format_check_response("OK"),
    ]
    mock_openai_cls.return_value = client

    reply, _, _ = run_agent("anything tomorrow?")

    assert reply == "Nothing for tomorrow."
    # 3 LLM calls: tool round, final reply, format check on the final reply.
    # NOT 4 (no format check on the tool-call round).
    assert client.chat.completions.create.call_count == 3


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_does_not_loop_forever_on_repeated_ghost_preview(
    mock_openai_cls, mock_call_tool
):
    """If the model writes a ghost preview a second time, return it as-is
    rather than looping forever."""
    from agentic_tasks.agent.loop import run_agent

    fake = (
        'About to:\n• Create "X"\n\nReply y to proceed or n to cancel.'
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(content=fake),
        _response(content=fake),
    ]
    mock_openai_cls.return_value = client

    reply, _, pending = run_agent("create X")

    assert reply == fake
    assert pending is None
    assert client.chat.completions.create.call_count == 2
