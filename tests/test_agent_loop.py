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
    # Find the date-hint system message (not the main system prompt)
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
    """When the model emits create_task / update_task / complete_task, the
    loop must NOT execute it. It returns a preview + pending plan instead."""
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call("create_task", '{"name": "Buy markers"}', tc_id="tc-w1"),
        ]
    )
    mock_openai_cls.return_value = client

    reply, agent_messages, pending = run_agent("create buy markers")

    # The write must NOT have been executed.
    mock_call_tool.assert_not_called()
    # The loop should make exactly one LLM call before exiting.
    assert client.chat.completions.create.call_count == 1
    assert pending == [
        {"tool": "create_task", "arguments": {"name": "Buy markers"}}
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
            _tool_call("create_task", '{"name": "X"}', tc_id="tc-w"),
        ]
    )
    mock_openai_cls.return_value = client

    _, _, pending = run_agent("...")

    mock_call_tool.assert_not_called()
    assert pending == [{"tool": "create_task", "arguments": {"name": "X"}}]


def test_looks_degenerate_detects_ellipsis_repetition():
    from agentic_tasks.agent.loop import _looks_degenerate

    # Real failure mode observed from gpt-oss-120b on Groq.
    bad = "Saturday, May 2:\n• Prepare ... ... … … … … … … … …"
    assert _looks_degenerate(bad)

    # Mixed ASCII and Unicode ellipses count too.
    assert _looks_degenerate("foo ... ... ... ... ...")
    assert _looks_degenerate("a … … … … … b")


def test_looks_degenerate_allows_normal_ellipsis_use():
    from agentic_tasks.agent.loop import _looks_degenerate

    # Regular prose with a single ellipsis is fine.
    assert not _looks_degenerate("Done... I'll call you back.")
    assert not _looks_degenerate("Wait — and then it happened…")
    assert not _looks_degenerate(None)
    assert not _looks_degenerate("")


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
    # Correction note must NOT leak into persisted history.
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
    # Only 2 of 7 days shown.
    reply = "<b>Friday, May 1:</b>\n• A\n\n<b>Saturday, May 2:</b>\n• B"
    feedback = _check_guardrails(reply, args, [{"name": "A"}, {"name": "B"}])
    assert feedback is not None
    assert "2 of the 7" in feedback


def test_check_guardrails_flags_missing_task_names():
    from agentic_tasks.agent.loop import _check_guardrails

    args = {"due_on": "2026-05-01"}
    tasks = [{"name": "Buy markers"}, {"name": "Print insurance docs"}]
    reply = "Today:\n• Buy markers — at 5 PM"  # second task missing
    feedback = _check_guardrails(reply, args, tasks)
    assert feedback is not None
    assert "Print insurance docs" in feedback


def test_check_guardrails_tolerates_html_escaped_task_names():
    from agentic_tasks.agent.loop import _check_guardrails

    args = {"due_on": "2026-05-01"}
    tasks = [{"name": "Joe & Alberto"}]
    # Reply has the &amp; HTML entity instead of literal &.
    reply = "Today:\n• Joe &amp; Alberto"
    assert _check_guardrails(reply, args, tasks) is None


def test_check_guardrails_no_op_when_no_query_args():
    from agentic_tasks.agent.loop import _check_guardrails

    assert _check_guardrails("anything", None, None) is None


def test_count_action_verbs_recognizes_multi_action_request():
    from agentic_tasks.agent.loop import _count_action_verbs

    # The exact failure mode that prompted this guardrail.
    assert (
        _count_action_verbs(
            "Can you mark as done go to the doctor And also reschedule cinema for tomorrow"
        )
        == 2
    )
    assert _count_action_verbs("complete X and update Y") == 2
    assert _count_action_verbs("create A and reschedule B") == 2


def test_count_action_verbs_returns_one_for_single_action():
    from agentic_tasks.agent.loop import _count_action_verbs

    assert _count_action_verbs("mark the doctor as done") == 1
    assert _count_action_verbs("Reschedule respond offer from APD") == 1
    assert _count_action_verbs("create a task to call John and Mary") == 1


def test_count_action_verbs_is_zero_for_queries_and_chitchat():
    from agentic_tasks.agent.loop import _count_action_verbs

    assert _count_action_verbs("what do I have today?") == 0
    assert _count_action_verbs("And the cinema?") == 0
    assert _count_action_verbs("Cool thank u") == 0


def test_count_action_verbs_handles_spanish():
    from agentic_tasks.agent.loop import _count_action_verbs

    assert _count_action_verbs("marca la tarea como hecha y reagenda cinema") == 2
    assert _count_action_verbs("crea una tarea") == 1


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_retries_when_user_asked_for_two_actions_but_model_did_one(
    mock_openai_cls, mock_call_tool
):
    """User asks for 2 actions, model emits 1 write → guardrail injects
    feedback and a second LLM round produces both writes. The final preview
    contains both."""
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        # First write attempt: only one of two actions. Use create_task so
        # format_preview doesn't fan out to a real Notion get_task call.
        _response(
            tool_calls=[
                _tool_call(
                    "create_task",
                    '{"name": "doctor follow-up"}',
                    tc_id="tc-1",
                ),
            ]
        ),
        # After feedback, model emits both writes.
        _response(
            tool_calls=[
                _tool_call(
                    "create_task",
                    '{"name": "doctor follow-up"}',
                    tc_id="tc-2",
                ),
                _tool_call(
                    "create_task",
                    '{"name": "cinema reschedule"}',
                    tc_id="tc-3",
                ),
            ]
        ),
    ]
    mock_openai_cls.return_value = client

    # Message has exactly 2 action verbs ("create" twice). Avoid messages
    # with hidden verb counts ("reschedule" inside a task name) that would
    # raise the expectation above what the second mock can satisfy.
    reply, _agent_messages, plan = run_agent("create X and create Y")

    assert plan is not None
    assert len(plan) == 2
    # Two LLM calls: the failing first write attempt + the corrected one.
    assert client.chat.completions.create.call_count == 2


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_loop_does_not_retry_for_single_action_request(
    mock_openai_cls, mock_call_tool
):
    """One-verb request should NOT trigger the multi-action retry, even
    when the model emits one write — that's the expected flow."""
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call("create_task", '{"name": "X"}', tc_id="tc-1"),
        ]
    )
    mock_openai_cls.return_value = client

    _reply, _agent_messages, plan = run_agent("create X")

    assert plan is not None
    assert len(plan) == 1
    assert client.chat.completions.create.call_count == 1


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_multi_action_guardrail_caps_retries(mock_openai_cls, mock_call_tool):
    """If both retries are also incomplete, accept the third attempt rather
    than looping forever. The user can ask for the missing action separately."""
    from agentic_tasks.agent.loop import _MAX_MULTI_ACTION_RETRIES, run_agent

    client = MagicMock()
    only_one = _response(
        tool_calls=[
            _tool_call("create_task", '{"name": "X"}', tc_id="tc-1"),
        ]
    )
    # Original attempt + N retries = N+1 total LLM calls.
    client.chat.completions.create.side_effect = [
        only_one for _ in range(_MAX_MULTI_ACTION_RETRIES + 1)
    ]
    mock_openai_cls.return_value = client

    _reply, _agent_messages, plan = run_agent("create X and create Y")

    assert plan is not None
    assert len(plan) == 1
    assert (
        client.chat.completions.create.call_count == _MAX_MULTI_ACTION_RETRIES + 1
    )


@patch("agentic_tasks.agent.loop.call_tool")
@patch("agentic_tasks.agent.loop.OpenAI")
def test_multi_action_feedback_quotes_user_message(mock_openai_cls, mock_call_tool):
    """The retry-feedback system message must include the user's verbatim
    message and the verbs we detected — that grounding is what helps the
    model catch the dropped clause on the second try."""
    from agentic_tasks.agent.loop import run_agent

    client = MagicMock()
    user_msg = "complete X and reschedule Y"
    client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[_tool_call("create_task", '{"name": "X"}', tc_id="tc-1")]
        ),
        # Second call returns both — sufficient.
        _response(
            tool_calls=[
                _tool_call("create_task", '{"name": "X"}', tc_id="tc-2"),
                _tool_call("create_task", '{"name": "Y"}', tc_id="tc-3"),
            ]
        ),
    ]
    mock_openai_cls.return_value = client

    run_agent(user_msg)

    sent = client.chat.completions.create.call_args.kwargs["messages"]
    feedback_msgs = [
        m for m in sent
        if m.get("role") == "system" and "MULTI-ACTION RECOVERY" in m.get("content", "")
    ]
    assert len(feedback_msgs) == 1
    body = feedback_msgs[0]["content"]
    assert user_msg in body  # verbatim user message
    assert "complete" in body and "reschedule" in body  # detected verbs


def test_check_guardrails_skips_day_check_for_single_day_query():
    from agentic_tasks.agent.loop import _check_guardrails

    args = {"due_on": "2026-05-01"}  # not a range
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

    # Tool result for the range query — 1 task per day for 3 days.
    mock_call_tool.return_value = (
        '{"tasks": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}'
    )

    incomplete_reply = (
        "<b>Friday, May 1:</b>\n• A"
        # 6 missing days
    )
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
        # 1: model decides to call query_tasks for the range.
        _response(
            tool_calls=[
                _tool_call(
                    "query_tasks",
                    '{"due_on_or_after": "2026-05-01", "due_on_or_before": "2026-05-07"}',
                    tc_id="tc-1",
                )
            ]
        ),
        # 2: model produces incomplete reply — guardrail catches it.
        _response(content=incomplete_reply),
        # 3: after feedback is injected, model produces the full reply.
        _response(content=complete_reply),
    ]
    mock_openai_cls.return_value = client

    reply, agent_messages, _ = run_agent("what's coming up?")

    assert reply == complete_reply
    # 3 LLM calls in total: query, broken reply, corrected reply.
    assert client.chat.completions.create.call_count == 3
    # Persisted history should include the corrected reply, NOT the broken
    # one (the failed attempt is internal-only).
    final_assistant_msgs = [
        m for m in agent_messages
        if m["role"] == "assistant" and m.get("content") == complete_reply
    ]
    assert len(final_assistant_msgs) == 1
    # The broken reply must not be persisted.
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
    bad = "Today:\n• Buy markers"  # second task dropped
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
    bad = "Today:\n• A"  # missing B both times

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("query_tasks", '{"due_on": "2026-05-01"}')]),
        _response(content=bad),
        _response(content=bad),
    ]
    mock_openai_cls.return_value = client

    reply, _, _ = run_agent("what's today?")

    assert reply == bad
    # Exactly 3 LLM calls — no third retry.
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
