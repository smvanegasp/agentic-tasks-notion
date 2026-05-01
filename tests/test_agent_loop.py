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

    reply, agent_messages = run_agent("hi")

    assert reply == "Hello."
    assert agent_messages == [{"role": "assistant", "content": "Hello."}]
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

    reply, agent_messages = run_agent("any tasks today?")

    assert reply == "No tasks today."
    mock_call_tool.assert_called_once_with("query_tasks", {"due_filter": "today"})

    # agent_messages should contain: assistant-with-tool-call, tool, assistant-final
    assert len(agent_messages) == 3
    assert agent_messages[0]["role"] == "assistant"
    assert agent_messages[0]["tool_calls"][0]["function"]["name"] == "query_tasks"
    assert agent_messages[1]["role"] == "tool"
    assert agent_messages[1]["tool_call_id"] == "tc-1"
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

    reply, _ = run_agent("loop forever", max_iterations=3)

    assert "iteration limit" in reply
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
def test_loop_handles_invalid_tool_arguments_json(mock_openai_cls, mock_call_tool):
    from agentic_tasks.agent.loop import run_agent

    mock_call_tool.return_value = "{}"
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("list_projects", "not-json")]),
        _response(content="ok"),
    ]
    mock_openai_cls.return_value = client

    reply, _ = run_agent("...")

    assert reply == "ok"
    mock_call_tool.assert_called_once_with("list_projects", {})
