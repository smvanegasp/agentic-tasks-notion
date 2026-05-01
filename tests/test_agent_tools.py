from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

from agentic_tasks.agent.tools import call_tool
from agentic_tasks.notion_io.tasks import Task


def _task() -> Task:
    return Task(
        page_id="page-1",
        name="Test task",
        status="To Do",
        priority="High",
        due=date(2026, 5, 1),
        project_ids=[],
        labels=[],
        description="",
        my_day=False,
        url="",
    )


@patch("agentic_tasks.agent.tools.query_tasks")
def test_query_tasks_default_excludes_done(mock_query):
    mock_query.return_value = []

    call_tool("query_tasks", {})

    filter_ = mock_query.call_args.kwargs["filter_"]
    assert filter_ == {"property": "Status", "status": {"does_not_equal": "Done"}}


@patch("agentic_tasks.agent.tools.query_tasks")
def test_query_tasks_sorts_by_priority_then_due(mock_query):
    """High > Medium > Low; within priority, soonest due first."""
    import json as _json

    def make(name, priority, due):
        return Task(
            page_id=name,
            name=name,
            status="To Do",
            priority=priority,
            due=due,
            project_ids=[],
            labels=[],
            description="",
            my_day=False,
            url="",
        )

    mock_query.return_value = [
        make("low-late", "Low", date(2026, 5, 5)),
        make("high-late", "High", date(2026, 5, 10)),
        make("medium-early", "Medium", date(2026, 5, 1)),
        make("high-early", "High", date(2026, 5, 1)),
        make("no-priority", None, date(2026, 5, 1)),
    ]

    result = call_tool("query_tasks", {})
    names = [t["name"] for t in _json.loads(result)["tasks"]]

    assert names == [
        "high-early",
        "high-late",
        "medium-early",
        "low-late",
        "no-priority",
    ]


@patch("agentic_tasks.agent.tools.query_tasks")
def test_query_tasks_due_on_specific_date(mock_query):
    mock_query.return_value = []

    call_tool("query_tasks", {"due_on": "2026-05-01"})

    filter_ = mock_query.call_args.kwargs["filter_"]
    # Default exclude-Done is still applied, so we expect an "and"
    assert "and" in filter_
    has_due = any(
        c.get("property") == "Due" and c.get("date", {}).get("equals") == "2026-05-01"
        for c in filter_["and"]
    )
    assert has_due


@patch("agentic_tasks.agent.tools.query_tasks")
def test_query_tasks_date_range(mock_query):
    """Range query: due >= 2026-05-01 AND due <= 2026-05-07 AND status != Done."""
    mock_query.return_value = []

    call_tool(
        "query_tasks",
        {"due_on_or_after": "2026-05-01", "due_on_or_before": "2026-05-07"},
    )

    filter_ = mock_query.call_args.kwargs["filter_"]
    assert "and" in filter_
    cond_strs = [str(c) for c in filter_["and"]]
    assert any("'on_or_after': '2026-05-01'" in s for s in cond_strs)
    assert any("'on_or_before': '2026-05-07'" in s for s in cond_strs)


@patch("agentic_tasks.agent.tools.query_tasks")
def test_query_tasks_include_done_drops_default_filter(mock_query):
    mock_query.return_value = []

    call_tool("query_tasks", {"include_done": True})

    filter_ = mock_query.call_args.kwargs["filter_"]
    assert filter_ is None  # no conditions → no filter, returns everything


@patch("agentic_tasks.agent.tools.query_tasks")
def test_query_tasks_combined_conditions(mock_query):
    mock_query.return_value = []

    call_tool("query_tasks", {"my_day": True, "status": "Doing"})

    filter_ = mock_query.call_args.kwargs["filter_"]
    assert "and" in filter_
    assert any(c.get("property") == "My Day" for c in filter_["and"])
    assert any(c.get("property") == "Status" for c in filter_["and"])


@patch("agentic_tasks.agent.tools.find_project_by_name")
@patch("agentic_tasks.agent.tools.create_task")
def test_create_task_resolves_project(mock_create, mock_find):
    mock_find.return_value = MagicMock(page_id="proj-1", name="MBA Prep")
    mock_create.return_value = _task()

    call_tool("create_task", {"name": "Study", "project_name": "mba"})

    assert mock_create.call_args.kwargs["project_ids"] == ["proj-1"]


@patch("agentic_tasks.agent.tools.find_project_by_name")
def test_create_task_unknown_project_returns_error(mock_find):
    mock_find.return_value = None

    result = call_tool("create_task", {"name": "X", "project_name": "zzzz"})

    assert "error" in result
    assert "zzzz" in result


@patch("agentic_tasks.agent.tools.create_task")
def test_create_task_parses_iso_date(mock_create):
    mock_create.return_value = _task()

    call_tool("create_task", {"name": "X", "due": "2026-05-01"})

    assert mock_create.call_args.kwargs["due"] == date(2026, 5, 1)


@patch("agentic_tasks.agent.tools.create_task")
def test_create_task_attaches_configured_tz_to_naive_datetime(mock_create):
    """If the LLM forgets the offset, the dispatcher must attach the user's
    configured timezone — never let Notion default to UTC."""
    from agentic_tasks.config import get_settings

    mock_create.return_value = _task()

    call_tool("create_task", {"name": "X", "due": "2026-04-30T21:30:00"})

    due = mock_create.call_args.kwargs["due"]
    assert isinstance(due, datetime)
    assert due.tzinfo == get_settings().timezone


@patch("agentic_tasks.agent.tools.create_task")
def test_create_task_keeps_explicit_offset(mock_create):
    mock_create.return_value = _task()

    call_tool("create_task", {"name": "X", "due": "2026-04-30T21:30:00-04:00"})

    due = mock_create.call_args.kwargs["due"]
    assert isinstance(due, datetime)
    assert due.utcoffset() == timedelta(hours=-4)


@patch("agentic_tasks.agent.tools.complete_task")
def test_complete_task_dispatches(mock_complete):
    mock_complete.return_value = _task()

    call_tool("complete_task", {"page_id": "page-1"})

    mock_complete.assert_called_once_with("page-1")


@patch("agentic_tasks.agent.tools.update_task")
def test_update_task_clears_due_on_empty_string(mock_update):
    mock_update.return_value = _task()

    call_tool("update_task", {"page_id": "page-1", "due": ""})

    assert mock_update.call_args.kwargs["due"] is None


@patch("agentic_tasks.agent.tools.update_task")
def test_update_task_status_change(mock_update):
    mock_update.return_value = _task()

    call_tool("update_task", {"page_id": "page-1", "status": "Doing"})

    assert mock_update.call_args.kwargs["status"].value == "Doing"


@patch("agentic_tasks.agent.tools.query_tasks")
def test_find_tasks_returns_acronym_match(mock_query):
    """Sanity-check the user's reported case: 'APD' should match 'Respond to
    the offer from APD' (rapidfuzz partial_ratio gives 100 for substring)."""
    import json as _json

    def make(name):
        return Task(
            page_id=f"id-{name}",
            name=name,
            status="To Do",
            priority=None,
            due=None,
            project_ids=[],
            labels=[],
            description="",
            my_day=False,
            url="",
        )

    mock_query.return_value = [
        make("Schedule the cinema"),
        make("Respond to the offer from APD"),
        make("go to the doctor"),
    ]

    result = _json.loads(call_tool("find_tasks", {"name_query": "APD"}))
    names = [t["name"] for t in result["tasks"]]
    assert names[0] == "Respond to the offer from APD"


@patch("agentic_tasks.agent.tools.query_tasks")
def test_find_tasks_returns_empty_below_threshold(mock_query):
    import json as _json

    def make(name):
        return Task(
            page_id=f"id-{name}",
            name=name,
            status="To Do",
            priority=None,
            due=None,
        )

    mock_query.return_value = [make("Buy markers"), make("Doctor appointment")]
    result = _json.loads(call_tool("find_tasks", {"name_query": "xzqyx"}))
    assert result["tasks"] == []


@patch("agentic_tasks.agent.tools.query_tasks")
def test_find_tasks_excludes_done_by_default(mock_query):
    mock_query.return_value = []
    call_tool("find_tasks", {"name_query": "anything"})

    filter_ = mock_query.call_args.kwargs["filter_"]
    assert filter_ == {"property": "Status", "status": {"does_not_equal": "Done"}}


@patch("agentic_tasks.agent.tools.query_tasks")
def test_find_tasks_includes_done_when_requested(mock_query):
    mock_query.return_value = []
    call_tool("find_tasks", {"name_query": "x", "include_done": True})

    assert mock_query.call_args.kwargs["filter_"] is None


def test_find_tasks_rejects_empty_query():
    result = call_tool("find_tasks", {"name_query": "  "})
    assert "error" in result


def test_unknown_tool_returns_error():
    result = call_tool("does_not_exist", {})
    assert "Unknown tool" in result


@patch("agentic_tasks.agent.tools.create_task", side_effect=RuntimeError("boom"))
def test_tool_exception_caught_and_returned_as_error(_mock):
    result = call_tool("create_task", {"name": "X"})
    assert "error" in result
    assert "boom" in result
