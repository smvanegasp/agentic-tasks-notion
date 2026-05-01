"""Confirmation-gate preview formatting + plan execution."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest


def test_is_confirmation_matches_common_yes_forms():
    from agentic_tasks.agent.preview import is_confirmation

    for word in ("y", "Y", "yes", "YES", "yeah", "yep", "ok", "okay", "sí", "si", "sure"):
        assert is_confirmation(word), word
    assert is_confirmation(" yes ")
    assert is_confirmation("y!")


def test_is_confirmation_rejects_non_confirmation():
    from agentic_tasks.agent.preview import is_confirmation

    for word in ("yes please", "definitely", "go ahead", "n", "no", ""):
        assert not is_confirmation(word), word


def test_is_rejection_matches_no_forms():
    from agentic_tasks.agent.preview import is_rejection

    for word in (
        "n", "N", "no", "NO", "nope", "nah", " no ",
        "cancel", "Cancel", "CANCEL",
        "cancela", "cancelar",
        "stop", "abort",
    ):
        assert is_rejection(word), word


def test_is_rejection_rejects_non_no_forms():
    from agentic_tasks.agent.preview import is_rejection

    assert not is_rejection("not now")
    assert not is_rejection("no, change due")  # clarification, not pure rejection
    assert not is_rejection("y")
    assert not is_rejection("cancel that one")  # has trailing words


def test_format_preview_create_basic():
    from agentic_tasks.agent.preview import format_preview

    plan = [{"tool": "create_task", "arguments": {"name": "Buy markers"}}]
    out = format_preview(plan)
    assert "About to:" in out
    assert "Buy markers" in out
    assert "<b>y</b>" in out
    assert "<b>n</b>" in out


def test_format_preview_create_with_due_priority_project():
    from agentic_tasks.agent.preview import format_preview

    plan = [
        {
            "tool": "create_task",
            "arguments": {
                "name": "Submit report",
                "due": "2026-05-08",
                "priority": "High",
                "project_name": "MBA",
            },
        }
    ]
    out = format_preview(plan)
    assert "Submit report" in out
    assert "priority High" in out
    assert "project: MBA" in out
    # date should be friendly, not ISO
    assert "2026-05-08" not in out


def test_format_preview_create_escapes_html_in_name():
    from agentic_tasks.agent.preview import format_preview

    plan = [{"tool": "create_task", "arguments": {"name": "<script>alert(1)</script>"}}]
    out = format_preview(plan)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_format_preview_update_resolves_task_name():
    from agentic_tasks.agent.preview import format_preview
    from agentic_tasks.notion_io.tasks import Task

    fake_task = Task(
        page_id="page-123",
        name="Vietnam docs",
        status="To Do",
        priority=None,
        due=None,
    )
    plan = [
        {
            "tool": "update_task",
            "arguments": {"page_id": "page-123", "due": "2026-05-05", "priority": "High"},
        }
    ]
    with patch("agentic_tasks.agent.preview.get_task", return_value=fake_task):
        out = format_preview(plan)
    assert "Vietnam docs" in out
    assert "priority → High" in out
    assert "due →" in out


def test_format_preview_update_falls_back_when_get_task_fails():
    from agentic_tasks.agent.preview import format_preview

    plan = [{"tool": "update_task", "arguments": {"page_id": "abcdef12345", "due": ""}}]
    with patch(
        "agentic_tasks.agent.preview.get_task", side_effect=RuntimeError("api down")
    ):
        out = format_preview(plan)
    # Should not crash; should include a hint with the page-id prefix.
    assert "abcdef12" in out
    assert "due → cleared" in out


def test_format_preview_complete():
    from agentic_tasks.agent.preview import format_preview
    from agentic_tasks.notion_io.tasks import Task

    fake_task = Task(
        page_id="page-9",
        name="Schedule Phil",
        status="To Do",
        priority=None,
        due=None,
    )
    plan = [{"tool": "complete_task", "arguments": {"page_id": "page-9"}}]
    with patch("agentic_tasks.agent.preview.get_task", return_value=fake_task):
        out = format_preview(plan)
    assert "Complete" in out
    assert "Schedule Phil" in out


def test_serialize_plan_keeps_only_writes():
    from unittest.mock import MagicMock

    from agentic_tasks.agent.preview import serialize_plan

    def _tc(name: str, args_json: str = "{}") -> MagicMock:
        tc = MagicMock()
        tc.function.name = name
        tc.function.arguments = args_json
        return tc

    tool_calls = [
        _tc("query_tasks", '{"limit": 10}'),
        _tc("create_task", '{"name": "X"}'),
        _tc("list_projects", "{}"),
        _tc("update_task", '{"page_id": "p1", "due": "2026-05-05"}'),
    ]
    plan = serialize_plan(tool_calls)
    assert [e["tool"] for e in plan] == ["create_task", "update_task"]
    assert plan[0]["arguments"] == {"name": "X"}


def test_execute_plan_creates_and_returns_summary_with_link():
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    created = Task(
        page_id="page-x",
        name="Buy markers",
        status="To Do",
        priority=None,
        due=None,
        url="https://www.notion.so/Buy-markers-pagex",
    )
    with patch("agentic_tasks.agent.preview.create_task", return_value=created):
        out = execute_plan([{"tool": "create_task", "arguments": {"name": "Buy markers"}}])
    assert out.startswith("Done.")
    assert "Created" in out
    assert "Buy markers" in out
    assert '<a href="https://www.notion.so/Buy-markers-pagex">open</a>' in out


def test_execute_plan_handles_partial_failure():
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    good = Task(
        page_id="p-ok",
        name="Good one",
        status="To Do",
        priority=None,
        due=None,
        url="https://www.notion.so/p-ok",
    )

    create_calls = {"n": 0}

    def fake_create(*args, **kwargs):
        create_calls["n"] += 1
        if create_calls["n"] == 1:
            return good
        raise RuntimeError("notion is down")

    with patch("agentic_tasks.agent.preview.create_task", side_effect=fake_create):
        out = execute_plan(
            [
                {"tool": "create_task", "arguments": {"name": "Good one"}},
                {"tool": "create_task", "arguments": {"name": "Bad one"}},
            ]
        )
    assert "Partial. 1 done, 1 failed." in out
    assert "Good one" in out
    assert "notion is down" in out


@pytest.mark.parametrize(
    "iso,expected_substr",
    [
        ("2026-05-08", "Friday, May 8"),
        ("2026-05-08T17:00:00", "at 5 PM"),
        ("2026-05-08T09:30:00", "at 9:30 AM"),
    ],
)
def test_format_date_human_friendly_format(iso, expected_substr):
    from agentic_tasks.agent.preview import _format_date_human

    today = date(2026, 5, 1)
    assert expected_substr in _format_date_human(iso, today)


def test_format_preview_shift_due_dates_lists_each_task():
    """The shift preview resolves the filter against Notion at preview time
    and lists each affected task with old → new dates."""
    from datetime import date as _date

    from agentic_tasks.agent.preview import format_preview
    from agentic_tasks.notion_io.tasks import Task

    targets = [
        Task(
            page_id="p1",
            name="Buy markers",
            status="To Do",
            priority=None,
            due=_date(2026, 5, 1),
        ),
        Task(
            page_id="p2",
            name="No due here",
            status="To Do",
            priority=None,
            due=None,
        ),
    ]
    plan = [
        {
            "tool": "shift_due_dates",
            "arguments": {"delta_days": 7, "due_on_or_after": "2026-05-01"},
        }
    ]
    with patch(
        "agentic_tasks.agent.preview.resolve_shift_targets",
        return_value=targets,
    ):
        out = format_preview(plan)
    assert "Shift 2 task(s) by +7 day(s):" in out
    assert "Buy markers" in out
    assert "no due date, skipped" in out


def test_format_preview_shift_due_dates_handles_no_matches():
    from agentic_tasks.agent.preview import format_preview

    plan = [{"tool": "shift_due_dates", "arguments": {"delta_days": 1, "due_on": "2026-05-01"}}]
    with patch("agentic_tasks.agent.preview.resolve_shift_targets", return_value=[]):
        out = format_preview(plan)
    assert "no matching open tasks" in out


def test_execute_plan_shift_due_dates_flattens_results():
    """Each shifted task should produce its own success entry in the summary."""
    from datetime import date as _date

    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    shifted = [
        Task(
            page_id="p1",
            name="A",
            status="To Do",
            priority=None,
            due=_date(2026, 5, 8),
            url="https://notion.so/p1",
        ),
        Task(
            page_id="p2",
            name="B",
            status="To Do",
            priority=None,
            due=_date(2026, 5, 9),
            url="https://notion.so/p2",
        ),
    ]
    with patch(
        "agentic_tasks.agent.preview.execute_shift_due_dates",
        return_value=shifted,
    ):
        out = execute_plan(
            [{"tool": "shift_due_dates", "arguments": {"delta_days": 7, "due_on": "2026-05-01"}}]
        )
    assert "Done." in out
    assert out.count("Shifted") == 2
    assert "A" in out
    assert "B" in out


def test_format_date_human_today_and_tomorrow():
    from agentic_tasks.agent.preview import _format_date_human

    today = date(2026, 5, 1)
    assert _format_date_human("2026-05-01", today) == "today"
    assert _format_date_human("2026-05-02", today) == "tomorrow"
