"""Confirmation-gate preview formatting + plan execution."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _stub_get_task(monkeypatch):
    """Default: ``get_task`` raises so ``_resolve_task_name`` falls back to
    "this task". Tests that need a real resolved name patch get_task
    explicitly inside the test body — those overrides win over this stub
    because ``with patch(...)`` re-binds the same module attribute."""

    def _raise(page_id):
        raise RuntimeError(f"get_task not stubbed for {page_id}")

    monkeypatch.setattr("agentic_tasks.agent.preview.get_task", _raise)


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


# ---- create_tasks preview -------------------------------------------------


def test_format_preview_create_basic():
    from agentic_tasks.agent.preview import format_preview

    plan = [{"tool": "create_tasks", "arguments": {"tasks": [{"name": "Buy markers"}]}}]
    out = format_preview(plan)
    assert "About to:" in out
    assert "Buy markers" in out
    assert "<b>y</b>" in out
    assert "<b>n</b>" in out


def test_format_preview_create_with_due_priority_project():
    from agentic_tasks.agent.preview import format_preview

    plan = [
        {
            "tool": "create_tasks",
            "arguments": {
                "tasks": [
                    {
                        "name": "Submit report",
                        "due": "2026-05-08",
                        "priority": "High",
                        "project_name": "MBA",
                    }
                ]
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

    plan = [
        {
            "tool": "create_tasks",
            "arguments": {"tasks": [{"name": "<script>alert(1)</script>"}]},
        }
    ]
    out = format_preview(plan)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_format_preview_create_lists_each_task_in_batch():
    """A batch of N creates should produce N bullet lines under one header."""
    from agentic_tasks.agent.preview import format_preview

    plan = [
        {
            "tool": "create_tasks",
            "arguments": {
                "tasks": [
                    {"name": "Task A"},
                    {"name": "Task B"},
                    {"name": "Task C"},
                ]
            },
        }
    ]
    out = format_preview(plan)
    assert out.count("About to:") == 1
    assert "Task A" in out
    assert "Task B" in out
    assert "Task C" in out
    # One y/n footer for the whole batch
    assert out.count("<b>y</b>") == 1


# ---- update_tasks preview -------------------------------------------------


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
            "tool": "update_tasks",
            "arguments": {
                "updates": [
                    {"page_id": "page-123", "due": "2026-05-05", "priority": "High"}
                ]
            },
        }
    ]
    with patch("agentic_tasks.agent.preview.get_task", return_value=fake_task):
        out = format_preview(plan)
    assert "Vietnam docs" in out
    assert "priority → High" in out
    assert "due →" in out


def test_format_preview_update_falls_back_when_get_task_fails():
    """When get_task fails (page deleted, stale id, permissions), the
    preview must NOT leak the page id — we show a neutral phrase instead."""
    from agentic_tasks.agent.preview import format_preview

    plan = [
        {
            "tool": "update_tasks",
            "arguments": {"updates": [{"page_id": "abcdef12345", "due": ""}]},
        }
    ]
    with patch(
        "agentic_tasks.agent.preview.get_task", side_effect=RuntimeError("api down")
    ):
        out = format_preview(plan)
    assert "abcdef12" not in out
    assert "this task" in out
    assert "due → cleared" in out


def test_format_preview_update_lists_each_in_batch():
    """Reschedule three tasks at once — preview should list all three."""
    from agentic_tasks.agent.preview import format_preview
    from agentic_tasks.notion_io.tasks import Task

    def fake_get(page_id):
        return Task(
            page_id=page_id,
            name=f"Task {page_id[-1].upper()}",
            status="To Do",
            priority=None,
            due=None,
        )

    plan = [
        {
            "tool": "update_tasks",
            "arguments": {
                "updates": [
                    {"page_id": "p-a", "due": "2026-05-09"},
                    {"page_id": "p-b", "due": "2026-05-09"},
                    {"page_id": "p-c", "due": "2026-05-09"},
                ]
            },
        }
    ]
    with patch("agentic_tasks.agent.preview.get_task", side_effect=fake_get):
        out = format_preview(plan)

    assert "Task A" in out
    assert "Task B" in out
    assert "Task C" in out
    assert out.count("<b>y</b>") == 1


# ---- complete_tasks preview ----------------------------------------------


def test_format_preview_complete_single():
    from agentic_tasks.agent.preview import format_preview
    from agentic_tasks.notion_io.tasks import Task

    fake_task = Task(
        page_id="page-9",
        name="Schedule Phil",
        status="To Do",
        priority=None,
        due=None,
    )
    plan = [
        {"tool": "complete_tasks", "arguments": {"page_ids": ["page-9"]}}
    ]
    with patch("agentic_tasks.agent.preview.get_task", return_value=fake_task):
        out = format_preview(plan)
    assert "Complete" in out
    assert "Schedule Phil" in out


def test_format_preview_complete_lists_each_in_batch():
    from agentic_tasks.agent.preview import format_preview
    from agentic_tasks.notion_io.tasks import Task

    def fake_get(page_id):
        return Task(
            page_id=page_id,
            name=f"task-{page_id}",
            status="To Do",
            priority=None,
            due=None,
        )

    plan = [
        {"tool": "complete_tasks", "arguments": {"page_ids": ["p1", "p2", "p3"]}}
    ]
    with patch("agentic_tasks.agent.preview.get_task", side_effect=fake_get):
        out = format_preview(plan)

    assert "task-p1" in out
    assert "task-p2" in out
    assert "task-p3" in out
    assert out.count("<b>y</b>") == 1


# ---- delete_tasks preview ------------------------------------------------


def test_format_preview_delete_resolves_task_name():
    from agentic_tasks.agent.preview import format_preview
    from agentic_tasks.notion_io.tasks import Task

    fake_task = Task(
        page_id="page-9",
        name="Old reminder",
        status="To Do",
        priority=None,
        due=None,
    )
    plan = [{"tool": "delete_tasks", "arguments": {"page_ids": ["page-9"]}}]
    with patch("agentic_tasks.agent.preview.get_task", return_value=fake_task):
        out = format_preview(plan)
    assert "Delete" in out
    assert "Old reminder" in out


def test_format_preview_delete_lists_each_in_batch():
    from agentic_tasks.agent.preview import format_preview
    from agentic_tasks.notion_io.tasks import Task

    def fake_get(page_id):
        return Task(
            page_id=page_id,
            name=f"task-{page_id}",
            status="To Do",
            priority=None,
            due=None,
        )

    plan = [{"tool": "delete_tasks", "arguments": {"page_ids": ["p1", "p2"]}}]
    with patch("agentic_tasks.agent.preview.get_task", side_effect=fake_get):
        out = format_preview(plan)

    assert "task-p1" in out
    assert "task-p2" in out
    assert out.count("<b>y</b>") == 1
    assert out.count("About to:") == 1


# ---- mixed batch (one preview, multiple actions) -------------------------


def test_format_preview_combines_complete_and_update_under_one_header():
    """The user's golden path: complete X AND reschedule Y in one message,
    one preview, one confirmation."""
    from agentic_tasks.agent.preview import format_preview
    from agentic_tasks.notion_io.tasks import Task

    def fake_get(page_id):
        names = {"page-x": "Doctor visit", "page-y": "Cinema"}
        return Task(
            page_id=page_id,
            name=names.get(page_id, "?"),
            status="To Do",
            priority=None,
            due=None,
        )

    plan = [
        {"tool": "complete_tasks", "arguments": {"page_ids": ["page-x"]}},
        {
            "tool": "update_tasks",
            "arguments": {"updates": [{"page_id": "page-y", "due": "2026-05-02"}]},
        },
    ]
    with patch("agentic_tasks.agent.preview.get_task", side_effect=fake_get):
        out = format_preview(plan)

    assert out.count("About to:") == 1
    assert "Doctor visit" in out
    assert "Cinema" in out
    assert out.count("<b>y</b>") == 1


# ---- serialize_plan -------------------------------------------------------


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
        _tc("create_tasks", '{"tasks": [{"name": "X"}]}'),
        _tc("list_projects", "{}"),
        _tc(
            "update_tasks",
            '{"updates": [{"page_id": "p1", "due": "2026-05-05"}]}',
        ),
    ]
    plan = serialize_plan(tool_calls)
    assert [e["tool"] for e in plan] == ["create_tasks", "update_tasks"]
    assert plan[0]["arguments"] == {"tasks": [{"name": "X"}]}


# ---- execute_plan ---------------------------------------------------------


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
    with patch("agentic_tasks.agent.tools.create_task", return_value=created):
        out = execute_plan(
            [{"tool": "create_tasks", "arguments": {"tasks": [{"name": "Buy markers"}]}}]
        )
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

    with patch("agentic_tasks.agent.tools.create_task", side_effect=fake_create):
        out = execute_plan(
            [
                {
                    "tool": "create_tasks",
                    "arguments": {
                        "tasks": [{"name": "Good one"}, {"name": "Bad one"}]
                    },
                }
            ]
        )
    assert "Partial. 1 done, 1 failed." in out
    assert "Good one" in out
    # The verbose Notion exception text is sanitized — the user sees a
    # short clean phrase, not the raw ``str(e)``.
    assert "notion is down" not in out
    assert "Couldn't create" in out
    assert "Bad one" in out


def test_execute_plan_deletes_and_returns_summary_with_link():
    """Deleted (archived) task still returns its URL — preserve it in the
    summary so the user can navigate to the trashed page in Notion."""
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    archived = Task(
        page_id="page-x",
        name="Old reminder",
        status="To Do",
        priority=None,
        due=None,
        url="https://www.notion.so/Old-reminder-pagex",
        archived=True,
    )
    with patch("agentic_tasks.agent.tools.delete_task", return_value=archived):
        out = execute_plan(
            [{"tool": "delete_tasks", "arguments": {"page_ids": ["page-x"]}}]
        )
    assert out.startswith("Done.")
    assert "Deleted" in out
    assert "Old reminder" in out
    assert (
        '<a href="https://www.notion.so/Old-reminder-pagex">open</a>' in out
    )


def test_execute_plan_deletes_each_in_batch():
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    def fake_delete(page_id):
        return Task(
            page_id=page_id,
            name=f"task-{page_id}",
            status="To Do",
            priority=None,
            due=None,
            url=f"https://www.notion.so/{page_id}",
            archived=True,
        )

    with patch(
        "agentic_tasks.agent.tools.delete_task", side_effect=fake_delete
    ) as mock_delete:
        out = execute_plan(
            [{"tool": "delete_tasks", "arguments": {"page_ids": ["p1", "p2", "p3"]}}]
        )

    assert mock_delete.call_count == 3
    assert out.startswith("Done.")
    assert out.count("Deleted") == 3


def test_execute_plan_completes_each_in_batch():
    """Mark three tasks as Done in one confirmed plan."""
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    def fake_complete(page_id):
        return Task(
            page_id=page_id,
            name=f"task-{page_id}",
            status="Done",
            priority=None,
            due=None,
            url=f"https://www.notion.so/{page_id}",
        )

    with patch(
        "agentic_tasks.agent.tools.complete_task", side_effect=fake_complete
    ) as mock_complete:
        out = execute_plan(
            [{"tool": "complete_tasks", "arguments": {"page_ids": ["p1", "p2", "p3"]}}]
        )

    assert mock_complete.call_count == 3
    assert out.startswith("Done.")
    assert out.count("Completed") == 3


def test_execute_plan_updates_each_in_batch():
    """Reschedule three tasks in one confirmed plan."""
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    def fake_update(page_id, **_kw):
        return Task(
            page_id=page_id,
            name=f"task-{page_id}",
            status="To Do",
            priority=None,
            due=date(2026, 5, 9),
        )

    plan = [
        {
            "tool": "update_tasks",
            "arguments": {
                "updates": [
                    {"page_id": "p1", "due": "2026-05-09"},
                    {"page_id": "p2", "due": "2026-05-09"},
                    {"page_id": "p3", "due": "2026-05-09"},
                ]
            },
        }
    ]
    with patch(
        "agentic_tasks.agent.tools.update_task", side_effect=fake_update
    ) as mock_update:
        out = execute_plan(plan)

    assert mock_update.call_count == 3
    assert out.count("Updated") == 3


def test_execute_plan_runs_combined_complete_and_update():
    """One pending plan can include multiple batch tools — execute all."""
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    def fake_complete(page_id):
        return Task(
            page_id=page_id, name=f"done-{page_id}", status="Done",
            priority=None, due=None,
        )

    def fake_update(page_id, **_kw):
        return Task(
            page_id=page_id, name=f"upd-{page_id}", status="To Do",
            priority=None, due=date(2026, 5, 2),
        )

    plan = [
        {"tool": "complete_tasks", "arguments": {"page_ids": ["x"]}},
        {
            "tool": "update_tasks",
            "arguments": {"updates": [{"page_id": "y", "due": "2026-05-02"}]},
        },
    ]
    with patch("agentic_tasks.agent.tools.complete_task", side_effect=fake_complete), \
         patch("agentic_tasks.agent.tools.update_task", side_effect=fake_update):
        out = execute_plan(plan)

    assert "Completed" in out
    assert "Updated" in out
    assert "done-x" in out
    assert "upd-y" in out


# ---- date helpers ---------------------------------------------------------


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


def test_format_date_human_today_and_tomorrow():
    from agentic_tasks.agent.preview import _format_date_human

    today = date(2026, 5, 1)
    assert _format_date_human("2026-05-01", today) == "today"
    assert _format_date_human("2026-05-02", today) == "tomorrow"


# ---- shift_due_dates preview / execute -----------------------------------


def test_format_preview_shift_due_dates_lists_each_task():
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

    plan = [
        {
            "tool": "shift_due_dates",
            "arguments": {"delta_days": 1, "due_on": "2026-05-01"},
        }
    ]
    with patch("agentic_tasks.agent.preview.resolve_shift_targets", return_value=[]):
        out = format_preview(plan)
    assert "no matching open tasks" in out


@pytest.fixture
def _no_retry_sleep(monkeypatch):
    """Make _attempt_with_retry's backoff zero so failure-path tests stay fast."""
    monkeypatch.setattr(
        "agentic_tasks.agent.preview._RETRY_BACKOFF_SECONDS", 0
    )


# ---- verification + retry ------------------------------------------------


def test_execute_plan_retries_when_update_did_not_land(_no_retry_sleep):
    """Notion returned the page but the field we asked to change didn't
    actually change — the helper retries until a later attempt verifies
    cleanly. The user sees plain "Done." not a soft failure."""
    from datetime import date as _date

    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    calls = {"n": 0}

    def fake_update(page_id, **_kw):
        calls["n"] += 1
        # First two attempts return an unchanged due; third one lands.
        due = _date(2026, 5, 9) if calls["n"] >= 3 else _date(2026, 4, 1)
        return Task(
            page_id=page_id,
            name="Vietnam docs",
            status="To Do",
            priority=None,
            due=due,
            url="https://www.notion.so/Vietnam-docs",
        )

    plan = [
        {
            "tool": "update_tasks",
            "arguments": {
                "updates": [{"page_id": "p1", "due": "2026-05-09"}]
            },
        }
    ]
    with patch("agentic_tasks.agent.tools.update_task", side_effect=fake_update):
        out = execute_plan(plan)

    assert calls["n"] == 3
    assert out.startswith("Done.")
    assert "Vietnam docs" in out


def test_execute_plan_surfaces_persistent_verification_mismatch(_no_retry_sleep):
    """If verification fails on every attempt, the user sees an honest
    'didn't land in Notion' error, NOT a fake 'Done.' — and the failure
    line uses the task's NAME, never the page id."""
    from datetime import date as _date

    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    stuck = Task(
        page_id="p1",
        name="Stuck task",
        status="To Do",
        priority=None,
        due=_date(2026, 4, 1),
    )

    def fake_update(page_id, **_kw):
        # Always returns the page WITHOUT the requested due change.
        return stuck

    plan = [
        {
            "tool": "update_tasks",
            "arguments": {
                "updates": [{"page_id": "p1", "due": "2026-05-09"}]
            },
        }
    ]
    with patch(
        "agentic_tasks.agent.tools.update_task", side_effect=fake_update
    ) as mock_update, patch(
        "agentic_tasks.agent.preview.get_task", return_value=stuck
    ):
        out = execute_plan(plan)

    assert mock_update.call_count == 3  # MAX_MUTATION_ATTEMPTS
    assert "Failed." in out
    assert "Couldn't update" in out
    assert "Stuck task" in out
    assert "didn't land in Notion" in out
    assert "Please check the page" in out


def test_execute_plan_retries_complete_until_status_is_done(_no_retry_sleep):
    """Notion returned the page with the wrong status — retry until status
    actually becomes Done."""
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    calls = {"n": 0}

    def fake_complete(page_id):
        calls["n"] += 1
        status = "Done" if calls["n"] >= 2 else "To Do"
        return Task(
            page_id=page_id,
            name="Pay rent",
            status=status,
            priority=None,
            due=None,
            url="https://www.notion.so/Pay-rent",
        )

    with patch(
        "agentic_tasks.agent.tools.complete_task", side_effect=fake_complete
    ):
        out = execute_plan(
            [{"tool": "complete_tasks", "arguments": {"page_ids": ["p1"]}}]
        )

    assert calls["n"] == 2
    assert out.startswith("Done.")
    assert "Pay rent" in out


def test_execute_plan_retries_delete_until_archived(_no_retry_sleep):
    """If the first attempt comes back with archived=False (silent no-op),
    retry until the page actually goes to trash."""
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    calls = {"n": 0}

    def fake_delete(page_id):
        calls["n"] += 1
        archived = calls["n"] >= 2
        return Task(
            page_id=page_id,
            name="Stuck delete",
            status="To Do",
            priority=None,
            due=None,
            url="https://www.notion.so/Stuck-delete",
            archived=archived,
        )

    with patch("agentic_tasks.agent.tools.delete_task", side_effect=fake_delete):
        out = execute_plan(
            [{"tool": "delete_tasks", "arguments": {"page_ids": ["p1"]}}]
        )

    assert calls["n"] == 2
    assert out.startswith("Done.")
    assert "Stuck delete" in out


def test_execute_plan_delete_clean_summary_on_object_not_found(_no_retry_sleep):
    """Reproduces the production failure: Notion returns the verbose
    ``Could not find page with ID: <uuid>. Make sure the relevant pages
    and databases are shared with your integration "agentic-tasks-bot"``
    error. The summary must show the task NAME (not the uuid) and a clean
    sanitized reason — never the raw integration name or the scary share
    instructions."""
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    page_id = "355cdbaa-bf8d-81e0-978f-e04e13adec5b"
    notion_error = (
        f"Could not find page with ID: {page_id}. Make sure the relevant"
        ' pages and databases are shared with your integration'
        ' "agentic-tasks-bot".'
    )

    def fake_delete(_page_id):
        raise RuntimeError(notion_error)

    resolved = Task(
        page_id=page_id,
        name="TRY TASK",
        status="To Do",
        priority=None,
        due=None,
    )

    with patch("agentic_tasks.agent.tools.delete_task", side_effect=fake_delete), \
         patch("agentic_tasks.agent.preview.get_task", return_value=resolved):
        out = execute_plan(
            [{"tool": "delete_tasks", "arguments": {"page_ids": [page_id]}}]
        )

    assert "Failed." in out
    assert 'Couldn\'t delete "TRY TASK"' in out
    assert "Notion couldn't find this page" in out
    # No id leak, no integration name leak, no scary share instructions.
    assert page_id not in out
    assert "355cdbaa" not in out
    assert "agentic-tasks-bot" not in out
    assert "Make sure the relevant pages" not in out


def test_execute_plan_delete_falls_back_to_neutral_label(_no_retry_sleep):
    """When ``get_task`` itself also fails (the production case — the
    page_id is unreachable), the failure label uses "this task" and still
    avoids leaking the page_id."""
    from agentic_tasks.agent.preview import execute_plan

    page_id = "355cdbaa-bf8d-81e0-978f-e04e13adec5b"

    def fake_delete(_page_id):
        raise RuntimeError(f"Could not find page with ID: {page_id}.")

    # Note: the autouse _stub_get_task fixture already makes get_task raise.
    with patch("agentic_tasks.agent.tools.delete_task", side_effect=fake_delete):
        out = execute_plan(
            [{"tool": "delete_tasks", "arguments": {"page_ids": [page_id]}}]
        )

    assert "Failed." in out
    assert 'Couldn\'t delete "this task"' in out
    assert "Notion couldn't find this page" in out
    assert page_id not in out
    assert "355cdbaa" not in out


def test_sanitize_notion_error_maps_known_phrases():
    from agentic_tasks.agent.preview import _sanitize_notion_error

    cases = [
        (
            RuntimeError("Could not find page with ID: abc-def. Make sure..."),
            "couldn't find",
        ),
        (RuntimeError("APIResponseError: object_not_found"), "couldn't find"),
        (RuntimeError("HTTPStatusError: 401 unauthorized"), "denied access"),
        (RuntimeError("rate limit exceeded"), "rate-limiting"),
        (RuntimeError("validation_error: bad property"), "rejected the request"),
        (RuntimeError("kafkaesque random error"), "something went wrong"),
    ]
    for err, expected_substr in cases:
        assert expected_substr in _sanitize_notion_error(err), err


def test_execute_plan_surfaces_persistent_delete_failure(_no_retry_sleep):
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    stubborn = Task(
        page_id="p1",
        name="Stubborn task",
        status="To Do",
        priority=None,
        due=None,
        archived=False,
    )

    def fake_delete(page_id):
        # Always returns archived=False — verification will never pass.
        return stubborn

    with patch(
        "agentic_tasks.agent.tools.delete_task", side_effect=fake_delete
    ) as mock_delete, patch(
        "agentic_tasks.agent.preview.get_task", return_value=stubborn
    ):
        out = execute_plan(
            [{"tool": "delete_tasks", "arguments": {"page_ids": ["p1"]}}]
        )

    assert mock_delete.call_count == 3
    assert "Failed." in out
    assert "Couldn't delete" in out
    assert "Stubborn task" in out
    assert "didn't land in Notion" in out


def test_execute_plan_retries_create_on_exception_then_succeeds(_no_retry_sleep):
    """A transient Notion exception on the first try is retried; the second
    attempt succeeds and the user sees a clean 'Done.'"""
    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    calls = {"n": 0}
    created = Task(
        page_id="p-x",
        name="Buy markers",
        status="To Do",
        priority=None,
        due=None,
        url="https://www.notion.so/p-x",
    )

    def fake_create(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient flake")
        return created

    with patch("agentic_tasks.agent.tools.create_task", side_effect=fake_create):
        out = execute_plan(
            [{"tool": "create_tasks", "arguments": {"tasks": [{"name": "Buy markers"}]}}]
        )

    assert calls["n"] == 2
    assert out.startswith("Done.")
    assert "Buy markers" in out


def test_execute_plan_retries_due_clear_until_actually_cleared(_no_retry_sleep):
    """Clearing a due date is a common 'silent no-op' shape — make sure the
    verifier catches it and retries."""
    from datetime import date as _date

    from agentic_tasks.agent.preview import execute_plan
    from agentic_tasks.notion_io.tasks import Task

    calls = {"n": 0}

    def fake_update(page_id, **_kw):
        calls["n"] += 1
        # First two return the old date; third actually clears.
        due = None if calls["n"] >= 3 else _date(2026, 5, 1)
        return Task(
            page_id=page_id,
            name="Ambiguous task",
            status="To Do",
            priority=None,
            due=due,
        )

    plan = [
        {
            "tool": "update_tasks",
            "arguments": {"updates": [{"page_id": "p1", "due": ""}]},
        }
    ]
    with patch("agentic_tasks.agent.tools.update_task", side_effect=fake_update):
        out = execute_plan(plan)

    assert calls["n"] == 3
    assert out.startswith("Done.")


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
            [
                {
                    "tool": "shift_due_dates",
                    "arguments": {"delta_days": 7, "due_on": "2026-05-01"},
                }
            ]
        )
    assert "Done." in out
    assert out.count("Shifted") == 2
    assert "A" in out
    assert "B" in out
