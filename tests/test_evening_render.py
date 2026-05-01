"""Evening reflection digest formatter."""

from __future__ import annotations

from datetime import date, datetime

from agentic_tasks.notion_io.tasks import Task


def _task(name: str, due=None, priority: str | None = None) -> Task:
    return Task(
        page_id=f"id-{name}",
        name=name,
        status="To Do",
        priority=priority,
        due=due,
    )


def test_evening_digest_clean_slate_when_no_open_tasks():
    from agentic_tasks.digest.evening import render_evening_digest

    out = render_evening_digest(date(2026, 5, 1), [])
    assert "Clean slate" in out
    assert "Friday, May 1" in out


def test_evening_digest_lists_open_tasks_sorted():
    from agentic_tasks.digest.evening import render_evening_digest

    tasks = [
        _task("Low priority later", due=date(2026, 5, 1), priority="Low"),
        _task("High priority", due=date(2026, 5, 1), priority="High"),
        _task("Medium", due=date(2026, 5, 1), priority="Medium"),
    ]
    out = render_evening_digest(date(2026, 5, 1), tasks)
    assert "Still open (3)" in out
    high_idx = out.index("High priority")
    medium_idx = out.index("Medium")
    low_idx = out.index("Low priority later")
    assert high_idx < medium_idx < low_idx


def test_evening_digest_no_call_to_action_text():
    """Evening reflection is just a list — no buttons, no 'tap a button' line."""
    from agentic_tasks.digest.evening import render_evening_digest

    out = render_evening_digest(
        date(2026, 5, 1), [_task("Buy markers", due=date(2026, 5, 1))]
    )
    assert "Tap" not in out
    assert "button" not in out.lower()


def test_evening_digest_renders_time_when_due_is_datetime():
    from agentic_tasks.digest.evening import render_evening_digest

    tasks = [_task("Submit report", due=datetime(2026, 5, 1, 17, 0))]
    out = render_evening_digest(date(2026, 5, 1), tasks)
    assert "5 PM" in out


def test_evening_digest_escapes_html_in_names():
    from agentic_tasks.digest.evening import render_evening_digest

    tasks = [_task("Joe & <Alberto>", due=date(2026, 5, 1))]
    out = render_evening_digest(date(2026, 5, 1), tasks)
    assert "Joe &amp; &lt;Alberto&gt;" in out
    assert "<Alberto>" not in out
