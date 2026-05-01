from datetime import date, datetime
from zoneinfo import ZoneInfo

from agentic_tasks.digest.render import render_digest
from agentic_tasks.notion_io.tasks import Task

TZ = ZoneInfo("America/New_York")


def _task(
    name: str,
    page_id: str = "p1",
    priority: str | None = "Medium",
    due=None,
    my_day: bool = False,
):
    return Task(
        page_id=page_id,
        name=name,
        status="To Do",
        priority=priority,
        due=due,
        project_ids=[],
        labels=[],
        description="",
        my_day=my_day,
        url="",
    )


def test_renders_all_clear_when_empty():
    text = render_digest(date(2026, 5, 1), [], [], [])
    assert "Friday, May 1" in text
    assert "All clear" in text or "nothing" in text.lower()


def test_renders_today_section_with_time():
    tasks = [
        _task(
            "Submit report",
            priority="High",
            due=datetime(2026, 5, 1, 17, 0, tzinfo=TZ),
        )
    ]
    text = render_digest(date(2026, 5, 1), tasks, [], [])

    assert "<b>Today (1):</b>" in text
    assert "Submit report" in text
    assert "5 PM" in text


def test_today_task_with_minutes_shows_them():
    tasks = [
        _task("Meeting", due=datetime(2026, 5, 1, 9, 30, tzinfo=TZ)),
    ]
    text = render_digest(date(2026, 5, 1), tasks, [], [])
    assert "9:30 AM" in text


def test_today_task_without_due_time_has_no_time_suffix():
    tasks = [_task("No-time task", due=date(2026, 5, 1))]
    text = render_digest(date(2026, 5, 1), tasks, [], [])
    line = next(line for line in text.split("\n") if "No-time task" in line)
    assert "—" not in line  # no "— at X PM"


def test_sorts_within_section_by_priority():
    tasks = [
        _task("Low task", page_id="l", priority="Low"),
        _task("High task", page_id="h", priority="High"),
        _task("Medium task", page_id="m", priority="Medium"),
    ]
    text = render_digest(date(2026, 5, 1), tasks, [], [])
    high = text.index("High task")
    medium = text.index("Medium task")
    low = text.index("Low task")
    assert high < medium < low


def test_overdue_section_shows_when_was_due():
    overdue = [_task("Pay bill", due=date(2026, 4, 25))]
    text = render_digest(date(2026, 5, 1), [], [], overdue)

    assert "<b>Overdue (1):</b>" in text
    assert "was due Apr 25" in text


def test_overdue_with_datetime_shows_date_and_time():
    overdue = [
        _task(
            "Missed meeting",
            due=datetime(2026, 4, 28, 14, 30, tzinfo=TZ),
        )
    ]
    text = render_digest(date(2026, 5, 1), [], [], overdue)
    assert "was due Apr 28" in text
    assert "2:30 PM" in text


def test_dedups_my_day_tasks_already_in_today():
    today_tasks = [_task("Both", page_id="p1")]
    my_day_tasks = [
        _task("Both", page_id="p1"),
        _task("Only my day", page_id="p2"),
    ]
    text = render_digest(date(2026, 5, 1), today_tasks, my_day_tasks, [])

    assert text.count("Both") == 1  # appears in Today, not My Day
    assert "<b>My Day (1):</b>" in text  # only the unique my-day task
    assert "Only my day" in text


def test_dedups_my_day_tasks_already_in_overdue():
    overdue = [_task("Both", page_id="x")]
    my_day_tasks = [_task("Both", page_id="x")]
    text = render_digest(date(2026, 5, 1), [], my_day_tasks, overdue)

    assert text.count("Both") == 1
    assert "<b>My Day" not in text  # filtered to empty → section skipped


def test_html_escapes_special_characters_in_task_names():
    tasks = [_task("Buy <coffee> & tea")]
    text = render_digest(date(2026, 5, 1), tasks, [], [])

    assert "&lt;coffee&gt;" in text
    assert "&amp;" in text
    # Raw chars must not leak through
    assert "Buy <coffee>" not in text


def test_section_order_overdue_today_my_day():
    overdue = [_task("Old", page_id="o")]
    today_tasks = [_task("Now", page_id="t")]
    my_day = [_task("Focus", page_id="m")]
    text = render_digest(date(2026, 5, 1), today_tasks, my_day, overdue)

    overdue_idx = text.index("<b>Overdue")
    today_idx = text.index("<b>Today")
    my_day_idx = text.index("<b>My Day")
    assert overdue_idx < today_idx < my_day_idx


def test_skips_empty_sections():
    """A digest with only My Day items should have only the My Day section."""
    my_day = [_task("Focus", page_id="m")]
    text = render_digest(date(2026, 5, 1), [], my_day, [])
    assert "Overdue" not in text
    assert "Today (" not in text
    assert "<b>My Day (1):</b>" in text
