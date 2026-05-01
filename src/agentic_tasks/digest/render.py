"""Pure formatter for the daily morning digest.

Takes today's date and three task lists (due today, My Day, overdue), returns
a Telegram-HTML string. No I/O, no LLM, no external state — all the rendering
choices live here so they're easy to test and tweak.
"""

from __future__ import annotations

import html
from datetime import date, datetime

from agentic_tasks.notion_io.tasks import Task

_PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}


def _sort(tasks: list[Task]) -> list[Task]:
    def key(t: Task) -> tuple:
        p = _PRIORITY_RANK.get(t.priority or "", 99)
        due_str = t.due.isoformat() if t.due else "9999-12-31"
        return (p, due_str, t.name)

    return sorted(tasks, key=key)


def _format_long_date(d: date) -> str:
    """Friday, May 1 — platform-neutral (no %-d / %#d differences)."""
    return f"{d.strftime('%A')}, {d.strftime('%B')} {d.day}"


def _format_short_date(d: date) -> str:
    """May 1"""
    return f"{d.strftime('%b')} {d.day}"


def _format_time(dt: datetime) -> str:
    """5 PM, 9:30 AM. Strips the leading zero portably."""
    if dt.minute == 0:
        return dt.strftime("%I %p").lstrip("0")
    return dt.strftime("%I:%M %p").lstrip("0")


def _render_task(t: Task) -> str:
    line = f"• {html.escape(t.name)}"
    if isinstance(t.due, datetime):
        line += f" — at {_format_time(t.due)}"
    return line


def _render_overdue_task(t: Task) -> str:
    line = f"• {html.escape(t.name)}"
    if t.due is None:
        return line
    due_date = t.due.date() if isinstance(t.due, datetime) else t.due
    line += f" — was due {_format_short_date(due_date)}"
    if isinstance(t.due, datetime):
        line += f" at {_format_time(t.due)}"
    return line


def _section(header: str, tasks: list[Task], line_fn) -> str:
    lines = [f"<b>{header} ({len(tasks)}):</b>"]
    lines.extend(line_fn(t) for t in tasks)
    return "\n".join(lines)


def render_digest(
    today: date,
    today_tasks: list[Task],
    my_day_tasks: list[Task],
    overdue_tasks: list[Task],
) -> str:
    """Build the morning digest message in Telegram HTML."""
    # A task already shown under Overdue or Today shouldn't appear under My Day.
    exclude = {t.page_id for t in today_tasks} | {t.page_id for t in overdue_tasks}
    my_day_filtered = [t for t in my_day_tasks if t.page_id not in exclude]

    today_sorted = _sort(today_tasks)
    overdue_sorted = _sort(overdue_tasks)
    my_day_sorted = _sort(my_day_filtered)

    greeting = f"Good morning. Today is <b>{_format_long_date(today)}</b>."

    if not (today_sorted or overdue_sorted or my_day_sorted):
        return f"{greeting}\n\nAll clear — nothing on the books."

    sections: list[str] = [greeting]
    if overdue_sorted:
        sections.append(_section("Overdue", overdue_sorted, _render_overdue_task))
    if today_sorted:
        sections.append(_section("Today", today_sorted, _render_task))
    if my_day_sorted:
        sections.append(_section("My Day", my_day_sorted, _render_task))

    return "\n\n".join(sections)
