"""Pure formatter for the end-of-day reflection digest.

Sent in the evening to surface tasks that started the day open and are
still not Done. The user replies via text to act on anything (mark done,
reschedule, snooze).

No I/O, no LLM, no external state — same shape as ``digest/render.py``.
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
    return f"{d.strftime('%A')}, {d.strftime('%B')} {d.day}"


def _format_time(dt: datetime) -> str:
    if dt.minute == 0:
        return dt.strftime("%I %p").lstrip("0")
    return dt.strftime("%I:%M %p").lstrip("0")


def _render_task(t: Task) -> str:
    line = f"• {html.escape(t.name)}"
    if isinstance(t.due, datetime):
        line += f" — at {_format_time(t.due)}"
    return line


def render_evening_digest(today: date, open_tasks: list[Task]) -> str:
    """Build the evening reflection message in Telegram HTML.

    ``open_tasks`` should be the set of tasks that were nominally on today's
    plate (due today or My Day) and are still not Done. The caller picks
    the filter; this formatter just renders.
    """
    sorted_tasks = _sort(open_tasks)
    greeting = (
        f"Evening check-in for <b>{_format_long_date(today)}</b>."
    )
    if not sorted_tasks:
        return f"{greeting}\n\nClean slate — everything from today is closed."
    lines = [
        greeting,
        "",
        f"<b>Still open ({len(sorted_tasks)}):</b>",
    ]
    lines.extend(_render_task(t) for t in sorted_tasks)
    return "\n".join(lines)
