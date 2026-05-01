"""Confirmation gate: preview formatting + pending-plan execution.

When the agent emits write tool calls (create_task, update_task,
complete_task), the loop short-circuits without executing them and stashes a
``pending_plan`` in the conversation store. The user sees a
deterministically-formatted preview built here and replies y/n. On y, the
dispatcher calls :func:`execute_plan` to run the writes and produce a
summary message.

All Telegram-facing strings here use HTML parse mode and only the tags
allowed by Telegram (``<b>``, ``<a>``). Task names are HTML-escaped.
"""

from __future__ import annotations

import html
import json
import logging
import re
from datetime import date, datetime, timedelta  # noqa: F401
from typing import Any

from pydantic import ValidationError

from agentic_tasks.agent.tools import (
    CompleteTaskArgs,
    CreateTaskArgs,
    ShiftDueDatesArgs,
    UpdateTaskArgs,
    _parse_iso_date,
    execute_shift_due_dates,
    resolve_shift_targets,
)
from agentic_tasks.config import get_settings
from agentic_tasks.notion_io.projects import find_project_by_name
from agentic_tasks.notion_io.schema import Priority, Status
from agentic_tasks.notion_io.tasks import (
    Task,
    complete_task,
    create_task,
    get_task,
    update_task,
)

log = logging.getLogger(__name__)

WRITE_TOOL_NAMES = frozenset(
    {"create_task", "update_task", "complete_task", "shift_due_dates"}
)

_CONFIRM_RE = re.compile(
    r"^\s*(y|yes|yeah|yep|sí|si|ok|okay|sure|👍)\s*[!.]?\s*$",
    re.IGNORECASE,
)
_REJECT_RE = re.compile(
    r"^\s*(n|no|nope|nah|cancel|cancela|cancelar|stop|abort)\s*[!.]?\s*$",
    re.IGNORECASE,
)


def is_confirmation(text: str) -> bool:
    return bool(_CONFIRM_RE.match(text))


def is_rejection(text: str) -> bool:
    return bool(_REJECT_RE.match(text))


def serialize_plan(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Convert OpenAI tool_call objects into a JSON-safe pending plan.

    Only entries whose tool name is in :data:`WRITE_TOOL_NAMES` are kept.
    """
    plan: list[dict[str, Any]] = []
    for tc in tool_calls:
        if tc.function.name not in WRITE_TOOL_NAMES:
            continue
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {}
        plan.append({"tool": tc.function.name, "arguments": args})
    return plan


# ---- Preview formatting ----------------------------------------------------


def format_preview(plan: list[dict[str, Any]]) -> str:
    """Build the user-visible preview text for a pending plan."""
    today = datetime.now(get_settings().timezone).date()
    lines: list[str] = ["About to:"]
    for entry in plan:
        tool = entry["tool"]
        args = entry.get("arguments") or {}
        if tool == "create_task":
            lines.append(_format_create_line(args, today))
        elif tool == "update_task":
            lines.append(_format_update_line(args, today))
        elif tool == "complete_task":
            lines.append(_format_complete_line(args))
        elif tool == "shift_due_dates":
            lines.extend(_format_shift_lines(args, today))
        else:
            lines.append(f"• {html.escape(tool)}")
    lines.append("")
    lines.append("Reply <b>y</b> to proceed or <b>n</b> to cancel.")
    return "\n".join(lines)


def _format_create_line(args: dict[str, Any], today: date) -> str:
    name = html.escape(str(args.get("name") or "(unnamed)"))
    details: list[str] = []
    if args.get("due"):
        details.append(f"due {_format_date_human(args['due'], today)}")
    if args.get("priority"):
        details.append(f"priority {args['priority']}")
    if args.get("project_name"):
        details.append(f"project: {html.escape(str(args['project_name']))}")
    if args.get("my_day"):
        details.append("My Day")
    if args.get("labels"):
        details.append(
            "labels: " + ", ".join(html.escape(str(label)) for label in args["labels"])
        )
    suffix = f" — {'; '.join(details)}" if details else ""
    return f'• Create "{name}"{suffix}'


def _format_update_line(args: dict[str, Any], today: date) -> str:
    page_id = args.get("page_id", "")
    name = _resolve_task_name(page_id)

    changes: list[str] = []
    if args.get("name") is not None:
        changes.append(f"name → {html.escape(str(args['name']))}")
    if args.get("description") is not None:
        changes.append("description updated")
    if args.get("priority") is not None:
        changes.append(f"priority → {args['priority']}")
    if args.get("status") is not None:
        changes.append(f"status → {args['status']}")
    if args.get("project_name") is not None:
        changes.append(f"project → {html.escape(str(args['project_name']))}")
    if args.get("due") is not None:
        if args["due"] == "":
            changes.append("due → cleared")
        else:
            changes.append(f"due → {_format_date_human(args['due'], today)}")
    if args.get("my_day") is not None:
        changes.append("My Day → on" if args["my_day"] else "My Day → off")
    if args.get("labels") is not None:
        labels = args["labels"] or []
        if labels:
            changes.append(
                "labels → " + ", ".join(html.escape(str(label)) for label in labels)
            )
        else:
            changes.append("labels → cleared")

    if not changes:
        changes.append("(no changes)")
    return f'• Update "{name}": {"; ".join(changes)}'


def _format_complete_line(args: dict[str, Any]) -> str:
    return f'• Complete "{_resolve_task_name(args.get("page_id", ""))}"'


def _format_shift_lines(args_dict: dict[str, Any], today: date) -> list[str]:
    """Build the multi-line preview for a shift_due_dates plan entry.

    Resolves the filter against Notion at preview time so the user sees the
    actual tasks that will move (and reads the old → new dates per task).
    Tasks without a current due date are listed as skipped.
    """
    try:
        args = ShiftDueDatesArgs.model_validate(args_dict)
    except ValidationError as e:
        return [f"• shift_due_dates: invalid args ({html.escape(str(e))})"]
    try:
        targets = resolve_shift_targets(args)
    except (ValueError, Exception) as e:  # noqa: BLE001
        log.warning("preview: resolve_shift_targets failed: %s", e)
        return [f"• shift_due_dates: {html.escape(str(e))}"]
    if not targets:
        return [
            f"• Shift {args.delta_days:+d} day(s): no matching open tasks."
        ]
    lines: list[str] = [
        f"• Shift {len(targets)} task(s) by {args.delta_days:+d} day(s):"
    ]
    delta = timedelta(days=args.delta_days)
    for task in targets:
        name = html.escape(task.name) if task.name else "(no title)"
        if task.due is None:
            lines.append(f'  – "{name}" — no due date, skipped')
            continue
        old_iso = task.due.isoformat()
        new_iso = (task.due + delta).isoformat()
        lines.append(
            f'  – "{name}" — {_format_date_human(old_iso, today)} →'
            f" {_format_date_human(new_iso, today)}"
        )
    return lines


def _resolve_task_name(page_id: str) -> str:
    """Look up a Notion page and return its HTML-escaped title.

    Falls back to a short page-id hint if the lookup fails so the preview
    still renders.
    """
    if not page_id:
        return "(unknown task)"
    try:
        task = get_task(page_id)
    except Exception as e:  # noqa: BLE001
        log.warning("preview: get_task(%s) failed: %s", page_id, e)
        return f"(task {html.escape(page_id[:8])})"
    return html.escape(task.name) if task.name else "(no title)"


def _format_date_human(iso_value: str, today: date) -> str:
    """ISO 8601 → 'Friday, May 1' / 'Friday, May 1 at 5 PM' / 'today' / 'tomorrow'."""
    try:
        if "T" not in iso_value:
            d = date.fromisoformat(iso_value)
            return _format_date_only(d, today)
        dt = datetime.fromisoformat(iso_value)
        return f"{_format_date_only(dt.date(), today)} at {_format_time(dt)}"
    except ValueError:
        return iso_value


def _format_date_only(d: date, today: date) -> str:
    if d == today:
        return "today"
    if d == today + timedelta(days=1):
        return "tomorrow"
    return f"{d.strftime('%A')}, {d.strftime('%B')} {d.day}"


def _format_time(dt: datetime) -> str:
    hour_12 = dt.hour % 12 or 12
    suffix = "AM" if dt.hour < 12 else "PM"
    if dt.minute == 0:
        return f"{hour_12} {suffix}"
    return f"{hour_12}:{dt.minute:02d} {suffix}"


# ---- Plan execution --------------------------------------------------------


def execute_plan(plan: list[dict[str, Any]]) -> str:
    """Execute a confirmed pending plan; return a Telegram-HTML summary.

    Each entry is dispatched independently — one failing entry does not abort
    the rest. Successes and failures are reported together in the summary.
    """
    successes: list[tuple[str, Task]] = []
    errors: list[str] = []
    for entry in plan:
        tool = entry["tool"]
        args = entry.get("arguments") or {}
        try:
            if tool == "create_task":
                successes.append(("Created", _execute_create(args)))
            elif tool == "update_task":
                successes.append(("Updated", _execute_update(args)))
            elif tool == "complete_task":
                successes.append(("Completed", _execute_complete(args)))
            elif tool == "shift_due_dates":
                for task in _execute_shift(args):
                    successes.append(("Shifted", task))
            else:
                errors.append(f"unknown action: {html.escape(str(tool))}")
        except ValidationError as e:
            errors.append(f"invalid args for {tool}: {html.escape(str(e))}")
        except Exception as e:  # noqa: BLE001
            log.exception("execute_plan: %s failed", tool)
            errors.append(f"{tool} failed: {html.escape(str(e))}")
    return _format_summary(successes, errors)


def _execute_create(args: dict[str, Any]) -> Task:
    parsed = CreateTaskArgs.model_validate(args)
    parsed_due = _parse_iso_date(parsed.due) if parsed.due else None
    parsed_priority = Priority(parsed.priority) if parsed.priority else None
    project_ids: list[str] | None = None
    if parsed.project_name:
        proj = find_project_by_name(parsed.project_name)
        if proj is None:
            raise ValueError(f"no project matching '{parsed.project_name}'")
        project_ids = [proj.page_id]
    return create_task(
        name=parsed.name,
        description=parsed.description,
        due=parsed_due,
        priority=parsed_priority,
        project_ids=project_ids,
        labels=parsed.labels,
        my_day=parsed.my_day,
    )


def _execute_update(args: dict[str, Any]) -> Task:
    parsed = UpdateTaskArgs.model_validate(args)
    kwargs: dict[str, Any] = {}
    if parsed.name is not None:
        kwargs["name"] = parsed.name
    if parsed.description is not None:
        kwargs["description"] = parsed.description
    if parsed.due is not None:
        kwargs["due"] = None if parsed.due == "" else _parse_iso_date(parsed.due)
    if parsed.priority is not None:
        kwargs["priority"] = Priority(parsed.priority)
    if parsed.status is not None:
        kwargs["status"] = Status(parsed.status)
    if parsed.project_name is not None:
        proj = find_project_by_name(parsed.project_name)
        if proj is None:
            raise ValueError(f"no project matching '{parsed.project_name}'")
        kwargs["project_ids"] = [proj.page_id]
    if parsed.labels is not None:
        kwargs["labels"] = parsed.labels
    if parsed.my_day is not None:
        kwargs["my_day"] = parsed.my_day
    return update_task(parsed.page_id, **kwargs)


def _execute_complete(args: dict[str, Any]) -> Task:
    parsed = CompleteTaskArgs.model_validate(args)
    return complete_task(parsed.page_id)


def _execute_shift(args: dict[str, Any]) -> list[Task]:
    parsed = ShiftDueDatesArgs.model_validate(args)
    return execute_shift_due_dates(parsed)


def _format_summary(
    successes: list[tuple[str, Task]],
    errors: list[str],
) -> str:
    lines: list[str] = []
    if errors and successes:
        lines.append(f"Partial. {len(successes)} done, {len(errors)} failed.")
    elif errors and not successes:
        lines.append("Failed.")
    else:
        lines.append("Done.")
    for action, task in successes:
        name = html.escape(task.name) if task.name else "(no title)"
        if task.url:
            link = f' — <a href="{html.escape(task.url, quote=True)}">open</a>'
        else:
            link = ""
        lines.append(f'• {action} "{name}"{link}')
    for err in errors:
        lines.append(f"• {err}")
    return "\n".join(lines)
