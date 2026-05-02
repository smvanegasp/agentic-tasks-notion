"""Confirmation gate: preview formatting + pending-plan execution.

When the agent emits any write tool call (``create_tasks``, ``update_tasks``,
``complete_tasks``, ``shift_due_dates``), the loop short-circuits without
executing them and stashes a ``pending_plan`` in the conversation store.
The user sees ONE deterministic preview of every action in the round and
replies y/n once. On y, the dispatcher calls :func:`execute_plan`.

All Telegram-facing strings here use HTML parse mode and only the tags
allowed by Telegram (``<b>``, ``<a>``). Task names are HTML-escaped.
"""

from __future__ import annotations

import html
import json
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from agentic_tasks.agent.tools import (
    CompleteTasksArgs,
    CreateTasksArgs,
    NewTask,
    ShiftDueDatesArgs,
    TaskUpdate,
    UpdateTasksArgs,
    execute_complete_tasks,
    execute_create_tasks,
    execute_shift_due_dates,
    execute_update_tasks,
    resolve_shift_targets,
)
from agentic_tasks.config import get_settings
from agentic_tasks.notion_io.tasks import Task, get_task

log = logging.getLogger(__name__)

WRITE_TOOL_NAMES = frozenset(
    {"create_tasks", "update_tasks", "complete_tasks", "shift_due_dates"}
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
    Reads in the same round are dropped — the model can re-issue them next
    turn after confirmation.
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
    """Build the user-visible preview text for a pending plan.

    Each plan entry is a batch tool call (``create_tasks`` with N tasks,
    ``update_tasks`` with N updates, etc.). All entries are flattened into
    one bullet list under a single ``About to:`` header so the user sees
    the entire change set in one message and confirms with one ``y``.
    """
    today = datetime.now(get_settings().timezone).date()
    lines: list[str] = ["About to:"]
    for entry in plan:
        tool = entry["tool"]
        args = entry.get("arguments") or {}
        if tool == "create_tasks":
            lines.extend(_format_create_lines(args, today))
        elif tool == "update_tasks":
            lines.extend(_format_update_lines(args, today))
        elif tool == "complete_tasks":
            lines.extend(_format_complete_lines(args))
        elif tool == "shift_due_dates":
            lines.extend(_format_shift_lines(args, today))
        else:
            lines.append(f"• {html.escape(tool)}")
    lines.append("")
    lines.append("Reply <b>y</b> to proceed or <b>n</b> to cancel.")
    return "\n".join(lines)


def _format_create_lines(args_dict: dict[str, Any], today: date) -> list[str]:
    try:
        args = CreateTasksArgs.model_validate(args_dict)
    except ValidationError as e:
        return [f"• create_tasks: invalid args ({html.escape(str(e))})"]
    return [_format_create_line(spec, today) for spec in args.tasks]


def _format_create_line(spec: NewTask, today: date) -> str:
    name = html.escape(spec.name or "(unnamed)")
    details: list[str] = []
    if spec.due:
        details.append(f"due {_format_date_human(spec.due, today)}")
    if spec.priority:
        details.append(f"priority {spec.priority}")
    if spec.project_name:
        details.append(f"project: {html.escape(spec.project_name)}")
    if spec.my_day:
        details.append("My Day")
    if spec.labels:
        details.append(
            "labels: " + ", ".join(html.escape(label) for label in spec.labels)
        )
    suffix = f" — {'; '.join(details)}" if details else ""
    return f'• Create "{name}"{suffix}'


def _format_update_lines(args_dict: dict[str, Any], today: date) -> list[str]:
    try:
        args = UpdateTasksArgs.model_validate(args_dict)
    except ValidationError as e:
        return [f"• update_tasks: invalid args ({html.escape(str(e))})"]
    return [_format_update_line(spec, today) for spec in args.updates]


def _format_update_line(spec: TaskUpdate, today: date) -> str:
    name = _resolve_task_name(spec.page_id)
    changes: list[str] = []
    if spec.name is not None:
        changes.append(f"name → {html.escape(spec.name)}")
    if spec.description is not None:
        changes.append("description updated")
    if spec.priority is not None:
        changes.append(f"priority → {spec.priority}")
    if spec.status is not None:
        changes.append(f"status → {spec.status}")
    if spec.project_name is not None:
        changes.append(f"project → {html.escape(spec.project_name)}")
    if spec.due is not None:
        if spec.due == "":
            changes.append("due → cleared")
        else:
            changes.append(f"due → {_format_date_human(spec.due, today)}")
    if spec.my_day is not None:
        changes.append("My Day → on" if spec.my_day else "My Day → off")
    if spec.labels is not None:
        if spec.labels:
            changes.append(
                "labels → " + ", ".join(html.escape(label) for label in spec.labels)
            )
        else:
            changes.append("labels → cleared")

    if not changes:
        changes.append("(no changes)")
    return f'• Update "{name}": {"; ".join(changes)}'


def _format_complete_lines(args_dict: dict[str, Any]) -> list[str]:
    try:
        args = CompleteTasksArgs.model_validate(args_dict)
    except ValidationError as e:
        return [f"• complete_tasks: invalid args ({html.escape(str(e))})"]
    return [
        f'• Complete "{_resolve_task_name(page_id)}"' for page_id in args.page_ids
    ]


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
    except Exception as e:  # noqa: BLE001
        log.warning("preview: resolve_shift_targets failed: %s", e)
        return [f"• shift_due_dates: {html.escape(str(e))}"]
    if not targets:
        return [f"• Shift {args.delta_days:+d} day(s): no matching open tasks."]
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

    Each batch entry is dispatched independently, and within a batch each
    task is dispatched independently — one failing entry does not abort the
    rest. Successes and failures are reported together in the summary.
    """
    successes: list[tuple[str, Task]] = []
    errors: list[str] = []
    for entry in plan:
        tool = entry["tool"]
        args = entry.get("arguments") or {}
        if tool == "create_tasks":
            _run_create_batch(args, successes, errors)
        elif tool == "update_tasks":
            _run_update_batch(args, successes, errors)
        elif tool == "complete_tasks":
            _run_complete_batch(args, successes, errors)
        elif tool == "shift_due_dates":
            _run_shift_batch(args, successes, errors)
        else:
            errors.append(f"unknown action: {html.escape(str(tool))}")
    return _format_summary(successes, errors)


def _run_create_batch(
    args_dict: dict[str, Any],
    successes: list[tuple[str, Task]],
    errors: list[str],
) -> None:
    try:
        args = CreateTasksArgs.model_validate(args_dict)
    except ValidationError as e:
        errors.append(f"invalid create_tasks args: {html.escape(str(e))}")
        return
    for spec in args.tasks:
        try:
            task = execute_create_tasks(CreateTasksArgs(tasks=[spec]))[0]
            successes.append(("Created", task))
        except Exception as e:  # noqa: BLE001
            log.exception("create_tasks: %r failed", spec.name)
            errors.append(f'create "{html.escape(spec.name)}" failed: {html.escape(str(e))}')


def _run_update_batch(
    args_dict: dict[str, Any],
    successes: list[tuple[str, Task]],
    errors: list[str],
) -> None:
    try:
        args = UpdateTasksArgs.model_validate(args_dict)
    except ValidationError as e:
        errors.append(f"invalid update_tasks args: {html.escape(str(e))}")
        return
    for spec in args.updates:
        try:
            task = execute_update_tasks(UpdateTasksArgs(updates=[spec]))[0]
            successes.append(("Updated", task))
        except Exception as e:  # noqa: BLE001
            log.exception("update_tasks: %s failed", spec.page_id)
            errors.append(f"update {html.escape(spec.page_id[:8])} failed: {html.escape(str(e))}")


def _run_complete_batch(
    args_dict: dict[str, Any],
    successes: list[tuple[str, Task]],
    errors: list[str],
) -> None:
    try:
        args = CompleteTasksArgs.model_validate(args_dict)
    except ValidationError as e:
        errors.append(f"invalid complete_tasks args: {html.escape(str(e))}")
        return
    for page_id in args.page_ids:
        try:
            task = execute_complete_tasks(CompleteTasksArgs(page_ids=[page_id]))[0]
            successes.append(("Completed", task))
        except Exception as e:  # noqa: BLE001
            log.exception("complete_tasks: %s failed", page_id)
            errors.append(f"complete {html.escape(page_id[:8])} failed: {html.escape(str(e))}")


def _run_shift_batch(
    args_dict: dict[str, Any],
    successes: list[tuple[str, Task]],
    errors: list[str],
) -> None:
    try:
        args = ShiftDueDatesArgs.model_validate(args_dict)
    except ValidationError as e:
        errors.append(f"invalid shift_due_dates args: {html.escape(str(e))}")
        return
    try:
        for task in execute_shift_due_dates(args):
            successes.append(("Shifted", task))
    except Exception as e:  # noqa: BLE001
        log.exception("shift_due_dates failed")
        errors.append(f"shift_due_dates failed: {html.escape(str(e))}")


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
