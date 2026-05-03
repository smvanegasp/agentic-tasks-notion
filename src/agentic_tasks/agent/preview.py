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
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from agentic_tasks.agent.tools import (
    CompleteTasksArgs,
    CreateTasksArgs,
    DeleteTasksArgs,
    NewTask,
    ShiftDueDatesArgs,
    TaskUpdate,
    UpdateTasksArgs,
    _parse_iso_date,
    execute_complete_tasks,
    execute_create_tasks,
    execute_delete_tasks,
    execute_shift_due_dates,
    execute_update_tasks,
    resolve_shift_targets,
)
from agentic_tasks.config import get_settings
from agentic_tasks.notion_io.tasks import Task, get_task

log = logging.getLogger(__name__)

WRITE_TOOL_NAMES = frozenset(
    {
        "create_tasks",
        "update_tasks",
        "complete_tasks",
        "delete_tasks",
        "shift_due_dates",
    }
)

# Each individual mutation gets re-attempted on failure (exception OR
# verification mismatch). The user reported "says done but isn't done"
# behavior; an immediate retry with verification catches transient Notion
# flakes without needing another LLM round-trip.
MAX_MUTATION_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.3


def _sanitize_notion_error(e: Exception) -> str:
    """Translate a Notion / SDK exception into one short, ID-free sentence
    for the user. The raw Notion error mentions the page UUID, the
    integration name, and tells the user to "share pages with your
    integration" — none of that belongs in the chat reply.
    """
    msg = str(e).lower()
    if "could not find" in msg or "object_not_found" in msg or "not_found" in msg:
        return "Notion couldn't find this page (it may have been deleted or moved)."
    if "unauthorized" in msg or "restricted_resource" in msg or "permission" in msg:
        return "Notion denied access to this page."
    if "rate" in msg and "limit" in msg:
        return "Notion is rate-limiting us — try again in a moment."
    if "validation" in msg or "invalid" in msg:
        return "Notion rejected the request as invalid."
    return "something went wrong on Notion's side."

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
        elif tool == "delete_tasks":
            lines.extend(_format_delete_lines(args))
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


def _format_delete_lines(args_dict: dict[str, Any]) -> list[str]:
    try:
        args = DeleteTasksArgs.model_validate(args_dict)
    except ValidationError as e:
        return [f"• delete_tasks: invalid args ({html.escape(str(e))})"]
    return [
        f'• Delete "{_resolve_task_name(page_id)}"' for page_id in args.page_ids
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


_UNRESOLVED_TASK_LABEL = "this task"


def _resolve_task_name(page_id: str) -> str:
    """Look up a Notion page and return its HTML-escaped title.

    Falls back to a neutral phrase ("this task") when the lookup fails —
    the page_id is an internal identifier and must never appear in
    user-facing text. Failures here usually mean the page is no longer
    reachable (deleted, moved, model emitted a stale id) — we still want
    the preview / summary to render rather than crash, but the user
    shouldn't see the raw uuid.
    """
    if not page_id:
        return _UNRESOLVED_TASK_LABEL
    try:
        task = get_task(page_id)
    except Exception as e:  # noqa: BLE001
        log.warning("preview: get_task(%s) failed: %s", page_id, e)
        return _UNRESOLVED_TASK_LABEL
    return html.escape(task.name) if task.name else _UNRESOLVED_TASK_LABEL


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
        elif tool == "delete_tasks":
            _run_delete_batch(args, successes, errors)
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
        name = html.escape(spec.name) if spec.name else _UNRESOLVED_TASK_LABEL

        def _label(name: str = name) -> str:
            return f'Couldn\'t create "{name}"'

        task, error = _attempt_with_retry(
            do=lambda spec=spec: execute_create_tasks(CreateTasksArgs(tasks=[spec]))[0],
            verify=lambda task, spec=spec: _verify_create(spec, task),
            label_fn=_label,
        )
        if task is not None:
            successes.append(("Created", task))
        else:
            errors.append(error)


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

        def _label(spec: TaskUpdate = spec) -> str:
            return f'Couldn\'t update "{_resolve_task_name(spec.page_id)}"'

        task, error = _attempt_with_retry(
            do=lambda spec=spec: execute_update_tasks(UpdateTasksArgs(updates=[spec]))[0],
            verify=lambda task, spec=spec: _verify_update(spec, task),
            label_fn=_label,
            log_id=spec.page_id[:8],
        )
        if task is not None:
            successes.append(("Updated", task))
        else:
            errors.append(error)


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

        def _label(page_id: str = page_id) -> str:
            return f'Couldn\'t complete "{_resolve_task_name(page_id)}"'

        task, error = _attempt_with_retry(
            do=lambda page_id=page_id: execute_complete_tasks(
                CompleteTasksArgs(page_ids=[page_id])
            )[0],
            verify=_verify_complete,
            label_fn=_label,
            log_id=page_id[:8],
        )
        if task is not None:
            successes.append(("Completed", task))
        else:
            errors.append(error)


def _run_delete_batch(
    args_dict: dict[str, Any],
    successes: list[tuple[str, Task]],
    errors: list[str],
) -> None:
    try:
        args = DeleteTasksArgs.model_validate(args_dict)
    except ValidationError as e:
        errors.append(f"invalid delete_tasks args: {html.escape(str(e))}")
        return
    for page_id in args.page_ids:

        def _label(page_id: str = page_id) -> str:
            return f'Couldn\'t delete "{_resolve_task_name(page_id)}"'

        task, error = _attempt_with_retry(
            do=lambda page_id=page_id: execute_delete_tasks(
                DeleteTasksArgs(page_ids=[page_id])
            )[0],
            verify=_verify_delete,
            label_fn=_label,
            log_id=page_id[:8],
        )
        if task is not None:
            successes.append(("Deleted", task))
        else:
            errors.append(error)


# ---- Mutation retry with verification --------------------------------------


def _attempt_with_retry(
    *,
    do: Any,
    verify: Any,
    label_fn: Callable[[], str],
    log_id: str = "",
) -> tuple[Task | None, str]:
    """Run a single mutation up to MAX_MUTATION_ATTEMPTS times, verifying the
    returned page after each successful call.

    ``label_fn`` is invoked only when an error message is actually built —
    so the success path doesn't pay the extra Notion lookup needed to
    render the task's name. The returned phrase is user-facing and must
    never contain raw page ids or scary backend text; verbose Notion
    errors are funnelled through :func:`_sanitize_notion_error` before
    being shown. ``log_id`` is the short page-id prefix, used only in
    server logs for debugging.

    A retry is triggered by either an exception OR a verification mismatch
    (the page came back but the field we asked to change didn't actually
    change). On persistent failure, returns ``(None, html_error)`` so the
    caller can append it to the error list and surface it honestly to the
    user instead of claiming "Done."
    """
    last_error = ""
    for attempt in range(1, MAX_MUTATION_ATTEMPTS + 1):
        try:
            task = do()
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[%s] attempt %d/%d raised: %s",
                log_id, attempt, MAX_MUTATION_ATTEMPTS, e,
            )
            last_error = f"{label_fn()}: {_sanitize_notion_error(e)}"
            if attempt < MAX_MUTATION_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
            continue

        mismatch = verify(task)
        if mismatch is None:
            if attempt > 1:
                log.info("[%s] succeeded on attempt %d", log_id, attempt)
            return task, ""

        log.warning(
            "[%s] attempt %d/%d verification failed: %s",
            log_id, attempt, MAX_MUTATION_ATTEMPTS, mismatch,
        )
        last_error = (
            f"{label_fn()}: the change didn't land in Notion."
            " Please check the page."
        )
        if attempt < MAX_MUTATION_ATTEMPTS:
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    return None, last_error


def _verify_create(spec: NewTask, returned: Task) -> str | None:
    """Return a short reason if the created page doesn't reflect the spec.

    Verifies the fields most likely to silently no-op: name, due, priority,
    my_day. Description / labels / project are skipped (looser matching, less
    common failure mode).
    """
    if returned.name != spec.name:
        return f"name → {returned.name!r}"
    if spec.priority and returned.priority != spec.priority:
        return f"priority → {returned.priority!r}"
    if spec.my_day and not returned.my_day:
        return "my_day → off"
    if spec.due:
        try:
            expected = _parse_iso_date(spec.due)
        except ValueError:
            return None
        if not _due_matches(expected, returned.due):
            return f"due → {returned.due}"
    return None


def _verify_update(spec: TaskUpdate, returned: Task) -> str | None:
    """Return a short reason if the updated page doesn't reflect the spec."""
    if spec.name is not None and returned.name != spec.name:
        return f"name → {returned.name!r}"
    if spec.priority is not None and returned.priority != spec.priority:
        return f"priority → {returned.priority!r}"
    if spec.status is not None and returned.status != spec.status:
        return f"status → {returned.status!r}"
    if spec.my_day is not None and returned.my_day != spec.my_day:
        return f"my_day → {returned.my_day}"
    if spec.due is not None:
        if spec.due == "":
            if returned.due is not None:
                return f"due not cleared (still {returned.due})"
        else:
            try:
                expected = _parse_iso_date(spec.due)
            except ValueError:
                return None
            if not _due_matches(expected, returned.due):
                return f"due → {returned.due}"
    return None


def _verify_complete(returned: Task) -> str | None:
    if returned.status != "Done":
        return f"status → {returned.status!r}"
    return None


def _verify_delete(returned: Task) -> str | None:
    if not returned.archived:
        return "still not archived"
    return None


def _due_matches(expected: date | datetime, actual: date | datetime | None) -> bool:
    """Compare a parsed expected date/datetime against a returned ``Task.due``.

    Date specs match against either the date part of a returned datetime or a
    returned date. Datetime specs require both to be datetimes and equal at
    second resolution.
    """
    if actual is None:
        return False
    if isinstance(expected, datetime):
        if not isinstance(actual, datetime):
            return False
        return actual.replace(microsecond=0) == expected.replace(microsecond=0)
    # expected is plain date
    if isinstance(actual, datetime):
        return actual.date() == expected
    return actual == expected


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
