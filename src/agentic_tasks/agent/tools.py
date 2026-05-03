"""Agent tools: Pydantic-backed function-calling schemas and a dispatcher.

Tool argument shapes are declared as Pydantic models. Each model becomes a
non-strict tool schema the LLM sees, and ``call_tool`` validates incoming
JSON arguments against the same model before dispatching.

Write tools (``create_tasks``, ``update_tasks``, ``complete_tasks``,
``shift_due_dates``) are batch-shaped on purpose: every write takes a list,
even for a single item. That removes the "emit N calls together" prompt
problem — the model just adds entries to one list, and the confirmation gate
previews + executes one batch.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from time import perf_counter
from typing import Any, Literal

from openai.types.chat import ChatCompletionFunctionToolParam
from pydantic import BaseModel, Field, ValidationError, field_validator
from rapidfuzz import fuzz, process

from agentic_tasks.config import get_settings
from agentic_tasks.notion_io.projects import find_project_by_name, list_projects
from agentic_tasks.notion_io.schema import Priority, Status, TaskProperty
from agentic_tasks.notion_io.tasks import (
    Task,
    complete_task,
    create_task,
    delete_task,
    query_tasks,
    update_task,
)

log = logging.getLogger(__name__)

_PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}

StatusLiteral = Literal["To Do", "Doing", "Done"]
PriorityLiteral = Literal["Low", "Medium", "High"]


def _normalize_task_name(name: str) -> str:
    """Tidy task names so they read intentionally rather than as raw
    transcription.

    Rule: if the name has at least one uppercase letter the user/model
    capitalized something on purpose ("HEA: Get a haircut", "iPhone
    setup", "Llamar a mamá") — leave it alone. Otherwise uppercase the
    first letter character (skipping any leading punctuation or digits)
    so "buy markers" → "Buy markers" and "(test) buy milk" → "(Test) buy
    milk".
    """
    stripped = name.strip()
    if not stripped:
        return stripped
    if any(c.isupper() for c in stripped):
        return stripped
    chars = list(stripped)
    for i, c in enumerate(chars):
        if c.isalpha():
            chars[i] = c.upper()
            return "".join(chars)
    return stripped


def _task_to_dict(t: Task) -> dict:
    return {
        "page_id": t.page_id,
        "name": t.name,
        "status": t.status,
        "priority": t.priority,
        "due": t.due.isoformat() if t.due else None,
        "project_ids": t.project_ids,
        "labels": t.labels,
        "my_day": t.my_day,
        "description": t.description,
    }


def _sort_tasks(tasks: list[Task]) -> list[Task]:
    """Sort by priority (High first), then by due (soonest first), then name."""

    def key(t: Task) -> tuple:
        p_rank = _PRIORITY_RANK.get(t.priority or "", 99)
        # ISO strings sort lexicographically the same as chronologically.
        # Tasks with no due land at the end of their priority group.
        due_str = t.due.isoformat() if t.due else "9999-12-31"
        return (p_rank, due_str, t.name)

    return sorted(tasks, key=key)


def _parse_iso_date(value: str) -> date | datetime:
    if "T" not in value:
        return date.fromisoformat(value)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        # The LLM is asked to include the offset, but if it forgets we treat
        # the wall-clock time as the user's configured timezone — never UTC.
        dt = dt.replace(tzinfo=get_settings().timezone)
    return dt


# ---- Pydantic argument models ----------------------------------------------


class QueryTasksArgs(BaseModel):
    """Query tasks from the user's Notion to-do database. By default excludes
    completed tasks. Use due_on for a specific date and due_on_or_after /
    due_on_or_before for ranges — compute concrete YYYY-MM-DD dates from the
    user's words yourself."""

    status: StatusLiteral | None = None
    include_done: bool = Field(
        False,
        description=(
            "Set true to also include completed tasks. Default false — done"
            " tasks are filtered out."
        ),
    )
    due_on: str | None = Field(
        None,
        description=(
            "Match tasks whose Due date equals this YYYY-MM-DD. Use for"
            " queries like 'tomorrow' or a specific day."
        ),
    )
    due_on_or_after: str | None = Field(
        None,
        description=(
            "Match tasks whose Due date is on or after this YYYY-MM-DD."
            " Combine with due_on_or_before for ranges."
        ),
    )
    due_on_or_before: str | None = Field(
        None,
        description="Match tasks whose Due date is on or before this YYYY-MM-DD.",
    )
    my_day: bool | None = Field(None, description="Only tasks marked 'My Day'.")
    project_name: str | None = Field(
        None, description="Filter by project (fuzzy matched)."
    )
    limit: int = 25


class FindTasksArgs(BaseModel):
    """Fuzzy-search for open tasks by partial name, acronym, or keyword.

    Use this BEFORE update_tasks / complete_tasks whenever the user references
    a task by less than its full name (e.g., 'APD task', 'the doctor
    appointment', 'cinema'). Returns the best matches with their page_ids
    so you can act on the right one without guessing from earlier
    conversation history. If the user names multiple tasks, call find_tasks
    in parallel — one call per task — in the same round.
    """

    name_query: str = Field(
        ..., description="Text the user used to refer to the task."
    )
    include_done: bool = Field(
        False, description="Set true to also search completed tasks."
    )
    limit: int = Field(5, description="Maximum number of matches to return.")


class NewTask(BaseModel):
    """One task to create. Used inside create_tasks."""

    name: str
    description: str | None = None
    due: str | None = Field(
        None,
        description=(
            "ISO 8601 date (YYYY-MM-DD) or datetime "
            "(YYYY-MM-DDTHH:MM:SS with optional offset)."
        ),
    )
    priority: PriorityLiteral | None = None
    project_name: str | None = Field(
        None, description="Project to attach (fuzzy matched)."
    )
    labels: list[str] | None = None
    my_day: bool = False

    @field_validator("name")
    @classmethod
    def _tidy_name(cls, v: str) -> str:
        return _normalize_task_name(v)


class CreateTasksArgs(BaseModel):
    """Create one or more new tasks in a single batch.

    Always pass a list — even for a single task, send tasks=[{...}]. The
    confirmation gate previews the whole batch and the user approves once.
    """

    tasks: list[NewTask] = Field(..., min_length=1)


class TaskUpdate(BaseModel):
    """One task update. Only fields you set will change. ``due=""`` clears
    the due date. Used inside update_tasks."""

    page_id: str
    name: str | None = None
    description: str | None = None
    due: str | None = None
    priority: PriorityLiteral | None = None
    status: StatusLiteral | None = None
    project_name: str | None = None
    labels: list[str] | None = None
    my_day: bool | None = None

    @field_validator("name")
    @classmethod
    def _tidy_name(cls, v: str | None) -> str | None:
        return _normalize_task_name(v) if v is not None else None


class UpdateTasksArgs(BaseModel):
    """Update one or more tasks in a single batch.

    Use this for rescheduling (set ``due``), changing priority/status/project,
    renaming, etc. Always pass a list — even for a single update. The
    confirmation gate previews the whole batch and the user approves once.
    """

    updates: list[TaskUpdate] = Field(..., min_length=1)


class CompleteTasksArgs(BaseModel):
    """Mark one or more tasks as Done in a single batch.

    Always pass a list of page_ids — even for one task, send page_ids=[...].
    The confirmation gate previews the whole batch and the user approves once.
    """

    page_ids: list[str] = Field(..., min_length=1)


class DeleteTasksArgs(BaseModel):
    """Move one or more tasks to Notion's Trash in a single batch.

    Use this when the user wants to ELIMINATE a task entirely (delete /
    remove / borrar / eliminar), NOT when they want to mark it as Done —
    that's complete_tasks. Always pass a list of page_ids — even for one
    task, send page_ids=[...]. The confirmation gate previews the whole
    batch and the user approves once.
    """

    page_ids: list[str] = Field(..., min_length=1)


class ListProjectsArgs(BaseModel):
    """List every project in the projects DB (for disambiguation)."""


class ShiftDueDatesArgs(BaseModel):
    """Shift the due date of every task matching the filter by an integer
    number of days. Use for filter-based batch reschedules — for example
    'push everything from this week to next week' (delta_days=7), 'pull
    tomorrow back to today' (delta_days=-1).

    Open tasks only — completed tasks are never shifted. Tasks with no due
    date are skipped silently. Provide at least one filter (due_on,
    due_on_or_after, due_on_or_before, project_name, or my_day).
    """

    delta_days: int = Field(
        ...,
        description=(
            "Number of days to shift. Positive moves dates later, negative"
            " earlier."
        ),
    )
    due_on: str | None = Field(
        None,
        description=(
            "Match tasks whose Due date equals this YYYY-MM-DD before the"
            " shift."
        ),
    )
    due_on_or_after: str | None = Field(
        None,
        description="Match tasks whose Due date is on or after this YYYY-MM-DD.",
    )
    due_on_or_before: str | None = Field(
        None,
        description="Match tasks whose Due date is on or before this YYYY-MM-DD.",
    )
    project_name: str | None = Field(
        None, description="Filter by project (fuzzy matched)."
    )
    my_day: bool | None = Field(None, description="Only tasks marked 'My Day'.")


# ---- Tool implementations --------------------------------------------------


def _query_tasks_tool(args: QueryTasksArgs) -> str:
    conditions: list[dict] = []

    if args.status:
        conditions.append(
            {"property": TaskProperty.STATUS, "status": {"equals": args.status}}
        )
    elif not args.include_done:
        conditions.append(
            {"property": TaskProperty.STATUS, "status": {"does_not_equal": "Done"}}
        )

    if args.due_on:
        conditions.append({"property": TaskProperty.DUE, "date": {"equals": args.due_on}})
    if args.due_on_or_after:
        conditions.append(
            {"property": TaskProperty.DUE, "date": {"on_or_after": args.due_on_or_after}}
        )
    if args.due_on_or_before:
        conditions.append(
            {"property": TaskProperty.DUE, "date": {"on_or_before": args.due_on_or_before}}
        )

    if args.my_day is True:
        conditions.append({"property": TaskProperty.MY_DAY, "checkbox": {"equals": True}})

    if args.project_name:
        proj = find_project_by_name(args.project_name)
        if proj is None:
            return json.dumps(
                {"error": f"No project matching '{args.project_name}'.", "tasks": []}
            )
        conditions.append(
            {"property": TaskProperty.PROJECT, "relation": {"contains": proj.page_id}}
        )

    filter_: dict | None
    if not conditions:
        filter_ = None
    elif len(conditions) == 1:
        filter_ = conditions[0]
    else:
        filter_ = {"and": conditions}

    tasks = _sort_tasks(query_tasks(filter_=filter_, page_size=args.limit))
    return json.dumps({"tasks": [_task_to_dict(t) for t in tasks]})


# rapidfuzz partial_ratio cutoff for find_tasks. Below this, matches are
# usually noise (substrings happen to overlap). 60 catches acronyms like
# "APD" matching "Respond to the offer from APD" (which scores 100).
_FIND_TASKS_SCORE_CUTOFF = 60


def _find_tasks_tool(args: FindTasksArgs) -> str:
    if not args.name_query.strip():
        return json.dumps({"error": "name_query is empty.", "tasks": []})

    filter_: dict | None = None
    if not args.include_done:
        filter_ = {
            "property": TaskProperty.STATUS,
            "status": {"does_not_equal": "Done"},
        }
    candidates = query_tasks(filter_=filter_, page_size=200)

    by_name: dict[str, Task] = {}
    for t in candidates:
        if t.name and t.name not in by_name:
            by_name[t.name] = t
    if not by_name:
        return json.dumps({"tasks": []})

    matches = process.extract(
        args.name_query,
        list(by_name.keys()),
        scorer=fuzz.partial_ratio,
        limit=args.limit,
        score_cutoff=_FIND_TASKS_SCORE_CUTOFF,
    )
    matched = [by_name[name] for name, _score, _i in matches]
    return json.dumps({"tasks": [_task_to_dict(t) for t in matched]})


def _resolve_create_kwargs(spec: NewTask) -> dict[str, Any]:
    """Translate one ``NewTask`` spec into kwargs for ``create_task``.

    Resolves the project name to a page_id and parses the ISO date here so
    both the LLM-facing tool and the confirmation-gate executor share the
    same logic.
    """
    parsed_due = _parse_iso_date(spec.due) if spec.due else None
    parsed_priority = Priority(spec.priority) if spec.priority else None
    project_ids: list[str] | None = None
    if spec.project_name:
        proj = find_project_by_name(spec.project_name)
        if proj is None:
            raise ValueError(f"No project matching '{spec.project_name}'.")
        project_ids = [proj.page_id]
    return {
        "name": spec.name,
        "description": spec.description,
        "due": parsed_due,
        "priority": parsed_priority,
        "project_ids": project_ids,
        "labels": spec.labels,
        "my_day": spec.my_day,
    }


def _resolve_update_kwargs(spec: TaskUpdate) -> dict[str, Any]:
    """Translate one ``TaskUpdate`` spec into kwargs for ``update_task``."""
    kwargs: dict[str, Any] = {}
    if spec.name is not None:
        kwargs["name"] = spec.name
    if spec.description is not None:
        kwargs["description"] = spec.description
    if spec.due is not None:
        kwargs["due"] = None if spec.due == "" else _parse_iso_date(spec.due)
    if spec.priority is not None:
        kwargs["priority"] = Priority(spec.priority)
    if spec.status is not None:
        kwargs["status"] = Status(spec.status)
    if spec.project_name is not None:
        proj = find_project_by_name(spec.project_name)
        if proj is None:
            raise ValueError(f"No project matching '{spec.project_name}'.")
        kwargs["project_ids"] = [proj.page_id]
    if spec.labels is not None:
        kwargs["labels"] = spec.labels
    if spec.my_day is not None:
        kwargs["my_day"] = spec.my_day
    return kwargs


def execute_create_tasks(args: CreateTasksArgs) -> list[Task]:
    """Run a confirmed create_tasks batch. Each entry is created independently;
    a failure on one entry raises out of this function — callers wrap in
    try/except per entry if they want partial-success semantics."""
    return [create_task(**_resolve_create_kwargs(spec)) for spec in args.tasks]


def execute_update_tasks(args: UpdateTasksArgs) -> list[Task]:
    """Run a confirmed update_tasks batch."""
    return [
        update_task(spec.page_id, **_resolve_update_kwargs(spec))
        for spec in args.updates
    ]


def execute_complete_tasks(args: CompleteTasksArgs) -> list[Task]:
    """Run a confirmed complete_tasks batch."""
    return [complete_task(page_id) for page_id in args.page_ids]


def execute_delete_tasks(args: DeleteTasksArgs) -> list[Task]:
    """Run a confirmed delete_tasks batch."""
    return [delete_task(page_id) for page_id in args.page_ids]


# These tool entry points are called only when the confirmation gate is ever
# bypassed. In the normal flow, ``agent/loop.py`` intercepts every write tool
# call before execution and routes through the gate.


def _create_tasks_tool(args: CreateTasksArgs) -> str:
    return json.dumps(
        {"created": [_task_to_dict(t) for t in execute_create_tasks(args)]}
    )


def _update_tasks_tool(args: UpdateTasksArgs) -> str:
    return json.dumps(
        {"updated": [_task_to_dict(t) for t in execute_update_tasks(args)]}
    )


def _complete_tasks_tool(args: CompleteTasksArgs) -> str:
    return json.dumps(
        {"completed": [_task_to_dict(t) for t in execute_complete_tasks(args)]}
    )


def _delete_tasks_tool(args: DeleteTasksArgs) -> str:
    return json.dumps(
        {"deleted": [_task_to_dict(t) for t in execute_delete_tasks(args)]}
    )


def _build_shift_filter(args: ShiftDueDatesArgs) -> dict | None:
    """Translate a ShiftDueDatesArgs into a Notion filter. Always excludes
    Done tasks. Raises ``ValueError`` if ``project_name`` is supplied but no
    project matches.
    """
    conditions: list[dict] = [
        {"property": TaskProperty.STATUS, "status": {"does_not_equal": "Done"}}
    ]
    if args.due_on:
        conditions.append(
            {"property": TaskProperty.DUE, "date": {"equals": args.due_on}}
        )
    if args.due_on_or_after:
        conditions.append(
            {"property": TaskProperty.DUE, "date": {"on_or_after": args.due_on_or_after}}
        )
    if args.due_on_or_before:
        conditions.append(
            {"property": TaskProperty.DUE, "date": {"on_or_before": args.due_on_or_before}}
        )
    if args.my_day is True:
        conditions.append(
            {"property": TaskProperty.MY_DAY, "checkbox": {"equals": True}}
        )
    if args.project_name:
        proj = find_project_by_name(args.project_name)
        if proj is None:
            raise ValueError(f"No project matching '{args.project_name}'.")
        conditions.append(
            {"property": TaskProperty.PROJECT, "relation": {"contains": proj.page_id}}
        )
    return {"and": conditions}


def resolve_shift_targets(args: ShiftDueDatesArgs) -> list[Task]:
    """Find every task that ``shift_due_dates`` would touch. Used by both the
    confirmation-gate preview (to show the user what will change) and the
    execute path (to do the shifting). May return tasks with ``due is None``
    — callers should skip those."""
    return query_tasks(filter_=_build_shift_filter(args), page_size=100)


def execute_shift_due_dates(args: ShiftDueDatesArgs) -> list[Task]:
    """Apply ``delta_days`` to every matching task with a due date. Returns
    only the tasks actually updated (no-due tasks are skipped). May return
    fewer rows than the preview showed if Notion state changed in between."""
    targets = resolve_shift_targets(args)
    delta = timedelta(days=args.delta_days)
    updated: list[Task] = []
    for task in targets:
        if task.due is None:
            continue
        updated.append(update_task(task.page_id, due=task.due + delta))
    return updated


def _shift_due_dates_tool(args: ShiftDueDatesArgs) -> str:
    return json.dumps(
        {"updated": [_task_to_dict(t) for t in execute_shift_due_dates(args)]}
    )


def _list_projects_tool(_args: ListProjectsArgs) -> str:
    projects = list_projects()
    return json.dumps(
        {"projects": [{"id": p.page_id, "name": p.name} for p in projects]}
    )


# ---- Schema + dispatch wiring ----------------------------------------------

_TOOLS: dict[str, tuple[type[BaseModel], Any]] = {
    "query_tasks": (QueryTasksArgs, _query_tasks_tool),
    "find_tasks": (FindTasksArgs, _find_tasks_tool),
    "create_tasks": (CreateTasksArgs, _create_tasks_tool),
    "update_tasks": (UpdateTasksArgs, _update_tasks_tool),
    "complete_tasks": (CompleteTasksArgs, _complete_tasks_tool),
    "delete_tasks": (DeleteTasksArgs, _delete_tasks_tool),
    "shift_due_dates": (ShiftDueDatesArgs, _shift_due_dates_tool),
    "list_projects": (ListProjectsArgs, _list_projects_tool),
}


def _build_tool_schema(
    model: type[BaseModel], tool_name: str
) -> ChatCompletionFunctionToolParam:
    """Build a tool schema directly from a Pydantic model — non-strict.

    We deliberately do NOT use ``openai.pydantic_function_tool`` because it
    sets ``strict: True``, which forces every property (including optionals)
    to be present in the model's emitted JSON. Groq's gpt-oss-120b
    occasionally omits a single optional field, and strict mode turns that
    into a hard 400 ``tool_use_failed`` error that crashes the turn.

    Non-strict schemas let the model emit only the fields it cares about;
    Pydantic in :func:`call_tool` still validates and applies defaults, so
    we keep the safety without the brittleness.
    """
    schema: dict[str, Any] = model.model_json_schema()
    schema.pop("title", None)
    function: dict[str, Any] = {"name": tool_name, "parameters": schema}
    description = (model.__doc__ or "").strip()
    if description:
        function["description"] = description
    return {"type": "function", "function": function}  # type: ignore[typeddict-item]


TOOL_SCHEMAS: list[ChatCompletionFunctionToolParam] = [
    _build_tool_schema(model, tool_name)
    for tool_name, (model, _fn) in _TOOLS.items()
]


def call_tool(name: str, arguments: dict) -> str:
    """Validate arguments against the tool's Pydantic model and dispatch.

    Returns a JSON string suitable for the model.
    """
    entry = _TOOLS.get(name)
    if entry is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    model_cls, fn = entry
    start = perf_counter()
    try:
        args = model_cls.model_validate(arguments)
    except ValidationError as e:
        log.warning("tool %s arg validation failed: %s", name, e)
        return json.dumps({"error": f"Invalid arguments for {name}: {e}"})
    try:
        result = fn(args)
    except Exception as e:  # noqa: BLE001
        log.warning(
            "tool %s failed in %.0fms: %s",
            name,
            (perf_counter() - start) * 1000,
            e,
        )
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    log.info(
        "tool done",
        extra={
            "tool": name,
            "duration_ms": int((perf_counter() - start) * 1000),
        },
    )
    return result
