"""Agent tools: OpenAI function-calling schemas and a dispatcher to local code.

Adding a new tool: append a schema to ``TOOL_SCHEMAS`` and a callable to
``DISPATCH``. Each tool returns a JSON string suitable for the model.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from time import perf_counter
from typing import Any

from agentic_tasks.config import get_settings
from agentic_tasks.notion_io.projects import find_project_by_name, list_projects
from agentic_tasks.notion_io.schema import Priority, Status, TaskProperty
from agentic_tasks.notion_io.tasks import (
    Task,
    complete_task,
    create_task,
    query_tasks,
    update_task,
)

log = logging.getLogger(__name__)

_PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}


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


def _query_tasks_tool(
    *,
    status: str | None = None,
    include_done: bool = False,
    due_on: str | None = None,
    due_on_or_after: str | None = None,
    due_on_or_before: str | None = None,
    my_day: bool | None = None,
    project_name: str | None = None,
    limit: int = 25,
) -> str:
    conditions: list[dict] = []

    if status:
        conditions.append({"property": TaskProperty.STATUS, "status": {"equals": status}})
    elif not include_done:
        conditions.append(
            {"property": TaskProperty.STATUS, "status": {"does_not_equal": "Done"}}
        )

    if due_on:
        conditions.append({"property": TaskProperty.DUE, "date": {"equals": due_on}})
    if due_on_or_after:
        conditions.append(
            {"property": TaskProperty.DUE, "date": {"on_or_after": due_on_or_after}}
        )
    if due_on_or_before:
        conditions.append(
            {"property": TaskProperty.DUE, "date": {"on_or_before": due_on_or_before}}
        )

    if my_day is True:
        conditions.append({"property": TaskProperty.MY_DAY, "checkbox": {"equals": True}})

    if project_name:
        proj = find_project_by_name(project_name)
        if proj is None:
            return json.dumps({"error": f"No project matching '{project_name}'.", "tasks": []})
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

    tasks = _sort_tasks(query_tasks(filter_=filter_, page_size=limit))
    return json.dumps({"tasks": [_task_to_dict(t) for t in tasks]})


def _create_task_tool(
    *,
    name: str,
    description: str | None = None,
    due: str | None = None,
    priority: str | None = None,
    project_name: str | None = None,
    labels: list[str] | None = None,
    my_day: bool = False,
) -> str:
    parsed_due = _parse_iso_date(due) if due else None
    parsed_priority = Priority(priority) if priority else None

    project_ids: list[str] | None = None
    if project_name:
        proj = find_project_by_name(project_name)
        if proj is None:
            return json.dumps({"error": f"No project matching '{project_name}'."})
        project_ids = [proj.page_id]

    task = create_task(
        name=name,
        description=description,
        due=parsed_due,
        priority=parsed_priority,
        project_ids=project_ids,
        labels=labels,
        my_day=my_day,
    )
    return json.dumps({"created": _task_to_dict(task)})


def _update_task_tool(
    *,
    page_id: str,
    name: str | None = None,
    description: str | None = None,
    due: str | None = None,
    priority: str | None = None,
    status: str | None = None,
    project_name: str | None = None,
    labels: list[str] | None = None,
    my_day: bool | None = None,
) -> str:
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if description is not None:
        kwargs["description"] = description
    if due is not None:
        kwargs["due"] = None if due == "" else _parse_iso_date(due)
    if priority is not None:
        kwargs["priority"] = Priority(priority)
    if status is not None:
        kwargs["status"] = Status(status)
    if project_name is not None:
        proj = find_project_by_name(project_name)
        if proj is None:
            return json.dumps({"error": f"No project matching '{project_name}'."})
        kwargs["project_ids"] = [proj.page_id]
    if labels is not None:
        kwargs["labels"] = labels
    if my_day is not None:
        kwargs["my_day"] = my_day

    task = update_task(page_id, **kwargs)
    return json.dumps({"updated": _task_to_dict(task)})


def _complete_task_tool(*, page_id: str) -> str:
    task = complete_task(page_id)
    return json.dumps({"completed": _task_to_dict(task)})


def _list_projects_tool() -> str:
    projects = list_projects()
    return json.dumps(
        {"projects": [{"id": p.page_id, "name": p.name} for p in projects]}
    )


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_tasks",
            "description": (
                "Query tasks from the user's Notion to-do database. By default"
                " excludes completed tasks. Use due_on for a specific date and"
                " due_on_or_after / due_on_or_before for ranges — compute"
                " concrete YYYY-MM-DD dates from the user's words yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["To Do", "Doing", "Done"],
                    },
                    "include_done": {
                        "type": "boolean",
                        "description": (
                            "Set true to also include completed tasks. Default"
                            " false — done tasks are filtered out."
                        ),
                    },
                    "due_on": {
                        "type": "string",
                        "description": (
                            "Match tasks whose Due date equals this YYYY-MM-DD."
                            " Use for queries like 'tomorrow' or a specific day."
                        ),
                    },
                    "due_on_or_after": {
                        "type": "string",
                        "description": (
                            "Match tasks whose Due date is on or after this"
                            " YYYY-MM-DD. Combine with due_on_or_before for"
                            " ranges."
                        ),
                    },
                    "due_on_or_before": {
                        "type": "string",
                        "description": (
                            "Match tasks whose Due date is on or before this"
                            " YYYY-MM-DD."
                        ),
                    },
                    "my_day": {
                        "type": "boolean",
                        "description": "Only tasks marked 'My Day'.",
                    },
                    "project_name": {
                        "type": "string",
                        "description": "Filter by project (fuzzy matched).",
                    },
                    "limit": {"type": "integer", "default": 25},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a new task in Notion.",
            "parameters": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "due": {
                        "type": "string",
                        "description": (
                            "ISO 8601 date (YYYY-MM-DD) or datetime "
                            "(YYYY-MM-DDTHH:MM:SS with optional offset)."
                        ),
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["Low", "Medium", "High"],
                    },
                    "project_name": {
                        "type": "string",
                        "description": "Project to attach (fuzzy matched).",
                    },
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "my_day": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_task",
            "description": (
                "Update fields on an existing task. Only provided fields "
                "change. Pass an empty string for 'due' to clear it."
            ),
            "parameters": {
                "type": "object",
                "required": ["page_id"],
                "properties": {
                    "page_id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "due": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["Low", "Medium", "High"],
                    },
                    "status": {
                        "type": "string",
                        "enum": ["To Do", "Doing", "Done"],
                    },
                    "project_name": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "my_day": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark a task as Done.",
            "parameters": {
                "type": "object",
                "required": ["page_id"],
                "properties": {"page_id": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "List every project in the projects DB (for disambiguation).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


DISPATCH: dict[str, Any] = {
    "query_tasks": _query_tasks_tool,
    "create_task": _create_task_tool,
    "update_task": _update_task_tool,
    "complete_task": _complete_task_tool,
    "list_projects": _list_projects_tool,
}


def call_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call. Returns a JSON string for the model."""
    fn = DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    start = perf_counter()
    try:
        result = fn(**arguments)
    except Exception as e:  # noqa: BLE001
        log.warning(
            "tool %s failed in %.0fms: %s",
            name,
            (perf_counter() - start) * 1000,
            e,
        )
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    log.info("tool %s done in %.0fms", name, (perf_counter() - start) * 1000)
    return result
