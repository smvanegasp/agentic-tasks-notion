"""Read/write operations for the Notion tasks database.

Functions return / accept small dataclasses, not raw Notion JSON. The agent
layer should only see :class:`Task`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from agentic_tasks.notion_io.client import get_client, get_tasks_data_source_id
from agentic_tasks.notion_io.schema import Priority, Status, TaskProperty

_UNSET: Any = object()


@dataclass(frozen=True)
class Task:
    page_id: str
    name: str
    status: str | None
    priority: str | None
    due: date | datetime | None
    project_ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    description: str = ""
    my_day: bool = False
    url: str = ""
    archived: bool = False


def _plain_text(rich_text: list[dict]) -> str:
    return "".join(t.get("plain_text", "") for t in rich_text or [])


def _parse_date(date_value: dict | None) -> date | datetime | None:
    if not date_value or not date_value.get("start"):
        return None
    start = date_value["start"]
    if "T" in start:
        return datetime.fromisoformat(start)
    return date.fromisoformat(start)


def _parse_task(page: dict) -> Task:
    props = page.get("properties", {})

    name = _plain_text(props.get(TaskProperty.NAME, {}).get("title", []))

    status_obj = props.get(TaskProperty.STATUS, {}).get("status")
    status = status_obj["name"] if status_obj else None

    priority_obj = props.get(TaskProperty.PRIORITY, {}).get("status")
    priority = priority_obj["name"] if priority_obj else None

    due = _parse_date(props.get(TaskProperty.DUE, {}).get("date"))

    project_ids = [r["id"] for r in props.get(TaskProperty.PROJECT, {}).get("relation", [])]

    labels = [
        item["name"] for item in props.get(TaskProperty.LABELS, {}).get("multi_select", [])
    ]

    description = _plain_text(props.get(TaskProperty.DESCRIPTION, {}).get("rich_text", []))

    my_day = bool(props.get(TaskProperty.MY_DAY, {}).get("checkbox", False))

    return Task(
        page_id=page["id"],
        name=name,
        status=status,
        priority=priority,
        due=due,
        project_ids=project_ids,
        labels=labels,
        description=description,
        my_day=my_day,
        url=page.get("url", ""),
        archived=bool(page.get("archived") or page.get("in_trash")),
    )


def get_task(page_id: str) -> Task:
    """Retrieve a single task by Notion page ID.

    Used by the confirmation gate so previews can display the human-readable
    task name for update / complete operations.
    """
    response: Any = get_client().pages.retrieve(page_id=page_id)
    return _parse_task(response)


def query_tasks(
    filter_: dict | None = None,
    sorts: list[dict] | None = None,
    page_size: int = 50,
) -> list[Task]:
    kwargs: dict[str, Any] = {
        "data_source_id": get_tasks_data_source_id(),
        "page_size": page_size,
    }
    if filter_:
        kwargs["filter"] = filter_
    if sorts:
        kwargs["sorts"] = sorts
    response: Any = get_client().data_sources.query(**kwargs)
    return [_parse_task(p) for p in response.get("results", [])]


def query_today(today: date | None = None) -> list[Task]:
    """Tasks due today OR with `My Day` set, status != Done.

    Returns Notion's natural order — callers are expected to sort by priority
    in Python, since Notion's sort on a status-typed property follows the
    option order in the database schema, not semantic priority order.
    """
    today = today or date.today()
    today_str = today.isoformat()
    return query_tasks(
        filter_={
            "and": [
                {
                    "property": TaskProperty.STATUS,
                    "status": {"does_not_equal": Status.DONE.value},
                },
                {
                    "or": [
                        {"property": TaskProperty.DUE, "date": {"equals": today_str}},
                        {"property": TaskProperty.MY_DAY, "checkbox": {"equals": True}},
                    ]
                },
            ]
        },
    )


def query_due_today(today: date | None = None) -> list[Task]:
    """Tasks due today only (not My Day), status != Done."""
    today = today or date.today()
    return query_tasks(
        filter_={
            "and": [
                {
                    "property": TaskProperty.STATUS,
                    "status": {"does_not_equal": Status.DONE.value},
                },
                {"property": TaskProperty.DUE, "date": {"equals": today.isoformat()}},
            ]
        },
    )


def query_my_day() -> list[Task]:
    """Tasks flagged ``My Day``, status != Done. Independent of due date."""
    return query_tasks(
        filter_={
            "and": [
                {
                    "property": TaskProperty.STATUS,
                    "status": {"does_not_equal": Status.DONE.value},
                },
                {"property": TaskProperty.MY_DAY, "checkbox": {"equals": True}},
            ]
        },
    )


def query_overdue(today: date | None = None) -> list[Task]:
    """Tasks with `Due` < today, status != Done."""
    today = today or date.today()
    today_str = today.isoformat()
    return query_tasks(
        filter_={
            "and": [
                {
                    "property": TaskProperty.STATUS,
                    "status": {"does_not_equal": Status.DONE.value},
                },
                {"property": TaskProperty.DUE, "date": {"before": today_str}},
            ]
        },
        sorts=[{"property": TaskProperty.DUE, "direction": "ascending"}],
    )


def _date_value(value: date | datetime) -> dict:
    if isinstance(value, datetime):
        return {"start": value.isoformat()}
    return {"start": value.isoformat()}


def create_task(
    name: str,
    *,
    description: str | None = None,
    due: date | datetime | None = None,
    priority: Priority | None = None,
    status: Status = Status.TO_DO,
    project_ids: list[str] | None = None,
    labels: list[str] | None = None,
    my_day: bool = False,
) -> Task:
    properties: dict[str, Any] = {
        TaskProperty.NAME: {"title": [{"text": {"content": name}}]},
        TaskProperty.STATUS: {"status": {"name": status.value}},
    }
    if priority is not None:
        properties[TaskProperty.PRIORITY] = {"status": {"name": priority.value}}
    if due is not None:
        properties[TaskProperty.DUE] = {"date": _date_value(due)}
    if project_ids:
        properties[TaskProperty.PROJECT] = {
            "relation": [{"id": pid} for pid in project_ids]
        }
    if labels:
        properties[TaskProperty.LABELS] = {
            "multi_select": [{"name": label} for label in labels]
        }
    if description:
        properties[TaskProperty.DESCRIPTION] = {
            "rich_text": [{"text": {"content": description}}]
        }
    if my_day:
        properties[TaskProperty.MY_DAY] = {"checkbox": True}

    response: Any = get_client().pages.create(
        parent={"data_source_id": get_tasks_data_source_id()},
        properties=properties,
    )
    return _parse_task(response)


def update_task(
    page_id: str,
    *,
    name: Any = _UNSET,
    description: Any = _UNSET,
    due: Any = _UNSET,
    priority: Any = _UNSET,
    status: Any = _UNSET,
    project_ids: Any = _UNSET,
    labels: Any = _UNSET,
    my_day: Any = _UNSET,
) -> Task:
    """Update only the fields you pass. Pass ``None`` to clear a clearable field."""
    properties: dict[str, Any] = {}

    if name is not _UNSET:
        properties[TaskProperty.NAME] = {"title": [{"text": {"content": name}}]}
    if status is not _UNSET:
        properties[TaskProperty.STATUS] = {"status": {"name": status.value}}
    if priority is not _UNSET:
        properties[TaskProperty.PRIORITY] = (
            {"status": None} if priority is None else {"status": {"name": priority.value}}
        )
    if due is not _UNSET:
        properties[TaskProperty.DUE] = (
            {"date": None} if due is None else {"date": _date_value(due)}
        )
    if project_ids is not _UNSET:
        properties[TaskProperty.PROJECT] = {
            "relation": [{"id": pid} for pid in (project_ids or [])]
        }
    if labels is not _UNSET:
        properties[TaskProperty.LABELS] = {
            "multi_select": [{"name": label} for label in (labels or [])]
        }
    if description is not _UNSET:
        properties[TaskProperty.DESCRIPTION] = {
            "rich_text": [{"text": {"content": description or ""}}]
        }
    if my_day is not _UNSET:
        properties[TaskProperty.MY_DAY] = {"checkbox": bool(my_day)}

    response: Any = get_client().pages.update(page_id=page_id, properties=properties)
    return _parse_task(response)


def complete_task(page_id: str) -> Task:
    return update_task(page_id, status=Status.DONE)


def delete_task(page_id: str) -> Task:
    """Move a task page to Notion's Trash via ``in_trash=True``.

    ``in_trash`` is the documented field name in Notion API version
    2025-09-03 (the version pinned by ``notion-client`` 3.x). The page is
    recoverable from Notion's Trash UI; its URL keeps working (it just
    lands on the "moved to Trash" view), so callers can still link to it
    in the confirmation summary.
    """
    response: Any = get_client().pages.update(page_id=page_id, in_trash=True)
    return _parse_task(response)
