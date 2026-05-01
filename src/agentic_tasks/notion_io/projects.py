"""Read-only operations for the Notion projects database.

The projects DB title property name is auto-detected at runtime, so we don't
hardcode whether it's called "Name", "Project", or anything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rapidfuzz import fuzz, process

from agentic_tasks.notion_io.client import (
    find_title_property,
    get_client,
    get_projects_data_source_id,
    get_projects_db_schema,
)


@dataclass(frozen=True)
class Project:
    page_id: str
    name: str


def _parse_project(page: dict, title_prop: str) -> Project:
    title_data = page.get("properties", {}).get(title_prop, {}).get("title", [])
    name = "".join(t.get("plain_text", "") for t in title_data)
    return Project(page_id=page["id"], name=name)


@lru_cache(maxsize=1)
def list_projects() -> list[Project]:
    """Return every project page in the Projects DB.

    Cached for the process lifetime so repeated lookups (especially within a
    single agent turn that needs both list_projects and find_project_by_name)
    don't repeatedly hit Notion. The bot needs a restart to pick up new
    projects added in Notion — acceptable trade-off for personal use.
    """
    title_prop = find_title_property(get_projects_db_schema())
    data_source_id = get_projects_data_source_id()
    client = get_client()

    projects: list[Project] = []
    cursor: str | None = None
    while True:
        kwargs: dict = {"data_source_id": data_source_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        response: Any = client.data_sources.query(**kwargs)
        projects.extend(_parse_project(p, title_prop) for p in response.get("results", []))
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")

    return projects


def find_project_by_name(query: str, *, score_cutoff: int = 70) -> Project | None:
    """Fuzzy-match a project by name. Returns ``None`` below the score cutoff."""
    if not query:
        return None
    projects = list_projects()
    if not projects:
        return None

    by_name = {p.name: p for p in projects if p.name}
    if not by_name:
        return None

    result = process.extractOne(
        query, list(by_name.keys()), scorer=fuzz.WRatio, score_cutoff=score_cutoff
    )
    if result is None:
        return None
    matched_name, _score, _index = result
    return by_name[matched_name]
