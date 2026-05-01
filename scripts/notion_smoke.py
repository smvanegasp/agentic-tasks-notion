"""Read-only smoke test against the real Notion databases.

Verifies credentials work and inspects whether the property names + types in
your tasks DB match what `schema.py` assumes. Lists a handful of tasks and
projects. Writes nothing.

Run from the repo root:

    uv run python scripts/notion_smoke.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from agentic_tasks.notion_io.client import (
    find_title_property,
    get_projects_db_schema,
    get_tasks_db_schema,
)
from agentic_tasks.notion_io.projects import list_projects
from agentic_tasks.notion_io.schema import TaskProperty
from agentic_tasks.notion_io.tasks import query_tasks, query_today


def _stub_unused_env() -> None:
    """The smoke test only exercises the Notion layer. Stub the rest so that
    ``Settings.from_env()`` (which validates every required field upfront) does
    not crash because, say, ``TELEGRAM_BOT_TOKEN`` is empty."""
    placeholders = {
        "TELEGRAM_BOT_TOKEN": "smoke-placeholder",
        "LLM_API_KEY": "smoke-placeholder",
        "ALLOWED_TELEGRAM_USER_ID": "0",
    }
    for key, fallback in placeholders.items():
        if not os.environ.get(key):
            os.environ[key] = fallback

EXPECTED_TASK_PROPS = {
    TaskProperty.NAME: "title",
    TaskProperty.STATUS: "status",
    TaskProperty.PRIORITY: "status",
    TaskProperty.DUE: "date",
    TaskProperty.PROJECT: "relation",
    TaskProperty.DESCRIPTION: "rich_text",
    TaskProperty.LABELS: "multi_select",
    TaskProperty.MY_DAY: "checkbox",
}


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    # Force UTF-8 output so non-ASCII task names (Spanish accents, ñ, etc.)
    # render correctly on Windows consoles, which default to legacy codepages.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    load_dotenv()
    _stub_unused_env()

    _section("Tasks DB schema check")
    schema = get_tasks_db_schema()
    mismatches: list[str] = []
    for prop_name, expected_type in EXPECTED_TASK_PROPS.items():
        if prop_name not in schema:
            print(f"  MISSING       {prop_name!r}")
            mismatches.append(prop_name)
            continue
        actual_type = schema[prop_name].get("type")
        marker = "OK   " if actual_type == expected_type else "WRONG"
        print(f"  {marker:<13} {prop_name:<14} expected={expected_type:<12} actual={actual_type}")
        if actual_type != expected_type:
            mismatches.append(f"{prop_name} (expected {expected_type}, got {actual_type})")

    if mismatches:
        print()
        print("  Schema differences detected. Update schema.py constants or the")
        print("  parse / build logic in tasks.py to match the real database.")

    _section("Projects DB title property")
    proj_schema = get_projects_db_schema()
    title_prop = find_title_property(proj_schema)
    print(f"  Title property: {title_prop!r}")

    _section("First 5 tasks (any status)")
    tasks = query_tasks(page_size=5)
    if not tasks:
        print("  (none)")
    for t in tasks:
        print(f"  - {t.name!r} status={t.status} priority={t.priority} due={t.due}")

    _section("Today's tasks (Due=today OR My Day, status != Done)")
    today_tasks = query_today()
    if not today_tasks:
        print("  (none)")
    for t in today_tasks:
        print(f"  - {t.name!r} my_day={t.my_day} due={t.due} priority={t.priority}")

    _section("Projects (first 5)")
    projects = list_projects()
    for p in projects[:5]:
        print(f"  - {p.name!r} ({p.page_id})")
    print(f"  total: {len(projects)}")


if __name__ == "__main__":
    main()
