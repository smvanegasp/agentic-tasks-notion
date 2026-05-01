"""Shared Notion client and schema introspection.

In Notion's current data model, a "database" is a thin wrapper around one or
more "data sources". The schema (properties) and the rows live on the data
source, so reads/writes go through the ``data_sources`` endpoint with the
data-source ID — not the database ID.

The mapping {database_id -> primary data_source_id} is fetched once per process
and cached.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from notion_client import Client

from agentic_tasks.config import get_settings


@lru_cache(maxsize=1)
def get_client() -> Client:
    return Client(auth=get_settings().notion_token)


def _primary_data_source_id(database_id: str) -> str:
    db: Any = get_client().databases.retrieve(database_id=database_id)
    sources = db.get("data_sources") or []
    if not sources:
        raise RuntimeError(
            f"Database {database_id!r} has no data sources. "
            "Either the integration lacks access or the database is malformed."
        )
    return sources[0]["id"]


@lru_cache(maxsize=1)
def get_tasks_data_source_id() -> str:
    return _primary_data_source_id(get_settings().notion_tasks_db_id)


@lru_cache(maxsize=1)
def get_projects_data_source_id() -> str:
    return _primary_data_source_id(get_settings().notion_projects_db_id)


@lru_cache(maxsize=1)
def get_tasks_db_schema() -> dict:
    """Return {property_name: property_schema} for the tasks data source."""
    ds: Any = get_client().data_sources.retrieve(data_source_id=get_tasks_data_source_id())
    return ds["properties"]


@lru_cache(maxsize=1)
def get_projects_db_schema() -> dict:
    """Return {property_name: property_schema} for the projects data source."""
    ds: Any = get_client().data_sources.retrieve(
        data_source_id=get_projects_data_source_id()
    )
    return ds["properties"]


def find_title_property(schema: dict) -> str:
    """Notion DBs have exactly one title property. Return its name."""
    for name, prop in schema.items():
        if prop.get("type") == "title":
            return name
    raise RuntimeError("Database has no title property")
