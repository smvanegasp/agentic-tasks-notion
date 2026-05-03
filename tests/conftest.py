"""Shared pytest fixtures.

Stubs the env vars `Settings.from_env()` requires and clears the lru_caches
between tests so each test gets fresh settings/clients.
"""

import pytest


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch):
    for k, v in {
        "TELEGRAM_BOT_TOKEN": "test-bot-token",
        "NOTION_TOKEN": "test-notion-token",
        "LLM_API_KEY": "test-llm-key",
        "NOTION_TASKS_DB_ID": "tasks-db-id",
        "NOTION_PROJECTS_DB_ID": "projects-db-id",
        "ALLOWED_TELEGRAM_USER_ID": "12345",
        # The LLM-based format check is opt-in for tests — exercising tests
        # that need it set their own value with monkeypatch.
        "LLM_FORMAT_CHECK_MODEL": "",
    }.items():
        monkeypatch.setenv(k, v)

    from agentic_tasks.config import get_settings
    from agentic_tasks.conversation.store import get_store
    from agentic_tasks.notion_io.client import (
        get_client,
        get_projects_data_source_id,
        get_projects_db_schema,
        get_tasks_data_source_id,
        get_tasks_db_schema,
    )
    from agentic_tasks.notion_io.projects import list_projects

    for cached in (
        get_settings,
        get_client,
        get_tasks_data_source_id,
        get_projects_data_source_id,
        get_tasks_db_schema,
        get_projects_db_schema,
        get_store,
        list_projects,
    ):
        cached.cache_clear()
