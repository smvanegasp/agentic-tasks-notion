"""Configuration loaded from environment.

In production, SAM template parameters become Lambda env vars.
For local dev, populate `.env` and load it at the entry point with python-dotenv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from zoneinfo import ZoneInfo


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name!r} is not set")
    return value


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    notion_token: str
    llm_api_key: str

    notion_tasks_db_id: str
    notion_projects_db_id: str

    allowed_telegram_user_id: int
    telegram_webhook_secret: str | None

    llm_base_url: str  # empty string means "use the OpenAI SDK default"
    llm_model: str
    transcription_model: str
    timezone: ZoneInfo
    digest_time: str
    conversation_history_limit: int
    agent_timeout_seconds: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            telegram_bot_token=_required("TELEGRAM_BOT_TOKEN"),
            notion_token=_required("NOTION_TOKEN"),
            llm_api_key=_required("LLM_API_KEY"),
            notion_tasks_db_id=_required("NOTION_TASKS_DB_ID"),
            notion_projects_db_id=_required("NOTION_PROJECTS_DB_ID"),
            allowed_telegram_user_id=int(_required("ALLOWED_TELEGRAM_USER_ID")),
            telegram_webhook_secret=os.environ.get("TELEGRAM_WEBHOOK_SECRET"),
            llm_base_url=os.environ.get(
                "LLM_BASE_URL", "https://api.groq.com/openai/v1"
            ),
            llm_model=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
            transcription_model=os.environ.get(
                "TRANSCRIPTION_MODEL", "whisper-large-v3"
            ),
            timezone=ZoneInfo(os.environ.get("TIMEZONE", "America/New_York")),
            digest_time=os.environ.get("DIGEST_TIME", "08:00"),
            conversation_history_limit=int(
                os.environ.get("CONVERSATION_HISTORY_LIMIT", "40")
            ),
            agent_timeout_seconds=int(os.environ.get("AGENT_TIMEOUT_SECONDS", "60")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
