# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Telegram bot that lets a single user manage their Notion to-do list in natural
language. Inbound Telegram messages drive an OpenAI tool-calling loop that reads
and writes a Notion tasks database. A scheduled job sends a daily morning digest
of today's tasks. Hosted on AWS Lambda.

The repository is **public**. Anyone may fork and deploy their own instance, so
secrets must never be committed and the bot must reject anyone who is not the
owner.

## Architecture

Two Lambda entry points, one shared codebase:

- **`webhook` Lambda** (API Gateway → Lambda) — receives Telegram updates, runs
  the agent loop, replies on Telegram. The webhook handler must reject any
  `chat.id` that does not match `ALLOWED_TELEGRAM_USER_ID`.
- **`digest` Lambda** (EventBridge Scheduler → Lambda) — fires daily at the
  configured local time. Queries Notion for today's / overdue / "My Day" tasks
  and sends a single formatted Telegram message.

The agent uses the **OpenAI SDK with tool calling** pointed at any
OpenAI-compatible endpoint. Default provider is **Groq** (`LLM_BASE_URL=
https://api.groq.com/openai/v1`) with `openai/gpt-oss-120b` for chat and
`whisper-large-v3` for speech-to-text — picked for low latency with solid
tool-calling and date reasoning. (`gpt-oss-20b` is faster but unreliable at
date math; we hit drift with it.)

To use OpenAI directly: set `LLM_BASE_URL=` (empty), use an OpenAI key, and
swap model names (`LLM_MODEL=gpt-4o-mini`, `TRANSCRIPTION_MODEL=
gpt-4o-mini-transcribe`). The SDK handles both paths transparently.

Both `agent/loop.py` and `agent/transcribe.py` build their `OpenAI()` client
from `Settings.llm_base_url` + `Settings.llm_api_key` — keep that pattern; do
not hardcode endpoints elsewhere.

Project relations are resolved by **fuzzy-matching** project names mentioned in
user messages against the live Projects DB (via `rapidfuzz`). The Projects DB
schema is inspected at runtime — there is no hard-coded title property name.

## Configuration model

Two sources, one per environment:

- **Local dev** — copy `.env.example` to `.env` (gitignored) and fill in values.
  `python-dotenv` loads it at the entry point.
- **Production** — `template.yaml` SAM Parameters define the digest time,
  timezone, model, and other behavior knobs. They flow into Lambda environment
  variables and into the EventBridge schedule. **To change the digest time or
  timezone in production, edit `template.yaml` and run `sam deploy`** — do not
  introduce a separate runtime config file. Secrets live in AWS Secrets Manager
  and are wired into Lambda env vars by the SAM template.

`src/agentic_tasks/config.py` is the single place that reads env vars. Use
`get_settings()` (cached) rather than reading `os.environ` directly elsewhere.

## Notion DB conventions

The user's Notion to-do database is based on a Thomas Frank-style template with
~50 properties (formulas, rollups, recurrence machinery, etc.). The agent only
reads/writes a focused subset:

- **Read & write:** `Name`, `Status`, `Priority`, `Due`, `Project`, `Description`, `Labels`, `My Day`
- **Ignored:** every formula, rollup, system-managed field (`Created`, `Edited*`,
  `Time Tracked*`, `Sub-Task Sorter`, `Smart List`, etc.). Recurrence is left to
  Notion's own machinery — the agent only creates one-off tasks.

Property names and option strings are **case-sensitive** and must match the user's
database exactly. They are centralized in `src/agentic_tasks/notion_io/schema.py`:

- `Status`: `To Do`, `Doing`, `Done`
- `Priority`: `Low`, `Medium`, `High`
- `Project` is a relation to a separate Projects database (its ID is in
  `NOTION_PROJECTS_DB_ID`).

If the user reports schema drift (renamed property, new status option), update
`schema.py` — never paper over it with string literals at the call site.

**`Priority` is a `status`-typed property**, not `select`. Both `Status` and
`Priority` are written as `{"status": {"name": "<value>"}}`.

**Notion's data_sources model:** the user's DB IDs are *database* IDs, but the
schema and rows live on a *data source*. All reads/writes go through
`client.data_sources` (query, retrieve), and page creation uses
`parent={"data_source_id": ...}`. The mapping is cached in `client.py` via
`get_tasks_data_source_id()` / `get_projects_data_source_id()`. Do not revert to
`databases.query` — that endpoint is being phased out.

## Public-repo safety rules

- Never write a real secret into any file under version control. `.env` is
  gitignored; `.env.example` shows variable *names* only.
- Never put secrets, the user's Telegram ID, or DB IDs in the SAM template as
  literals — surface them as parameters resolved from Secrets Manager / passed
  at deploy time.
- The webhook handler **must** check `chat.id == ALLOWED_TELEGRAM_USER_ID` and
  reject otherwise. Do not log message bodies from unauthorized senders.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency
management. SAM commands land once `template.yaml` exists.

```bash
# Install dependencies (creates .venv, writes uv.lock)
uv sync

# Install with dev dependencies
uv sync --group dev

# Run a one-off command in the project env
uv run python -m agentic_tasks.handlers.webhook

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_notion_tasks.py::test_create_task -v

# Lint / type-check
uv run ruff check .
uv run mypy src
```

## Build sequence (in progress)

Implementation lands in stages. The agreed sequence:

1. ✅ **Scaffold** — `pyproject.toml`, `.env.example`, README, `config.py`, `notion_io/schema.py`.
2. ✅ **Notion layer** — `notion_io/client.py`, `tasks.py`, `projects.py`. 13 tests, all mocked. `scripts/notion_smoke.py` exercises the real DB read-only.
3. ✅ **Agent layer** — `agent/loop.py`, `tools.py`, `prompts.py`. 15 tests, mocked. `scripts/agent_repl.py` for interactive end-to-end testing.
4. ✅ **Telegram layer** — `telegram_io/dispatcher.py` (shared) + `telegram_io/auth.py` + `handlers/webhook.py` (Lambda). 10 tests, mocked. `scripts/telegram_local.py` runs the bot via long-polling for local testing — no public URL required.
5. ✅ **Digest layer** — `digest/render.py` (pure formatter, 12 tests) + `handlers/digest.py` (Lambda handler, 3 tests). `scripts/digest_local.py` previews/sends the digest locally with `--dry-run` and `--date` flags. No LLM call — deterministic and free to run.
6. **SAM template** — `template.yaml` with both Lambdas, EventBridge cron, Secrets Manager wiring, parameters for `DIGEST_TIME`, `TIMEZONE`, `OPENAI_MODEL`.
7. **Deploy + register webhook** — first AWS deploy; `scripts/set_webhook.py` points Telegram at the API Gateway URL.
8. **README polish** — fork-and-deploy instructions for public users.

## Things that are deliberately NOT in this project

- No multi-user support. Single-user only by design.
- No conversation memory across Telegram messages (each message is a fresh
  agent turn, with state reconstructed from Notion as needed).
- No recurring task creation logic — Notion's template handles recurrence.
- No framework on top of the OpenAI SDK (no LangChain, etc.). Tool calling is
  used directly.

## Reference: prior art

Tom Frank's `pipedream-notion-voice-tasks` repo (JavaScript on Pipedream) inspired
the project. Reusable ideas borrowed from it: task-parsing prompt structure,
ISO 8601 due-date handling with relative-date interpretation, fuzzy assignee
and project matching, defensive JSON-repair fallback. Differences: this project
is conversational (two-way), supports read/update/complete (not just create),
uses tool calling instead of multi-round JSON prompts, and runs on AWS.
