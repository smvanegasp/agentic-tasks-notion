# agentic-tasks-notion

A Telegram bot that lets you manage your Notion to-do list in natural language.
Send a message, and an LLM agent (OpenAI tool calling) creates, updates, queries,
or completes tasks in your Notion database. A daily morning digest of today's
tasks is delivered automatically.

Designed to be deployed on AWS Lambda — the free tier comfortably covers personal use.

## Status

Early development. Implementation lands in stages; see commit history for design context.

## Stack

- **Runtime:** Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- **Telegram:** `python-telegram-bot` (webhook mode)
- **LLM:** OpenAI SDK with tool calling (default `gpt-4o-mini`, configurable)
- **Notion:** `notion-client` SDK
- **Hosting:** AWS Lambda + API Gateway + EventBridge Scheduler, deployed via AWS SAM

## Configuration

Two layers, by environment:

- **Local dev** — copy `.env.example` to `.env` and fill in values.
- **Production (AWS)** — values flow from `template.yaml` SAM Parameters into Lambda
  environment variables. To change the digest time, timezone, or model, edit
  `template.yaml` and run `sam deploy`. Secrets live in AWS Secrets Manager.

## Local setup

A one-time onboarding so you can run the bot locally before deploying anywhere.

### 1. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/) for Python packaging:

```sh
uv sync --group dev
```

That creates `.venv/`, resolves the dependency tree into `uv.lock`, and installs
runtime and dev dependencies in one shot.

### 2. Set up Notion access

1. **Create a Notion integration** at <https://www.notion.so/my-integrations>.
   Click "+ New integration", give it a name (e.g. `agentic-tasks-bot`), and copy
   the **Internal Integration Token** — it starts with `ntn_` or `secret_`.
2. **Share both databases with the integration.** In Notion, open your Tasks DB,
   click `•••` (top right) → **Connections** → add your integration. Repeat for
   the Projects DB. Without this step the integration cannot read or write either
   database.
3. **Grab the database IDs.** Open each database as a full page. The URL looks
   like `notion.so/<workspace>/<32-hex-id>?v=...`. The 32-character hex chunk
   before the `?` is the database ID.
4. **Create your local env file** by copying the template:

   ```sh
   cp .env.example .env
   ```

   Fill in `NOTION_TOKEN`, `NOTION_TASKS_DB_ID`, and `NOTION_PROJECTS_DB_ID`.
   The other variables are needed only when the Telegram and OpenAI layers are
   running — leave them blank for now. `.env` is gitignored, so your secrets
   stay local.

### 3. Verify with the smoke test

```sh
uv run python scripts/notion_smoke.py
```

This is a read-only sanity check. It inspects your Tasks DB schema (flagging any
property name or type that does not match what the code assumes), lists a few
tasks and projects, and writes nothing.

### 4. Run the test suite

```sh
uv run pytest
```

Tests are fully mocked — they don't hit Notion or any other network.
