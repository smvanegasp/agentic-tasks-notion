# agentic-tasks-notion

A Telegram bot that lets you manage your Notion to-do list in natural language.
Send a text or voice message and an LLM agent creates, updates, queries, or
completes tasks in your Notion database. A daily morning digest of today's
tasks is delivered automatically.

Designed to run on AWS Lambda — the free tier comfortably covers personal use.

## Stack

- **Runtime:** Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- **Telegram:** `python-telegram-bot` (webhook in production, long-polling locally)
- **LLM:** OpenAI SDK pointed at any OpenAI-compatible endpoint. Default
  provider is **Groq** (`openai/gpt-oss-120b` for chat, `whisper-large-v3` for
  voice transcription). Switch to OpenAI by setting `LLM_BASE_URL` empty in
  your env.
- **Notion:** `notion-client` SDK using the new `data_sources` endpoints.
- **Hosting:** AWS Lambda + API Gateway + EventBridge Scheduler + DynamoDB +
  Secrets Manager, deployed via AWS SAM.

## Configuration

Two layers, by environment:

- **Local dev** — copy `.env.example` to `.env` and fill in values.
- **Production (AWS)** — non-secret config lives in `template.yaml`
  parameters (digest time, timezone, model). Secrets (tokens, API keys, your
  Telegram user ID) live in AWS Secrets Manager and are pulled into Lambda
  env vars on cold start. To change non-secret config, edit `template.yaml`
  and run `sam deploy`. To rotate secrets, edit `.env` and re-run
  `scripts/seed_secrets.py`.

---

## Local setup

A one-time onboarding so you can run the bot on your laptop before deploying
anywhere.

### 1. Install dependencies

```sh
uv sync --group dev
```

Creates `.venv/`, resolves the dependency tree into `uv.lock`, and installs
runtime + dev dependencies.

### 2. Set up Notion access

1. **Create a Notion integration** at <https://www.notion.so/my-integrations>.
   Click "+ New integration", name it (e.g. `agentic-tasks-bot`), and copy
   the **Internal Integration Token** — starts with `ntn_` or `secret_`.
2. **Share both databases with the integration.** In Notion, open your Tasks
   DB, click `•••` (top right) → **Connections** → add your integration.
   Repeat for the Projects DB.
3. **Grab the database IDs.** Open each database as a full page. The URL
   looks like `notion.so/<workspace>/<32-hex-id>?v=...`. The 32-character hex
   chunk before `?` is the database ID.

### 3. Set up the Telegram bot

1. Open Telegram, message **@BotFather**.
2. Send `/newbot`, follow the prompts. Copy the **bot token** (looks like
   `123456:ABC-...`).
3. Message **@userinfobot** to get your numeric Telegram user ID.

### 4. Get an LLM API key

Default provider is **Groq** (fast and cheap). Sign up at
<https://console.groq.com> and create an API key. Keys start with `gsk_`.

To use OpenAI instead, get a key from <https://platform.openai.com/api-keys>
and override `LLM_BASE_URL`, `LLM_MODEL`, `TRANSCRIPTION_MODEL` in `.env`.

### 5. Create your `.env`

```sh
cp .env.example .env
```

Fill in the keys you just collected:
- `TELEGRAM_BOT_TOKEN` (from BotFather)
- `ALLOWED_TELEGRAM_USER_ID` (from @userinfobot)
- `NOTION_TOKEN`, `NOTION_TASKS_DB_ID`, `NOTION_PROJECTS_DB_ID`
- `LLM_API_KEY`

`.env` is gitignored — secrets stay local.

### 6. Verify with the Notion smoke test

```sh
uv run python scripts/notion_smoke.py
```

Read-only check — confirms credentials work and your DB schema matches what
the code expects. Writes nothing.

### 7. Run the test suite

```sh
uv run pytest
```

Fully mocked, no network calls.

### 8. Run the bot locally

```sh
uv run python scripts/telegram_local.py
```

Uses Telegram long-polling so you don't need a public URL. Send your bot a
message from your phone — it should reply within a few seconds.

To preview the daily digest without scheduling:
```sh
uv run python scripts/digest_local.py --dry-run
```

---

## Deploy to AWS

The bot lives on Lambda once deployed. EventBridge fires the morning digest
on schedule, API Gateway routes Telegram webhook POSTs to the agent.

### Prerequisites

- An AWS account with the AWS CLI installed and configured (`aws configure`).
- The SAM CLI:
  ```sh
  uv tool install aws-sam-cli
  sam --version
  ```
  (Or use `pipx`, the official MSI from
  <https://github.com/aws/aws-sam-cli/releases>, `scoop`, `choco`, etc.)
- Python 3.12 on your `PATH` for `sam build` to find. If you used `uv` to
  install Python, it lives in a uv-managed location SAM can't see by default.
  Two fixes:
  - **Per-session (PowerShell):**
    ```powershell
    $env:Path = "$env:USERPROFILE\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none;$env:Path"
    ```
  - **Permanent:** install Python 3.12 from <https://www.python.org/downloads/>
    and tick *Add python.exe to PATH* during install.

  Verify with `python --version` → `Python 3.12.x`.

### 1. Add a webhook secret to your `.env`

A random string Telegram echoes back so the Lambda can verify each incoming
webhook came from Telegram:

```sh
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the output into `.env`:
```
TELEGRAM_WEBHOOK_SECRET=<paste here>
```

### 2. Build the Lambda zip

```sh
sam build
```

Installs `src/requirements.txt` and packages your code. ~30-60 seconds.

### 3. First deploy (interactive)

```sh
sam deploy --guided
```

Answer the prompts (defaults are fine):
- Stack Name → press Enter (`agentic-tasks-notion`)
- AWS Region → press Enter (`us-east-2`)
- All `Parameter*` prompts → press Enter
- Confirm changes before deploy → `y`
- Allow SAM CLI IAM role creation → `y`
- Disable rollback → `n`
- Save arguments to configuration file → `y`

It'll show a CloudFormation changeset — type `y` to deploy. Takes ~2-3
minutes. When done, scroll up to find the **Outputs** block:

```
WebhookUrl              https://<id>.execute-api.us-east-2.amazonaws.com/webhook
SecretsName             agentic-tasks-notion-secrets
ConversationTableName   agentic-tasks-notion-conversation
```

You'll need `WebhookUrl` and `SecretsName` next. **Copy them somewhere
safe** before moving on.

> Lost the outputs? Re-fetch any time:
> ```sh
> sam list stack-outputs --stack-name agentic-tasks-notion --region us-east-2
> ```

### 4. Push secrets from `.env` to Secrets Manager

```sh
uv run python scripts/seed_secrets.py --secret-name agentic-tasks-notion-secrets
```

(Substitute your actual `SecretsName` from the deploy output.)

### 5. Register the webhook with Telegram

```sh
uv run python scripts/set_webhook.py --url https://<your-id>.execute-api.us-east-2.amazonaws.com/webhook
```

**Substitute your actual `WebhookUrl`** from the deploy output (the full
`https://...amazonaws.com/webhook` value). The `<your-id>` placeholder is
not meant to be typed literally.

### 6. Test the webhook

In Telegram, send a message to your bot. **Stop your local long-polling
runner first** if it's still running (`Ctrl-C` in the terminal running
`telegram_local.py`) — both can't be active at the same time.

You should get a reply within a few seconds. If not, tail the logs:
```sh
sam logs --stack-name agentic-tasks-notion --name WebhookFunction --tail
```

### 7. Manually trigger the digest

Don't wait until tomorrow morning to find out if it works:

```sh
sam remote invoke DigestFunction --stack-name agentic-tasks-notion --region us-east-2
```

The morning digest should land in Telegram immediately.

---

## Operating it

- **Update code:** `sam build && sam deploy`.
- **Change non-secret config** (digest time, timezone, model): edit
  `template.yaml`, then `sam deploy`.
- **Rotate a secret:** update `.env`, re-run `scripts/seed_secrets.py`.
  Lambdas pick up the new value on next cold start.
- **Update Lambda dependencies:** `pyproject.toml` is the source of truth.
  After changing it, regenerate `src/requirements.txt`:
  ```sh
  uv export --format requirements-txt --no-emit-project --no-dev --no-hashes -o src/requirements.txt
  ```
  Then `sam build && sam deploy`.
- **Tear it all down:** `sam delete --stack-name agentic-tasks-notion --region us-east-2`.

---

## Use this for your own setup

This repo is public so others can fork and deploy their own instance. The
single-user guard (`ALLOWED_TELEGRAM_USER_ID`) means each fork only responds
to its owner — sharing the bot's username with someone else doesn't give
them access.
