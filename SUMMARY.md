# Project Summary

> Snapshot for picking up in a fresh Claude Code session without losing
> context. Pair this with `CLAUDE.md` (architecture + conventions) to be
> productive immediately.

## Current state (snapshot 2026-05-01, post-reliability-pass)

- **Deployed to AWS** in `us-east-2`. CloudFormation stack:
  `agentic-tasks-notion`.
- Two Lambdas live: webhook (Telegram → API Gateway → agent) and digest
  (EventBridge fires daily at 8:00 ET).
- Bot replies to **text and voice** messages on Telegram. Single-user via
  `ALLOWED_TELEGRAM_USER_ID`. **Replies always in English**, understands
  English + Spanish input.
- DynamoDB persists conversation history AND a per-chat `pending_plan`
  field for the confirmation gate. Day-boundary reset in user's TZ.
- **Confirmation gate enforced in code**: every write tool
  (`create_task` / `update_task` / `complete_task`) is intercepted by the
  agent loop, previewed deterministically, and only executed after the
  user replies y/yes/sí (with widened "no" matching: n, no, nope, nah,
  cancel, cancela, cancelar, stop, abort).
- **Code-based reply guardrails** retry once with feedback when the model
  drops days from a multi-day reply or omits tasks the tool returned.
- **Six tools**: `query_tasks`, `find_tasks` (fuzzy by partial name),
  `create_task`, `update_task`, `complete_task`, `list_projects`. Tool
  schemas auto-generated from Pydantic models (non-strict mode for Groq
  compatibility).
- **157 tests passing locally**; ruff + mypy clean.
- Public repo. Anyone can fork and deploy their own instance — secrets are
  user-supplied via `.env` → Secrets Manager (never committed).

## Build sequence (all done)

1. **Scaffold** — `pyproject.toml`, `.env.example`, `config.py`, schema
   constants for Notion property names, Status, Priority enums.
2. **Notion layer** — `notion_io/{client, tasks, projects, schema}.py`.
   Uses Notion's new `data_sources` endpoints (not the deprecated
   `databases.query`). Smoke script `scripts/notion_smoke.py` exercises
   the real DB read-only.
3. **Agent layer** — `agent/{loop, tools, prompts, transcribe}.py`.
   OpenAI SDK with tool calling against any OpenAI-compatible endpoint.
   Six tools (above). Voice transcription via Whisper. Local CLI REPL
   `scripts/agent_repl.py`.
4. **Telegram layer** — shared `telegram_io/dispatcher.py` (auth check,
   typing indicator, async-to-thread agent run, HTML reply with
   parse-mode fallback that strips tags). `handlers/webhook.py` for
   Lambda; `scripts/telegram_local.py` long-polls for local development.
5. **Digest layer** — `digest/render.py` (pure formatter, no LLM call) +
   `handlers/digest.py` (Lambda). Three sections: Overdue → Today → My
   Day, bold day-group subheaders, friendly date format, dedup so a task
   appears once.
6. **SAM template** — `template.yaml` defines both Lambdas, API Gateway
   HTTP API, EventBridge ScheduleV2 (timezone-aware), DynamoDB
   conversation table, Secrets Manager secret. Parameters: `Timezone`,
   `DigestHour`, `DigestMinute`, `LlmModel`, `LlmBaseUrl`,
   `TranscriptionModel`, `ConversationHistoryLimit`,
   `AgentTimeoutSeconds`.
7. **First deploy + webhook registration** — `sam deploy --guided` →
   `scripts/seed_secrets.py` → `scripts/set_webhook.py`.
8. **Pydantic-based tool schemas** — replaced hand-written JSON schemas
   with Pydantic `BaseModel` classes; tool schemas built non-strict from
   `model.model_json_schema()`. `call_tool` validates arguments via
   `model_validate` and dispatches to typed implementations.
9. **Code-enforced confirmation gate** — `agent/preview.py` module
   handles serialization, preview formatting, and execution of pending
   plans. The agent loop short-circuits whenever the model emits a write
   tool call; reads in the same round are dropped (model can re-issue
   them next turn). Dispatcher routes y/n/cancel/clarification before
   reaching the LLM. The user's persisted history only contains the
   final preview text and the post-execution summary — the underlying
   tool_calls aren't stored.
10. **Code-based reply guardrails** — `agent/loop.py:_check_guardrails`
    checks (a) day-completeness for range queries, (b) task-name
    coverage from the most recent `query_tasks` result. On failure,
    injects a system feedback message and runs one extra LLM round; the
    broken first reply is never persisted.
11. **find_tasks tool** — fuzzy lookup by partial name / acronym (uses
    rapidfuzz `partial_ratio`, score cutoff 60). Prompt instructs the
    model to call this BEFORE update/complete whenever the user
    references a task by partial name (avoids the "wrong page_id from
    history" failure mode).

## Key design choices (and why)

### LLM provider: Groq (OpenAI-compatible)
Settled on `openai/gpt-oss-120b` on Groq. Same OpenAI SDK, just
`LLM_BASE_URL=https://api.groq.com/openai/v1`. To switch back to OpenAI:
set `LLM_BASE_URL=` (empty), use OpenAI key, swap models to
`gpt-4o-mini` / `gpt-4o-mini-transcribe`.

### Tool schemas: Pydantic + non-strict
Schemas come from `model.model_json_schema()` wrapped manually — NOT from
`openai.pydantic_function_tool()`. Reason: the SDK helper sets
`strict: true`, which forces the model to emit every property in JSON
(including optionals as `null`). Groq's gpt-oss-120b occasionally omits a
single field, and strict mode turns that into a hard 400
`tool_use_failed` error. Non-strict schemas let the model emit only the
fields it cares about; Pydantic in `call_tool` still validates and
applies defaults, so we keep the safety without the brittleness.

### Confirmation gate: code-enforced, not prompt-enforced
Every write tool is intercepted by `agent/loop.py` — it does NOT execute
in the same turn the model emits it. The flow:

1. Model emits a write tool call → loop builds a preview via
   `preview.format_preview` (resolves task names for update/complete via
   one extra `notion_io.tasks.get_task` call), serializes the plan, and
   returns `(preview_text, agent_messages, pending_plan)`.
2. Dispatcher persists the plan in DynamoDB (`pending_plan` field).
3. On the next user message, dispatcher routes BEFORE the LLM:
   - `is_confirmation` (y/yes/yeah/sí/ok/sure/👍) → call
     `preview.execute_plan`, clear pending state, send deterministic
     summary with `<a href="...">open</a>` Notion links.
   - `is_rejection` (n/no/nope/cancel/cancela/cancelar/stop/abort) →
     clear pending plan, reply "What would you like to change?".
   - Anything else → clear pending and run agent normally with a
     transient `correction_note` system message ("user rejected your
     previous proposal — re-evaluate, do not re-propose; if they used a
     partial name, call find_tasks").

The confirmation gate also fixed the iteration-limit issue — batch
creations exit at round 1 with the preview, then `execute_plan`
sequentially calls each write directly (no LLM loop).

### Reply guardrails: deterministic checks, capped retry
Pure code (no LLM judge). Two checks fire on the model's final reply:

1. **Day-completeness**: if the most recent `query_tasks` had a date
   range, count `<b>Weekday, Mon Day:</b>` subheaders in the reply.
   Below expected → retry with feedback.
2. **Task-name coverage**: every task name from the tool result must
   appear in the reply (case-insensitive, HTML-entity-tolerant).
   Missing tasks → retry with feedback.

On failure, the loop injects a system message describing what's wrong
and runs one more LLM round. Capped at one retry per turn. Neither the
broken first reply nor the synthetic feedback is persisted to
`agent_messages` — only the corrected reply ends up in user-visible
history.

### Degeneracy retry: ellipsis spam detection
gpt-oss-120b on Groq occasionally produces final replies that degenerate
into runs of `...` / `…`. We detect 5+ consecutive ellipsis tokens with
a regex and silently retry the same LLM call once.

### Conversation history: per-chat, day-bounded, dual-backend
- 40 messages max (env-configurable). Resets at midnight in user's TZ.
- `ConversationStore` Protocol with `InMemoryConversationStore` (local)
  and `DynamoDBConversationStore` (Lambda). `get_store()` factory picks
  the backend based on `CONVERSATION_TABLE_NAME` env.
- Both backends now also support `get/set/clear_pending_plan` for the
  confirmation gate. DynamoDB stores `pending_plan` alongside
  `messages` on the same item.

### Date handling: triple-defense
LLMs (especially smaller ones) regularly miscount days. Three layers:
1. **Pre-computed 14-day date table in system prompt** — model looks up
   "Saturday → 2026-05-02" instead of computing.
2. **Per-turn regex hint injection** in `agent/loop.py` —
   `_extract_date_hints()` detects weekday names + today/tomorrow/
   yesterday in EN and ES, computes the date in Python, injects as a
   system message.
3. **Auto-tz on naive datetimes** in `agent/tools._parse_iso_date` — if
   the LLM forgets the offset, we attach the configured TZ. Never UTC.

### Reply formatting: Telegram HTML, sparing bold, splittable
Allowed tags only: `<b>`, `<i>`, `<u>`, `<s>`, `<code>`,
`<a href="...">`. The dispatcher's `_send`:
- Splits messages over 4000 chars at paragraph boundaries (Telegram's
  hard cap is 4096).
- On `BadRequest` parse-mode failure, strips HTML tags and resends as
  plain text so the user never sees raw `<b>` markers.

Multi-day queries always include EVERY day in the requested range — empty
days appear with their bold subheader and `Nothing.` underneath.

### Notion API: data_sources, not databases
The user's DB IDs are *database* IDs but schema and rows live on a *data
source*. Reads/writes via `client.data_sources` (query, retrieve). Page
creation uses `parent={"data_source_id": ...}`. Mapping cached in
`client.py`. **Do not revert to `databases.query`** — that endpoint is
being phased out.
**`Priority` in this DB is type `status` (not `select`)** — both Status
and Priority are written as `{"status": {"name": "<value>"}}`.

### find_tasks: fuzzy lookup before write
The model is unreliable at picking the right `page_id` from earlier
conversation history when the user references a task by partial name
("APD task", "the doctor appointment"). `find_tasks(name_query)`
queries Notion, fuzzy-matches with rapidfuzz `partial_ratio`, returns
top 5 matches. The prompt instructs: call `find_tasks` BEFORE
update/complete whenever the user used a partial name; if multiple
match, ask which one.

### Performance defenses
- `list_projects` `@lru_cache(maxsize=1)` — restart the bot to refresh
  after adding new Notion projects.
- Schemas fetched once per process.
- `max_completion_tokens=4096` on every Groq call (gpt-oss-120b's
  default reasoning budget caused multi-day replies to truncate
  mid-heading).
- 60-second agent timeout with polite "Sorry for the delayed response,
  try again." Configurable via `AGENT_TIMEOUT_SECONDS`.

## Bugs we hit and fixed (don't regress)

| Symptom | Root cause | Fix location |
|---|---|---|
| `ZoneInfoNotFoundError` on Windows | Python's `zoneinfo` needs `tzdata` on Windows / minimal Linux containers | `pyproject.toml` (`tzdata` runtime dep) |
| `KeyError: 'properties'` on Notion retrieve | Notion split databases into databases + data sources | `notion_io/client.py` (data_sources endpoints) |
| Saturday tasks scheduled for Friday | Smaller LLMs do bad date arithmetic | Date table in prompt + `_extract_date_hints()` + bigger model |
| Times appearing 4–5 hours off in Notion | Naive datetimes treated as UTC | `agent/tools._parse_iso_date` attaches configured TZ |
| Spanish accents printed as `?` in console | Windows cp1252 default | `sys.stdout.reconfigure(encoding="utf-8")` in scripts |
| Tests blew up across new env vars | `lru_cache` on `get_settings()` retained stale | `tests/conftest.py` clears caches |
| `HarmonyError: Tools should have a name!` on every turn after a long batch | Tool messages were missing `name` field — Harmony (gpt-oss tokenizer) requires it; OpenAI treats it as optional. Persisted broken history blocked all subsequent turns. | `agent/loop.py:tool_msg` always includes `name` |
| Iteration limit hit on 13-task batch | `MAX_ITERATIONS=8` + model serializing tool calls | Bumped to 25 + prompt nudge for parallel calls. Confirmation gate now also makes this near-impossible (writes exit at round 1). |
| Empty `list_projects` schema rejected by Groq strict mode | Pydantic emits `required: []` for empty models, Groq rejects | Drop strict + remove empty `required` for zero-arg tools |
| `tool_use_failed: missing properties: 'due_on'` | Strict mode required model to emit every optional as null; gpt-oss-120b sometimes omits one | Switched to non-strict tool schemas (`model.model_json_schema()` + manual wrap) |
| Replies showing literal `\n` characters | Prompt examples used `\\n` (literal backslash-n) instead of real newlines | Rewrite examples with real newlines, add explicit "never write backslash-n" rule |
| Multi-day replies truncating mid-heading | Groq's default completion-token budget is too small for reasoning model | `max_completion_tokens=4096` on every call |
| Reply degenerating into ellipsis spam | gpt-oss-120b occasionally produces runs of `…` instead of valid output | `_looks_degenerate` regex + auto-retry once |
| Multi-day reply skipping days with no tasks | Old prompt said "skip empty days" | Prompt rewritten: include EVERY day, write `Nothing.` for empty ones |
| Multi-day reply dropping days even when prompt says include all | Model occasionally still skips | Code guardrail: count subheaders vs expected days, retry with feedback |
| Reschedule picking the wrong task | Model guessed page_id from conversation context, picked wrong | New `find_tasks` tool + prompt rule to use it for partial-name references |
| "Cancel" treated as clarification rather than rejection | Rejection regex didn't include "cancel" | Widened regex to include cancel/cancela/cancelar/stop/abort |
| User correcting after wrong proposal: model re-proposes same wrong thing | No signal to model that prior proposal was rejected | Dispatcher injects transient `correction_note` system message when "anything else" path fires after a pending plan |

## Reliability layers (in order they fire on a single turn)

1. **Pre-LLM routing** in dispatcher: `/reset`, pending-plan
   confirmation/rejection.
2. **Date hints** injected before user message (transient).
3. **Correction note** injected if pending plan was just rejected
   (transient).
4. **Loop iteration**: LLM call → tool calls → tool results, repeat.
5. **Confirmation gate**: any write tool call exits with a preview.
6. **Degeneracy retry**: ellipsis-spam reply → retry same call once.
7. **Reply guardrails**: incomplete days or missing tasks → inject
   feedback, run one more LLM round (capped).
8. **Send-side**: split >4000 chars, strip HTML on parse failure.

## Performance baselines

5-query read-only benchmark (`scripts/benchmark.py`) — pre-confirmation
gate, but qualitatively unchanged for read-only queries:

| Config | Total time | Per-query avg | LLM share |
|---|---|---|---|
| gpt-4o (OpenAI) | 94.5s | ~19s | 98% |
| gpt-4o-mini (OpenAI) | 17.0s | 3.4s | 89% |
| gpt-oss-20b (Groq) | 8.7s | 1.7s | 66% |
| **gpt-oss-120b (Groq) — current** | **11.0s** | **2.2s** | **80%** |

Notion API calls average ~500ms; remaining time is LLM round-trips.
Write turns now have an extra round trip for confirmation
(preview → user → execute) but `execute_plan` itself is direct Notion
calls, no LLM.

## Expected AWS cost

~**$0.40–$0.50/month** for personal use:
- Lambda / API Gateway / DynamoDB / EventBridge / Logs / S3 — within
  free tier.
- Secrets Manager — `$0.40/month per secret` (only meaningful line item).

## How to run things

```sh
# Local dev
uv sync --group dev
uv run pytest                                    # 157 tests
uv run python scripts/notion_smoke.py            # check Notion creds
uv run python scripts/agent_repl.py              # interactive agent CLI
uv run python scripts/telegram_local.py          # bot via long-polling
uv run python scripts/digest_local.py --dry-run  # preview daily digest
uv run python scripts/benchmark.py               # perf measurement

# Deploy
sam build && sam deploy
sam list stack-outputs --stack-name agentic-tasks-notion --region us-east-2
sam logs --stack-name agentic-tasks-notion --name WebhookFunction --tail
sam remote invoke DigestFunction --stack-name agentic-tasks-notion \
  --region us-east-2

# Rotate a secret
# (edit .env first)
uv run python scripts/seed_secrets.py --secret-name agentic-tasks-notion-secrets

# Clear stuck DynamoDB conversation (rare; usually `/reset` from Telegram suffices)
# Use a key file because PowerShell mangles inline JSON quoting:
echo '{"chat_id": {"N": "<your-chat-id>"}}' > .tmp-key.json
aws dynamodb delete-item --region us-east-2 \
  --table-name agentic-tasks-notion-conversation \
  --key file://.tmp-key.json

# Tear down (preserves nothing)
sam delete --stack-name agentic-tasks-notion --region us-east-2
```

## Open ideas / future work

Roughly in order of value:

1. **Streaming LLM responses** — show tokens as they generate (perceived
   speedup, no total-time savings). Telegram's `editMessageText`
   supports it.
2. **Tighter IAM** — currently using `AdministratorAccess` for deploys;
   could write a least-privilege policy.
3. **`reasoning_effort` tuning** — gpt-oss-120b accepts low/medium/high.
   Default is medium. For our use case (lookup + format), "low" might
   reduce latency further. Worth A/B testing.
4. **Notion API retry with backoff** — currently a transient 5xx kills
   the turn. Wrap `client.data_sources.*` and `client.pages.*` with a
   simple retry.
5. **LLM-as-judge for subjective quality** — only if we observe
   subjective failures (tone, hallucination of dates not in tool
   results). Not preemptive — the code-based guardrails handle the
   mechanical issues.
6. **Recurring task creation** — out of scope; Notion's template
   handles recurrence today.

Done since last summary:
- ✅ `/reset` command
- ✅ Confirmation gate
- ✅ English-only replies
- ✅ Show URLs as clickable "open" links
- ✅ Smarter iteration-limit fallback (now mostly moot)

## Files worth knowing (orientation)

- `CLAUDE.md` — canonical architecture, conventions, build sequence;
  documents the confirmation gate, Pydantic tool layer, Harmony
  compatibility note.
- `README.md` — public-facing local-setup + AWS deploy guide.
- `template.yaml` — SAM/CloudFormation infra.
- `samconfig.toml` — SAM CLI defaults (us-east-2).
- `src/agentic_tasks/agent/prompts.py` — system prompt (date table,
  formatting rules with REAL newlines, behavioral rules including
  "use find_tasks before update/complete for partial names").
- `src/agentic_tasks/agent/tools.py` — Pydantic tool models, non-strict
  schema builder, six tool implementations, `call_tool` dispatcher.
- `src/agentic_tasks/agent/loop.py` — tool-calling loop, write-tool
  short-circuit, degeneracy retry, code-based guardrails with feedback
  retry, transient correction-note support, `max_completion_tokens`.
- `src/agentic_tasks/agent/preview.py` — confirmation-gate module:
  serialize_plan, format_preview (resolves task names via `get_task`),
  is_confirmation/is_rejection regexes, execute_plan with
  HTML-formatted summary including `<a href>open</a>` links.
- `src/agentic_tasks/notion_io/tasks.py` — `Task` dataclass + CRUD;
  `get_task(page_id)` for preview name resolution.
- `src/agentic_tasks/notion_io/projects.py` — `find_project_by_name`
  (rapidfuzz, lru_cache).
- `src/agentic_tasks/conversation/store.py` — `ConversationStore`
  Protocol, `InMemoryConversationStore` and
  `DynamoDBConversationStore`, plus `get/set/clear_pending_plan` for
  the gate.
- `src/agentic_tasks/telegram_io/dispatcher.py` — `/reset` route,
  pending-plan routing, `_split_message` + `_strip_html`,
  correction-note pass-through.
- `src/agentic_tasks/digest/render.py` — pure morning-digest formatter.
- `scripts/benchmark.py` — perf measurement.

## Test layout

| File | What it covers |
|---|---|
| `tests/test_agent_tools.py` | All six tool implementations, including new find_tasks (acronym match, threshold, default exclusions, empty query) |
| `tests/test_agent_loop.py` | Tool-calling loop, date hints, write-tool short-circuit, degeneracy retry, guardrail behavior (day-completeness, task coverage, retry cap, correction note plumbing) |
| `tests/test_preview.py` | Confirmation gate: serialize/format/execute, rejection regex (now includes cancel/cancelar/etc.), date formatter |
| `tests/test_dispatcher_confirmation.py` | `/reset`, y/n/cancel routing, clarification path injecting correction_note |
| `tests/test_dispatcher_send.py` | Message splitter at paragraph boundaries, HTML tag stripping, multi-chunk send |
| `tests/test_telegram_dispatcher.py` | Auth, voice transcription, plain-text fallback, agent error handling |
| `tests/test_conversation_store.py` + `_dynamodb.py` | Both backends including pending_plan round-trip and day-boundary clearing |
| `tests/test_digest_render.py` + `_handler.py` | Daily digest formatter and Lambda handler |
| `tests/test_notion_tasks.py` + `_projects.py` | Notion CRUD layer (mocked) |
| `tests/test_webhook_handler.py`, `tests/test_transcribe.py` | Lambda entry, Whisper integration |

## Known sharp edges for a reviewer

1. **Strict-mode tool schemas were intentionally dropped** for Groq
   compatibility. If you swap back to OpenAI, strict mode is fine and
   would catch more validation issues server-side. See
   `_build_tool_schema` in `agent/tools.py`.
2. **Confirmation gate drops reads in mixed read+write rounds** — if
   the model does `query_tasks` AND `create_task` in one response, both
   are deferred (gate exits with the preview). The model must re-issue
   any reads on the next turn. This is rare in practice because the
   prompt encourages parallel writes only.
3. **Guardrail retry is capped at 1** per turn. If both attempts are
   incomplete, we accept the second. Logs `guardrail violation` so it's
   visible in CloudWatch.
4. **Correction-note transience**: it goes into the in-memory `messages`
   list (sent to the LLM) but NOT into `agent_messages` (persisted). If
   you change how those two lists are kept in sync, double-check this.
5. **`pending_plan` survives `append`** in DynamoDB — the
   `DynamoDBConversationStore.append` reads the existing item, mutates
   `messages`, and re-writes preserving `pending_plan`. Test:
   `test_append_preserves_pending_plan`.
6. **Day-boundary reset clears pending_plan too** — both backends. If
   you reproduce this in another store backend, mirror that behavior.
7. **`find_tasks` cutoff is 60** (rapidfuzz partial_ratio). Lower is
   permissive (more false-positive matches), higher is strict. 60 was
   chosen so 3-letter acronyms ("APD") still match; tune if you see
   too many wrong matches surfacing for the user to disambiguate.

## Memory (auto-loaded across sessions)

Saved at `~/.claude/projects/C--Users-smvan-repos-agentic-tasks-notion/memory/`:
- User profile (HBS MBA, public portfolio orientation, dictation is
  common in messages — interpret transcription artifacts charitably).
- Python tooling depth (familiar with `uv venv` + `uv pip`; learning
  project-style `uv sync`).
- Python gotchas (`tzdata` is a runtime dep; notion-client v3 has loose
  type stubs — annotate `Any` at the boundary, ignore `attr-defined`).
- Notion data_sources model + `Priority` is `status`-typed.
- Naive datetimes must get TZ attached before Notion calls.

These load automatically — no need to re-derive.
