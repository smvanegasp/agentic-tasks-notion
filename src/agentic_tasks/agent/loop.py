"""LLM tool-calling loop for the task-management agent.

Uses the OpenAI SDK pointed at any OpenAI-compatible endpoint (Groq by default
in this project, but OpenAI itself also works — see ``config.llm_base_url``).

Stateless on its own — the caller passes any prior conversation history.
The function appends the new user message to the model's messages internally,
runs the tool-call loop, and returns ``(reply_text, agent_messages)`` where
``agent_messages`` contains every assistant + tool message produced during
this run (so the caller can persist them to its conversation store). The user
message is the caller's responsibility to persist.
"""

from __future__ import annotations

import html
import json
import logging
import re
from datetime import date, datetime, timedelta
from time import perf_counter
from typing import Any

from openai import OpenAI

from agentic_tasks.agent.preview import (
    WRITE_TOOL_NAMES,
    format_preview,
    serialize_plan,
)
from agentic_tasks.agent.prompts import system_prompt
from agentic_tasks.agent.tools import TOOL_SCHEMAS, call_tool
from agentic_tasks.config import get_settings

log = logging.getLogger(__name__)

MAX_ITERATIONS = 25

# Cap completion tokens per LLM call. gpt-oss-120b is a reasoning model whose
# default output budget on Groq is too small for multi-day grouped replies —
# without this cap we observed responses cut off mid-heading. 4096 is well
# above any single Telegram message we'd plausibly send.
MAX_COMPLETION_TOKENS = 4096

# gpt-oss-120b on Groq occasionally degenerates into ellipsis repetition or
# echoed gibberish on the final natural-language reply — typically once per
# many turns. We detect that pattern (5+ consecutive ellipsis tokens, ASCII
# "..." or Unicode "…", optionally separated by whitespace) and retry the
# same call once. A short literal "…" inside normal prose is fine; only
# runs of them flag.
_DEGENERATE_REPLY_RE = re.compile(r"(?:\.{3}|…)(?:\s*(?:\.{3}|…)){4,}")


def _looks_degenerate(text: str | None) -> bool:
    return bool(text and _DEGENERATE_REPLY_RE.search(text))


# Matches a bold day-subheader the model produces in multi-day replies, e.g.
# ``<b>Friday, May 1:</b>`` or ``<b>Sunday, May 4</b>``.
_DAY_SUBHEADER_RE = re.compile(r"<b>\s*[A-Za-z]+,\s*[A-Za-z]+\s+\d+:?\s*</b>")


# Action-verb vocabulary used by :func:`_count_action_verbs` to detect
# multi-action user messages ("mark X done AND reschedule Y"). Imperfect — a
# list-of-objects pattern ("complete A, B, and C") only contains the verb
# once and won't trip the count. We tune the list to be a lower bound: when
# the count is >= 2, the user almost certainly wants multiple writes.
_ACTION_VERBS = frozenset(
    {
        # English
        "complete", "completed", "completes",
        "mark",
        "create", "created", "creates", "add", "added", "adds",
        "update", "updated", "updates", "change", "changed", "changes",
        "edit", "edited", "edits",
        "reschedule", "rescheduled", "reschedules",
        "push", "pushed", "pushes",
        "move", "moved", "moves",
        "shift", "shifted", "shifts",
        "snooze", "snoozed", "snoozes",
        "delete", "deleted", "deletes",
        "remove", "removed", "removes",
        "eliminate", "eliminated", "eliminates",
        "cancel", "cancelled", "canceled", "cancels",
        "schedule", "scheduled", "schedules",
        "set", "sets",
        "finish", "finished", "finishes",
        # Spanish
        "completa", "completar", "completado",
        "marca", "marcar",
        "crea", "crear", "agrega", "agregar",
        "actualiza", "actualizar", "cambia", "cambiar", "edita", "editar",
        "reagenda", "reagendar", "reprograma", "reprogramar",
        "mueve", "mover", "empuja", "empujar",
        "borra", "borrar", "elimina", "eliminar",
        "cancela", "cancelar",
        "programa", "programar",
        "establece", "establecer",
        "termina", "terminar",
    }
)

_WORD_RE = re.compile(r"\b[a-záéíóúñü]+\b")

# How many times we'll inject feedback and re-run the LLM on a multi-action
# turn before giving up and previewing whatever the model produced. Each
# retry adds one Groq round-trip; 2 keeps worst-case latency reasonable
# while giving the model a fair shot at re-emitting all the writes.
_MAX_MULTI_ACTION_RETRIES = 2


def _count_action_verbs(user_message: str) -> int:
    """Lower-bound estimate of how many distinct write actions the user is
    asking for. Used by the multi-action guardrail to detect when the model
    dropped one of N requested writes.

    Counts occurrences (not distinct verbs), so "complete X and complete Y"
    correctly returns 2 even though both are the same verb.
    """
    return sum(
        1 for w in _WORD_RE.findall(user_message.lower()) if w in _ACTION_VERBS
    )


def _matched_action_verbs(user_message: str) -> list[str]:
    """The action-verb tokens we found in the user's message, in order, with
    duplicates preserved. Used in the retry feedback so the model sees
    exactly which verbs we expect it to address."""
    return [
        w for w in _WORD_RE.findall(user_message.lower()) if w in _ACTION_VERBS
    ]


def _check_guardrails(
    reply: str,
    last_query_args: dict[str, Any] | None,
    last_query_tasks: list[dict[str, Any]] | None,
) -> str | None:
    """Code-based quality checks on the model's final reply.

    Returns a feedback string describing what's wrong (to be injected as a
    system message and used to drive a single retry round), or ``None`` if
    the reply passes. Only fires when the loop has data to compare against —
    i.e. there was at least one ``query_tasks`` call this turn.
    """
    if last_query_args is None:
        return None

    issues: list[str] = []

    # 1. Day-completeness: if the user asked for a date range, the reply
    #    should have one bold subheader per day in that range — even empty
    #    days, per the prompt rule.
    after = last_query_args.get("due_on_or_after")
    before = last_query_args.get("due_on_or_before")
    if isinstance(after, str) and isinstance(before, str):
        start: date | None
        end: date | None
        try:
            start = date.fromisoformat(after)
            end = date.fromisoformat(before)
        except ValueError:
            start = end = None
        if start and end and end > start:
            expected = (end - start).days + 1
            actual = len(_DAY_SUBHEADER_RE.findall(reply))
            if actual < expected:
                issues.append(
                    f"Your last reply showed only {actual} of the {expected}"
                    f" expected day subheaders. Include EVERY day from"
                    f" {after} through {before} (inclusive). Days with no"
                    f" tasks must still appear with their <b>Weekday, Mon"
                    f" Day:</b> subheader and the single line \"Nothing.\""
                    f" beneath it."
                )

    # 2. Task-name coverage: every task returned by the most recent
    #    query_tasks call must appear by name in the reply, otherwise the
    #    model dropped tasks. Match case-insensitively after unescaping HTML
    #    entities so &amp; / &lt; etc. are tolerated.
    if last_query_tasks:
        normalized_reply = html.unescape(reply).casefold()
        missing: list[str] = []
        for task in last_query_tasks:
            name = (task.get("name") or "").strip()
            if not name:
                continue
            if name.casefold() not in normalized_reply:
                missing.append(name)
        if missing:
            shown = ", ".join(f'"{n}"' for n in missing[:5])
            extra = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
            issues.append(
                f"Your last reply omitted these tasks{extra}: {shown}. List"
                f" every task returned by the query — do not skip any."
            )

    if not issues:
        return None
    return "Your previous reply has problems:\n- " + "\n- ".join(issues)

# Weekday name → Python weekday index (Mon=0..Sun=6). Lowercase keys.
# Includes Spanish forms with and without accents.
_WEEKDAY_NAMES: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "lunes": 0, "martes": 1,
    "miércoles": 2, "miercoles": 2,
    "jueves": 3, "viernes": 4,
    "sábado": 5, "sabado": 5,
    "domingo": 6,
}


def _extract_date_hints(user_message: str, today: date) -> list[str]:
    """Detect weekday and relative-date words in the user's message and return
    a pre-computed mapping. Defends against LLM date-arithmetic errors by
    handing the answer to the model rather than asking it to compute.
    """
    text = user_message.lower()
    hints: list[str] = []

    def _add(label: str, target: date) -> None:
        hints.append(
            f"  \"{label}\" -> {target.isoformat()} ({target.strftime('%A')})"
        )

    # Today / tonight (both English and Spanish, first match wins)
    for word in ("tonight", "esta noche", "today", "hoy"):
        if re.search(rf"\b{re.escape(word)}\b", text):
            _add(word, today)
            break

    for word in ("tomorrow", "mañana"):
        if re.search(rf"\b{re.escape(word)}\b", text):
            _add(word, today + timedelta(days=1))
            break

    for word in ("yesterday", "ayer"):
        if re.search(rf"\b{re.escape(word)}\b", text):
            _add(word, today - timedelta(days=1))
            break

    # Weekday names — emit each unique weekday (and "next <weekday>") only once.
    seen_weekdays: set[int] = set()
    for name, idx in _WEEKDAY_NAMES.items():
        if idx in seen_weekdays:
            continue
        if not re.search(rf"\b{re.escape(name)}\b", text):
            continue
        seen_weekdays.add(idx)
        days_ahead = (idx - today.weekday()) % 7
        upcoming = today + timedelta(days=days_ahead)
        _add(name, upcoming)
        if re.search(rf"\bnext\s+{re.escape(name)}\b", text):
            _add(f"next {name}", upcoming + timedelta(days=7))

    return hints


def _llm_client() -> OpenAI:
    s = get_settings()
    kwargs: dict[str, Any] = {"api_key": s.llm_api_key}
    if s.llm_base_url:
        kwargs["base_url"] = s.llm_base_url
    return OpenAI(**kwargs)


def run_agent(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    *,
    max_iterations: int = MAX_ITERATIONS,
    correction_note: str | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Run the tool-calling agent for a single user turn.

    Returns ``(reply_text, agent_messages, pending_plan)``.

    ``pending_plan`` is non-None when the model attempted any write tool
    (``create_task`` / ``update_task`` / ``complete_task`` /
    ``shift_due_dates``). In that case the loop short-circuits without
    executing any tool calls from the same round (reads in the same round
    are dropped — the model can re-issue them next turn) and ``reply_text``
    is a deterministic preview the caller should show to the user. The
    caller persists the pending plan and waits for confirmation before
    running :func:`agent.preview.execute_plan`.

    ``correction_note`` is an optional transient system message injected
    before the user message. The dispatcher passes one when the user has
    just rejected a pending plan with feedback (e.g., "no, the X task")
    so the model knows to re-evaluate rather than re-propose. It is not
    persisted to ``agent_messages``.
    """
    s = get_settings()
    client = _llm_client()

    messages: list[Any] = [{"role": "system", "content": system_prompt()}]
    if history:
        messages.extend(history)

    # Per-turn date hints, computed in Python (LLM-proof).
    today = datetime.now(s.timezone).date()
    hints = _extract_date_hints(user_message, today)
    if hints:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Date references detected in the user's current message."
                    " Use these exact dates — they are computed correctly:\n"
                    + "\n".join(hints)
                ),
            }
        )

    if correction_note:
        messages.append({"role": "system", "content": correction_note})

    messages.append({"role": "user", "content": user_message})

    agent_messages: list[dict[str, Any]] = []
    last_query_args: dict[str, Any] | None = None
    last_query_tasks: list[dict[str, Any]] | None = None
    guardrail_retry_done = False
    multi_action_retries = 0
    expected_action_count = _count_action_verbs(user_message)
    matched_verbs = _matched_action_verbs(user_message)

    for iteration in range(max_iterations):
        api_start = perf_counter()
        response: Any = client.chat.completions.create(
            model=s.llm_model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        log.info(
            "llm call",
            extra={
                "iteration": iteration + 1,
                "duration_ms": int((perf_counter() - api_start) * 1000),
            },
        )
        msg: Any = response.choices[0].message

        if not msg.tool_calls:
            reply = msg.content or "(empty reply)"
            if _looks_degenerate(reply):
                log.warning("degenerate reply detected; retrying once")
                api_start = perf_counter()
                response = client.chat.completions.create(
                    model=s.llm_model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )
                log.info(
                    "llm retry",
                    extra={
                        "duration_ms": int((perf_counter() - api_start) * 1000),
                        "reason": "degenerate",
                    },
                )
                msg = response.choices[0].message
                if not msg.tool_calls:
                    reply = msg.content or "(empty reply)"
                    if _looks_degenerate(reply):
                        log.warning("retry was also degenerate; returning anyway")

            # If we still have a final-reply (no tool_calls), check the
            # code-based guardrails. If they fail and we haven't retried for
            # them yet, inject feedback and continue the loop so the model
            # produces a corrected reply. Neither the broken reply nor the
            # feedback message are persisted to agent_messages — only the
            # final clean reply ends up in the user's conversation history.
            if not msg.tool_calls:
                if not guardrail_retry_done:
                    feedback = _check_guardrails(
                        reply, last_query_args, last_query_tasks
                    )
                    if feedback:
                        log.warning(
                            "guardrail violation; injecting feedback: %s",
                            feedback.replace("\n", " | ")[:300],
                        )
                        guardrail_retry_done = True
                        messages.append(
                            {"role": "assistant", "content": reply}
                        )
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    feedback
                                    + "\n\nNow output the corrected reply"
                                    " directly — do not call any tools."
                                ),
                            }
                        )
                        continue
                agent_messages.append({"role": "assistant", "content": reply})
                return reply, agent_messages, None

        # Confirmation gate: if any write tool is in this round, defer the
        # whole round. Replace the assistant tool-call message with a clean
        # preview so the persisted history stays internally consistent (no
        # orphan tool_calls without responses).
        write_present = any(
            tc.function.name in WRITE_TOOL_NAMES for tc in msg.tool_calls
        )
        if write_present:
            plan = serialize_plan(msg.tool_calls)

            # Multi-action guardrail: the user's message contained 2+ action
            # verbs but the model only proposed N<expected writes. Inject
            # the model's incomplete attempt + fake tool responses + a
            # feedback system message and run another LLM round so it can
            # re-emit ALL the writes together. Capped per turn — open-weight
            # models sometimes need two passes before catching the full set.
            if (
                expected_action_count >= 2
                and len(plan) < expected_action_count
                and multi_action_retries < _MAX_MULTI_ACTION_RETRIES
            ):
                multi_action_retries += 1
                log.warning(
                    "multi-action guardrail violation",
                    extra={
                        "expected": expected_action_count,
                        "actual": len(plan),
                        "retry": multi_action_retries,
                    },
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                # Fake tool responses are required by the OpenAI message
                # protocol — every emitted tool_call needs a paired tool
                # message before the next assistant turn. We don't run the
                # writes; the next assistant turn will re-emit the full set.
                deferred_response = json.dumps(
                    {
                        "deferred": (
                            "user requested multiple actions; emit ALL"
                            " writes together in your next response"
                        )
                    }
                )
                for tc in msg.tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": deferred_response,
                        }
                    )
                emitted_tools = ", ".join(tc.function.name for tc in msg.tool_calls)
                verbs_str = ", ".join(f"'{v}'" for v in matched_verbs)
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"MULTI-ACTION RECOVERY (retry"
                            f" {multi_action_retries}/{_MAX_MULTI_ACTION_RETRIES}):"
                            f" the user said exactly:\n"
                            f"  \"{user_message}\"\n"
                            f"That message contains {expected_action_count}"
                            f" action verbs ({verbs_str}), so you must emit at"
                            f" LEAST {expected_action_count} write tool calls"
                            f" in your next response. You only emitted"
                            f" {len(plan)} ({emitted_tools}).\n\n"
                            f"Re-read the user's message clause by clause."
                            f" Each clause separated by 'and' / 'y' / 'también'"
                            f" / a comma is a SEPARATE action. If you already"
                            f" called find_tasks and have the page_ids you"
                            f" need, emit ALL the writes now. If you still"
                            f" need to look up tasks, call find_tasks for"
                            f" every missing target in parallel — then in"
                            f" the round after that, emit every write"
                            f" together in a single response."
                        ),
                    }
                )
                continue

            preview = format_preview(plan)
            agent_messages.append({"role": "assistant", "content": preview})
            log.info(
                "confirmation gate deferred",
                extra={"writes": len(plan)},
            )
            return preview, agent_messages, plan

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        }
        messages.append(assistant_msg)
        agent_messages.append(assistant_msg)

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            log.info("tool call", extra={"tool": tc.function.name})
            result = call_tool(tc.function.name, args)
            # Track the most recent query_tasks call so guardrails can check
            # the model's final reply against the args (date range) and the
            # tool result (task list) it was given.
            if tc.function.name == "query_tasks":
                last_query_args = args if isinstance(args, dict) else None
                try:
                    parsed = json.loads(result)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    tasks = parsed.get("tasks")
                    last_query_tasks = (
                        list(tasks) if isinstance(tasks, list) else None
                    )
                else:
                    last_query_tasks = None
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                # Harmony (gpt-oss tokenizer used by Groq) requires `name` on
                # tool messages; OpenAI treats it as optional. Always include.
                "name": tc.function.name,
                "content": result,
            }
            messages.append(tool_msg)
            agent_messages.append(tool_msg)

    fallback = "Sorry, I needed too many steps to answer that. Could you simplify or break it up?"
    agent_messages.append({"role": "assistant", "content": fallback})
    return fallback, agent_messages, None
