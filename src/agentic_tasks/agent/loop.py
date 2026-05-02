"""LLM tool-calling loop for the task-management agent.

Uses the OpenAI SDK pointed at any OpenAI-compatible endpoint (Groq by default
in this project, but OpenAI itself also works — see ``config.llm_base_url``).

Stateless on its own — the caller passes any prior conversation history.
The function appends the new user message to the model's messages internally,
runs the tool-call loop, and returns ``(reply_text, agent_messages, pending_plan)``.

If the model emits any write tool call (``create_tasks`` / ``update_tasks`` /
``complete_tasks`` / ``shift_due_dates``), the loop short-circuits the round,
returns a deterministic preview, and ``pending_plan`` carries the plan the
caller should persist for the confirmation gate. Reads in the same round are
dropped — the model can re-issue them next turn.
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
# echoed gibberish on the final natural-language reply. We detect that pattern
# (5+ consecutive ellipsis tokens) and retry the same call once.
_DEGENERATE_REPLY_RE = re.compile(r"(?:\.{3}|…)(?:\s*(?:\.{3}|…)){4,}")


def _looks_degenerate(text: str | None) -> bool:
    return bool(text and _DEGENERATE_REPLY_RE.search(text))


# Matches a bold day-subheader the model produces in multi-day replies, e.g.
# ``<b>Friday, May 1:</b>`` or ``<b>Sunday, May 4</b>``.
_DAY_SUBHEADER_RE = re.compile(r"<b>\s*[A-Za-z]+,\s*[A-Za-z]+\s+\d+:?\s*</b>")


def _check_guardrails(
    reply: str,
    last_query_args: dict[str, Any] | None,
    last_query_tasks: list[dict[str, Any]] | None,
) -> str | None:
    """Code-based quality checks on the model's final reply for read queries.

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

    ``pending_plan`` is non-None when the model emitted any write tool call.
    The loop short-circuits the whole round (reads in the same round are
    dropped — the model can re-issue them next turn), ``reply_text`` is a
    deterministic preview, and the caller persists the plan and waits for
    user confirmation before running :func:`agent.preview.execute_plan`.

    ``correction_note`` is an optional transient system message injected
    before the user message. The dispatcher passes one when the user has
    just rejected a pending plan with feedback so the model knows to
    re-evaluate rather than re-propose. It is not persisted to
    ``agent_messages``.
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

    for _iteration in range(max_iterations):
        api_start = perf_counter()
        response: Any = client.chat.completions.create(
            model=s.llm_model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        log.info(
            "llm call",
            extra={"duration_ms": int((perf_counter() - api_start) * 1000)},
        )
        msg: Any = response.choices[0].message

        if not msg.tool_calls:
            reply = msg.content or "(empty reply)"
            if _looks_degenerate(reply):
                log.warning("degenerate reply detected; retrying once")
                response = client.chat.completions.create(
                    model=s.llm_model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )
                msg = response.choices[0].message
                if not msg.tool_calls:
                    reply = msg.content or "(empty reply)"

            # If we still have a final-reply (no tool_calls), check the
            # code-based guardrails. If they fail and we haven't retried for
            # them yet, inject feedback and continue the loop so the model
            # produces a corrected reply.
            if not msg.tool_calls:
                if not guardrail_retry_done:
                    feedback = _check_guardrails(
                        reply, last_query_args, last_query_tasks
                    )
                    if feedback:
                        log.warning("guardrail violation; injecting feedback")
                        guardrail_retry_done = True
                        messages.append({"role": "assistant", "content": reply})
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
        if any(tc.function.name in WRITE_TOOL_NAMES for tc in msg.tool_calls):
            plan = serialize_plan(msg.tool_calls)
            preview = format_preview(plan)
            agent_messages.append({"role": "assistant", "content": preview})
            log.info("confirmation gate deferred", extra={"writes": len(plan)})
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
