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

# gpt-oss-120b on Groq occasionally produces broken final replies. We detect
# them and retry the same call once. Three signal classes:
#
# 1. Consecutive ellipsis runs (the original failure mode).
# 2. Scattered ellipses (3+ total in one reply, possibly interleaved with
#    other punctuation like "?"). One real reply has at most one ellipsis;
#    three is anomalous.
# 3. Self-coaching / analysis-channel leakage. gpt-oss models have a hidden
#    reasoning channel that Groq normally strips; when stripping fails, the
#    final reply contains phrases the assistant has no legitimate reason to
#    write to a user ("the previous answer appears garbled", "provide bullet
#    list", "use 'Tomorrow:' heading", "must not include ellipsis"). These
#    are how the model talks to itself, not to the user.
_DEGENERATE_REPLY_RE = re.compile(r"(?:\.{3}|…)(?:\s*(?:\.{3}|…)){4,}")
_ELLIPSIS_TOKEN_RE = re.compile(r"\.{3}|…")
_THINKING_LEAK_RE = re.compile(
    r"(?:"
    r"the previous (?:answer|reply|response)"
    r"|previous (?:answer|reply|response) (?:appears|was|got|seems)"
    r"|appears\s+(?:garbled|truncated|incomplete|broken)"
    r"|got\s+(?:truncated|cut\s+off|garbled)"
    r"|need\s+to\s+(?:correctly|just|simply|properly)\s+(?:format|provide|show|reply|output|use)"
    r"|provide\s+(?:a\s+|the\s+)?bullet\s+list"
    r"|use\s+\"[A-Za-z][A-Za-z\s]{0,20}:?\"\s+(?:as\s+)?(?:the\s+)?heading"
    r"|must\s+not\s+include\s+(?:ellipsis|dots|extra|backslash|escape)"
    r")",
    re.IGNORECASE,
)


def _looks_degenerate(text: str | None) -> bool:
    if not text:
        return False
    if _DEGENERATE_REPLY_RE.search(text):
        return True
    if _THINKING_LEAK_RE.search(text):
        return True
    if len(_ELLIPSIS_TOKEN_RE.findall(text)) >= 3:
        return True
    return False


# The model occasionally hallucinates the system's confirmation preview in
# plain chat instead of emitting the write tool call — the user then sees
# the same plan twice (once fake, once real after they reply Y). This
# pattern catches the unique footer phrase the system uses; the model has
# no legitimate reason to write it.
_GHOST_PREVIEW_RE = re.compile(
    r"reply\s+(?:<b>)?\s*y\s*(?:</b>)?\s+to\s+proceed",
    re.IGNORECASE,
)


def _looks_like_ghost_preview(text: str | None) -> bool:
    return bool(text and _GHOST_PREVIEW_RE.search(text))


# Last-line-of-defense format validator. A small model judges whether the
# main model's reply is clean Telegram-HTML output or broken/garbled. Cheap
# heuristics (degenerate regex, ghost-preview, day-completeness guardrails)
# catch known patterns; this catches anything novel that still looks wrong.
# Output is binary "OK" / "BAD: <reason>" so parsing is trivial and the
# small model has minimal room to drift.
_FORMAT_CHECK_SYSTEM = (
    "You are a strict format validator for a Telegram task-assistant. You"
    " receive an assistant reply and must judge whether it is clean output"
    " ready to send to the user.\n\n"
    "Output EXACTLY one line, beginning with one of:\n"
    "- \"OK\" — clean reply, ready to send.\n"
    "- \"BAD: <one short reason>\" — broken; must be rewritten.\n\n"
    "Mark a reply BAD if any of these are present:\n"
    "- Ellipsis tokens (\"...\" or \"…\") used as truncation, mid-word"
    " breaks, or appearing 3+ times in one reply.\n"
    "- Self-coaching / meta language: \"the previous reply\", \"needs to"
    " be\", \"should provide\", \"must use\", \"let me try\", \"the format"
    " should\", \"correctly format\", or any text describing how the reply"
    " ought to look.\n"
    "- Repeated \"?\" or \"!\" tokens not part of normal punctuation.\n"
    "- Garbled, partial, or non-prose noise; truncated bullets.\n"
    "- Malformed Telegram HTML (unclosed or unknown tags).\n"
    "- Mixed-language gibberish or untranslated scratch-pad text.\n\n"
    "Mark a reply OK if it is natural Telegram-HTML output: bullets"
    " prefixed \"•\", optional <b>day:</b> subheaders, complete prose."
    " Single-line replies like \"Nothing.\" or \"Nothing for today.\" are"
    " OK. Headings like \"Today:\" / \"Tomorrow:\" on their own line are"
    " OK. Asking the user a clarifying question is OK.\n\n"
    "Reply with nothing else. Just \"OK\" or \"BAD: <reason>\"."
)


def _check_format(
    client: OpenAI,
    model: str,
    user_message: str,
    reply: str,
) -> str | None:
    """Binary format validation by a small fast model.

    Returns ``None`` if the reply is judged OK or if the validator itself
    errors out (fail open — never block a legitimate reply on validator
    flake). Returns a short feedback string when the reply is judged BAD,
    so the caller can inject it as a system message and retry the main
    model once.
    """
    try:
        check_start = perf_counter()
        response: Any = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FORMAT_CHECK_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"User question:\n{user_message}\n\n"
                        f"Assistant reply:\n{reply}\n\n"
                        "Verdict:"
                    ),
                },
            ],
            max_completion_tokens=64,
            temperature=0,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("format check call failed: %s", e)
        return None

    log.info(
        "format check",
        extra={"duration_ms": int((perf_counter() - check_start) * 1000)},
    )

    verdict = (response.choices[0].message.content or "").strip()
    upper = verdict.upper()
    if upper.startswith("OK"):
        return None
    if upper.startswith("BAD"):
        reason = verdict.split(":", 1)[1].strip() if ":" in verdict else ""
        return reason or "format judged BAD by validator"
    log.warning("format check returned unparseable verdict: %r", verdict[:80])
    return None


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
    ghost_preview_retry_done = False
    format_check_retry_done = False

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
                # Ghost-preview defense: the model wrote a fake preview in
                # chat instead of emitting the actual write tool. Feed back
                # and force a retry so the user only ever sees the real
                # system-generated preview.
                if (
                    not ghost_preview_retry_done
                    and _looks_like_ghost_preview(reply)
                ):
                    log.warning("ghost preview detected; injecting feedback")
                    ghost_preview_retry_done = True
                    messages.append({"role": "assistant", "content": reply})
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Your previous reply imitated the system's"
                                " confirmation preview in chat (it contained"
                                " 'Reply y to proceed'). The system shows"
                                " that preview AUTOMATICALLY when you emit a"
                                " write tool call. Do not write previews"
                                " yourself. Emit the actual write tool call"
                                " (create_tasks / update_tasks /"
                                " complete_tasks / shift_due_dates) now,"
                                " with no chat text alongside it."
                            ),
                        }
                    )
                    continue

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

                # Final line of defense: ask a small model to validate the
                # reply's format. Catches novel garbled/leaked-thinking
                # shapes the cheap regex checks above missed. Skipped if
                # the format-check model is unset (env opt-out) or the
                # reply is the placeholder for an empty model response.
                if (
                    not format_check_retry_done
                    and s.llm_format_check_model
                    and reply != "(empty reply)"
                ):
                    feedback = _check_format(
                        client, s.llm_format_check_model, user_message, reply
                    )
                    if feedback:
                        log.warning(
                            "format check rejected reply; injecting feedback",
                            extra={"reason": feedback[:80]},
                        )
                        format_check_retry_done = True
                        messages.append({"role": "assistant", "content": reply})
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "A format validator rejected your"
                                    f" previous reply: {feedback}. Re-output"
                                    " the answer in clean Telegram-HTML"
                                    " format only — no ellipsis, no"
                                    " meta-commentary, no scratch-pad text,"
                                    " no descriptions of the format itself."
                                    " Just produce the corrected reply"
                                    " directly. Do not call any tools."
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
