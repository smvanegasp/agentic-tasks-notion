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

import json
import logging
import re
from datetime import date, datetime, timedelta
from time import perf_counter
from typing import Any

from openai import OpenAI

from agentic_tasks.agent.prompts import system_prompt
from agentic_tasks.agent.tools import TOOL_SCHEMAS, call_tool
from agentic_tasks.config import get_settings

log = logging.getLogger(__name__)

MAX_ITERATIONS = 8

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
) -> tuple[str, list[dict[str, Any]]]:
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

    messages.append({"role": "user", "content": user_message})

    agent_messages: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        api_start = perf_counter()
        response: Any = client.chat.completions.create(
            model=s.llm_model,
            messages=messages,
            tools=TOOL_SCHEMAS,  # type: ignore[arg-type]
        )
        log.info(
            "llm call %d: %.0fms",
            iteration + 1,
            (perf_counter() - api_start) * 1000,
        )
        msg: Any = response.choices[0].message

        if not msg.tool_calls:
            reply = msg.content or "(empty reply)"
            agent_messages.append({"role": "assistant", "content": reply})
            return reply, agent_messages

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
            log.info("tool call: %s %s", tc.function.name, args)
            result = call_tool(tc.function.name, args)
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
            messages.append(tool_msg)
            agent_messages.append(tool_msg)

    fallback = "(agent reached the iteration limit without producing a final reply)"
    agent_messages.append({"role": "assistant", "content": fallback})
    return fallback, agent_messages
