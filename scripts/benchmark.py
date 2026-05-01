"""Where does the agent spend its time? LLM vs Notion.

Runs a handful of read-only queries through the real agent loop, captures the
existing timing log lines from ``agent.loop`` and ``agent.tools`` via a custom
log handler, and prints a per-query and aggregate breakdown.

    uv run python scripts/benchmark.py

Costs a few cents in LLM provider usage. Hits your real Notion DB read-only —
no tasks are created or modified.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Edit this list to test other queries.
QUERIES = [
    "What's on my plate today?",
    "What do I have tomorrow?",
    "What's on my plate this week?",
    "What projects do I have?",
    "Show me overdue tasks",
]


@dataclass
class TurnStats:
    query: str
    total_ms: float = 0.0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    tool_times: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )


class _CapturingHandler(logging.Handler):
    """Parses the timing log lines emitted by the agent and tools modules."""

    _LLM = re.compile(r"llm call \d+: ([\d.]+)ms")
    _TOOL = re.compile(r"tool (\S+) done in ([\d.]+)ms")

    def __init__(self) -> None:
        super().__init__()
        self.current: TurnStats | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if self.current is None:
            return
        msg = record.getMessage()
        if m := self._LLM.search(msg):
            self.current.llm_ms += float(m.group(1))
            self.current.llm_calls += 1
        elif m := self._TOOL.search(msg):
            name = m.group(1)
            elapsed = float(m.group(2))
            self.current.tool_ms += elapsed
            self.current.tool_calls += 1
            self.current.tool_times[name].append(elapsed)


def _stub_telegram_env() -> None:
    for k, v in {
        "TELEGRAM_BOT_TOKEN": "bench",
        "ALLOWED_TELEGRAM_USER_ID": "0",
    }.items():
        if not os.environ.get(k):
            os.environ[k] = v


def _print_row(query: str, stats: TurnStats) -> None:
    other = max(stats.total_ms - stats.llm_ms - stats.tool_ms, 0)
    print(
        f"  {query[:43]:<43} "
        f"{stats.total_ms:>7.0f}ms  "
        f"{stats.llm_ms:>7.0f}ms ({stats.llm_calls})  "
        f"{stats.tool_ms:>7.0f}ms ({stats.tool_calls})  "
        f"{other:>7.0f}ms"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    load_dotenv()
    _stub_telegram_env()

    # Capture only agentic_tasks logs; suppress other noise.
    handler = _CapturingHandler()
    agent_logger = logging.getLogger("agentic_tasks")
    agent_logger.setLevel(logging.INFO)
    agent_logger.addHandler(handler)
    agent_logger.propagate = False

    from agentic_tasks.agent.loop import run_agent
    from agentic_tasks.config import get_settings
    from agentic_tasks.notion_io.client import get_projects_db_schema, get_tasks_db_schema
    from agentic_tasks.notion_io.projects import list_projects

    settings = get_settings()
    print(f"LLM model:           {settings.llm_model}")
    print(f"LLM base URL:        {settings.llm_base_url or '(OpenAI default)'}")
    print(f"Transcription model: {settings.transcription_model}")
    print(f"History limit:       {settings.conversation_history_limit}")
    print()
    print("Warming caches (schemas + projects list)...")
    get_tasks_db_schema()
    get_projects_db_schema()
    list_projects()
    print()

    print("Running queries:")
    results: list[TurnStats] = []
    for query in QUERIES:
        handler.current = TurnStats(query=query)
        start = time.perf_counter()
        try:
            run_agent(query)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR  {query!r}: {type(e).__name__}: {e}")
            handler.current = None
            continue
        handler.current.total_ms = (time.perf_counter() - start) * 1000
        results.append(handler.current)
        print(f"  done   {query!r}")
        handler.current = None
    print()

    print("=" * 100)
    header = (
        f"  {'Query':<43} "
        f"{'Total':>9}  "
        f"{'LLM (n)':>13}  "
        f"{'Tools (n)':>12}  "
        f"{'Other':>9}"
    )
    print(header)
    print("-" * 100)
    total_t = total_l = total_n = 0.0
    for r in results:
        _print_row(r.query, r)
        total_t += r.total_ms
        total_l += r.llm_ms
        total_n += r.tool_ms
    print("-" * 100)
    other = max(total_t - total_l - total_n, 0)
    print(
        f"  {'TOTAL':<43} "
        f"{total_t:>7.0f}ms  "
        f"{total_l:>7.0f}ms        "
        f"{total_n:>7.0f}ms       "
        f"{other:>7.0f}ms"
    )

    if total_t:
        print()
        print(f"  LLM:      {total_l / total_t:>5.1%} of total")
        print(f"  Notion:   {total_n / total_t:>5.1%} of total")
        print(
            f"  Other:    {other / total_t:>5.1%} of total"
            " (Python overhead, network setup, etc.)"
        )

    # Per-tool breakdown across all queries
    per_tool: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for name, times in r.tool_times.items():
            per_tool[name].extend(times)

    if per_tool:
        print()
        print("Tool breakdown (across all queries):")
        for name in sorted(per_tool, key=lambda n: -sum(per_tool[n])):
            times = per_tool[name]
            print(
                f"  {name:<20} "
                f"{sum(times):>6.0f}ms total, "
                f"{len(times):>2} call(s), "
                f"{sum(times) / len(times):>5.0f}ms avg"
            )


if __name__ == "__main__":
    main()
