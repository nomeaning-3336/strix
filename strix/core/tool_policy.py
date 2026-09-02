"""Tool execution policy for wide turns (Efficiency v1).

Which tools may run concurrently inside one model turn, which are pure enough
to re-run/cache, and how everything else is kept serial:

  - ``parallel_safe``   the tool has no shared mutable state, so issuing
                        several of its calls (or mixing it with other safe
                        calls) in one reasoning turn is fine.
  - ``side_effect_free`` the call changes nothing (read-only).
  - ``cacheable``       the result is a pure function of its arguments against
                        immutable inputs (safe to replay from a result cache).
  - ``serial_group``    tools sharing a group must never overlap each other
                        (e.g. the sandbox shell pair). Unknown tools default to
                        serialized under their own name.

Rule of thumb: reads are parallel, everything that mutates shared state —
shell, file writes, exploit/mutating HTTP, report creation/revision, agent
lifecycle — stays serial. The executor serializes every non-``parallel_safe``
tool by acquiring a per-(event-loop, group) lock around its invocation, so
enabling provider-level ``parallel_tool_calls`` cannot interleave two mutating
calls, even when the model emits several in one turn.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    parallel_safe: bool = False
    side_effect_free: bool = False
    cacheable: bool = False
    serial_group: str | None = None


DEFAULT_POLICY = ToolPolicy()

# Read-only coordination readers: deterministic, no side effects, safe to issue
# together and safe to re-run against an unchanged snapshot.
_PARALLEL_SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "list_reports",
        "get_report",
        "list_todos",
        "get_note",
        "list_notes",
        "get_threat_model",
        "list_coverage",
        "list_mcps",
        "describe_mcp",
        "view_agent_graph",
        "source_inspect_many",
        # sandbox filesystem readers
        "read_file",
        "list_dir",
        "file_search",
    }
)

# Tools that must never overlap each other even across different names: the
# sandbox shell pair shares one interactive/session channel.
_SHELL_TOOLS: frozenset[str] = frozenset({"exec_command", "write_stdin"})

# Every parallel-safe reader is a pure function of immutable inputs here.
_CACHEABLE_TOOLS: frozenset[str] = frozenset(_PARALLEL_SAFE_TOOLS)


def policy_for(tool_name: str | None) -> ToolPolicy:
    name = (tool_name or "").strip()
    if not name:
        return DEFAULT_POLICY
    if name in _SHELL_TOOLS:
        return ToolPolicy(
            parallel_safe=False,
            side_effect_free=False,
            cacheable=False,
            serial_group="shell",
        )
    if name in _PARALLEL_SAFE_TOOLS:
        return ToolPolicy(
            parallel_safe=True,
            side_effect_free=True,
            cacheable=name in _CACHEABLE_TOOLS,
        )
    # Unknown tools are assumed to mutate or depend on shared state: they run
    # alone (serialized under their own name).
    return DEFAULT_POLICY


def is_parallel_safe(tool_name: str | None) -> bool:
    return policy_for(tool_name).parallel_safe


def serial_group_for(tool_name: str | None) -> str:
    """The lock group serializing this tool: policy group, else the tool name."""
    name = (tool_name or "").strip() or "?"
    policy = policy_for(name)
    return policy.serial_group or name


def parallel_safe_tool_names() -> list[str]:
    return sorted(_PARALLEL_SAFE_TOOLS)


def wide_turn_guidance(width: int) -> str:
    """System-prompt guidance for parallel turns, rendered only when enabled.

    Kept as prose the model can act on: safe tools may batch, everything else
    stays strictly one-per-turn.
    """
    safe = ", ".join(parallel_safe_tool_names())
    return (
        "Efficient parallel turns are enabled. You may issue up to "
        f"{max(1, int(width))} independent tool calls in a single turn — but ONLY "
        "when every call is read-only and independent (they must not depend on "
        "each other's output). The tools safe to batch are: "
        f"{safe}. "
        "NEVER issue exec_command/write_stdin (shell), file writes, mutating HTTP, "
        "report creation/revision, dependency/agent lifecycle, or any other "
        "state-changing call in parallel — those stay strictly one-per-turn and "
        "sequential. When one result informs the next call, keep the turn narrow "
        "and wait for the result."
    )


_LOCAL = threading.local()


def _per_loop_locks() -> dict[str, asyncio.Lock]:
    """Per-(event-loop, group) lock registry, cached on the current thread.

    Agents run on their own asyncio loops (one per agent), and asyncio.Lock is
    loop-bound: a process-wide lock would raise across loops. Same-loop
    concurrent calls (exactly what provider ``parallel_tool_calls`` produces)
    must contend on one lock; calls from different agents are already
    orchestrated apart and do not need a shared lock.
    """
    locks = getattr(_LOCAL, "serial_locks", None)
    if locks is None:
        locks = {}
        _LOCAL.serial_locks = locks
    return locks


@asynccontextmanager
async def serialized(group: str) -> AsyncIterator[None]:
    """Run one tool call under its group's serialization lock (no-op is none)."""
    lock = _per_loop_locks().setdefault(group, asyncio.Lock())
    async with lock:
        yield
