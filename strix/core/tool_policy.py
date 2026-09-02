"""Tool execution policy for wide turns (Efficiency v1).

Which tools may run concurrently inside one model turn, which are pure enough
to re-run/cache, and how everything else is kept serial:

  - ``parallel_safe``    the tool is read-only and concurrency-safe, so issuing
                         several of its calls (or mixing it with other safe
                         calls) in one reasoning turn is fine.
  - ``side_effect_free`` the call changes nothing (read-only).
  - ``cacheable``        the result is a pure function of its arguments against
                         an immutable snapshot (safe to replay from a result
                         cache). Read-only is NOT enough: runtime-state readers
                         (list_reports, view_agent_graph, list_coverage, …)
                         change as the scan runs and are deliberately NOT
                         cacheable.
  - ``serial_group``     tools sharing a group must never overlap each other.
                         All non-parallel-safe / unknown / mutating tools share
                         ONE group (``mutation``) so two different mutating
                         tool names can never interleave; the sandbox shell
                         pair keeps its own ``shell`` group.

Two independent bounds are enforced by the executor, never left to the model:

  1. Safety — every non-``parallel_safe`` call acquires its serial group lock.
  2. Width — every tool call (safe or not) additionally acquires a per-loop
     semaphore sized by STRIX_TOOL_WIDTH, so the number of concurrently
     executing tool calls on an agent can never exceed the configured width,
     even when the provider returns more tool calls than the guidance asked for.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    parallel_safe: bool = False
    side_effect_free: bool = False
    cacheable: bool = False
    serial_group: str | None = None


MUTATION_GROUP = "mutation"

# Every non-parallel-safe / unknown tool is presumed to mutate shared state and
# shares one serialization group: different mutating tool names must never
# overlap inside a wide turn. Safer to under-parallelize than to race.
DEFAULT_POLICY = ToolPolicy(
    parallel_safe=False,
    side_effect_free=False,
    cacheable=False,
    serial_group=MUTATION_GROUP,
)

# Read-only, concurrency-safe readers. They are safe to issue together, but
# most read *runtime state* that changes as the scan runs (reports, todos,
# notes, coverage, agent graph) — so they are NOT cacheable.
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

# Tools whose result is stable for an immutable snapshot key. Only the
# deterministic source-inspection operations qualify today: the code checkout
# does not change between two identical calls, unlike every runtime reader.
_CACHEABLE_TOOLS: frozenset[str] = frozenset({"source_inspect_many"})

# Tools that must never overlap each other even across different names: the
# sandbox shell pair shares one interactive/session channel.
_SHELL_TOOLS: frozenset[str] = frozenset({"exec_command", "write_stdin"})


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
    # Unknown tools are presumed mutating: parallel_safe=False and grouped with
    # every other mutation.
    return DEFAULT_POLICY


def is_parallel_safe(tool_name: str | None) -> bool:
    return policy_for(tool_name).parallel_safe


def is_cacheable(tool_name: str | None) -> bool:
    return policy_for(tool_name).cacheable


def serial_group_for(tool_name: str | None) -> str:
    """The lock group serializing this tool (policy group when set)."""
    name = (tool_name or "").strip() or "?"
    policy = policy_for(name)
    return policy.serial_group or name


def parallel_safe_tool_names() -> list[str]:
    return sorted(_PARALLEL_SAFE_TOOLS)


def wide_turn_guidance(width: int) -> str:
    """System-prompt guidance for parallel turns, rendered only when enabled.

    Guidance only — the executor enforces both the serial groups and the width
    semaphore, so the model cannot widen the turn beyond the configured bound
    even if it ignores this prose.
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


# --- executor primitives ------------------------------------------------------
#
# Both locks and the width semaphore are per asyncio loop (agents run one loop
# per agent, and asyncio primitives are loop-bound), cached on the current
# thread — the thread that owns that loop.


_LOCAL = threading.local()


def _per_thread(attr: str) -> dict[Any, Any]:
    store = getattr(_LOCAL, attr, None)
    if store is None:
        store = {}
        setattr(_LOCAL, attr, store)
    return store


@asynccontextmanager
async def serialized(group: str) -> AsyncIterator[None]:
    """Run one tool call under its group's serialization lock."""
    locks = _per_thread("serial_locks")
    key = (id(asyncio.get_running_loop()), group)
    lock = locks.setdefault(key, asyncio.Lock())
    async with lock:
        yield


_OVERRIDE_WIDTH: int | None = None


def set_tool_width_override(value: int | None) -> None:
    """Test seam: fix the width semaphore capacity without touching settings."""
    global _OVERRIDE_WIDTH  # noqa: PLW0603
    _OVERRIDE_WIDTH = value


def _tool_width() -> int:
    if _OVERRIDE_WIDTH is not None:
        return max(1, int(_OVERRIDE_WIDTH))
    try:
        from strix.config.loader import load_settings  # noqa: PLC0415

        width = int(getattr(load_settings().llm, "tool_width", 3) or 3)
    except Exception:  # noqa: BLE001 - width is a bound, never a failure
        width = 3
    return max(1, width)


@asynccontextmanager
async def tool_slot() -> AsyncIterator[None]:
    """Per-agent-loop width bound: at most STRIX_TOOL_WIDTH tool calls execute
    concurrently on this loop, safe or not (mutating calls additionally hold
    their serial lock inside the slot)."""
    semaphores = _per_thread("tool_slots")
    key = id(asyncio.get_running_loop())
    semaphore = semaphores.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(_tool_width())
        semaphores[key] = semaphore
    async with semaphore:
        yield
