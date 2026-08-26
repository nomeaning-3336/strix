"""Per-run registry of the MCP connections a scan may reach.

Replaces per-tool registration. The old model turned every tool of every
connected MCP server into its own agent tool, so a run with a handful of
connections put dozens of provider tool schemas on the root agent's first LLM
request. Instead, a run holds its live connections here, keyed by the name the
user gave each connection, and every agent reaches them through two generic
dispatch tools: ``describe_mcp`` to learn one connection's tool schemas on
demand, and ``call_mcp`` to run one of its tools.

One :class:`McpRegistry` is built per run in :mod:`strix.core.runner`, stored in
the run context under :data:`MCP_REGISTRY_CONTEXT_KEY`, and shared by the root
agent and every child (the child context is a copy of the parent's, so it
carries the same registry object).

strix-pro imports :class:`McpRegistry` to add its cloud connections into the
same registry and to attach a per-connection ``result_transform`` (its
sanitizer), which :func:`strix.tools.mcp.client.dispatch_mcp_call` applies at the
single dispatch point.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from agents.mcp import MCPServer

    from strix.tools.mcp.client import ResultTransform


# The run-context key under which the runner stores the per-run registry, and
# the two dispatch tools read it back. Kept here so the tools, the runner, and
# strix-pro all agree on one name.
MCP_REGISTRY_CONTEXT_KEY = "mcp_registry"


@dataclasses.dataclass(frozen=True)
class McpConnectionEntry:
    """One live MCP connection a scan may reach, keyed by ``name``.

    ``server`` is the connected SDK session the dispatch tools list tools on and
    call tools through. ``purpose`` is the human label shown in the prompt
    inventory (the user's connection notes, or whatever the caller supplies).
    ``tool_count`` is how many tools the connection offers, for the inventory
    line. ``result_transform``, when set, runs on each call's structured result
    at the single dispatch point (strix-pro's sanitizer uses it).
    """

    server: MCPServer
    name: str
    purpose: str | None = None
    tool_count: int = 0
    result_transform: ResultTransform | None = None


@dataclasses.dataclass(frozen=True)
class McpConnectionSummary:
    """One inventory line: what an agent needs to decide whether to
    ``describe_mcp`` a connection, with no tool schemas."""

    name: str
    purpose: str | None
    tool_count: int


class McpRegistry:
    """Connection name -> live MCP connection, built per run and shared by every
    agent in the run.

    Public API (strix-pro builds against it): the constructor, :meth:`add`,
    :meth:`get`, and :meth:`summaries`.
    """

    def __init__(self) -> None:
        self._entries: dict[str, McpConnectionEntry] = {}

    def add(
        self,
        *,
        name: str,
        server: MCPServer,
        purpose: str | None = None,
        tool_count: int = 0,
        result_transform: ResultTransform | None = None,
    ) -> McpConnectionEntry:
        """Register one connection under ``name`` (last write wins)."""
        entry = McpConnectionEntry(
            server=server,
            name=name,
            purpose=purpose,
            tool_count=tool_count,
            result_transform=result_transform,
        )
        self._entries[name] = entry
        return entry

    def get(self, name: str) -> McpConnectionEntry | None:
        """The connection registered under ``name``, or ``None``."""
        return self._entries.get(name)

    def names(self) -> list[str]:
        """The registered connection names, in insertion order."""
        return list(self._entries)

    def summaries(self) -> list[McpConnectionSummary]:
        """One inventory summary per connection, in insertion order."""
        return [
            McpConnectionSummary(
                name=entry.name, purpose=entry.purpose, tool_count=entry.tool_count
            )
            for entry in self._entries.values()
        ]

    def clear(self) -> None:
        """Drop every connection (the sessions themselves are closed by the
        runner)."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)


def mcp_inventory_context(registry: McpRegistry | None) -> list[dict[str, Any]]:
    """Build the prompt inventory data for a run's connections.

    Returns one dict per connection with its ``name``, ``purpose``, and
    ``tool_count`` (no tool schemas), ready to thread into the system-prompt
    context under ``mcp_connections`` so both the root agent and every child
    render the same inventory. Returns an empty list when there is no registry
    or no connection, and the template's inventory section is then not rendered.
    """
    if not registry:
        return []
    return [
        {"name": summary.name, "purpose": summary.purpose, "tool_count": summary.tool_count}
        for summary in registry.summaries()
    ]
