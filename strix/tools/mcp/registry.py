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
from typing import TYPE_CHECKING, Any, NamedTuple


if TYPE_CHECKING:
    from agents.mcp import MCPServer

    from strix.tools.mcp.client import ResultTransform
    from strix.tools.mcp.config import McpConnectionConfig


# The run-context key under which the runner stores the per-run registry, and
# the two dispatch tools read it back. Kept here so the tools, the runner, and
# strix-pro all agree on one name.
MCP_REGISTRY_CONTEXT_KEY = "mcp_registry"


# The two generic dispatch tools every agent reaches its MCP connections through.
# ``call_mcp`` runs one tool on a connection; ``describe_mcp`` lists a
# connection's tool schemas. Kept here (not in the interface layer) so the engine,
# the OSS viewer, and strix-pro's tracer all recognise a dispatch call by the same
# names.
CALL_MCP_TOOL = "call_mcp"
DESCRIBE_MCP_TOOL = "describe_mcp"
MCP_DISPATCH_TOOLS = frozenset({CALL_MCP_TOOL, DESCRIBE_MCP_TOOL})


@dataclasses.dataclass(frozen=True)
class McpConnectionEntry:
    """One live MCP connection a scan may reach, keyed by ``name``.

    ``server`` is the connected SDK session the dispatch tools list tools on and
    call tools through. ``purpose`` is the human label shown in the prompt
    inventory (the user's connection notes, or whatever the caller supplies).
    ``tool_count`` is how many tools the connection offers, for the inventory
    line. ``result_transform``, when set, runs on each call's structured result
    at the single dispatch point (strix-pro's sanitizer uses it). ``provider`` is
    an optional source label (e.g. ``"supabase"``) the caller tags the connection
    with; the command-line path leaves it ``None``, and event tagging surfaces it
    when set.
    """

    server: MCPServer
    name: str
    purpose: str | None = None
    tool_count: int = 0
    result_transform: ResultTransform | None = None
    provider: str | None = None


@dataclasses.dataclass(frozen=True)
class McpConnectionSummary:
    """One inventory line: what an agent needs to decide whether to
    ``describe_mcp`` a connection, with no tool schemas."""

    name: str
    purpose: str | None
    tool_count: int
    provider: str | None = None


@dataclasses.dataclass(frozen=True)
class McpConnectionRequest:
    """A source-agnostic request to attach one MCP connection to a run.

    The caller hands the engine an inert ``config`` (how to reach the server, its
    name, and any auth token) plus metadata, and never a live session: the engine
    owns connecting and cleaning up. ``provider`` is an optional source label
    (e.g. ``"supabase"``; empty for the command-line path). ``result_transform``
    is an optional per-connection transform run on each call's structured result
    at the single dispatch point (strix-pro's sanitizer; empty for the
    command-line path). ``purpose`` is the human label shown in the prompt
    inventory; when unset it falls back to ``config.notes``.
    """

    config: McpConnectionConfig
    provider: str | None = None
    result_transform: ResultTransform | None = None
    purpose: str | None = None


class McpCallInfo(NamedTuple):
    """What one MCP dispatch call resolved to: the connection name, the
    underlying tool (empty for ``describe_mcp``), and the connection's provider
    label (``None`` when unknown or untagged)."""

    connection: str
    tool: str
    provider: str | None


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
        provider: str | None = None,
    ) -> McpConnectionEntry:
        """Register one connection under ``name`` (last write wins)."""
        entry = McpConnectionEntry(
            server=server,
            name=name,
            purpose=purpose,
            tool_count=tool_count,
            result_transform=result_transform,
            provider=provider,
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
                name=entry.name,
                purpose=entry.purpose,
                tool_count=entry.tool_count,
                provider=entry.provider,
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


def resolve_mcp_call(
    tool_name: str,
    args: dict[str, Any],
    registry: McpRegistry | None = None,
) -> McpCallInfo | None:
    """Resolve one tool call to the MCP connection/tool/provider it went out to.

    The single resolver both the OSS viewer and strix-pro's tracer read a
    dispatch call through, so a call is attributed the same way everywhere. Every
    MCP call an agent makes goes through ``call_mcp`` or ``describe_mcp``, and the
    connection (and, for ``call_mcp``, the server's own tool name) ride in the
    call's ``args`` rather than the tool name, so they are read from there.

    Returns ``None`` when ``tool_name`` is not one of the two dispatch tools, when
    the call carries no connection name, or when a ``registry`` is supplied and
    has no connection under that name. ``tool`` is the underlying tool for
    ``call_mcp`` and empty for ``describe_mcp`` (which inspects the connection
    itself). ``provider`` comes from the registry entry; it is ``None`` when no
    ``registry`` is supplied (the viewer projects calls without one) or when the
    connection carries no provider label.
    """
    if tool_name not in MCP_DISPATCH_TOOLS:
        return None
    connection = args.get("connection")
    if not isinstance(connection, str) or not connection:
        return None
    provider: str | None = None
    if registry is not None:
        entry = registry.get(connection)
        if entry is None:
            return None
        provider = entry.provider
    raw_tool = args.get("tool") if tool_name == CALL_MCP_TOOL else ""
    tool = raw_tool if isinstance(raw_tool, str) else ""
    return McpCallInfo(connection=connection, tool=tool, provider=provider)
