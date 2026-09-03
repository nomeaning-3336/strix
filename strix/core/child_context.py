"""Deterministic compact child context handoff (Child Context v1).

Stop copying huge parent trajectories into newly spawned specialists.
This module provides a pure structured handoff (``ChildContextPacket``)
that ``create_agent`` can substitute for ``parent_history`` when the
parent opts into ``inherit_context=False``.

Hard purity rules:

- No imports of :mod:`strix.team`, :mod:`strix.tools.source_partition`,
  :mod:`strix.core.agents`, or any runtime / report state. The packet
  is pure data.
- No LLM calls, no transcript summarization, no semantic inference.
- The rendering is byte-deterministic: stable key ordering,
  order-preserving dedupe across tuple fields, canonical JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ChildContextPacket:
    """Pure structured handoff for a focused specialist subagent.

    The compact rendering replaces a copy of the parent's full trajectory
    when the parent sets ``inherit_context=False`` on ``create_agent``.
    Optional fields default to empty tuples - later work can populate
    them from explicit structured state. v1 establishes the transport
    contract without guessing knowledge from arbitrary transcript text.
    """

    objective: str
    scope: tuple[str, ...]
    known_facts: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    do_not_repeat: tuple[str, ...] = ()


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    """Order-preserving dedupe (deterministic for identical inputs)."""
    return tuple(dict.fromkeys(values))


#: The default do-not-repeat list, surfaced on every compact packet in v1
#: so specialists do not redo broad parent-level discovery by accident.
_DEFAULT_DO_NOT_REPEAT: tuple[str, ...] = (
    "broad repository-level discovery already performed by the parent",
    "broadly scanning files that belong to another agent's scope",
)


def build_packet_from_task(
    *,
    task: str,
    scan_targets: Sequence[object] | None = (),
) -> ChildContextPacket:
    """Build the v1 compact packet from a parent task + scan scope.

    ``scan_targets`` arrives from untyped runtime context
    (``context["scan_targets"]``), so non-string entries are defensively
    dropped here rather than trusted. v1 only fills ``objective``,
    ``scope``, and ``do_not_repeat``. The other structured fields stay
    empty - later work feeds them from explicit state, never by
    summarizing transcript text.
    """
    return ChildContextPacket(
        objective=task,
        scope=_dedupe(
            tuple(t.strip() for t in (scan_targets or ()) if isinstance(t, str) and t.strip())
        ),
        do_not_repeat=_DEFAULT_DO_NOT_REPEAT,
    )


#: Deterministic header rendered above the JSON block. Same wording every
#: render, so the model cannot drift per caller.
_HEADER = (
    "Compact parent handoff.\n"
    "\n"
    "Do not repeat broad discovery already completed by the parent.\n"
    "Work from this packet and inspect additional evidence only as your task requires.\n"
    "\n"
    "Task packet:"
)


def render_packet(packet: ChildContextPacket) -> str:
    """Render the packet as a deterministic, human-readable + JSON block.

    Canonical JSON (``sort_keys=True``, ``ensure_ascii=False``) plus the
    fixed header. Same input -> byte-identical output.
    """
    payload = {
        "objective": packet.objective,
        "scope": list(packet.scope),
        "known_facts": list(packet.known_facts),
        "open_questions": list(packet.open_questions),
        "evidence_refs": list(packet.evidence_refs),
        "do_not_repeat": list(packet.do_not_repeat),
    }
    return f"{_HEADER}\n{json.dumps(payload, sort_keys=True, ensure_ascii=False)}"


__all__ = [
    "ChildContextPacket",
    "build_packet_from_task",
    "render_packet",
]
