"""Phase 3 - deterministic locality-aware partition assignment.

Pure function over already-built :class:`PartitionUnit` lists; the assignment
core does no filesystem work and knows nothing about agents, coordinators or
report state (fan-out is the *next* commit's job - see ``__init__``).  The
``roots``-based convenience wrapper (:func:`partition_source`) only threads the
three phases together and surfaces their notes.

Algorithm (documented so the determinism tests can assert it):

1. **Effective workers** ``E = min(requested, len(units))``; ``E == 0`` when
   there are no units.  Shards are never empty: after assignment, trailing
   empty shards (possible only when a single file heavier than the cap is
   placed alone) are dropped and shard ids are renumbered contiguously, so
   ``effective_workers`` can end up below ``E`` (the ``requested_workers``
   field still reports what was asked for).

2. **Weight rollup over the directory trie.**  Every file is a leaf of a trie
   keyed by its display path; every directory node carries the cumulative
   weight of its subtree (longest-prefix rollup).  This is what keeps
   locality: whole subtrees - not individual files - are the LPT jobs.

3. **Balanced LPT over subtrees.**  Jobs are processed in strict order
   ``weight desc, display path asc`` (case-folded tie-break).  A job is
   assigned to the currently least-loaded shard (ties -> lowest shard id)
   under one of three rules:

   - a *file* is always assigned whole (a file is never split, never
     duplicated) - a file is the only unit allowed to exceed ``cap``;
   - a *directory* is assigned whole when the least-loaded shard can take it
     within ``cap = ceil(total / E * balance_tolerance)`` - related
     directories (``src/server/auth`` + ``src/server/session``) therefore land
     in one shard whenever they fit together;
   - any directory that does not fit - including one whose subtree alone is
     heavier than ``cap`` - is *recursively expanded* into its children, which
     re-enter the job queue in the same deterministic order.  Expansion stops
     only when children are useful-sized subtrees that fit or individual
     files; nothing but an individual file ever stays whole above the cap, so
     a repo where nearly everything lives under one directory is distributed
     across the requested workers instead of being dumped onto one shard.

4. **Result.**  Files are grouped per shard; within a shard they are ordered
   ``(casefold(path), path)``; shards are ordered by id; ``file_to_shard``
   maps every included path to exactly one shard.  ``notes`` (inventory
   warnings, oversized/streamed flags, conservative data skips) are carried
   onto the manifest verbatim so the future team layer can surface them.

Tie-breaking summary (the contract tests assert): inventory order
``(casefold, raw)``; unit order ``(casefold(display), display)``; job order
``weight desc`` then ``display asc``; shard choice ``min (load, id)``.
"""

from __future__ import annotations

import heapq
import os
from fractions import Fraction
from typing import TYPE_CHECKING

from strix.tools.source_partition.inventory import inventory_source
from strix.tools.source_partition.models import (
    PartitionConfig,
    PartitionManifest,
    PartitionShard,
    PartitionUnit,
)
from strix.tools.source_partition.normalize import path_sort_key
from strix.tools.source_partition.units import build_partition_units


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["partition_source", "partition_units"]


class _Node:
    """Directory-trie node; a leaf carries its file's unit.

    Plain class (identity semantics) - nodes are heap/queue members and dict
    keys, so structural equality would be wrong.
    """

    __slots__ = ("children", "key", "parent", "unit", "weight")

    def __init__(self, key: str, parent: _Node | None = None) -> None:
        self.key = key
        self.weight = 0
        self.unit: PartitionUnit | None = None
        self.parent = parent
        self.children: dict[str, _Node] = {}


def partition_source(
    roots: Sequence[Path],
    *,
    workers: int,
    config: PartitionConfig | None = None,
) -> PartitionManifest:
    """Convenience wrapper: inventory -> units -> assignment in one call.

    ``workers <= 0`` means "auto" (see :func:`partition_units`).  The wrapper
    does not duplicate any phase logic - it threads the three stages and
    surfaces every deterministic note (inventory warnings, oversized/streamed
    flags) on the returned :class:`PartitionManifest` instead of swallowing
    them.
    """
    cfg = config or PartitionConfig()
    inventory = inventory_source(roots, config=cfg)
    units, unit_notes = build_partition_units(inventory, config=cfg)
    notes = (*inventory.notes, *unit_notes)
    return partition_units(units, workers=workers, config=cfg, notes=notes)


def partition_units(
    units: Sequence[PartitionUnit],
    *,
    workers: int,
    config: PartitionConfig | None = None,
    notes: Sequence[str] = (),
) -> PartitionManifest:
    """Deterministically partition an already-built unit list.

    The assignment core - pure over units, no roots, no filesystem access.
    ``workers <= 0`` means "auto": ``config.default_workers`` if set, else
    ``os.cpu_count()``.  ``notes`` (optional deterministic diagnostics from the
    earlier phases) are copied onto the manifest.
    """
    cfg = config or PartitionConfig()
    requested = workers if workers > 0 else _auto_workers(cfg)
    return _assign(list(units), requested_workers=requested, cfg=cfg, notes=tuple(notes))


def _auto_workers(cfg: PartitionConfig) -> int:
    if cfg.default_workers is not None and cfg.default_workers > 0:
        return cfg.default_workers
    return max(1, os.cpu_count() or 1)


def _build_trie(units: list[PartitionUnit]) -> _Node:
    """Insert units by display path and roll weights bottom-up."""
    root = _Node(key="")
    for unit in units:
        node = root
        for part in unit.display.split("/"):
            child = node.children.get(part)
            if child is None:
                key = f"{node.key}/{part}" if node.key else part
                child = _Node(key=key, parent=node)
                node.children[part] = child
            node = child
        node.unit = unit

    def roll(node: _Node) -> int:
        weight = node.unit.weight if node.unit is not None else 0
        for child in node.children.values():
            weight += roll(child)
        node.weight = weight
        return weight

    roll(root)
    return root


def _place(root: _Node, effective: int, cap: int) -> dict[_Node, int]:
    """Balanced LPT over subtrees; returns node -> shard for whole placements.

    Files are always placed whole.  Directories are placed whole only when the
    least-loaded shard has room within ``cap``; otherwise they are recursively
    expanded into their children (which re-enter the queue in deterministic
    order).  A directory is never the unit that overflows ``cap``.
    """
    loads = [0] * effective
    placed: dict[_Node, int] = {}
    heap: list[tuple[int, str, _Node]] = [
        (-child.weight, child.key, child) for child in root.children.values()
    ]
    heapq.heapify(heap)

    def least_loaded() -> int:
        return min(range(effective), key=lambda index: (loads[index], index))

    while heap:
        _neg_weight, _key, node = heapq.heappop(heap)
        weight = node.weight
        if node.unit is not None:
            shard = least_loaded()
            loads[shard] += weight
            placed[node] = shard
            continue
        least = least_loaded()
        if loads[least] + weight <= cap:
            loads[least] += weight
            placed[node] = least
            continue
        for child in sorted(node.children.values(), key=lambda child: (-child.weight, child.key)):
            heapq.heappush(heap, (-child.weight, child.key, child))
    return placed


def _collect_shards(
    root: _Node,
    effective: int,
    placed: dict[_Node, int],
) -> dict[int, list[PartitionUnit]]:
    """Resolve each unit to its nearest whole-placed ancestor's shard."""
    by_shard: dict[int, list[PartitionUnit]] = {shard_id: [] for shard_id in range(effective)}

    def collect(node: _Node, inherited: int | None) -> None:
        shard = placed.get(node, inherited)
        if node.unit is not None:
            if shard is None:  # pragma: no cover - invariant, see module docstring
                raise RuntimeError(f"internal error: unit without a shard: {node.unit.display!r}")
            by_shard[shard].append(node.unit)
        for child in node.children.values():
            collect(child, shard)

    for child in root.children.values():
        collect(child, None)
    return by_shard


def _assign(
    units: list[PartitionUnit],
    *,
    requested_workers: int,
    cfg: PartitionConfig,
    notes: tuple[str, ...],
) -> PartitionManifest:
    total_weight = sum(unit.weight for unit in units)
    total_loc = sum(unit.loc for unit in units)
    if not units:
        return PartitionManifest(
            requested_workers=requested_workers,
            effective_workers=0,
            total_weight=0,
            total_loc=0,
            shards=(),
            file_to_shard={},
            notes=notes,
        )

    effective = min(requested_workers, len(units))
    tolerance = Fraction(cfg.balance_tolerance).limit_denominator(1_000_000)
    cap = (total_weight * tolerance.numerator + effective * tolerance.denominator - 1) // (
        effective * tolerance.denominator
    )

    root = _build_trie(units)
    placed = _place(root, effective, cap)
    by_shard = _collect_shards(root, effective, placed)

    # A single file heavier than cap can pack most of the weight into one
    # shard; the leftover jobs then fill shards from the lowest id upward,
    # leaving only *trailing* shards empty.  Drop those and renumber
    # contiguously: shards are never empty, so ``effective_workers`` is the
    # number of shards that actually received files.
    used_ids = sorted(shard_id for shard_id, shard_units in by_shard.items() if shard_units)
    renumber = {old_id: new_id for new_id, old_id in enumerate(used_ids)}

    shards: list[PartitionShard] = []
    file_to_shard: dict[str, int] = {}
    for shard_id in used_ids:
        shard_units = by_shard[shard_id]
        shard_units.sort(key=lambda unit: path_sort_key(unit.display))
        files = tuple(unit.display for unit in shard_units)
        new_id = renumber[shard_id]
        shards.append(
            PartitionShard(
                shard_id=new_id,
                files=files,
                weight=sum(unit.weight for unit in shard_units),
                loc=sum(unit.loc for unit in shard_units),
            )
        )
        for unit in shard_units:
            file_to_shard[unit.display] = new_id

    return PartitionManifest(
        requested_workers=requested_workers,
        effective_workers=len(used_ids),
        total_weight=total_weight,
        total_loc=total_loc,
        shards=tuple(shards),
        file_to_shard=file_to_shard,
        notes=notes,
    )
