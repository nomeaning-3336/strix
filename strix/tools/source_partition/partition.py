"""Phase 3 - deterministic locality-aware partition assignment.

Pure function over units; the partitioner never spawns workers and knows
nothing about agents, coordinators or report state (fan-out is the *next*
commit's job - see ``__init__``).

Algorithm (documented so the determinism tests can assert it):

1. **Effective workers** ``E = min(requested, len(units))``; ``E == 0`` when
   there are no units.  Shards are never empty: after assignment, trailing
   empty shards (possible only when a giant carve packs nearly everything into
   one shard) are dropped and shard ids are renumbered contiguously, so
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
     duplicated);
   - a *directory* is assigned whole when the least-loaded shard can take it
     within ``cap = ceil(total / E * balance_tolerance)`` - related
     directories (``src/server/auth`` + ``src/server/session``) therefore land
     in one shard whenever they fit together;
   - a directory that exceeds ``cap`` *and* holds at least ``giant_share`` of
     the total weight is the "giant subsystem" carve: it is assigned whole to
     the least-loaded shard (fewest possible shards; at most one shard holds
     the bulk - the spec's fallback for ``> 50%`` subtrees);
   - any other directory that cannot fit is *expanded* into its children,
     which re-enter the job queue in the same deterministic order - a subtree
     is only ever split when it is too large to stay together.

4. **Result.**  Files are grouped per shard; within a shard they are ordered
   ``(casefold(path), path)``; shards are ordered by id; ``file_to_shard``
   maps every included path to exactly one shard.

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

__all__ = ["partition_source"]


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
    """Partition ``roots`` into ``workers`` locality-aware shards.

    ``workers <= 0`` means "auto": ``config.default_workers`` if set, else
    ``os.cpu_count()``.  Pure composition of :func:`inventory_source` ->
    :func:`build_partition_units` -> deterministic assignment.
    """
    cfg = config or PartitionConfig()
    requested = workers if workers > 0 else _auto_workers(cfg)
    inventory = inventory_source(roots, config=cfg)
    units = build_partition_units(inventory, config=cfg)
    return _assign(units, requested_workers=requested, cfg=cfg)


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


def _place(
    root: _Node, effective: int, total_weight: int, cap: int, giant_share: Fraction
) -> dict[_Node, int]:
    """Balanced LPT over subtrees; returns node -> shard for whole placements."""
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
        giant_numerator = giant_share.numerator
        giant_denominator = giant_share.denominator
        is_giant = weight > cap and weight * giant_denominator >= total_weight * giant_numerator
        if is_giant or loads[least] + weight <= cap:
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
        )

    effective = min(requested_workers, len(units))
    tolerance = Fraction(cfg.balance_tolerance).limit_denominator(1_000_000)
    cap = (total_weight * tolerance.numerator + effective * tolerance.denominator - 1) // (
        effective * tolerance.denominator
    )
    giant_share = Fraction(cfg.giant_share).limit_denominator(1_000_000)

    root = _build_trie(units)
    placed = _place(root, effective, total_weight, cap, giant_share)
    by_shard = _collect_shards(root, effective, placed)

    # A giant carve (or a single file heavier than cap) can pack nearly the
    # whole tree into one shard; the leftover jobs then fill shards from the
    # lowest id upward, leaving only *trailing* shards empty.  Drop those and
    # renumber contiguously: shards are never empty, so ``effective_workers``
    # is the number of shards that actually received files.
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
    )
