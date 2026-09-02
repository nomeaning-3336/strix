"""Deterministic source partitioning v1 - library API (no agent fan-out).

Given a source tree and a requested worker count ``N``, deterministically
divide the meaningful source into approximately balanced, locality-aware
worker manifests.  Pure deterministic Python: no agents spawned, no LLM calls,
no coordinator wiring (team fan-out is a later commit and is *not* part of
this package's contract).

Ignore handling is deliberately three separate layers (each with its own
module and tests):

1. ``exclusions`` - curated hard exclusions (VCS metadata, dependency/build/
   artifact trees, lock/minified files), sharing the canonical
   ``_IGNORE_DIRS`` set with ``source_inspect_many``;
2. ``gitignore`` - a real gitignore-style matcher (nested files, negation,
   anchored/unanchored patterns, ``**``), applied during the walk;
3. ``classify`` - source classification (tests/schemas/generated/vendor/data/
   text), a weight-policy concern that only ever sees surviving paths.

Phase pipeline (composable - the assignment core is pure over units)::

    roots = resolve_authorized_roots(report_state)   # caller concern
    inventory = inventory_source(roots, config=...)  # ignore layers 1-3
    units, notes = build_partition_units(inventory, config=...)  # weights
    manifest = partition_units(units, workers=N, config=..., notes=notes)
    # ...or the one-call convenience wrapper that threads the above:
    manifest = partition_source(roots, workers=N, config=...)

``partition_source(roots, ...)`` only *composes* the three phases - it never
re-implements them - and surfaces every deterministic diagnostic (inventory
warnings, oversized/streamed-file flags, conservative data skips) on
``manifest.notes`` instead of swallowing it.  ``build_partition_units``
returns ``(units, notes)``.

The engine is authorization-agnostic: it takes already-resolved ``Path``
roots and nothing else - it never knows about ``report_state``, a coordinator
or a scan context.  Determinism contract: same checkout + same config + same
worker count => byte-for-byte identical JSON, across reruns and across
Windows/Linux (paths are NFC-normalized with forward slashes, ordered
case-folded).
"""

from __future__ import annotations

from strix.tools.source_partition.inventory import inventory_source
from strix.tools.source_partition.models import (
    DEFAULT_MAX_FILE_BYTES,
    FileKind,
    InventoryEntry,
    PartitionConfig,
    PartitionManifest,
    PartitionShard,
    PartitionUnit,
    SourceInventory,
    WeightFactors,
)
from strix.tools.source_partition.partition import partition_source, partition_units
from strix.tools.source_partition.units import build_partition_units


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "FileKind",
    "InventoryEntry",
    "PartitionConfig",
    "PartitionManifest",
    "PartitionShard",
    "PartitionUnit",
    "SourceInventory",
    "WeightFactors",
    "build_partition_units",
    "inventory_source",
    "partition_source",
    "partition_units",
]
