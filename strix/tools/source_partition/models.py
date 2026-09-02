"""Data model for deterministic source partitioning.

Pure data containers only - no I/O, no logic beyond serialization.  Every
public shape is frozen so it can be shared between phases without accidental
mutation, and every field has a deterministic, documented meaning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

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
]

#: Files above this size are never fully read into memory: their LOC is
#: counted deterministically by streaming, so an oversized hand-written source
#: file still becomes a normal (flagged) partition unit instead of silently
#: disappearing.  Classification - not this byte threshold - decides whether a
#: file belongs in the scan at all.
DEFAULT_MAX_FILE_BYTES = 8 * 1024 * 1024


class FileKind(StrEnum):
    """Source classification - a *weight-policy* concept, separate from
    exclusion (see ``strix.tools.source_partition.classify``).

    ``GENERATED`` / ``VENDOR`` files are kept out of the partition units
    entirely; the remaining kinds stay in the manifest but contribute their
    effective weight scaled by ``PartitionConfig.weight``.
    """

    SOURCE = "source"
    TEST = "test"
    SCHEMA = "schema"
    DATA = "data"
    GENERATED = "generated"
    VENDOR = "vendor"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class WeightFactors:
    """Per-``FileKind`` effective-weight multipliers (fractions).

    A file's effective weight is ``round_half_up(loc * factor)`` with a floor
    of 1, so a file is never weightless (it would otherwise not count as a
    "useful unit").  ``GENERATED`` / ``VENDOR`` have no factor because those
    kinds are excluded from units, not weighted.
    """

    source: Fraction = Fraction(1, 1)
    test: Fraction = Fraction(1, 5)
    schema: Fraction = Fraction(1, 5)
    data: Fraction = Fraction(1, 20)
    text: Fraction = Fraction(1, 4)

    def factor_for(self, kind: FileKind) -> Fraction:
        if kind is FileKind.SOURCE:
            return self.source
        if kind is FileKind.TEST:
            return self.test
        if kind is FileKind.SCHEMA:
            return self.schema
        if kind is FileKind.DATA:
            return self.data
        if kind is FileKind.TEXT:
            return self.text
        raise ValueError(f"no weight factor for excluded kind {kind!r}")


@dataclass(frozen=True, slots=True)
class PartitionConfig:
    """Tuning knobs for inventory / unit building / partitioning.

    Defaults are chosen so that the deterministic tests and the documented
    guarantees hold; change them only with intent (they are all part of the
    byte-identity contract: same config => same manifest).
    """

    #: Locality/balance knob: a whole subtree is placed into a shard while
    #: ``least_loaded + subtree_weight <= ceil(total_weight / E * balance_tolerance)``.
    #: A directory that cannot fit (or is heavier than the cap) is recursively
    #: expanded into its children - only an individual *file* may stay whole
    #: above the cap (files are never split).
    balance_tolerance: float = 1.25
    #: Files strictly larger than this many bytes are LOC-counted by streaming
    #: (never fully read); oversized ``DATA`` artifacts are skipped with a note
    #: (conservative data safety limit - classification still decides
    #: inclusion for every other kind).
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    #: When ``partition_source(..., workers<=0)`` means "auto".
    default_workers: int | None = None
    #: Alternative to downweighting: drop test-kind files from units entirely.
    exclude_tests: bool = False
    #: Effective-weight multipliers per kind.
    weight: WeightFactors = WeightFactors()


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    """One surviving file from the inventory walk.

    ``rel`` is the normalized path relative to its root (NFC, forward slashes,
    no root-name prefix - the prefix is a display concern handled at manifest
    build time).  ``io_parts`` is the raw on-disk relative path (exact spelling
    of every segment) used only to reopen the file for LOC counting - it is
    never serialized and never used for matching, so filesystems that store
    decomposed unicode or unusual casing still read back correctly.  ``kind``
    is the classification layer's verdict (computed during inventory; used by
    the unit builder).
    """

    root_index: int
    rel: str
    io_parts: tuple[str, ...]
    kind: FileKind
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SourceInventory:
    """Result of :func:`inventory_source`.

    ``roots`` are the canonicalized (resolved, sorted, deduplicated) roots.
    ``entries`` is sorted by (casefolded rel, rel).  ``notes`` carries
    deterministic diagnostics (skips, warnings) in processing order.
    """

    roots: tuple[Path, ...]
    entries: tuple[InventoryEntry, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PartitionUnit:
    """A file that will be handed to exactly one worker shard.

    ``display`` is the manifest spelling of the path (root-prefixed when the
    source spans several roots); ``rel`` is the root-relative path used for
    classification and reading.  ``oversized`` flags units whose LOC was
    counted by streaming (the file exceeded ``PartitionConfig.max_file_bytes``
    and was never fully read into memory).
    """

    root_index: int
    rel: str
    display: str
    kind: FileKind
    loc: int
    weight: int
    oversized: bool = False


@dataclass(frozen=True, slots=True)
class PartitionShard:
    """One worker manifest: the file list plus aggregate metrics."""

    shard_id: int
    files: tuple[str, ...]
    weight: int
    loc: int


@dataclass(frozen=True, slots=True)
class PartitionManifest:
    """Deterministic partition output.

    Same checkout + same config + same worker count => byte-for-byte identical
    JSON (see :meth:`to_json`).  Shards are ordered by id; file lists and the
    ``file_to_shard`` mapping are ordered lexicographically case-folded.

    ``notes`` carries deterministic, machine-readable diagnostics (inventory
    warnings and oversized/streamed-file flags) so the team layer can surface
    them at the call site instead of relying on log files.
    """

    requested_workers: int
    effective_workers: int
    total_weight: int
    total_loc: int
    shards: tuple[PartitionShard, ...]
    file_to_shard: dict[str, int]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_workers": self.requested_workers,
            "effective_workers": self.effective_workers,
            "total_weight": self.total_weight,
            "total_loc": self.total_loc,
            "shards": [
                {
                    "id": shard.shard_id,
                    "files": list(shard.files),
                    "weight": shard.weight,
                    "loc": shard.loc,
                }
                for shard in self.shards
            ],
            "file_to_shard": dict(self.file_to_shard),
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        """Canonical serialization - deterministic across runs and platforms."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False)
