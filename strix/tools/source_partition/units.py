"""Phase 2 - build partition units from the inventory.

Deterministic, no agent coupling.  For every inventory entry that survives the
classification weight policy (``GENERATED`` / ``VENDOR`` are excluded
*entirely*; tests optionally too), read the file, count meaningful LOC with the
per-language rule, and apply the kind's weight factor.

Weight policy (documented defaults, all configurable):

- ``SOURCE``    -> factor 1/1  (the main signal: meaningful source LOC)
- ``TEST``      -> factor 1/5  (kept but substantially downweighted - the v1
                               default; set ``exclude_tests`` to drop them)
- ``SCHEMA``    -> factor 1/5  (contracts/API schemas stay visible)
- ``DATA``      -> factor 1/20 (dumps/fixtures barely shape the balance)
- ``TEXT``      -> factor 1/4  (docs/config noise stays out of the way)
- ``GENERATED`` / ``VENDOR`` -> excluded entirely (never a unit)

Effective weight is ``round_half_up(loc * factor)`` floored at 1, so every
unit is at least one "useful unit" of work; balance is driven by *weight*,
never by raw byte size (the spec's constraint).

Oversized files (``size_bytes > cfg.max_file_bytes``) are LOC-counted by
*streaming* so a huge hand-written source file still becomes a normal
partition unit without being loaded into memory; such units carry
``oversized=True`` and a note.  The one conservative size-based exclusion is
classification-driven: an oversized ``DATA`` artifact (giant dumps/fixtures)
is dropped with a note - inclusion is never decided by the byte threshold
alone for source-like kinds.

Returns ``(units, notes)``; ``notes`` are deterministic, machine-readable
diagnostics meant to be surfaced on the manifest by the convenience wrapper.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from strix.tools.source_partition.classify import suffix_of
from strix.tools.source_partition.loc import count_loc, count_loc_lines, language_for
from strix.tools.source_partition.models import (
    FileKind,
    InventoryEntry,
    PartitionConfig,
    PartitionUnit,
    SourceInventory,
)
from strix.tools.source_partition.normalize import display_rel, display_root_names
from strix.tools.source_partition.readio import decode_text, iter_text_lines, read_bytes


if TYPE_CHECKING:
    from fractions import Fraction
    from pathlib import Path

__all__ = ["build_partition_units", "effective_weight"]

logger = logging.getLogger("strix.tools.source_partition.units")


def effective_weight(loc_count: int, factor: Fraction) -> int:
    """Round-half-up ``loc * factor`` with a floor of 1 (integer weights keep
    the LPT pass and every manifest number exact)."""
    if loc_count <= 0:
        return 1
    numerator = loc_count * factor.numerator
    denominator = factor.denominator
    rounded = (2 * numerator + denominator) // (2 * denominator)
    return max(1, rounded)


def build_partition_units(
    inventory: SourceInventory,
    *,
    config: PartitionConfig | None = None,
) -> tuple[list[PartitionUnit], tuple[str, ...]]:
    """Turn a :class:`SourceInventory` into the weighted unit list + notes.

    The returned list is sorted by ``(casefold(display), display)`` - the same
    ordering the manifest uses, so downstream code never needs to re-sort.
    """
    cfg = config or PartitionConfig()
    root_names = display_root_names(inventory.roots)
    units: list[PartitionUnit] = []
    notes: list[str] = []
    for entry in inventory.entries:
        unit, note = _build_unit(inventory, entry, cfg, root_names)
        if unit is not None:
            units.append(unit)
        if note is not None:
            notes.append(note)
    units.sort(key=lambda unit: (unit.display.casefold(), unit.display))
    return units, tuple(notes)


def _count_oversized(path: Path, language: str) -> int:
    """Deterministic streaming LOC count - never loads the file whole."""
    return count_loc_lines(iter_text_lines(path), language)


def _try_read_loc(
    path: Path, entry: InventoryEntry, cfg: PartitionConfig, language: str
) -> int | None:
    """Count meaningful LOC (streaming when oversized); ``None`` if unreadable
    (TOCTOU between inventory and here - logged, never fatal)."""
    try:
        if entry.size_bytes > cfg.max_file_bytes:
            return _count_oversized(path, language)
        data = read_bytes(path)
        return count_loc(decode_text(data), language)
    except OSError as exc:
        logger.warning("source_partition: skipping unreadable file %r: %s", entry.rel, exc)
        return None


def _build_unit(
    inventory: SourceInventory,
    entry: InventoryEntry,
    cfg: PartitionConfig,
    root_names: tuple[str, ...],
) -> tuple[PartitionUnit | None, str | None]:
    display = display_rel(root_names, entry.root_index, entry.rel)
    if entry.kind in (FileKind.GENERATED, FileKind.VENDOR):
        logger.debug("source_partition: exclude %s-kind file %r", entry.kind.value, entry.rel)
        return None, None
    if cfg.exclude_tests and entry.kind is FileKind.TEST:
        logger.debug("source_partition: exclude tests (config) %r", entry.rel)
        return None, None
    oversized = entry.size_bytes > cfg.max_file_bytes
    if oversized and entry.kind is FileKind.DATA:
        # Conservative data-safety limit, applied *after* classification: a
        # giant dump/fixture is not meaningful source for any shard.
        note = (
            f"skipping oversized data artifact (kind=data, "
            f"{entry.size_bytes} bytes > {cfg.max_file_bytes} max): {entry.rel}"
        )
        logger.warning("source_partition: %s", note)
        return None, note

    path = inventory.roots[entry.root_index].joinpath(*entry.io_parts)
    language = language_for(suffix_of(entry.rel))
    meaningful = _try_read_loc(path, entry, cfg, language)
    if meaningful is None:
        return None, None
    factor = cfg.weight.factor_for(entry.kind)
    unit = PartitionUnit(
        root_index=entry.root_index,
        rel=entry.rel,
        display=display,
        kind=entry.kind,
        loc=meaningful,
        weight=effective_weight(meaningful, factor),
        oversized=oversized,
    )
    oversized_note: str | None = None
    if oversized:
        oversized_note = (
            f"oversized file counted by streaming LOC ({entry.size_bytes} bytes > "
            f"{cfg.max_file_bytes} max): {display}"
        )
    return unit, oversized_note
