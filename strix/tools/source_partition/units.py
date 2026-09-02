"""Phase 2 — build partition units from the inventory.

Deterministic, pure, no agent coupling.  For every inventory entry that
survives the classification weight policy (``GENERATED`` / ``VENDOR`` are
excluded *entirely*; tests optionally too), read the file, count meaningful
LOC with the per-language rule, and apply the kind's weight factor.

Weight policy (documented defaults, all configurable):

- ``SOURCE``    → factor 1/1  (the main signal: meaningful source LOC)
- ``TEST``      → factor 1/5  (kept but substantially downweighted — the v1
                              default; set ``exclude_tests`` to drop them)
- ``SCHEMA``    → factor 1/5  (contracts/API schemas stay visible)
- ``DATA``      → factor 1/20 (dumps/fixtures barely shape the balance)
- ``TEXT``      → factor 1/4  (docs/config noise stays out of the way)
- ``GENERATED`` / ``VENDOR`` → excluded entirely (never a unit)

Effective weight is ``round_half_up(loc * factor)`` floored at 1, so every
unit is at least one "useful unit" of work; balance is driven by *weight*,
never by raw byte size (the spec's constraint).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from strix.tools.source_partition.classify import suffix_of
from strix.tools.source_partition.loc import count_loc, language_for
from strix.tools.source_partition.models import (
    FileKind,
    InventoryEntry,
    PartitionConfig,
    PartitionUnit,
    SourceInventory,
)
from strix.tools.source_partition.normalize import display_rel, display_root_names
from strix.tools.source_partition.readio import decode_text, read_bytes


if TYPE_CHECKING:
    from fractions import Fraction

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
) -> list[PartitionUnit]:
    """Turn a :class:`SourceInventory` into the weighted unit list.

    The returned list is sorted by ``(casefold(display), display)`` — the same
    ordering the manifest uses, so downstream code never needs to re-sort.
    """
    cfg = config or PartitionConfig()
    root_names = display_root_names(inventory.roots)
    units: list[PartitionUnit] = []
    for entry in inventory.entries:
        unit = _build_unit(inventory, entry, cfg, root_names)
        if unit is not None:
            units.append(unit)
    units.sort(key=lambda unit: (unit.display.casefold(), unit.display))
    return units


def _build_unit(
    inventory: SourceInventory,
    entry: InventoryEntry,
    cfg: PartitionConfig,
    root_names: tuple[str, ...],
) -> PartitionUnit | None:
    if entry.kind in (FileKind.GENERATED, FileKind.VENDOR):
        logger.debug("source_partition: exclude %s-kind file %r", entry.kind.value, entry.rel)
        return None
    if cfg.exclude_tests and entry.kind is FileKind.TEST:
        logger.debug("source_partition: exclude tests (config) %r", entry.rel)
        return None
    path = inventory.roots[entry.root_index].joinpath(*entry.io_parts)
    try:
        data = read_bytes(path)
    except OSError as exc:
        # The file survived inventory but vanished / became unreadable between
        # the two phases (TOCTOU).  Skip it — never abort the partition.
        logger.warning("source_partition: skipping unreadable file %r: %s", entry.rel, exc)
        return None
    language = language_for(suffix_of(entry.rel))
    meaningful = count_loc(decode_text(data), language)
    factor = cfg.weight.factor_for(entry.kind)
    weight = effective_weight(meaningful, factor)
    return PartitionUnit(
        root_index=entry.root_index,
        rel=entry.rel,
        display=display_rel(root_names, entry.root_index, entry.rel),
        kind=entry.kind,
        loc=meaningful,
        weight=weight,
    )
