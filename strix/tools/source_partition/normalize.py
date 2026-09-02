"""Path normalization for deterministic manifests.

Rules (documented once here, referenced from inventory/partition):

- separators: backslashes are normalized to forward slashes on entry
  (Windows → POSIX) — the manifest never contains ``\\``;
- unicode: every path segment is NFC-normalized (macOS decomposed forms and
  Windows NFD inputs collapse to the same bytes);
- case folding: *containment* checks (ignore-dir membership, classification
  markers) compare ``segment.casefold()`` against lower-case constants; *
  ordering* compares ``(path.casefold(), path)`` so the sort is stable and
  case-insensitive without losing ties;
- traversal: a normalized relative path may never contain ``..`` (the walker
  only builds paths from real directory entries, and this is enforced again
  for any caller-supplied relative path).

Normalization notes intentionally mirror ``strix/tools/source_inspect/tool.py``
(``map_operation_path`` replaces ``\\`` with ``/``; containment is resolved
against roots).  The partitioner is root-discovery agnostic and therefore
normalizes *relative* paths only — host absolute paths never appear in the
manifest.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "canonical_roots",
    "display_rel",
    "display_root_names",
    "normalize_rel_path",
    "path_sort_key",
    "rel_from_parts",
]


def normalize_name(name: str) -> str:
    """NFC-normalize one path segment (no separators inside)."""
    return unicodedata.normalize("NFC", name)


def rel_from_parts(parts: Sequence[str]) -> str | None:
    """Canonical rel path from already-split path components.

    Unlike :func:`normalize_rel_path`, backslashes inside a component are
    *not* treated as separators (a component can legally contain one on
    POSIX); components are NFC-normalized only.  Returns ``None`` when the
    result is empty or escapes its root (``..``).
    """
    out: list[str] = []
    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            return None
        out.append(normalize_name(part))
    if not out:
        return None
    return "/".join(out)


def normalize_rel_path(raw: str) -> str | None:
    """Canonical within-root relative path from a possibly Windows-style input.

    Backslashes are normalized to forward slashes (the caller-provided path
    spelling, where ``\\`` always means a separator).  Returns ``None`` when
    the path is empty or escapes its root (``..``).
    """
    return rel_from_parts(raw.replace("\\", "/").split("/"))


def path_sort_key(path: str) -> tuple[str, str]:
    """Deterministic ordering key: case-folded first, raw string as tiebreak."""
    return (path.casefold(), path)


def canonical_roots(roots: Sequence[Path]) -> list[Path]:
    """Resolve + dedupe + sort the caller's roots into a stable order.

    Determinism rule: root order in the manifest never depends on the caller's
    argument order — roots are sorted by (casefolded resolved path, raw path),
    which makes permuting the input list a no-op.
    """
    seen: dict[tuple[str, str], Path] = {}
    for raw_root in roots:
        try:
            resolved = Path(raw_root).expanduser().resolve()
        except OSError:
            continue
        seen.setdefault((str(resolved).casefold(), str(resolved)), resolved)
    return sorted(seen.values(), key=lambda p: (str(p).casefold(), str(p)))


def display_root_names(roots: Sequence[Path]) -> tuple[str, ...]:
    """Stable, *globally unique* root-name prefixes for multi-root manifests.

    Single root => no prefix (manifest paths are plain repo-relative paths, the
    same spelling ``source_inspect_many`` accepts).  Multiple roots => every
    root gets a prefix, reserved one at a time in canonical-root order so the
    resulting prefix set is a set (no duplicates):

    - the first root to want a natural name keeps it;
    - later roots that collide with an already-reserved name get a ``-2`` /
      ``-3`` / ... suffix, and the suffix keeps incrementing until it collides
      with neither an already-reserved prefix *nor any other root's natural
      name* (a generated ``checkout-2`` never shadows a real ``checkout-2``
      root).

    Determinism: identical for a given root list regardless of how the caller
    ordered it (roots arrive canonical-sorted from the inventory).
    """
    if len(roots) <= 1:
        return ()
    naturals = [normalize_name(root.name) or f"root{index}" for index, root in enumerate(roots)]
    natural_set = set(naturals)
    reserved: set[str] = set()
    result: list[str] = []
    for natural in naturals:
        chosen = natural
        if chosen in reserved:
            suffix = 2
            candidate = f"{chosen}-{suffix}"
            while candidate in reserved or candidate in natural_set:
                suffix += 1
                candidate = f"{chosen}-{suffix}"
            chosen = candidate
        reserved.add(chosen)
        result.append(chosen)
    return tuple(result)


def display_rel(root_names: tuple[str, ...], root_index: int, rel: str) -> str:
    """Manifest spelling of a within-root rel path."""
    if not root_names:
        return rel
    return f"{root_names[root_index]}/{rel}"
