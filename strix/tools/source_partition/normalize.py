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
from collections import Counter
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
    """Stable root-name prefixes for multi-root manifests.

    Single root ⇒ no prefix (manifest paths are plain repo-relative paths, the
    same spelling ``source_inspect_many`` accepts).  Multiple roots ⇒ each path
    is prefixed with its root's directory name so files from different roots
    can never collide; duplicate root names get deterministic ``-2``/``-3``
    suffixes in canonical-root order.
    """
    if len(roots) <= 1:
        return ()
    base_names = [normalize_name(root.name) or f"root{index}" for index, root in enumerate(roots)]
    counts = Counter(base_names)
    if all(count == 1 for count in counts.values()):
        return tuple(base_names)
    occurrences: dict[str, int] = {}
    result: list[str] = []
    for name in base_names:
        occurrences[name] = occurrences.get(name, 0) + 1
        result.append(name if occurrences[name] == 1 else f"{name}-{occurrences[name]}")
    return tuple(result)


def display_rel(root_names: tuple[str, ...], root_index: int, rel: str) -> str:
    """Manifest spelling of a within-root rel path."""
    if not root_names:
        return rel
    return f"{root_names[root_index]}/{rel}"
