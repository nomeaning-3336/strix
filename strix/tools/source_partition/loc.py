"""Effective-LOC counting — the main weight signal for partitioning.

Per-extension rule, deliberately conservative (per the v1 spec):

- blank lines never count;
- *light* comment stripping (line comments plus non-nested block comments) is
  applied only to the major languages — Python, JavaScript/TypeScript, Go,
  Rust, Java.  Everything else strips blanks only, so a comment line in an
  unlisted language still counts as a meaningful line (better to over-count
  than to guess a comment syntax wrong);
- no string-literal awareness: a ``//`` or ``#`` inside a string is treated as
  a comment marker for the remainder of that line.  This only ever
  under-counts, deterministically, and is the documented limit of "light"
  stripping;
- Python docstrings (``\"\"\"`` / ``'''``) are treated as block comments.

The counted value is *meaningful lines*, stored as the unit's ``loc``; the
shard-level ``weight`` is the weighted sum used for balancing.
"""

from __future__ import annotations


__all__ = ["count_loc", "language_for"]

#: suffix → profile name.  Profiles with markers get light comment stripping;
#: everything else falls back to ``"text"`` (blank-line-only counting).
_LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    ".js": "c_style",
    ".jsx": "c_style",
    ".mjs": "c_style",
    ".cjs": "c_style",
    ".ts": "c_style",
    ".tsx": "c_style",
    ".mts": "c_style",
    ".cts": "c_style",
    ".go": "c_style",
    ".rs": "c_style",
    ".java": "c_style",
}

_LINE_COMMENT_MARKERS: dict[str, tuple[str, ...]] = {
    "python": ("#",),
    "c_style": ("//",),
    "text": (),
}

_BLOCK_COMMENT_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "python": (('"""', '"""'), ("'''", "'''")),
    "c_style": (("/*", "*/"),),
    "text": (),
}


def language_for(suffix: str) -> str:
    """Map a lower-cased suffix (with dot) to a counting profile."""
    return _LANGUAGE_BY_SUFFIX.get(suffix, "text")


def _scan_line(
    line: str,
    markers: tuple[str, ...],
    pairs: tuple[tuple[str, str], ...],
    in_block: bool,
) -> tuple[bool, bool]:
    meaningful = False
    index = 0
    length = len(line)
    while index < length:
        if in_block:
            closed = False
            for _start, end in pairs:
                if line.startswith(end, index):
                    in_block = False
                    index += len(end)
                    closed = True
                    break
            if not closed:
                index += 1
            continue
        opened = False
        for start, _end in pairs:
            if line.startswith(start, index):
                in_block = True
                index += len(start)
                opened = True
                break
        if opened:
            continue
        if any(line.startswith(marker, index) for marker in markers):
            break
        if not line[index].isspace():
            meaningful = True
        index += 1
    return meaningful, in_block


def count_loc(text: str, language: str) -> int:
    """Count meaningful (non-blank, lightly comment-stripped) lines."""
    markers = _LINE_COMMENT_MARKERS.get(language, ())
    pairs = _BLOCK_COMMENT_PAIRS.get(language, ())
    in_block = False
    count = 0
    for raw_line in text.splitlines():
        meaningful, in_block = _scan_line(raw_line, markers, pairs, in_block)
        if meaningful:
            count += 1
    return count
