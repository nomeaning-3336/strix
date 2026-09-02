"""Ignore-layer 2 - a real gitignore-style matcher (pure, deterministic).

This is a *matcher*, not an ignore-name list: patterns are parsed from
``.gitignore`` files at every directory level and applied with git semantics.
Layer 1 (curated hard exclusions, ``exclusions.py``) runs before this layer and
can never be overridden by a negation here; layer 3 (classification,
``classify.py``) runs after and only ever sees surviving paths.

Implemented gitignore semantics (documented subset):

- one pattern per line; blank lines and ``#`` comments are ignored (a leading
  ``\\`` escapes the first character, so ``\\#foo`` / ``\\!bar`` are literal);
- a leading ``!`` negates the pattern - the *last* matching pattern wins,
  across rules of a single file (root-to-leaf rule order across files);
- a trailing ``/`` restricts the pattern to directories;
- a leading ``/`` anchors the pattern to the directory of the ``.gitignore``;
  any pattern containing a ``/`` elsewhere is anchored as well;
- patterns without any ``/`` match the basename at any depth below the
  ``.gitignore`` location;
- globs: ``*`` and ``?`` never cross ``/``; ``[...]`` character classes (``!``
  means negation inside a class); ``**`` has git's three special forms -
  leading ``**/`` (any depth prefix), trailing ``/**`` (everything inside, and
  the directory itself), and ``/**/`` (zero or more directories in the
  middle); any other ``**`` behaves like ``*``;
- a file whose parent directory is excluded cannot be re-included - directory
  pruning happens during the walk, so files under an excluded directory are
  never evaluated (mirrors git);
- matching is case-sensitive on NFC-normalized paths (case folding is applied
  only to the *curated* containment checks and to ordering, never to pattern
  matching - that keeps results identical on case-insensitive filesystems).

Trailing unescaped spaces are trimmed; backslash escapes inside patterns are
honored for the next character only.  These are the conservative limits of the
subset: anything outside it is treated literally rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["GitignoreFile", "chain_decides", "parse_gitignore"]


@dataclass(frozen=True, slots=True)
class _Rule:
    negated: bool
    dir_only: bool
    anchored: bool
    regex: re.Pattern[str]

    def matches(self, rel_to_base: str, is_dir: bool) -> bool:
        if self.dir_only and not is_dir:
            return False
        if self.anchored:
            return self.regex.fullmatch(rel_to_base) is not None
        basename = rel_to_base.rsplit("/", 1)[-1]
        return self.regex.fullmatch(basename) is not None


@dataclass(frozen=True, slots=True)
class GitignoreFile:
    """Patterns parsed from one ``.gitignore`` file.

    ``base_rel`` is the normalized path of the directory containing the file,
    relative to the walk root (``""`` for the root itself); its patterns apply
    to everything *below* that directory.
    """

    base_rel: str
    rules: tuple[_Rule, ...]

    def relative(self, rel_from_root: str) -> str | None:
        """Relativize a root-relative candidate to this scope, or ``None``."""
        if not self.base_rel:
            return rel_from_root
        if rel_from_root == self.base_rel:
            return ""
        if rel_from_root.startswith(self.base_rel + "/"):
            return rel_from_root[len(self.base_rel) + 1 :]
        return None

    def decide(self, rel_to_base: str, is_dir: bool) -> bool | None:
        """Last matching rule wins: ``True`` = ignored, ``False`` = kept
        (negated), ``None`` = this file has no opinion."""
        decision: bool | None = None
        for rule in self.rules:
            if rule.matches(rel_to_base, is_dir):
                decision = not rule.negated
        return decision


def _char_class(segment: str, index: int) -> tuple[str, int] | None:
    """Regex body + next index for a ``[...]`` class at ``index``.

    Returns ``None`` for an unterminated class (git treats it as a literal
    ``[``).  A leading ``!`` inside the class negates it.
    """
    end = index + 1
    if end < len(segment) and segment[end] in ("!", "^"):
        end += 1
    if end < len(segment) and segment[end] == "]":
        end += 1
    while end < len(segment) and segment[end] != "]":
        end += 1
    if end >= len(segment):
        return None
    body = segment[index + 1 : end]
    if body.startswith("!"):
        body = "^" + body[1:]
    return "[" + body.replace("\\", "\\\\") + "]", end + 1


def _segment_regex(segment: str) -> str:
    """One path segment (never spans ``/``).  ``**`` inside a segment is just
    ``*`` (git: "other consecutive asterisks are considered regular
    asterisks")."""
    out: list[str] = []
    index = 0
    length = len(segment)
    while index < length:
        char = segment[index]
        if char == "\\" and index + 1 < length:
            out.append(re.escape(segment[index + 1]))
            index += 2
        elif char == "*":
            while index < length and segment[index] == "*":
                index += 1
            out.append(r"[^/]*")
        elif char == "?":
            out.append(r"[^/]")
            index += 1
        elif char == "[":
            parsed = _char_class(segment, index)
            if parsed is None:
                out.append(re.escape("["))
                index += 1
            else:
                body, index = parsed
                out.append(body)
        else:
            out.append(re.escape(char))
            index += 1
    return "".join(out)


def _full_path_regex(pattern: str) -> str:
    """Regex (body) matching an anchored pattern against a full relative path."""
    if pattern == "**":
        return ".*"
    segments = pattern.split("/")
    leading = segments[0] == "**"
    trailing = len(segments) > 1 and segments[-1] == "**"
    if leading:
        segments = segments[1:]
    if trailing:
        segments = segments[:-1]
    parts: list[str | None] = [_segment_regex(seg) if seg != "**" else None for seg in segments]
    body = ""
    for part in parts:
        if part is None:  # middle '**': zero or more directories
            body += "(?:/.*)?"
        elif body:
            body += "/" + part
        else:
            body = part
    if leading:
        body = "(?:.*/)?" + body
    if trailing:
        body += "(?:/.*)?"
    return body


def parse_gitignore(text: str, base_rel: str) -> GitignoreFile:
    """Parse ``.gitignore`` content into a deterministic rule set."""
    rules: list[_Rule] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip(" \t")
        if not line:
            continue
        if line.startswith("\\"):
            line = line[1:]  # escaped leading char ('\#', '\!', '\/'...)
        elif line.startswith("#"):
            continue
        negated = False
        if line.startswith("!"):
            negated = True
            line = line[1:]
        if not line:
            continue
        dir_only = line.endswith("/")
        pattern = line.rstrip("/")
        if not pattern:
            continue  # matches only the scope directory itself
        anchored = pattern.startswith("/")
        pattern = pattern[1:] if anchored else pattern
        if not pattern:
            continue
        if "/" in pattern:
            anchored = True
        if pattern == "**":
            body = ".*"
        elif anchored:
            body = _full_path_regex(pattern)
        else:
            body = _segment_regex(pattern)
        rules.append(
            _Rule(
                negated=negated,
                dir_only=dir_only,
                anchored=anchored,
                regex=re.compile(f"^{body}$"),
            )
        )
    return GitignoreFile(base_rel=base_rel, rules=tuple(rules))


def chain_decides(scopes: Sequence[GitignoreFile], rel_from_root: str, is_dir: bool) -> bool | None:
    """Apply the scope chain (root -> deepest) to one candidate path.

    Deeper scopes are evaluated later, so their verdicts override shallower
    ones - git precedence for nested ``.gitignore`` files.
    """
    decision: bool | None = None
    for scope in scopes:
        rel_to_base = scope.relative(rel_from_root)
        if rel_to_base is None:
            continue
        verdict = scope.decide(rel_to_base, is_dir)
        if verdict is not None:
            decision = verdict
    return decision
