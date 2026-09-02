"""Ignore-layer 1 — curated hard exclusions.

This is the *exclusion* layer: it decides, by name, what is never part of the
source inventory.  It is deliberately separate from:

- ignore-layer 2 (``.gitignore`` semantics — see ``gitignore.py``): gitignore
  patterns can never re-include something this layer drops (a negation rule
  like ``!node_modules/`` is ignored for curated directories);
- ignore-layer 3 (source classification — see ``classify.py``): classification
  is a weight-policy decision that runs *after* this layer, and it consults
  this layer's results (files under a curated directory never reach the
  classifier).

Directory names come from the single canonical source
``strix.tools.source_inspect.ignore_dirs.IGNORE_DIRS`` (shared with the
``source_inspect_many`` search walker — never forked).  File-level artifact
rules below are the partitioner's own extension of that dir-only set, matching
the same intent (lockfiles, minified bundles, bytecode, source maps).
"""

from __future__ import annotations

from strix.tools.source_inspect.ignore_dirs import IGNORE_DIRS


__all__ = [
    "CURATED_DIR_NAMES",
    "CURATED_FILE_NAMES",
    "CURATED_FILE_SUFFIXES",
    "is_curated_dir_name",
    "is_curated_file_name",
]

#: Directory basenames pruned wherever they appear (case-folded at compare
#: time).  Hard: no gitignore negation can resurrect them.
CURATED_DIR_NAMES: frozenset[str] = IGNORE_DIRS

#: Exact filenames (case-insensitive) treated as lock/checksum artifacts.
CURATED_FILE_NAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "uv.lock",
        "poetry.lock",
        "cargo.lock",
        "gemfile.lock",
        "composer.lock",
        "pipfile.lock",
        "go.sum",
        "podfile.lock",
        "mix.lock",
        "pubspec.lock",
        "deno.lock",
    }
)

#: Filename suffixes treated as build/generated artifacts (beyond the generic
#: binary sniff, these are textual but never meaningful partition source).
CURATED_FILE_SUFFIXES: frozenset[str] = frozenset(
    {".lock", ".pyc", ".pyo", ".map", ".min.js", ".min.css"}
)


def is_curated_dir_name(name: str) -> bool:
    """True when a directory/file basename must be hard-excluded by name."""
    return name.casefold() in CURATED_DIR_NAMES


def is_curated_file_name(name: str) -> bool:
    """True when a file basename is a curated artifact (lock/minified/bytecode)."""
    folded = name.casefold()
    if folded in CURATED_FILE_NAMES:
        return True
    return any(folded.endswith(suffix) for suffix in CURATED_FILE_SUFFIXES)
