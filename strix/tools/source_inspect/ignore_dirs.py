"""Canonical curated hard-exclusion directory names.

Single source of truth shared by:

- ``strix.tools.source_inspect.tool`` — the ``source_inspect_many`` search
  walker prunes these directory names while scanning for matches;
- ``strix.tools.source_partition.exclusions`` — the partitioner's curated
  hard-exclusion layer (layer 1 of the ignore pipeline) prunes them during
  source inventory, before any ``.gitignore`` semantics or classification.

Only *directory names* belong here (VCS metadata, dependency/build/artifact
trees).  File-level artifact rules (lock files, minified bundles, source maps)
live in the partitioner's ``exclusions`` module, which extends this set — the
source_inspect_many walker already skips those file kinds by suffix in its own
``_iter_source_files`` filter.
"""

IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".next",
    }
)
