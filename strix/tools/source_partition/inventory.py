"""Deterministic source inventory - the walking + ignore pipeline.

Phase 1 of source partitioning (see ``__init__`` for the full pipeline).  Pure
deterministic code, no agents, no LLM, no report/telemetry coupling.

Ignore handling is three separate layers, applied in a fixed order:

1. curated hard exclusions by name (``exclusions``) - VCS metadata,
   dependency/build/artifact trees, lock/minified files; a ``.gitignore``
   negation can never re-include these;
2. ``.gitignore`` semantics (``gitignore``) - real matcher applied at every
   directory with nested files, negation and anchored/unanchored patterns;
3. binary/asset skip + source classification (``classify``) - files that
   survive layers 1-2 are sniffed for binary content (extension hint second)
   and classified; classification is a weight-policy input, not an exclusion.

Determinism contract:

- roots are canonicalized (resolved, deduplicated, sorted) so argument order
  never matters; nested roots are dropped (the parent already covers them);
- directories and files are visited in case-folded lexicographic order;
- symlinked directories are never followed (cycle-safe by construction);
  symlinked files are followed only when their target stays inside the same
  root and is not a duplicate of an already-visited inode/realpath (the first
  path in sorted order wins, later aliases are noted and skipped);
- every path is NFC-normalized with forward slashes; ordering is
  ``(casefold(path), path)``;
- files that cannot be read (or are larger than ``max_file_bytes``) are
  skipped with a structured warning; they never abort the walk.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from strix.tools.source_partition.classify import classify_file
from strix.tools.source_partition.exclusions import (
    is_curated_dir_name,
    is_curated_file_name,
)
from strix.tools.source_partition.gitignore import (
    GitignoreFile,
    chain_decides,
    parse_gitignore,
)
from strix.tools.source_partition.models import (
    InventoryEntry,
    PartitionConfig,
    SourceInventory,
)
from strix.tools.source_partition.normalize import (
    canonical_roots,
    path_sort_key,
    rel_from_parts,
)
from strix.tools.source_partition.readio import (
    SNIFF_BYTES,
    decode_text,
    has_binary_extension,
    is_binary_head,
    read_bytes,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["inventory_source"]

logger = logging.getLogger("strix.tools.source_partition.inventory")


def _within(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


class _Walker:
    """Per-root walk state (kept small and deterministic).

    A plain class (not a frozen dataclass) on purpose: it is a mutable
    accumulator whose fields are set before use, and type checkers resolve it
    without indirection.
    """

    def __init__(
        self,
        root: Path,
        root_index: int,
        cfg: PartitionConfig,
        entries: list[InventoryEntry],
        notes: list[str],
    ) -> None:
        self.root = root
        self.root_index = root_index
        self.cfg = cfg
        self.entries = entries
        self.notes = notes
        self.visited: set[tuple[int, int] | str] = set()

    def note(self, message: str) -> None:
        self.notes.append(message)
        logger.warning("source_partition: %s", message)

    def read_scope(self, dir_path: Path, rel_dir: str) -> GitignoreFile | None:
        ignore_file = dir_path / ".gitignore"
        if not ignore_file.is_file():
            return None
        try:
            content = read_bytes(ignore_file)
        except OSError as exc:
            self.note(f"skipping unreadable .gitignore in {rel_dir or '.'}: {exc}")
            return None
        return parse_gitignore(decode_text(content), base_rel=rel_dir)

    def _name_excluded(self, name: str, rel: str, scopes: tuple[GitignoreFile, ...]) -> bool:
        """Layer 1 (curated) + layer 2 (gitignore) verdict for a file name."""
        if is_curated_file_name(name) or is_curated_dir_name(name):
            logger.debug("source_partition: skip curated file %r", rel)
            return True
        if chain_decides(scopes, rel, is_dir=False) is True:
            logger.debug("source_partition: gitignored file %r", rel)
            return True
        return False

    def _resolve_file(self, full: Path, rel: str) -> tuple[Path, os.stat_result] | None:
        """Resolve symlinks and stat; returns ``(read_path, stat)`` or ``None``
        (with a note) when the file cannot be read from."""
        try:
            link_stat = full.lstat()
        except OSError as exc:
            self.note(f"skipping file {rel!r}: cannot stat: {exc}")
            return None
        read_path = full
        if stat.S_ISLNK(link_stat.st_mode):
            try:
                target = full.resolve(strict=True)
            except OSError as exc:
                self.note(f"skipping dangling symlink {rel!r}: {exc}")
                return None
            if not _within(target, self.root):
                self.note(f"skipping symlink {rel!r}: target outside root")
                return None
            if not target.is_file():
                self.note(f"skipping symlink {rel!r}: target is not a file")
                return None
            read_path = target
        try:
            stat_result = read_path.stat()
        except OSError as exc:
            self.note(f"skipping file {rel!r}: cannot stat: {exc}")
            return None
        return read_path, stat_result

    def _already_seen(self, read_path: Path, stat_result: os.stat_result) -> bool:
        """Dedupe hardlinks/symlink aliases; first sorted path wins."""
        dedupe_key: tuple[int, int] | str = (
            (stat_result.st_dev, stat_result.st_ino)
            if stat_result.st_ino
            else str(read_path).casefold()
        )
        if dedupe_key in self.visited:
            logger.debug("source_partition: skip duplicate path %r", str(read_path))
            return True
        self.visited.add(dedupe_key)
        return False

    def _is_binary(self, name: str, read_path: Path, rel: str) -> bool:
        """Extension hint, then content sniff (sniffing is authoritative)."""
        if has_binary_extension(name):
            logger.debug("source_partition: skip asset/binary file %r", rel)
            return True
        try:
            head = read_bytes(read_path, limit=SNIFF_BYTES)
        except OSError as exc:
            self.note(f"skipping unreadable file {rel!r}: {exc}")
            return True
        if is_binary_head(head):
            logger.debug("source_partition: skip binary content in %r", rel)
            return True
        return False

    def process_file(
        self,
        dir_path: Path,
        name: str,
        io_parts: tuple[str, ...],
        scopes: tuple[GitignoreFile, ...],
    ) -> None:
        io_entry_parts = (*io_parts, name)
        rel = rel_from_parts(io_entry_parts)
        if rel is None or self._name_excluded(name, rel, scopes):
            return
        full = dir_path / name
        resolved = self._resolve_file(full, rel)
        if resolved is None:
            return
        read_path, stat_result = resolved
        size = stat_result.st_size
        if size == 0:
            logger.debug("source_partition: skip empty file %r", rel)
            return
        if size > self.cfg.max_file_bytes:
            self.note(f"skipping oversized file {rel!r}: {size} bytes")
            return
        if self._already_seen(read_path, stat_result):
            return
        if self._is_binary(name, read_path, rel):
            return
        kind = classify_file(rel)
        self.entries.append(
            InventoryEntry(
                root_index=self.root_index,
                rel=rel,
                io_parts=io_entry_parts,
                kind=kind,
                size_bytes=size,
            )
        )


def _walk_root(walker: _Walker) -> None:
    root_str = str(walker.root)
    # f_chain[rel] = tuple of .gitignore scopes that apply to entries *inside*
    # rel (root..rel inclusive).  Top-down walk => a dir's parent chain is
    # always stored before the dir itself is visited.
    f_chain: dict[str, tuple[GitignoreFile, ...]] = {}

    def on_error(exc: OSError) -> None:
        walker.note(f"cannot list directory during walk: {exc}")

    for dir_path_str, dirnames, filenames in os.walk(
        root_str, topdown=True, onerror=on_error, followlinks=False
    ):
        dir_path = Path(dir_path_str)
        io_dir_parts = dir_path.relative_to(walker.root).parts
        rel_dir = rel_from_parts(io_dir_parts) or ""
        if "/" in rel_dir:
            parent_chain = f_chain[rel_dir.rsplit("/", 1)[0]]
        elif rel_dir:
            parent_chain = f_chain[""]
        else:
            parent_chain = ()
        scope = walker.read_scope(dir_path, rel_dir)
        entry_chain = parent_chain + ((scope,) if scope is not None else ())
        f_chain[rel_dir] = entry_chain

        kept: list[str] = []
        for dir_name in sorted(dirnames, key=path_sort_key):
            child_rel = rel_from_parts((*io_dir_parts, dir_name))
            if child_rel is None:
                continue
            if is_curated_dir_name(dir_name):
                logger.debug("source_partition: skip curated dir %r", child_rel)
                continue
            if (dir_path / dir_name).is_symlink():
                logger.debug("source_partition: not following symlinked dir %r", child_rel)
                continue
            if chain_decides(entry_chain, child_rel, is_dir=True) is True:
                logger.debug("source_partition: gitignored dir %r", child_rel)
                continue
            kept.append(dir_name)
        dirnames[:] = kept
        for file_name in sorted(filenames, key=path_sort_key):
            walker.process_file(dir_path, file_name, io_dir_parts, entry_chain)


def inventory_source(
    roots: Sequence[Path],
    *,
    config: PartitionConfig | None = None,
) -> SourceInventory:
    """Walk ``roots`` and return the deterministic source inventory.

    ``roots`` are canonicalized first (resolve + dedupe + sort); argument
    order therefore never affects the result.  Non-directory and nested roots
    are dropped with a note.
    """
    cfg = config or PartitionConfig()
    notes: list[str] = []
    canonical = canonical_roots(roots)
    kept_roots: list[Path] = []
    for candidate in canonical:
        if not candidate.is_dir():
            notes.append(f"skipping non-directory root {str(candidate)!r}")
            logger.warning("source_partition: %s", notes[-1])
            continue
        if any(_within(candidate, earlier) for earlier in kept_roots):
            notes.append(f"skipping root {str(candidate)!r}: nested inside another root")
            logger.warning("source_partition: %s", notes[-1])
            continue
        kept_roots.append(candidate)

    entries: list[InventoryEntry] = []
    for root_index, root in enumerate(kept_roots):
        walker = _Walker(root=root, root_index=root_index, cfg=cfg, entries=entries, notes=notes)
        _walk_root(walker)

    entries.sort(key=lambda entry: path_sort_key(entry.rel))
    return SourceInventory(
        roots=tuple(kept_roots),
        entries=tuple(entries),
        notes=tuple(notes),
    )
