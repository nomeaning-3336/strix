"""Materialize writable, symlink-safe copies of user-owned source trees."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    root: Path,
    excluded: tuple[Path, ...],
    seen: frozenset[Path],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with os.scandir(source) as entries:
        for entry in entries:
            src = Path(entry.path)
            dst = destination / entry.name
            resolved = src.resolve(strict=False)
            if any(_is_within(resolved, blocked) for blocked in excluded):
                continue
            if entry.name == "strix_runs" and entry.is_dir(follow_symlinks=False):
                continue
            if entry.is_symlink():
                target = src.resolve(strict=False)
                if not target.exists() or not _is_within(target, root) or target in seen:
                    logger.warning("isolated workspace: dropping unsafe symlink %s", src)
                    continue
                if target.is_dir():
                    _copy_tree(
                        target,
                        dst,
                        root=root,
                        excluded=excluded,
                        seen=seen | {target},
                    )
                elif target.is_file():
                    shutil.copy2(target, dst)
                continue
            if entry.is_dir(follow_symlinks=False):
                _copy_tree(src, dst, root=root, excluded=excluded, seen=seen)
            elif entry.is_file(follow_symlinks=False):
                # Never hard-link: the destination is intentionally writable.
                shutil.copy2(src, dst)


def materialize_isolated_sources(
    local_sources: list[dict[str, Any]],
    *,
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Replace user-owned live mounts with durable per-run writable copies."""
    workspace_root = run_dir / ".state" / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    for source in local_sources:
        item = dict(source)
        if not item.get("protect_metadata"):
            result.append(item)
            continue
        source_path = Path(str(item.get("source_path") or "")).expanduser().resolve()
        subdir = str(item.get("workspace_subdir") or "workspace")
        destination = (workspace_root / subdir).resolve()
        complete_marker = workspace_root / f".{subdir}.complete"
        if destination.exists() and not complete_marker.is_file():
            shutil.rmtree(destination, ignore_errors=True)
        if not destination.exists():
            try:
                _copy_tree(
                    source_path,
                    destination,
                    root=source_path,
                    excluded=(run_dir.resolve(), destination),
                    seen=frozenset({source_path}),
                )
            except Exception:
                shutil.rmtree(destination, ignore_errors=True)
                complete_marker.unlink(missing_ok=True)
                raise
            complete_marker.write_text(str(source_path), encoding="utf-8")
            logger.info("materialized isolated workspace %s -> %s", source_path, destination)
        item["original_source_path"] = str(source_path)
        item["source_path"] = str(destination)
        item["workspace_mode"] = "isolated_copy"
        # `protect_metadata` is deliberately preserved: the copy's `.git`, `.agents`, and
        # `.codex` still stay read-only. They are agent-instruction and repository state
        # that persist across `--resume`, so a run that ingested injected target content
        # must not be able to rewrite them.
        result.append(item)
    return result
