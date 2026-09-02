"""Deterministic ``source_inspect_many`` tool (WideTurn / Efficiency v1 item b).

One model decision + one tool round trip performs several read-only source
operations — file slice reads and regex searches — concurrently on the
authorized host checkout(s), returning a single structured result.
Independent per-operation errors, deterministic ordering (input order; files
sorted), bounded concurrency, and hard output caps so a single call never
floods the context.

Only local-source / repository scans can use it (roots are the
``local_sources[].source_path`` / ``cloned_repo_path`` checkouts). URL/DAST
scans get a clear "use exec_command instead" error, never a silent no-op.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents import function_tool

from strix.tools.source_inspect.ignore_dirs import IGNORE_DIRS as _IGNORE_DIRS


if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# --- deterministic constants -------------------------------------------------

# Curated hard-exclusion directory names, kept identical to the search walker's
# prune set (single source of truth in strix/tools/source_inspect/ignore_dirs.py).

_READ_MAX_CHARS = 24_000
_READ_MAX_LINES = 1_500
_SEARCH_MAX_MATCHES = 60
_SEARCH_MAX_FILE_BYTES = 2_000_000
_OP_CONCURRENCY = 4
_MAX_READS = 12
_MAX_SEARCHES = 8


# --- root resolution (authorized host checkouts) ------------------------------


def resolve_authorized_roots(report_state: Any) -> list[Path]:
    """Host directories this scan may read: local sources + repo checkouts.

    Preserves order and deduplicates so op-path resolution is deterministic
    (first root containing the requested path wins).
    """
    roots: list[Path] = []
    seen: set[str] = set()

    def _add(candidate: Any) -> None:
        if isinstance(candidate, dict):
            candidate = candidate.get("source_path")
        if not isinstance(candidate, str) or not candidate.strip():
            return
        try:
            resolved = Path(candidate).expanduser().resolve()
        except OSError:
            return
        key = str(resolved).lower()
        if key in seen or not resolved.is_dir():
            return
        seen.add(key)
        roots.append(resolved)

    run_record = getattr(report_state, "run_record", None) or {}
    for source in run_record.get("local_sources") or []:
        _add(source)
    for target in run_record.get("targets_info") or []:
        if isinstance(target, dict) and target.get("type") == "repository":
            details = target.get("details") or {}
            if isinstance(details, dict):
                _add(details.get("cloned_repo_path"))
    return roots


def map_operation_path(raw: str, roots: list[Path]) -> Path | None:
    """Resolve an agent-supplied path against the authorized roots.

    Accepts bare repo-relative paths (``src/server/GameServer.ts``), sandbox
    paths (``/workspace/<subdir>/src/...`` — the subdir segment is dropped when
    it names one of the roots), and host-absolute paths that already live under
    a root. Returns None for anything that escapes every root.
    """
    text = (raw or "").strip().replace("\\", "/")
    if not text or "\x00" in text:
        return None

    result: Path | None = None
    if text.startswith("/workspace/"):
        result = _resolve_workspace_path(text[len("/workspace/") :], roots)
    elif text.startswith("/"):
        result = _resolve_host_absolute(text, roots)
    else:
        result = _resolve_within_roots(text, roots)
    return result


def _resolve_workspace_path(remainder: str, roots: list[Path]) -> Path | None:
    parts = remainder.split("/")
    variants: list[str] = []
    if not remainder:
        return None
    base_names = {root.name for root in roots}
    if len(parts) > 1 and parts[0] in base_names:
        # /workspace/<root-name>/rest → rest (the subdir names one of our roots,
        # and trying it first matches how the mount is laid out).
        variants.append("/".join(parts[1:]))
        variants.append(remainder)
    else:
        variants.append(remainder)
    for rel in variants:
        resolved = _resolve_within_roots(rel, roots)
        if resolved is not None:
            return resolved
    return None


def _resolve_host_absolute(text: str, roots: list[Path]) -> Path | None:
    try:
        resolved_abs = Path(text).resolve()
    except OSError:
        return None
    for root in roots:
        try:
            resolved_abs.relative_to(root)
        except ValueError:
            continue
        return resolved_abs
    return None


def _resolve_within_roots(rel: str, roots: list[Path]) -> Path | None:
    rel = rel.strip("/")
    if not rel or ".." in Path(rel).parts:
        return None
    for root in roots:
        try:
            candidate = (root / rel).resolve()
        except OSError:
            continue
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        return candidate
    return None


# --- reads --------------------------------------------------------------------


def _read_operation(path: Path, op: dict[str, Any]) -> dict[str, Any]:
    start_line = 1
    end_line: int | None = None
    max_chars = _READ_MAX_CHARS
    try:
        start_line = max(1, int(op.get("start_line") or 1))
        end_raw = op.get("end_line")
        if end_raw is not None:
            end_line = max(start_line, int(end_raw))
        raw_max = op.get("max_chars")
        if raw_max is not None:
            max_chars = min(max(1, int(raw_max)), _READ_MAX_CHARS)
    except (TypeError, ValueError):
        return {"ok": False, "error": "start_line/end_line/max_chars must be integers"}

    try:
        data = path.read_bytes()
    except OSError as exc:
        return {"ok": False, "error": f"cannot read {path.name}: {exc.strerror or exc}"}

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"decode failed: {exc}"}
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return {"ok": True, "path": str(path), "lines_total": 0, "content": "", "truncated": False}
    start_index = min(start_line - 1, total - 1)
    selected = (
        lines[start_index:]
        if end_line is None
        else lines[start_index : min(end_line, total)]
    )
    content = "\n".join(selected)
    over_chars = len(content) > max_chars
    over_lines = len(selected) > _READ_MAX_LINES
    content = content[:max_chars]
    truncated = bool(over_chars or over_lines)
    if truncated:
        content = f"{content}\n...[truncated]"
    result: dict[str, Any] = {
        "ok": True,
        "path": str(path),
        "start_line": start_index + 1,
        "lines_shown": len(selected),
        "lines_total": total,
        "content": content,
        "truncated": truncated,
    }
    # An explicit end_line is a requested slice, not truncation: the model
    # chose the window, so no marker is appended — just note more lines exist.
    if end_line is not None and end_line < total:
        result["note"] = f"more lines available below line {end_line} (total {total})"
    return result


# --- searches -----------------------------------------------------------------


def _iter_source_files(directory: Path) -> Iterable[Path]:
    for root_dir, dirnames, filenames in directory.walk(follow_symlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORE_DIRS)
        for filename in sorted(filenames):
            path = root_dir / filename
            if path.suffix.lower() in {
                ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
                ".woff", ".woff2", ".zip", ".gz", ".lock", ".min.js", ".min.css",
            }:
                continue
            yield path


def _search_file(path: Path, compiled: re.Pattern[str], rel_base: Path) -> list[dict[str, Any]]:
    try:
        if path.stat().st_size > _SEARCH_MAX_FILE_BYTES:
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    matches: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, start=1):
        if compiled.search(line):
            matches.append(
                {
                    "file": str(path.relative_to(rel_base)).replace("\\", "/"),
                    "line": lineno,
                    "text": line.strip()[:300],
                }
            )
    return matches


def _search_directory(
    root: Path,
    compiled: re.Pattern[str],
    include_re: re.Pattern[str] | None,
    *,
    max_matches: int,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    scanned_files = 0
    truncated = False
    for path in _iter_source_files(root):
        if len(matches) >= max_matches:
            truncated = True
            break
        rel = path.relative_to(root)
        if include_re is not None and not include_re.search(str(rel).replace("\\", "/")):
            continue
        scanned_files += 1
        for match in _search_file(path, compiled, root):
            if len(matches) >= max_matches:
                truncated = True
                break
            matches.append(match)
        if truncated:
            break
    return {"ok": True, "scanned_files": scanned_files, "matches": matches, "truncated": truncated}


def _search_operation(path: Path, op: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    pattern = op.get("pattern")
    compiled: re.Pattern[str] | None = None
    if not isinstance(pattern, str) or not pattern.strip():
        errors.append("pattern is required")
    else:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            errors.append(f"invalid regex: {pattern!r} ({exc})")
    try:
        max_matches = min(
            max(1, int(op.get("max_matches") or _SEARCH_MAX_MATCHES)),
            _SEARCH_MAX_MATCHES,
        )
    except (TypeError, ValueError):
        errors.append("max_matches must be an integer")
    include_re: re.Pattern[str] | None = None
    include = op.get("include")
    if include:
        try:
            include_re = re.compile(str(include))
        except re.error as exc:
            errors.append(f"invalid include regex: {exc}")
    if errors:
        return {"ok": False, "error": "; ".join(errors)}
    assert compiled is not None

    if path.is_file():
        return {
            "ok": True,
            "scanned_files": 1,
            "matches": _search_file(path, compiled, path.parent),
            "truncated": False,
        }
    if not path.is_dir():
        return {"ok": False, "error": f"search path not found: {op.get('path')}"}
    return _search_directory(path, compiled, include_re, max_matches=max_matches)


# --- bounded, ordered batch execution -----------------------------------------


async def _run_batch(
    kind: str,
    ops: list[dict[str, Any]],
    roots: list[Path],
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(_OP_CONCURRENCY)

    async def run(op: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(_dispatch, kind, op, roots)

    return list(await asyncio.gather(*(run(op) for op in ops)))


def _dispatch(kind: str, op: dict[str, Any], roots: list[Path]) -> dict[str, Any]:
    raw_path = str(op.get("path") or "")
    path = map_operation_path(raw_path, roots)
    if path is None:
        return {
            "ok": False,
            "error": (
                f"path {raw_path!r} is outside the authorized source roots; "
                "use a path relative to the source checkout (e.g. src/…)"
            ),
        }
    return _read_operation(path, op) if kind == "read" else _search_operation(path, op)


async def do_source_inspect_many(
    *,
    reads: list[dict[str, Any]],
    searches: list[dict[str, Any]],
    roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Run reads + searches concurrently (bounded), one structured result.

    ``roots`` (host checkouts) may be passed explicitly for tests; otherwise
    they are resolved from the current global report state by the tool wrapper.
    """
    valid_reads = [op for op in (reads or []) if isinstance(op, dict)]
    valid_searches = [op for op in (searches or []) if isinstance(op, dict)]
    notes: list[str] = []
    if len(valid_reads) > _MAX_READS:
        notes.append(f"too many reads; kept the first {_MAX_READS}")
        valid_reads = valid_reads[:_MAX_READS]
    if len(valid_searches) > _MAX_SEARCHES:
        notes.append(f"too many searches; kept the first {_MAX_SEARCHES}")
        valid_searches = valid_searches[:_MAX_SEARCHES]
    if not (valid_reads or valid_searches):
        return {
            "success": False,
            "error": "Provide at least one read or one search operation.",
            "errors": notes,
        }

    roots = list(roots or [])
    if not roots:
        return {
            "success": False,
            "error": (
                "No authorized local source root for this scan (it targets a "
                "remote URL/host). source_inspect_many reads the local checkout "
                "— for this scan use exec_command (e.g. grep/sed) instead."
            ),
            "errors": notes,
        }

    read_results = await _run_batch("read", valid_reads, roots)
    search_results = await _run_batch("search", valid_searches, roots)
    return {
        "success": True,
        "reads": read_results,
        "searches": search_results,
        "notes": notes or None,
    }


# --- agent-facing tool --------------------------------------------------------


@function_tool(timeout=120, strict_mode=False)
async def source_inspect_many(
    ctx: Any,
    reads: list[dict[str, Any]] | None = None,
    searches: list[dict[str, Any]] | None = None,
) -> str:
    """Inspect source code efficiently: batch several reads and searches in ONE call.

    Use this instead of issuing many separate grep/read shell commands when you
    need several independent pieces of source evidence at once (WideTurn): the
    harness runs the operations concurrently inside one tool call, so one model
    turn pays for many filesystem queries.

    Each entry is a dict:

      reads: [{ "path": "src/server/GameServer.ts", "start_line": 1, "end_line": 250 }]
        - path: relative to the source checkout (e.g. ``src/...``), or the
          sandbox path ``/workspace/<dir>/...``.
        - start_line/end_line: optional 1-based slice. Reads are capped
          (~24k chars / ~1.5k lines shown) and truncated content is flagged.
        - max_chars: optional per-read cap below the ceiling.

      searches: [{"pattern": "sharesBorderWith|clientID", "path": "src/core",
                  "include": ".*\\.ts$", "max_matches": 40}]
        - pattern: Python regular expression, searched per line.
        - path: directory or file to search (defaults to the checkout root).
        - include: optional regex matched against each file's path relative to
          the search root.
        - max_matches: cap on total matches returned (default 60; per line).

    Results come back as JSON in the SAME order you passed the operations, so
    ``reads[i]`` answers ``reads[i]``. Every operation is independent: a missing
    file or bad pattern fails just that entry (``ok: false``) and never aborts
    the batch. Searches skip dependency/build noise (.git, node_modules, dist,
    __pycache__, minified/binary files).

    Read-only and deterministic — safe to batch with other parallel-safe tools.
    Only available when this scan has an authorized local source checkout; for
    URL/host scans or anything needing a shell, keep using exec_command.
    """
    reads_list = list(reads or [])
    searches_list = list(searches or [])
    # Lazy: strix.skills can import this tool module during boot via the agent
    # factory; report.state must not load until a scan is actually running.
    from strix.report.state import get_global_report_state  # noqa: PLC0415

    report_state = get_global_report_state()
    roots: list[Path] = (
        resolve_authorized_roots(report_state) if report_state is not None else []
    )
    result = await do_source_inspect_many(
        reads=reads_list, searches=searches_list, roots=roots
    )
    return json.dumps(result, ensure_ascii=False, default=str)
