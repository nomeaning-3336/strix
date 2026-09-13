"""Local writeup corpus: FTS5 index build + retrieval.

The corpus is a gzipped JSONL file of public bug-bounty writeups. It ships
compressed (~2 MB for ~1900 writeups) rather than as a prebuilt SQLite file
(~10 MB), and the FTS5 index is built once into a user cache directory on first
use. Matching is BM25 over title/weakness/classes/body — deterministic, offline,
and free of model calls — which is the right default for a pentest agent that
must not spend tokens (or depend on the network) to recall prior art.

Resolution order for both files is: explicit argument, then environment
override, then the shipped default.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from strix.utils.resource_paths import get_strix_resource_path


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3
CORPUS_ENV = "STRIX_WRITEUPS_CORPUS"
INDEX_ENV = "STRIX_WRITEUPS_DB"

# Serializes first-use index construction inside this process. `search_writeups`
# is declared parallel-safe, so several agents can call it at once; without this
# lock every one of them could decide the index needs building and race on the
# same SQLite file. Not reentrant on purpose: `ensure_index` never calls itself,
# and a reentrant acquire would only hide that.
_BUILD_LOCK = threading.Lock()

# A replace can lose a race twice over: another process may have installed a
# current index (accepted, not an error), or a reader may hold the database open
# for the moment its query runs (transient). Retry briefly before surfacing.
_REPLACE_ATTEMPTS = 3
_REPLACE_RETRY_DELAY = 0.1

_COLUMNS = (
    "doc_id",
    "source",
    "title",
    "url",
    "program",
    "weakness",
    "severity",
    "bounty",
    "votes",
    "reporter",
    "asset",
    "reported",
    "disclosed",
    "resolution",
    "year",
    "classes",
    "body",
    "extra",
)

# FTS5 treats bare words as a query language; a model-supplied string can easily
# contain `"`, `-`, `:` or `*` and produce a syntax error instead of results.
_TOKEN = re.compile(r"[A-Za-z0-9_.+/-]{2,}")


def corpus_path(explicit: Path | None = None) -> Path:
    """The gzipped JSONL corpus to read."""
    if explicit is not None:
        return explicit
    override = os.environ.get(CORPUS_ENV)
    if override:
        return Path(override)
    return get_strix_resource_path("knowledge", "writeups.jsonl.gz")


def index_path(explicit: Path | None = None) -> Path:
    """Where the built SQLite index lives."""
    if explicit is not None:
        return explicit
    override = os.environ.get(INDEX_ENV)
    if override:
        return Path(override)
    return Path.home() / ".strix" / "writeups.db"


def _corpus_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fts_query(raw: str) -> str:
    """Turn free text into a safe FTS5 query of quoted, OR-joined tokens.

    Tokens are OR-ed rather than AND-ed: a natural-language query ("graphql
    introspection authorization bypass") has no document containing every term,
    and requiring all of them returned nothing. BM25 does the relevance work —
    a document matching more, rarer terms scores higher — so OR plus ranking
    beats a strict conjunction here.
    """
    tokens = _TOKEN.findall(raw or "")
    if not tokens:
        return ""
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique.append(token)
    return " OR ".join(f'"{token}"' for token in unique[:24])


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE writeups (
            doc_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL DEFAULT '',
            program TEXT NOT NULL DEFAULT '',
            weakness TEXT NOT NULL DEFAULT '',
            severity TEXT NOT NULL DEFAULT '',
            bounty TEXT NOT NULL DEFAULT '',
            votes TEXT NOT NULL DEFAULT '',
            reporter TEXT NOT NULL DEFAULT '',
            asset TEXT NOT NULL DEFAULT '',
            reported TEXT NOT NULL DEFAULT '',
            disclosed TEXT NOT NULL DEFAULT '',
            resolution TEXT NOT NULL DEFAULT '',
            year INTEGER,
            classes TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL,
            extra TEXT NOT NULL DEFAULT '{}'
        );
        CREATE VIRTUAL TABLE writeups_fts USING fts5(
            title, weakness, classes, body,
            content='writeups', content_rowid='rowid', tokenize='porter unicode61'
        );
        CREATE TABLE corpus_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )


def build_index(corpus: Path, index: Path) -> int:
    """Build ``index`` from ``corpus`` atomically; returns the writeup count."""
    index.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(suffix=".db", dir=str(index.parent))
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        conn = sqlite3.connect(tmp)
        try:
            _create_schema(conn)
            count = 0
            with gzip.open(corpus, "rt", encoding="utf-8") as stream:
                for raw_line in stream:
                    payload = raw_line.strip()
                    if not payload:
                        continue
                    row = json.loads(payload)
                    # Column names come from the fixed _COLUMNS tuple, never from
                    # input; the values are bound as placeholders.
                    conn.execute(
                        "INSERT OR REPLACE INTO writeups ("  # noqa: S608
                        + ", ".join(_COLUMNS)
                        + ") VALUES ("
                        + ", ".join("?" for _ in _COLUMNS)
                        + ")",
                        [row.get(column) for column in _COLUMNS],
                    )
                    count += 1
            conn.execute("INSERT INTO writeups_fts (writeups_fts) VALUES ('rebuild')")
            conn.executemany(
                "INSERT INTO corpus_meta (key, value) VALUES (?, ?)",
                [
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("corpus_sha256", _corpus_fingerprint(corpus)),
                    ("writeup_count", str(count)),
                ],
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    else:
        tmp.replace(index)
        return count


def _index_is_current(index: Path, corpus: Path) -> bool:
    if not index.is_file():
        return False
    try:
        conn = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
        try:
            meta = dict(conn.execute("SELECT key, value FROM corpus_meta").fetchall())
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return meta.get("schema_version") == str(SCHEMA_VERSION) and meta.get(
        "corpus_sha256"
    ) == _corpus_fingerprint(corpus)


def _lock_file(handle: Any, *, blocking: bool) -> None:
    # Loaded through importlib rather than `import fcntl` / `import msvcrt`: each
    # module exists on only one platform, and importing them conditionally makes
    # the other platform's stubs reject attributes that are real on that platform.
    if os.name == "nt":
        msvcrt = importlib.import_module("msvcrt")
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
    else:
        fcntl = importlib.import_module("fcntl")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))


def _unlock_file(handle: Any) -> None:
    if os.name == "nt":
        msvcrt = importlib.import_module("msvcrt")
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl = importlib.import_module("fcntl")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _cross_process_build_lock(index: Path) -> Any:
    """Advisory lock so two Strix processes do not build the index at once.

    The in-process lock already serializes threads; this covers two scans sharing
    ``~/.strix/writeups.db``. Losing the lock (an unwritable directory, or
    contention past the platform's wait) is deliberately not fatal: the caller
    still builds into a private temp file and replaces atomically, and reconciles
    a lost replacement race by re-checking whether a current index now exists.
    """
    try:
        index.parent.mkdir(parents=True, exist_ok=True)
        handle = (index.parent / f"{index.name}.lock").open("a+b")
    except OSError:
        logger.debug("writeup index lock file unavailable", exc_info=True)
        yield
        return

    try:
        try:
            _lock_file(handle, blocking=True)
        except OSError:
            logger.debug("writeup index lock not acquired; proceeding unlocked", exc_info=True)
            yield
            return
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                _unlock_file(handle)
    finally:
        handle.close()


def ensure_index(
    *,
    corpus: Path | None = None,
    index: Path | None = None,
    rebuild: bool = False,
) -> tuple[Path | None, str]:
    """Return ``(index_path, status)``.

    ``status`` is ``ready``, ``rebuilt``, or ``missing_corpus`` — callers surface
    the last one rather than raising, so a scan without the corpus still runs.

    The unsynchronized fast path is read-only and covers the steady state. Every
    path that can build takes the in-process lock and re-checks freshness inside
    it, so a second caller that arrived while the first was building finds the
    finished index instead of starting a competing build.
    """
    corpus_file = corpus_path(corpus)
    index_file = index_path(index)
    if not corpus_file.is_file():
        return None, "missing_corpus"
    if not rebuild and _index_is_current(index_file, corpus_file):
        return index_file, "ready"

    with _BUILD_LOCK:
        if not rebuild and _index_is_current(index_file, corpus_file):
            return index_file, "ready"
        with _cross_process_build_lock(index_file):
            # Second check inside the cross-process lock: another process may
            # have finished building while this one waited.
            if not rebuild and _index_is_current(index_file, corpus_file):
                return index_file, "ready"
            built_here = _build_with_replace_race_recovery(
                corpus_file, index_file, rebuild=rebuild
            )
    return index_file, "rebuilt" if built_here else "ready"


def _build_with_replace_race_recovery(
    corpus_file: Path, index_file: Path, *, rebuild: bool
) -> bool:
    """Build the index, tolerating a lost replace race.

    Returns True when this call built the index and False when a competing build
    had already installed a current one. Two things can make the final `replace`
    fail on Windows: that competing build (the outcome we wanted, so it is
    reconciled rather than raised), or a reader in another process holding the
    database open for the moment its query takes (transient, so it is retried
    briefly). Only a failure with no usable index after those retries surfaces.
    """
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            build_index(corpus_file, index_file)
        except OSError:
            if not rebuild and _index_is_current(index_file, corpus_file):
                logger.debug("writeup index was built by another process", exc_info=True)
                return False
            if attempt + 1 >= _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_RETRY_DELAY)
        else:
            return True
    return True


def _connect(index: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def search(
    query: str,
    *,
    limit: int = 5,
    vulnerability_class: str | None = None,
    since_year: int | None = None,
    program: str | None = None,
    corpus: Path | None = None,
    index: Path | None = None,
) -> list[dict[str, Any]]:
    """BM25-ranked writeups matching ``query``, newest metadata first on ties."""
    index_file, _status = ensure_index(corpus=corpus, index=index)
    if index_file is None:
        return []

    match = _fts_query(query)
    clauses: list[str] = []
    params: list[Any] = []
    if vulnerability_class:
        clauses.append("w.classes LIKE ?")
        params.append(f"%{vulnerability_class.strip().lower()}%")
    if since_year is not None:
        clauses.append("(w.year IS NULL OR w.year >= ?)")
        params.append(int(since_year))
    if program:
        clauses.append("w.program LIKE ?")
        params.append(f"%{program.strip()}%")
    where = f" AND {' AND '.join(clauses)}" if clauses else ""

    sql = (
        "SELECT w.doc_id, w.source, w.title, w.url, w.program, w.weakness, w.severity, "
        "w.bounty, w.votes, w.asset, w.reported, w.disclosed, w.resolution, w.year, w.classes, "
        "snippet(writeups_fts, 3, '[', ']', ' … ', 24) AS snippet, "
        "bm25(writeups_fts, 8.0, 4.0, 3.0, 1.0) AS score "
        "FROM writeups_fts JOIN writeups w ON w.rowid = writeups_fts.rowid "
    )
    if match:
        sql += f"WHERE writeups_fts MATCH ?{where} ORDER BY score, w.year DESC LIMIT ?"
        params = [match, *params, int(limit)]
    elif where:
        sql += f"WHERE 1=1{where} ORDER BY w.year DESC LIMIT ?"
        params = [*params, int(limit)]
    else:
        return []

    with _connect(index_file) as conn:
        # Values are always bound; only the fixed clause fragments are composed.
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def stats(*, corpus: Path | None = None, index: Path | None = None) -> dict[str, Any]:
    """Corpus/index summary, for tool output and health checks."""
    index_file, status = ensure_index(corpus=corpus, index=index)
    out: dict[str, Any] = {"status": status, "index_path": str(index_path(index))}
    if index_file is None:
        out["corpus_path"] = str(corpus_path(corpus))
        return out
    with _connect(index_file) as conn:
        meta = dict(conn.execute("SELECT key, value FROM corpus_meta").fetchall())
        out["writeups"] = int(meta.get("writeup_count", 0))
        out["sources"] = dict(
            conn.execute(
                "SELECT source, COUNT(*) FROM writeups GROUP BY source ORDER BY source"
            ).fetchall()
        )
        out["years"] = dict(
            conn.execute(
                "SELECT year, COUNT(*) FROM writeups WHERE year IS NOT NULL "
                "GROUP BY year ORDER BY year"
            ).fetchall()
        )
        out["top_classes"] = dict(
            conn.execute(
                "SELECT classes, COUNT(*) FROM writeups WHERE classes != '' "
                "GROUP BY classes ORDER BY COUNT(*) DESC LIMIT 1"
            ).fetchall()
        )
    return out
