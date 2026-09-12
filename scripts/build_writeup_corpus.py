#!/usr/bin/env python3
"""Build the local bug-bounty writeup corpus (SQLite FTS5) used by ``search_writeups``.

Corpus sources are public writeup archives that ship as markdown, which keeps
ingestion to plain text extraction with no HTML/JS scraping:

- ``h1-disclosed``  HackerOne disclosed reports (ajaysenr/HackerOne-Disclosed-Reports)
- ``h1-curated``    HackerOne reports curated by weakness (reddelexc/hackerone-reports)
- ``h1-reports``    HackerOne report writeups (codebygk/hackerone-bug-bounty-reports)

The H1 archive is read straight out of git objects (``git archive HEAD``) rather
than a worktree checkout: it carries ~13k deeply nested files, and checking them
out fails on Windows path limits while reading the archive does not.

Usage::

    python scripts/build_writeup_corpus.py --since-year 2024
    python scripts/build_writeup_corpus.py --reuse-clones --since-year 2024

Re-running is deterministic: the same sources and year window produce the same
rows in the same insertion order, so the output DB is reproducible.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_writeup_corpus")


SCHEMA_VERSION = 1

# Vulnerability classes worth tagging for retrieval. Order matters only for
# stable output; a writeup can carry several.
_CLASS_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("idor", ("idor", "insecure direct object", "broken object level", "bola")),
    ("ssrf", ("ssrf", "server-side request forgery", "server side request forgery")),
    ("xss", ("xss", "cross-site scripting", "cross site scripting")),
    ("sqli", ("sqli", "sql injection", "sql-injection")),
    ("rce", ("rce", "remote code execution", "command injection", "code execution")),
    ("xxe", ("xxe", "xml external entity")),
    ("ssti", ("ssti", "server-side template injection", "template injection")),
    ("csrf", ("csrf", "cross-site request forgery")),
    ("open_redirect", ("open redirect", "open-redirect", "unvalidated redirect")),
    ("path_traversal", ("path traversal", "directory traversal", "lfi", "local file inclusion")),
    (
        "auth_bypass",
        (
            "auth bypass",
            "authentication bypass",
            "broken authentication",
            "2fa bypass",
            "mfa bypass",
            "otp bypass",
            "password reset",
            "account takeover",
        ),
    ),
    (
        "privilege_escalation",
        (
            "privilege escalation",
            "privesc",
            "broken access control",
            "missing authorization",
            "authorization bypass",
        ),
    ),
    (
        "info_disclosure",
        (
            "information disclosure",
            "info disclosure",
            "sensitive data exposure",
            "data leak",
            "exposed secret",
            "leaked",
        ),
    ),
    ("race_condition", ("race condition", "toctou", "time-of-check")),
    ("prototype_pollution", ("prototype pollution",)),
    ("request_smuggling", ("request smuggling", "smuggling", "desync")),
    ("cache_poisoning", ("cache poisoning", "web cache deception")),
    ("subdomain_takeover", ("subdomain takeover", "dangling cname")),
    ("graphql", ("graphql", "introspection")),
    ("jwt", ("jwt", "json web token", "algorithm confusion")),
    ("business_logic", ("business logic", "logic flaw", "logic bug")),
    ("file_upload", ("file upload", "unrestricted upload", "arbitrary file")),
    ("deserialization", ("deserialization", "deserialisation", "unserialize")),
    ("cors", ("cors", "cross-origin resource sharing")),
    ("clickjacking", ("clickjacking", "ui redressing")),
    ("dos", ("denial of service", "dos ", "redos", "resource consumption")),
    (
        "memory_corruption",
        ("buffer overflow", "use-after-free", "heap overflow", "memory corruption"),
    ),
    ("supply_chain", ("supply chain", "dependency confusion", "typosquat", "npm package")),
)

_SOURCE_REPO = {
    "h1-disclosed": "https://github.com/ajaysenr/HackerOne-Disclosed-Reports.git",
    "h1-curated": "https://github.com/reddelexc/hackerone-reports.git",
    "h1-reports": "https://github.com/codebygk/hackerone-bug-bounty-reports.git",
}

_TABLE_ROW = re.compile(r"^\|\s*\*\*(?P<key>[^*]+)\*\*\s*\|\s*(?P<value>.*?)\s*\|\s*$")
_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^)]*)\)")
_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


@dataclass
class Writeup:
    """One retrieval unit: a writeup plus the metadata that scopes it."""

    doc_id: str
    source: str
    title: str
    url: str
    program: str = ""
    weakness: str = ""
    severity: str = ""
    bounty: str = ""
    votes: str = ""
    reporter: str = ""
    asset: str = ""
    reported: str = ""
    disclosed: str = ""
    resolution: str = ""
    classes: tuple[str, ...] = ()
    body: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def year(self) -> int | None:
        for value in (self.disclosed, self.reported):
            match = _DATE.search(value or "")
            if match:
                return int(match.group(1))
        return None


def _clean(value: str) -> str:
    return _MD_LINK.sub(r"\1", value).strip().strip("|").strip()


def _compile_class_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile each class's needles as word-bounded alternatives.

    Word boundaries are essential for the short acronyms: a bare substring test
    for ``rce`` also matches "resource", "source" and "force", which tagged most
    of the corpus as RCE. Bounding each needle's own ends still lets multi-word
    phrases match as phrases.
    """
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for name, needles in _CLASS_PATTERNS:
        parts = []
        for needle in needles:
            stripped = needle.strip()
            escaped = re.escape(stripped)
            leading = r"\b" if stripped[:1].isalnum() else ""
            trailing = r"\b" if stripped[-1:].isalnum() else ""
            parts.append(f"{leading}{escaped}{trailing}")
        compiled.append((name, re.compile("|".join(parts), re.IGNORECASE)))
    return tuple(compiled)


_CLASS_REGEXES = _compile_class_patterns()


def classify(text: str) -> tuple[str, ...]:
    """Deterministic vulnerability-class tags from a writeup's own words."""
    return tuple(name for name, pattern in _CLASS_REGEXES if pattern.search(text))


def parse_h1_report(text: str, *, fallback_id: str, source: str) -> Writeup | None:
    """Parse one HackerOne disclosed-report markdown file.

    The archive renders each report as a title heading, a metadata table, and the
    program summary plus the vulnerability details.
    """
    lines = text.splitlines()
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    if not title:
        return None

    fields: dict[str, str] = {}
    for line in lines:
        match = _TABLE_ROW.match(line)
        if match:
            fields[match.group("key").strip().lower()] = _clean(match.group("value"))

    url = ""
    report_cell = fields.get("report id", "")
    link = re.search(r"\((https?://[^)]+)\)", "\n".join(lines))
    if report_cell.startswith("http"):
        url = report_cell
    elif link:
        url = link.group(1)
    if not url:
        url = f"https://hackerone.com/reports/{fallback_id}"

    body_start = text.find("## Vulnerability Details")
    body = text[body_start + len("## Vulnerability Details") :] if body_start >= 0 else text
    summary_start = text.find("## Program Summary")
    program_summary = (
        text[summary_start:body_start] if summary_start >= 0 and body_start > summary_start else ""
    )

    writeup = Writeup(
        doc_id=f"{source}:{fallback_id}",
        source=source,
        title=title,
        url=url,
        program=fields.get("program", ""),
        weakness=fields.get("weakness", ""),
        severity=fields.get("severity", ""),
        bounty=fields.get("bounty", ""),
        votes=fields.get("votes", ""),
        reporter=fields.get("reporter", ""),
        asset=fields.get("asset", ""),
        reported=fields.get("reported", ""),
        disclosed=fields.get("disclosed", ""),
        resolution=fields.get("resolution", ""),
        extra={"cve": fields.get("cve ids", ""), "program_summary": program_summary.strip()},
    )
    writeup.body = body.strip()
    writeup.classes = classify(f"{writeup.title}\n{writeup.weakness}\n{writeup.body}")
    return writeup


def parse_markdown_writeup(text: str, *, path: str, source: str) -> Writeup | None:
    """Parse a plain markdown writeup (curated collections, no metadata table)."""
    lines = text.splitlines()
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    if not title:
        return None
    body = text
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]  # noqa: S324 - stable ids, not security
    writeup = Writeup(
        doc_id=f"{source}:{digest}",
        source=source,
        title=title,
        url="",
        body=body.strip(),
        extra={"path": path},
    )
    writeup.classes = classify(f"{writeup.title}\n{writeup.body}")
    return writeup


def _run_git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )


def clone_repo(source: str, work_dir: Path) -> Path | None:
    """Shallow, checkout-free clone so Windows path limits cannot break it."""
    dest = work_dir / source
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    proc = _run_git(
        [
            "-c",
            "http.sslBackend=openssl",
            "clone",
            "--depth",
            "1",
            "--no-checkout",
            "--quiet",
            _SOURCE_REPO[source],
            str(dest),
        ]
    )
    if proc.returncode != 0:
        logger.warning(
            "clone failed for %s: %s", source, (proc.stderr or proc.stdout).strip()[:200]
        )
        return None
    return dest


def iter_git_markdown(repo: Path, prefix: str) -> list[tuple[str, str]]:
    """Every ``prefix``-prefixed markdown file in HEAD, read from git objects.

    Blobs are streamed through ``cat-file --batch`` (oids on stdin) rather than
    ``git archive``: the archive refuses paths Windows considers invalid — the
    H1 archive contains e.g. ``by-weakness/asi05:-...rce.md`` — and passing 12k
    individual paths would exceed the Windows command-line limit.
    """
    listing = _run_git(["ls-tree", "-r", "-z", "HEAD", prefix or "."], cwd=repo)
    if listing.returncode != 0:
        logger.warning("ls-tree failed in %s: %s", repo, listing.stderr.strip()[:200])
        return []

    by_oid: dict[str, list[str]] = {}
    for entry in listing.stdout.split("\0"):
        if not entry or "\t" not in entry:
            continue
        meta, path = entry.split("\t", 1)
        if not path.endswith((".md", ".markdown")):
            continue
        parts = meta.split()
        if len(parts) < 3 or parts[1] != "blob":
            continue
        by_oid.setdefault(parts[2], []).append(path)

    if not by_oid:
        return []

    proc = subprocess.run(
        ["git", "cat-file", "--batch"],  # noqa: S607
        cwd=str(repo),
        input=("\n".join(by_oid) + "\n").encode("ascii"),
        capture_output=True,
        check=False,
        timeout=1800,
    )
    if proc.returncode != 0:
        logger.warning("cat-file failed in %s: %s", repo, proc.stderr.decode()[:200])
        return []

    out: list[tuple[str, str]] = []
    data = proc.stdout
    offset = 0
    while offset < len(data):
        newline = data.find(b"\n", offset)
        if newline < 0:
            break
        header = data[offset:newline].decode("utf-8", "replace").split()
        offset = newline + 1
        if len(header) < 3:
            continue
        oid, _kind, size_text = header[0], header[1], header[2]
        try:
            size = int(size_text)
        except ValueError:
            continue
        payload = data[offset : offset + size]
        offset += size + 1  # trailing newline
        text = payload.decode("utf-8", "replace")
        out.extend((path, text) for path in by_oid.get(oid, ()))
    return out


def _read_one_by_one(repo: Path, paths: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in paths:
        proc = _run_git(["show", f"HEAD:{path}"], cwd=repo)
        if proc.returncode == 0:
            out.append((path, proc.stdout))
    return out


def collect(
    source: str, repo: Path, since_year: int | None, *, skip_undated: bool
) -> list[Writeup]:
    if source == "h1-disclosed":
        files = iter_git_markdown(repo, "reports")
        writeups: list[Writeup] = []
        for path, text in files:
            report_id = Path(path).stem
            parsed = parse_h1_report(text, fallback_id=report_id, source=source)
            if parsed is None:
                continue
            # The disclosed/reported date is the window this corpus is scoped to.
            if since_year is not None and parsed.year is not None and parsed.year < since_year:
                continue
            if since_year is not None and parsed.year is None and skip_undated:
                continue
            writeups.append(parsed)
        return writeups

    files = iter_git_markdown(repo, "")
    writeups = []
    for path, text in files:
        parsed = parse_markdown_writeup(text, path=path, source=source)
        if parsed is None or len(parsed.body) < 200:
            continue
        # Curated collections carry no dates: keep them (year stays NULL) unless
        # the caller explicitly wants only date-scoped material.
        if since_year is not None and skip_undated and parsed.year is None:
            continue
        if since_year is not None and parsed.year is not None and parsed.year < since_year:
            continue
        writeups.append(parsed)
    return writeups


def create_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
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
    return conn


def insert(conn: sqlite3.Connection, writeups: list[Writeup]) -> int:
    rows = [
        (
            w.doc_id,
            w.source,
            w.title,
            w.url,
            w.program,
            w.weakness,
            w.severity,
            w.bounty,
            w.votes,
            w.reporter,
            w.asset,
            w.reported,
            w.disclosed,
            w.resolution,
            w.year,
            " ".join(w.classes),
            w.body,
            json.dumps(w.extra, sort_keys=True),
        )
        for w in writeups
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO writeups (doc_id, source, title, url, program, weakness, "
        "severity, bounty, votes, reporter, asset, reported, disclosed, resolution, year, "
        "classes, body, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.execute("INSERT INTO writeups_fts (writeups_fts) VALUES ('rebuild')")
    conn.commit()
    return len(rows)


def write_corpus(corpus: Path, writeups: list[Writeup]) -> None:
    """Write the shippable gzipped JSONL corpus (byte-stable for a given input).

    ``mtime=0`` drops the gzip timestamp so rebuilding an unchanged corpus twice
    produces identical bytes rather than a spurious diff.
    """
    corpus.parent.mkdir(parents=True, exist_ok=True)
    with (
        corpus.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8") as stream,
    ):
        for writeup in writeups:
            stream.write(
                json.dumps(
                    {
                        "doc_id": writeup.doc_id,
                        "source": writeup.source,
                        "title": writeup.title,
                        "url": writeup.url,
                        "program": writeup.program,
                        "weakness": writeup.weakness,
                        "severity": writeup.severity,
                        "bounty": writeup.bounty,
                        "votes": writeup.votes,
                        "reporter": writeup.reporter,
                        "asset": writeup.asset,
                        "reported": writeup.reported,
                        "disclosed": writeup.disclosed,
                        "resolution": writeup.resolution,
                        "year": writeup.year,
                        "classes": " ".join(writeup.classes),
                        "body": writeup.body,
                        "extra": json.dumps(writeup.extra, sort_keys=True),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since-year", type=int, default=2024)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("strix/knowledge/writeups.jsonl.gz"),
        help="output gzipped JSONL corpus (shipped with the package)",
    )
    parser.add_argument(
        "--db-out",
        type=Path,
        default=None,
        help="optionally also write a prebuilt SQLite index",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "strix-writeup-sources",
    )
    parser.add_argument(
        "--reuse-clones",
        action="store_true",
        help="reuse existing clones under --work-dir instead of re-cloning",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["h1-disclosed"],
        choices=list(_SOURCE_REPO),
        help="default is the prose archive; the other two are link tables, not writeups",
    )
    parser.add_argument(
        "--skip-undated",
        action="store_true",
        help="keep only writeups that carry a date inside the window",
    )
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    all_writeups: list[Writeup] = []
    for source in args.sources:
        repo = args.work_dir / source
        usable = repo.exists() and (repo / ".git").exists() if args.reuse_clones else False
        if not usable:
            repo = clone_repo(source, args.work_dir)  # type: ignore[assignment]
        if repo is None:
            continue
        found = collect(source, repo, args.since_year, skip_undated=args.skip_undated)
        dated = sum(1 for w in found if w.year is not None)
        logger.info(
            "%-14s %6d writeups >= %s (%d dated, %d undated)",
            source,
            len(found),
            args.since_year,
            dated,
            len(found) - dated,
        )
        all_writeups.extend(found)

    # Deterministic order: source, then id.
    all_writeups.sort(key=lambda w: (w.source, w.doc_id))
    deduped: dict[str, Writeup] = {}
    for writeup in all_writeups:
        deduped.setdefault(writeup.doc_id, writeup)
    ordered = list(deduped.values())

    write_corpus(args.out, ordered)
    size_mb = args.out.stat().st_size / 1e6
    logger.info("wrote %s: %d writeups, %.2f MB", args.out, len(ordered), size_mb)

    class_counts: dict[str, int] = {}
    for writeup in ordered:
        for name in writeup.classes:
            class_counts[name] = class_counts.get(name, 0) + 1
    logger.info("top classes: %s", sorted(class_counts.items(), key=lambda kv: -kv[1])[:12])

    if args.db_out is not None:
        conn = create_db(args.db_out)
        try:
            count = insert(conn, ordered)
            conn.executemany(
                "INSERT INTO corpus_meta (key, value) VALUES (?, ?)",
                [
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("since_year", str(args.since_year)),
                    ("writeup_count", str(count)),
                    ("class_counts", json.dumps(class_counts, sort_keys=True)),
                ],
            )
            conn.commit()
        finally:
            conn.close()
        logger.info("wrote %s (%.1f MB)", args.db_out, args.db_out.stat().st_size / 1e6)

    return 0


if __name__ == "__main__":
    sys.exit(main())
