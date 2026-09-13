"""Tests for the local writeup corpus (``search_writeups``).

Fully offline: the corpus is a small synthetic JSONL fixture and the index is
built into ``tmp_path``. No network, no model calls.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from agents.tool_context import ToolContext

from strix.core.tool_policy import is_parallel_safe, policy_for
from strix.tools.writeups import corpus
from strix.tools.writeups.tool import search_writeups


def _writeup(
    doc_id: str,
    title: str,
    body: str,
    *,
    classes: str = "",
    year: int = 2025,
    program: str = "example",
    severity: str = "high",
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "source": "fixture",
        "title": title,
        "url": f"https://hackerone.com/reports/{doc_id}",
        "program": program,
        "weakness": "",
        "severity": severity,
        "bounty": "$500",
        "votes": "10",
        "reporter": "tester",
        "asset": "api.example.com",
        "reported": f"{year}-01-01",
        "disclosed": f"{year}-02-01",
        "resolution": "Resolved",
        "year": year,
        "classes": classes,
        "body": body,
        "extra": "{}",
    }


@pytest.fixture
def fixture_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A three-writeup corpus plus a destination index path."""
    rows = [
        _writeup(
            "1",
            "GraphQL introspection enabled on the public API",
            "The introspection query returns the full schema, including a "
            "hidden mutation used for password reset. Request follows.",
            classes="graphql info_disclosure",
            year=2024,
        ),
        _writeup(
            "2",
            "IDOR in invoice download enumerates other tenants",
            "Swapping the numeric invoice id returns another tenant's PDF. "
            "Baseline request with own id returns 403 for theirs.",
            classes="idor privilege_escalation",
            year=2025,
        ),
        _writeup(
            "3",
            "Web cache poisoning via an unkeyed X-Forwarded-Host header",
            "A crafted header is reflected into the cached response and served to other users.",
            classes="cache_poisoning",
            year=2026,
            program="otherprogram",
        ),
    ]
    source = tmp_path / "writeups.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return source, tmp_path / "index.db"


def test_index_builds_and_reports_stats(fixture_corpus: tuple[Path, Path]) -> None:
    source, index = fixture_corpus
    index_file, status = corpus.ensure_index(corpus=source, index=index)

    assert status == "rebuilt"
    assert index_file == index
    assert index.is_file()
    stats = corpus.stats(corpus=source, index=index)
    assert stats["writeups"] == 3
    assert stats["years"] == {2024: 1, 2025: 1, 2026: 1}


def test_index_is_reused_while_the_corpus_is_unchanged(fixture_corpus: tuple[Path, Path]) -> None:
    source, index = fixture_corpus
    corpus.ensure_index(corpus=source, index=index)
    stamp = index.stat().st_mtime_ns

    _, status = corpus.ensure_index(corpus=source, index=index)

    assert status == "ready"
    assert index.stat().st_mtime_ns == stamp


def test_changed_corpus_rebuilds_the_index(fixture_corpus: tuple[Path, Path]) -> None:
    source, index = fixture_corpus
    corpus.ensure_index(corpus=source, index=index)

    with gzip.open(source, "at", encoding="utf-8") as stream:
        stream.write(json.dumps(_writeup("4", "A new writeup", "body text here")) + "\n")

    _, status = corpus.ensure_index(corpus=source, index=index)

    assert status == "rebuilt"
    assert corpus.stats(corpus=source, index=index)["writeups"] == 4


def test_missing_corpus_is_reported_not_raised(tmp_path: Path) -> None:
    path, status = corpus.ensure_index(
        corpus=tmp_path / "nope.jsonl.gz", index=tmp_path / "index.db"
    )

    assert path is None
    assert status == "missing_corpus"


def test_search_ranks_the_matching_writeup_first(fixture_corpus: tuple[Path, Path]) -> None:
    source, index = fixture_corpus
    hits = corpus.search("introspection schema mutation", corpus=source, index=index)

    assert hits
    assert hits[0]["doc_id"] == "1"
    assert hits[0]["url"].endswith("/1")
    assert hits[0]["year"] == 2024


def test_search_matches_multi_word_queries_without_every_token(
    fixture_corpus: tuple[Path, Path],
) -> None:
    """A natural-language query must not require every term to appear."""
    source, index = fixture_corpus
    hits = corpus.search("unkeyed header cache poisoning other users", corpus=source, index=index)

    assert hits
    assert hits[0]["doc_id"] == "3"


def test_search_filters_by_vulnerability_class(fixture_corpus: tuple[Path, Path]) -> None:
    source, index = fixture_corpus
    hits = corpus.search("", limit=5, vulnerability_class="idor", corpus=source, index=index)

    assert [hit["doc_id"] for hit in hits] == ["2"]


def test_search_filters_by_program_and_year(fixture_corpus: tuple[Path, Path]) -> None:
    source, index = fixture_corpus
    by_program = corpus.search("", limit=5, program="otherprogram", corpus=source, index=index)
    assert [hit["doc_id"] for hit in by_program] == ["3"]

    since_2025 = corpus.search("", limit=5, since_year=2025, corpus=source, index=index)
    assert {hit["doc_id"] for hit in since_2025} == {"2", "3"}


def test_special_characters_do_not_break_the_query(fixture_corpus: tuple[Path, Path]) -> None:
    """FTS5 treats bare operators specially; a model-supplied string is data."""
    source, index = fixture_corpus
    for query in ('"unclosed', "AND OR NOT", "a*", "title:foo", "???", "x-y:z/.."):
        assert isinstance(corpus.search(query, corpus=source, index=index), list)


def test_search_without_any_criteria_returns_nothing(fixture_corpus: tuple[Path, Path]) -> None:
    source, index = fixture_corpus
    assert corpus.search("", corpus=source, index=index) == []


@pytest.mark.asyncio
async def test_tool_returns_prior_art_with_a_verification_caveat(
    fixture_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, index = fixture_corpus
    monkeypatch.setenv(corpus.CORPUS_ENV, str(source))
    monkeypatch.setenv(corpus.INDEX_ENV, str(index))

    payload = json.loads(await search_writeups.on_invoke_tool(_ctx(), _args("introspection")))

    assert payload["success"] is True
    assert payload["count"] >= 1
    first = payload["writeups"][0]
    assert first["url"].startswith("https://hackerone.com/reports/")
    assert first["classes"] == ["graphql", "info_disclosure"]
    # The tool must not present prior art as evidence about the current target,
    # and must warn that the quoted excerpts are untrusted text.
    assert "not evidence about the current target" in payload["note"]
    assert "untrusted" in payload["note"]


@pytest.mark.asyncio
async def test_tool_reports_a_missing_corpus_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(corpus.CORPUS_ENV, str(tmp_path / "absent.jsonl.gz"))
    monkeypatch.setenv(corpus.INDEX_ENV, str(tmp_path / "index.db"))

    payload = json.loads(await search_writeups.on_invoke_tool(_ctx(), _args("any query")))

    assert payload["success"] is True
    assert payload["writeups"] == []
    assert "not installed" in payload["note"]


@pytest.mark.asyncio
async def test_tool_requires_some_criteria() -> None:
    # A blank query with no filters has nothing to match on.
    payload = json.loads(await search_writeups.on_invoke_tool(_ctx(), _args("   ")))

    assert payload["success"] is False
    assert "query" in payload["error"]


def test_tool_policy_treats_search_writeups_as_a_read_only_reader() -> None:
    # A local, immutable index: safe to run beside other calls and to cache.
    policy = policy_for("search_writeups")
    assert is_parallel_safe("search_writeups") is True
    assert policy.parallel_safe is True
    assert policy.side_effect_free is True
    assert policy.cacheable is True
    assert policy.serial_group is None


def test_classify_is_word_bounded() -> None:
    """Short acronyms must not match inside ordinary words."""
    sys.path.insert(0, "scripts")
    import build_writeup_corpus as builder  # noqa: PLC0415 - a script, not a package

    assert "rce" not in builder.classify("This resource is the source of force")
    assert "rce" in builder.classify("Remote Code Execution via the upload")
    assert "idor" in builder.classify("An IDOR lets you read others")
    # Multi-word phrases still match as phrases.
    assert "cache_poisoning" in builder.classify("web cache poisoning on the CDN")
    assert "rce" in builder.classify("achieved command injection")


def test_concurrent_first_use_builds_the_index_once(
    fixture_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`search_writeups` is parallel-safe, so first use must be serialized.

    Several agents can call the tool before any index exists. Without a lock each
    of them decides the index needs building and they race on the same SQLite
    file; with it, exactly one build happens and the rest observe the finished
    index.
    """
    source, index = fixture_corpus
    real_build = corpus.build_index
    builds: list[int] = []
    build_started = threading.Event()

    def slow_build(corpus_file: Path, index_file: Path) -> int:
        builds.append(1)
        build_started.set()
        time.sleep(0.05)  # widen the window so a lockless build would collide
        return real_build(corpus_file, index_file)

    monkeypatch.setattr(corpus, "build_index", slow_build)

    results: list[tuple[Path | None, str]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            results.append(corpus.ensure_index(corpus=source, index=index))
        except BaseException as exc:  # noqa: BLE001 - surfaced through `errors`
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert len(builds) == 1, f"expected a single build, got {len(builds)}"
    assert len(results) == 4
    for index_file, status in results:
        assert index_file == index
        assert status in {"rebuilt", "ready"}
        assert index.is_file()
    # The index is usable afterwards, not a half-written file.
    assert corpus.stats(corpus=source, index=index)["writeups"] == 3


def test_in_process_lock_alone_serializes_first_use(
    fixture_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-process lock must serialize first use on its own.

    The cross-process lock is disabled here, so this pins `_BUILD_LOCK`
    specifically: without it, four threads in one process all decide the index
    needs building and race on the same database file.
    """
    source, index = fixture_corpus
    real_build = corpus.build_index
    builds: list[int] = []

    def slow_build(corpus_file: Path, index_file: Path) -> int:
        builds.append(1)
        time.sleep(0.05)
        return real_build(corpus_file, index_file)

    monkeypatch.setattr(corpus, "build_index", slow_build)
    monkeypatch.setattr(
        corpus, "_cross_process_build_lock", lambda _index: contextlib.nullcontext()
    )

    results: list[str] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        barrier.wait(timeout=10)
        results.append(corpus.ensure_index(corpus=source, index=index)[1])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(builds) == 1, f"expected a single build, got {len(builds)}"
    assert sorted(results) == ["ready", "ready", "ready", "rebuilt"]
    assert corpus.stats(corpus=source, index=index)["writeups"] == 3


def test_lost_replace_race_is_reconciled_when_a_current_index_exists(
    fixture_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replace failure after a competing build must not surface as an error.

    Windows cannot replace a database another process still holds open, so the
    build can fail *after* another process produced a valid index. That outcome
    is what the caller wanted, so it is reported as ready rather than raised.
    """
    source, index = fixture_corpus
    real_build = corpus.build_index

    def build_then_lose_the_race(corpus_file: Path, index_file: Path) -> int:
        # The competing process finishes first and installs a current index...
        real_build(corpus_file, index_file)
        # ...then this process's replace fails.
        raise PermissionError(13, "the file is in use by another process")

    monkeypatch.setattr(corpus, "build_index", build_then_lose_the_race)

    index_file, status = corpus.ensure_index(corpus=source, index=index)

    assert status == "ready"
    assert index_file == index
    assert corpus.stats(corpus=source, index=index)["writeups"] == 3


def test_lost_replace_race_still_raises_without_a_usable_index(
    fixture_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, index = fixture_corpus

    def always_fails(_corpus_file: Path, _index_file: Path) -> int:
        raise PermissionError(13, "the file is in use by another process")

    monkeypatch.setattr(corpus, "build_index", always_fails)

    with pytest.raises(PermissionError):
        corpus.ensure_index(corpus=source, index=index)


def test_transient_replace_failure_is_retried(
    fixture_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reader holding the database open is transient, so the build is retried."""
    source, index = fixture_corpus
    real_build = corpus.build_index
    attempts: list[int] = []

    def fail_once(corpus_file: Path, index_file: Path) -> int:
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError(13, "the file is in use by another process")
        return real_build(corpus_file, index_file)

    monkeypatch.setattr(corpus, "build_index", fail_once)

    index_file, status = corpus.ensure_index(corpus=source, index=index)

    assert status == "rebuilt"
    assert index_file == index
    assert len(attempts) == 2
    assert corpus.stats(corpus=source, index=index)["writeups"] == 3


def test_forced_rebuild_does_not_swallow_a_replace_failure(
    fixture_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`rebuild=True` asked for a rebuild, so a failure to rebuild is an error."""
    source, index = fixture_corpus
    corpus.build_index(source, index)  # a current index already exists

    def refuse(_corpus_file: Path, _index_file: Path) -> int:
        raise PermissionError(13, "in use")

    monkeypatch.setattr(corpus, "build_index", refuse)

    with pytest.raises(PermissionError):
        corpus.ensure_index(corpus=source, index=index, rebuild=True)


def test_cross_process_lock_is_taken_and_released(fixture_corpus: tuple[Path, Path]) -> None:
    source, index = fixture_corpus

    # The lock is the unit under test here.
    with corpus._cross_process_build_lock(index):
        assert (index.parent / f"{index.name}.lock").is_file()

    # Released on exit: the same lock can be taken again in this process.
    with corpus._cross_process_build_lock(index):
        pass

    corpus.ensure_index(corpus=source, index=index)
    assert index.is_file()


def test_build_still_works_when_the_lock_file_cannot_be_created(
    fixture_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable lock path must degrade to the unlocked build, not fail."""
    source, index = fixture_corpus
    real_open = Path.open

    def refuse_lock_file(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name.endswith(".lock"):
            raise PermissionError(13, "lock file not writable")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse_lock_file)

    index_file, status = corpus.ensure_index(corpus=source, index=index)

    assert status == "rebuilt"
    assert index_file == index
    assert corpus.stats(corpus=source, index=index)["writeups"] == 3


def _ctx() -> ToolContext:
    return ToolContext(
        context={}, tool_name="search_writeups", tool_call_id="c1", tool_arguments="{}"
    )


def _args(query: str) -> str:
    return json.dumps({"query": query})
