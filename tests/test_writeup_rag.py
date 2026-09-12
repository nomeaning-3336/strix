"""Tests for the local writeup corpus (``search_writeups``).

Fully offline: the corpus is a small synthetic JSONL fixture and the index is
built into ``tmp_path``. No network, no model calls.
"""

from __future__ import annotations

import gzip
import json
import sys
from typing import TYPE_CHECKING, Any

import pytest
from agents.tool_context import ToolContext

from strix.core.tool_policy import is_parallel_safe, policy_for
from strix.tools.writeups import corpus
from strix.tools.writeups.tool import search_writeups


if TYPE_CHECKING:
    from pathlib import Path


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
    # The tool must not present prior art as evidence about the current target.
    assert "not evidence about the current target" in payload["note"]


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


def _ctx() -> ToolContext:
    return ToolContext(
        context={}, tool_name="search_writeups", tool_call_id="c1", tool_arguments="{}"
    )


def _args(query: str) -> str:
    return json.dumps({"query": query})
