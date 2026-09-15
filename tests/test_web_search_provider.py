"""Tests for the DeepSeek-native web search provider.

Response fixtures mirror the shapes returned by
``https://api.deepseek.com/anthropic/v1/messages`` when the
``web_search_20250305`` server tool runs: ``thinking`` and ``server_tool_use``
blocks, a ``web_search_tool_result`` carrying ``web_search_result`` items, and a
final ``text`` summary. DeepSeek does not currently attach ``citations[]`` to
that summary, so the snippet path is exercised with a separate Anthropic-shaped
fixture rather than assumed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Self

import pytest
import requests

from strix.config.settings import IntegrationSettings
from strix.tools.web_search import provider


def _result_item(url: str, title: str | None = None, page_age: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "web_search_result",
        "url": url,
        "encrypted_content": "opaque-blob",
    }
    if title is not None:
        item["title"] = title
    item["page_age"] = page_age
    return item


def _deepseek_response(
    items: list[dict[str, Any]],
    *,
    text: str | None = "Summary of the retrieved pages.",
    search_requests: int = 1,
    citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a response body shaped like DeepSeek's Anthropic-compatible reply."""
    content: list[dict[str, Any]] = [
        {"type": "thinking", "thinking": "...", "signature": "sig"},
        {
            "type": "server_tool_use",
            "id": "call_00_abc",
            "name": "web_search",
            "input": {"query": "example"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "call_00_abc",
            "content": items,
        },
    ]
    if text is not None:
        text_block: dict[str, Any] = {"type": "text", "text": text}
        if citations is not None:
            text_block["citations"] = citations
        content.append(text_block)
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "deepseek-v4-flash",
        "content": content,
        "usage": {
            "input_tokens": 6877,
            "output_tokens": 903,
            "server_tool_use": {"web_search_requests": search_requests},
        },
    }


# --------------------------------------------------------------------------- #
# build_messages_payload
# --------------------------------------------------------------------------- #


def test_payload_uses_native_web_search_tool() -> None:
    payload = provider.build_messages_payload(
        "xz backdoor versions", model="deepseek-v4-flash", max_tokens=4096, max_uses=5
    )

    assert payload["model"] == "deepseek-v4-flash"
    assert payload["max_tokens"] == 4096
    assert payload["tools"] == [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
    ]
    # The search is server-side, so no client tool choice may be sent.
    assert "tool_choice" not in payload
    assert len(payload["messages"]) == 1
    message = payload["messages"][0]
    assert message["role"] == "user"
    assert message["content"][0]["type"] == "text"


def test_payload_includes_query_and_security_framing() -> None:
    payload = provider.build_messages_payload(
        "OpenSSH 7.4 privesc", model="m", max_tokens=10, max_uses=1
    )
    text = payload["messages"][0]["content"][0]["text"]

    assert "OpenSSH 7.4 privesc" in text
    assert "CVE" in text
    assert "Kali" in text


# --------------------------------------------------------------------------- #
# parse_response
# --------------------------------------------------------------------------- #


def test_parse_returns_sources_and_summary() -> None:
    data = _deepseek_response(
        [
            _result_item("https://example.test/a", "A title"),
            _result_item("https://example.test/b", "B title"),
        ]
    )

    result = provider.parse_response(data)

    assert result["success"] is True
    assert result["content"] == "Summary of the retrieved pages."
    assert result["search_requests"] == 1
    assert [s["url"] for s in result["sources"]] == [
        "https://example.test/a",
        "https://example.test/b",
    ]
    assert result["sources"][0]["title"] == "A title"


def test_parse_dedupes_repeated_urls_across_result_blocks() -> None:
    # max_uses > 1 can surface the same page in more than one result block.
    data = _deepseek_response([_result_item("https://example.test/a", "A")])
    data["content"].append(
        {
            "type": "web_search_tool_result",
            "tool_use_id": "call_00_def",
            "content": [
                _result_item("https://example.test/a", "A again"),
                _result_item("https://example.test/c", "C"),
            ],
        }
    )

    sources = provider.parse_response(data)["sources"]

    assert [s["url"] for s in sources] == ["https://example.test/a", "https://example.test/c"]
    # First occurrence wins.
    assert sources[0]["title"] == "A"


def test_parse_joins_citation_excerpts_as_snippets() -> None:
    # Anthropic attaches citations[] to text blocks; DeepSeek currently does not.
    # When they are present the excerpt must be joined by url.
    data = _deepseek_response(
        [
            _result_item("https://example.test/a", "A"),
            _result_item("https://example.test/b", "B"),
        ],
        citations=[
            {
                "type": "web_search_result_location",
                "url": "https://example.test/a",
                "cited_text": "Excerpt A",
            },
            {
                "type": "web_search_result_location",
                "url": "https://example.test/b",
                "cited_text": "Excerpt B",
            },
            {
                "type": "web_search_result_location",
                "url": "https://example.test/a",
                "cited_text": "Later dup",
            },
        ],
    )

    sources = provider.parse_response(data)["sources"]

    assert sources[0]["snippet"] == "Excerpt A"
    assert sources[1]["snippet"] == "Excerpt B"


def test_parse_omits_snippet_and_published_at_when_absent() -> None:
    # The live DeepSeek endpoint omits citations and sends page_age: null, so
    # neither key may be invented.
    data = _deepseek_response([_result_item("https://example.test/a", "A", page_age=None)])

    source = provider.parse_response(data)["sources"][0]

    assert source == {"url": "https://example.test/a", "title": "A"}


def test_parse_keeps_page_age_when_present() -> None:
    data = _deepseek_response([_result_item("https://example.test/a", "A", page_age="2024-03-29")])

    assert provider.parse_response(data)["sources"][0]["publishedAt"] == "2024-03-29"


def test_parse_skips_non_result_items_in_result_block() -> None:
    # A result block can carry an error object instead of results.
    data = _deepseek_response(
        [
            {"type": "web_search_tool_error", "error_code": "max_uses_exceeded"},
            _result_item("https://example.test/a", "A"),
        ]
    )

    sources = provider.parse_response(data)["sources"]

    assert [s["url"] for s in sources] == ["https://example.test/a"]


def test_parse_tolerates_result_block_with_non_list_content() -> None:
    data = _deepseek_response([])
    data["content"][2]["content"] = {"error": "unavailable"}

    result = provider.parse_response(data)

    assert result["success"] is True
    assert result["sources"] == []


def test_parse_without_result_blocks_is_an_error() -> None:
    # Absence of retrieved pages must not be papered over with model prose.
    data = _deepseek_response([])
    data["content"] = [{"type": "text", "text": "I think the answer is 42."}]

    with pytest.raises(provider.ProviderError) as excinfo:
        provider.parse_response(data)

    assert excinfo.value.kind == "shape"


def test_parse_rejects_malformed_bodies() -> None:
    for body in (None, "nope", {"content": "nope"}, {}):
        with pytest.raises(provider.ProviderError):
            provider.parse_response(body)


def test_parse_allows_missing_summary_when_sources_exist() -> None:
    data = _deepseek_response([_result_item("https://example.test/a", "A")], text=None)

    result = provider.parse_response(data)

    assert "content" not in result
    assert result["sources"][0]["url"] == "https://example.test/a"


# --------------------------------------------------------------------------- #
# search — credential gate and error mapping
# --------------------------------------------------------------------------- #


def test_search_without_credential_does_not_call_the_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Settings:
        deepseek_search_api_key = None

    monkeypatch.setattr(
        provider,
        "load_settings",
        lambda: SimpleNamespace(integrations=_Settings()),
    )

    def _explode(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise AssertionError("must not reach the network without a credential")

    monkeypatch.setattr(provider.requests, "post", _explode)

    result = provider.search("anything")

    assert result["success"] is False
    assert "DEEPSEEK_SEARCH_API_KEY" in result["error"]


def test_search_rejects_blank_query() -> None:
    assert provider.search("   ") == {"success": False, "error": "Query cannot be empty"}


class _FakeResponse:
    """Minimal stand-in for :class:`requests.Response` used as a context manager."""

    def __init__(self, status_code: int = 200, body: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._body = body if body is not None else {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    settings = IntegrationSettings(DEEPSEEK_SEARCH_API_KEY="test-key", **overrides)
    monkeypatch.setattr(
        provider,
        "load_settings",
        lambda: SimpleNamespace(integrations=settings),
    )


def _post_returning(status_code: int, body: dict[str, Any] | None = None) -> Any:
    """Build a ``requests.post`` stand-in that always returns ``status_code``."""

    def _post(*args: Any, **kwargs: Any) -> _FakeResponse:  # noqa: ARG001
        return _FakeResponse(status_code, body)

    return _post


def test_search_maps_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)

    def _timeout(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise requests.exceptions.Timeout

    monkeypatch.setattr(provider.requests, "post", _timeout)

    result = provider.search("q")

    assert result["success"] is False
    assert "timed out" in result["error"]


def test_search_maps_client_error_to_query_advice(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)
    monkeypatch.setattr(provider.requests, "post", _post_returning(400))

    result = provider.search("q")

    assert result["success"] is False
    assert "rejected the query" in result["error"]


def test_search_maps_server_error_to_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)
    monkeypatch.setattr(provider.requests, "post", _post_returning(503))

    result = provider.search("q")

    assert result["success"] is False
    assert "unavailable" in result["error"]


def test_search_never_raises_on_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)

    def _boom(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise RuntimeError("kaboom")

    monkeypatch.setattr(provider.requests, "post", _boom)

    result = provider.search("q")

    assert result["success"] is False
    assert "RuntimeError" in result["error"]


def test_search_end_to_end_against_a_fake_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch)
    captured: dict[str, Any] = {}

    def _post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse(
            200,
            _deepseek_response(
                [_result_item("https://example.test/a", "A", page_age="2024-01-01")]
            ),
        )

    monkeypatch.setattr(provider.requests, "post", _post)

    result = provider.search("xz backdoor")

    assert result["success"] is True
    assert captured["url"] == "https://api.deepseek.com/anthropic/v1/messages"
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == provider.ANTHROPIC_API_VERSION
    assert captured["json"]["tools"][0]["type"] == "web_search_20250305"
    # The real message text must reach the request unchanged.
    assert "xz backdoor" in captured["json"]["messages"][0]["content"][0]["text"]
    assert json.loads(json.dumps(result)) == result


def test_search_honours_base_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, DEEPSEEK_SEARCH_API_BASE="https://gateway.test/anthropic/v1/")
    captured: dict[str, Any] = {}

    def _post(url: str, **kwargs: Any) -> _FakeResponse:  # noqa: ARG001
        captured["url"] = url
        return _FakeResponse(200, _deepseek_response([_result_item("https://example.test/a", "A")]))

    monkeypatch.setattr(provider.requests, "post", _post)

    provider.search("q")

    # A trailing slash must not produce a doubled separator.
    assert captured["url"] == "https://gateway.test/anthropic/v1/messages"
