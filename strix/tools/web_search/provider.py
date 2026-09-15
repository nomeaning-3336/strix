"""DeepSeek-native web search via the Anthropic-compatible Messages API.

The search itself is performed **server-side** by DeepSeek through the
``web_search_20250305`` server tool, so results are real retrieved pages rather
than model-generated URLs. The provider sends one Messages request and parses the
returned content blocks:

* ``web_search_tool_result`` blocks carry the retrieved pages
  (``url`` / ``title`` / ``page_age``).
* ``text`` blocks carry the answer DeepSeek synthesizes from those pages.

Snippets are best-effort. Anthropic's hosted implementation attaches
``citations[]`` (each with ``cited_text``) to the text blocks, and DeepSeek's
endpoint may omit them entirely; when they are absent a result still carries
``url`` and ``title``, just no ``snippet``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import requests

from strix.config import load_settings


if TYPE_CHECKING:
    from strix.config.settings import IntegrationSettings


logger = logging.getLogger(__name__)


ANTHROPIC_API_VERSION = "2023-06-01"

WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
WEB_SEARCH_TOOL_NAME = "web_search"

# DeepSeek enforces the tool's own search budget server-side and ignores a
# client-sent ``tool_choice`` for server tools, so no choice is sent. Keeping the
# request to a single round trip means one call stays well inside the tool's
# timeout even when several searches are used.
_MESSAGES_PATH = "/messages"

# The security framing previously sent as the Perplexity system prompt is folded
# into the request here, because this endpoint applies no separate system prompt.
_SEARCH_INSTRUCTIONS = (
    "Perform a web search for the query below and summarize what the retrieved "
    "pages actually say.\n\n"
    "Prioritize cybersecurity-relevant information: vulnerability details (CVE "
    "IDs, CVSS scores, affected versions, impact), security tooling and "
    "methodology, exploit proofs-of-concept, mitigations, and penetration-testing "
    "technique. Give technical depth suited to a security professional: name "
    "specific versions, configurations, and technical details, and cite reliable "
    "sources (NIST, OWASP, CVE databases, vendor advisories). When giving commands "
    "or install steps, prefer Kali Linux compatibility and the ``apt`` package "
    "manager or tools pre-installed in Kali. Always include the concrete commands, "
    "code, or configuration that applies.\n\n"
    "Query: {query}"
)


ErrorKind = Literal["not_configured", "timeout", "http", "network", "shape"]


class ProviderError(Exception):
    """A sanitized search failure.

    ``message`` is safe to hand to the model; it never embeds the API key or the
    raw provider payload.
    """

    def __init__(
        self,
        message: str,
        kind: ErrorKind,
        *,
        status: int | None = None,
        api_key_env: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status = status
        self.api_key_env = api_key_env


def build_messages_payload(
    query: str,
    *,
    model: str,
    max_tokens: int,
    max_uses: int,
) -> dict[str, Any]:
    """Build the DeepSeek Messages request body for one search."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": _SEARCH_INSTRUCTIONS.format(query=query)}],
            }
        ],
        "tools": [
            {
                "type": WEB_SEARCH_TOOL_TYPE,
                "name": WEB_SEARCH_TOOL_NAME,
                "max_uses": max_uses,
            }
        ],
    }


def _answer_text(blocks: list[dict[str, Any]]) -> str:
    """Join the response's ``text`` blocks into the model-visible answer."""
    parts: list[str] = []
    for block in blocks:
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _citation_snippets(blocks: list[dict[str, Any]]) -> dict[str, str]:
    """Map ``url -> cited_text`` from every text block's ``citations[]``.

    First occurrence wins, so a page cited by several blocks keeps one excerpt.
    """
    snippets: dict[str, str] = {}
    for block in blocks:
        if block.get("type") != "text":
            continue
        citations = block.get("citations")
        if not isinstance(citations, list):
            continue
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            url = citation.get("url")
            cited_text = citation.get("cited_text")
            if (
                isinstance(url, str)
                and url
                and isinstance(cited_text, str)
                and cited_text
                and url not in snippets
            ):
                snippets[url] = cited_text
    return snippets


def extract_sources(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize result blocks into deduped ``{url,title,snippet,publishedAt}``.

    Deduplication matters because a request with ``max_uses > 1`` can surface the
    same URL across several searches. Only ``web_search_result`` items are read;
    error items that can appear in a result block are skipped rather than raising.
    """
    snippets = _citation_snippets(blocks)
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in blocks:
        if block.get("type") != "web_search_tool_result":
            continue
        content = block.get("content")
        if not isinstance(content, list):
            # The API may put a tool-error object here instead of results.
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "web_search_result":
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url or url in seen:
                continue
            seen.add(url)
            source: dict[str, Any] = {"url": url}
            title = item.get("title")
            if isinstance(title, str) and title:
                source["title"] = title
            snippet = snippets.get(url)
            if snippet:
                source["snippet"] = snippet
            page_age = item.get("page_age")
            if isinstance(page_age, str) and page_age:
                source["publishedAt"] = page_age
            sources.append(source)
    return sources


def _search_requests(usage: Any) -> int:
    """Read how many server-side searches DeepSeek actually ran."""
    if not isinstance(usage, dict):
        return 0
    server_tool_use = usage.get("server_tool_use")
    if not isinstance(server_tool_use, dict):
        return 0
    count = server_tool_use.get("web_search_requests")
    return count if isinstance(count, int) else 0


def parse_response(data: Any) -> dict[str, Any]:
    """Turn a Messages response body into the tool's normalized result.

    Success requires at least one *retrieved* page. A result block that carries
    only an error, or no usable item at all, is a failure: model prose on its own
    must never be reported as a successful search.

    Raises:
        ProviderError: when the body is malformed, or carried no retrieved pages.
    """
    if not isinstance(data, dict):
        raise ProviderError(
            "Web search returned an unexpected response. Try again",
            "shape",
        )
    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise ProviderError(
            "Web search returned an unexpected response. Try again",
            "shape",
        )

    sources = extract_sources(blocks)
    if not sources:
        # Covers both "no result block at all" and "result block with zero
        # usable URLs". Either way nothing was actually retrieved, so returning
        # the synthesized text here would present model recall as search output.
        raise ProviderError(
            "Web search returned no usable retrieved sources. Try rephrasing the query",
            "shape",
        )

    result: dict[str, Any] = {
        "success": True,
        "sources": sources,
        "search_requests": _search_requests(data.get("usage")),
    }
    answer = _answer_text(blocks)
    if answer:
        result["content"] = answer
    return result


def _http_error(status: int) -> ProviderError:
    """Map an HTTP failure to the recovery step that can actually fix it.

    The distinction matters: telling the agent to rephrase its query is useless
    advice for a rejected credential or a rate limit.
    """
    if status in {401, 403}:
        return ProviderError(
            "Web search authentication failed. The operator needs to check "
            "DEEPSEEK_SEARCH_API_KEY (or DEEPSEEK_API_KEY). Proceed without web search",
            "http",
            status=status,
        )
    if status in {404, 405}:
        return ProviderError(
            "Web search endpoint appears misconfigured. The operator needs to check "
            "DEEPSEEK_SEARCH_API_BASE. Proceed without web search",
            "http",
            status=status,
        )
    if status == 429:
        return ProviderError(
            "Web search was rate limited. Wait and retry later",
            "http",
            status=status,
        )
    if 400 <= status < 500:
        return ProviderError(
            "Web search rejected the query. Refine it "
            "(more specific, shorter, no unusual characters) and retry",
            "http",
            status=status,
        )
    return ProviderError(
        "Web search service is unavailable. Try again later",
        "http",
        status=status,
    )


def _execute(
    query: str,
    *,
    api_key: str,
    settings: IntegrationSettings,
) -> dict[str, Any]:
    """Send one Messages request and parse it. Raises :class:`ProviderError`."""
    endpoint = f"{settings.deepseek_search_api_base.rstrip('/')}{_MESSAGES_PATH}"
    payload = build_messages_payload(
        query,
        model=settings.deepseek_search_model,
        max_tokens=settings.deepseek_search_max_tokens,
        max_uses=settings.deepseek_search_max_uses,
    )
    headers = {
        # DeepSeek expects x-api-key; an Anthropic-compatible gateway may expect
        # a bearer token, so send both and let either authenticate.
        "x-api-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "anthropic-version": ANTHROPIC_API_VERSION,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "strix",
    }

    try:
        with requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=settings.deepseek_search_timeout,
        ) as response:
            if not response.ok:
                raise _http_error(response.status_code)
            data = response.json()
    except requests.exceptions.Timeout as exc:
        raise ProviderError(
            "Web search timed out. Try again or shorten the query",
            "timeout",
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ProviderError(
            "Web search network error. Try again later",
            "network",
        ) from exc
    except ValueError as exc:
        raise ProviderError(
            "Web search returned an unexpected response. Try again",
            "shape",
        ) from exc

    return parse_response(data)


def search(query: str) -> dict[str, Any]:
    """Run one DeepSeek-native web search.

    Never raises: every failure comes back as ``{"success": False, "error": ...}``
    with a message the agent can act on.

    Args:
        query: The search query.

    Returns:
        On success, ``{"success": True, "sources": [...], "content": str,
        "search_requests": int}``. On failure, ``{"success": False, "error": str}``.
    """
    if not query or not query.strip():
        return {"success": False, "error": "Query cannot be empty"}

    settings = load_settings().integrations
    api_key = settings.deepseek_search_api_key
    if not api_key:
        logger.warning(
            "web_search invoked without DEEPSEEK_SEARCH_API_KEY/DEEPSEEK_API_KEY configured"
        )
        return {
            "success": False,
            "error": (
                "Web search is not configured for this scan (operator needs to set "
                "DEEPSEEK_SEARCH_API_KEY, or DEEPSEEK_API_KEY). Proceed without it"
            ),
        }

    logger.info("web_search query (len=%d): %s", len(query), query[:120])
    try:
        return _execute(
            query,
            api_key=api_key,
            settings=settings,
        )
    except ProviderError as exc:
        log = logger.warning if exc.kind in {"not_configured", "timeout"} else logger.error
        log("web_search failed kind=%s status=%s: %s", exc.kind, exc.status, exc.message)
        return {"success": False, "error": exc.message}
    except Exception as exc:
        logger.exception("web_search failed unexpectedly")
        return {
            "success": False,
            "error": f"Web search failed unexpectedly: {type(exc).__name__}",
        }
