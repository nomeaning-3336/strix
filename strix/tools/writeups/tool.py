"""``search_writeups`` — prior-art retrieval from the local bug-bounty corpus.

Read-only and offline: it searches a locally built index of public writeups and
returns matching reports with their source URLs. No model calls, no network.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agents import function_tool

from strix.tools.writeups import corpus


logger = logging.getLogger(__name__)

_MAX_LIMIT = 10


def _format_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Compact, model-facing view of one writeup."""
    return {
        "title": hit.get("title") or "",
        "url": hit.get("url") or "",
        "program": hit.get("program") or "",
        "weakness": hit.get("weakness") or "",
        "severity": hit.get("severity") or "",
        "bounty": hit.get("bounty") or "",
        "disclosed": hit.get("disclosed") or hit.get("reported") or "",
        "classes": [c for c in str(hit.get("classes") or "").split() if c],
        "asset": hit.get("asset") or "",
        "excerpt": hit.get("snippet") or "",
    }


@function_tool(timeout=30)
async def search_writeups(
    query: str,
    vulnerability_class: str | None = None,
    since_year: int | None = None,
    program: str | None = None,
    limit: int = 5,
) -> str:
    """Search public bug-bounty writeups for prior art on a technique or bug class.

    The corpus is a local index of disclosed HackerOne reports: how other
    researchers found, proved, and escalated a class of bug, with the payloads
    and request/response detail they used. Use it to

    - get technique ideas when a surface looks testable but you are unsure how to
      approach it (bypasses, parameter tricks, chaining a low-impact bug up);
    - check the shape of a valid proof before filing (what evidence a real report
      of this class carries);
    - recognise a known anti-pattern in source (e.g. trusting a client-supplied
      role field).

    This returns **prior art, not findings about your target**: a match means
    someone hit something similar elsewhere, never that your target is
    vulnerable. Verify independently on the actual target before filing.

    Prefer specific queries ("graphql introspection authorization bypass",
    "password reset token reuse") over broad ones ("xss"). If results look
    unrelated, try the technique vocabulary rather than the target's wording.

    Args:
        query: What to look for — bug class plus technique, e.g.
            "cache poisoning unkeyed header" or "idor uuid enumeration".
        vulnerability_class: Optional exact class filter, e.g. ``idor``,
            ``ssrf``, ``xss``, ``rce``, ``auth_bypass``, ``race_condition``,
            ``request_smuggling``, ``prototype_pollution``, ``business_logic``.
        since_year: Optional lower bound on the report's disclosure year.
        program: Optional program-name filter (substring), e.g. ``gitlab``.
        limit: Max writeups to return (1-10, default 5).
    """
    query = (query or "").strip()
    if not query and not (vulnerability_class or program):
        return json.dumps(
            {"success": False, "error": "Provide a query, a vulnerability_class, or a program"},
            ensure_ascii=False,
        )

    bounded = max(1, min(int(limit or 5), _MAX_LIMIT))
    try:
        # SQLite reads are blocking; keep them off the event loop so parallel
        # tool calls are not serialized behind this one.
        hits = await asyncio.to_thread(
            corpus.search,
            query,
            limit=bounded,
            vulnerability_class=vulnerability_class,
            since_year=since_year,
            program=program,
        )
    except Exception as exc:  # noqa: BLE001 - a corpus problem must not fail a scan
        logger.warning("writeup corpus search failed", exc_info=True)
        return json.dumps(
            {"success": False, "error": f"writeup corpus unavailable: {exc}"},
            ensure_ascii=False,
        )

    if not hits:
        _, status = await asyncio.to_thread(corpus.ensure_index)
        note = (
            "No matching writeups. Try broader or different technique vocabulary."
            if status != "missing_corpus"
            else (
                "The writeup corpus is not installed. Build it with "
                "'python scripts/build_writeup_corpus.py'."
            )
        )
        return json.dumps(
            {
                "success": True,
                "query": query,
                "count": 0,
                "writeups": [],
                "note": note,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "success": True,
            "query": query,
            "count": len(hits),
            "note": (
                "Prior art from other targets — not evidence about the current target. "
                "Verify any technique here against the actual target before filing."
            ),
            "writeups": [_format_hit(hit) for hit in hits],
        },
        ensure_ascii=False,
        default=str,
    )


__all__ = ["search_writeups"]
