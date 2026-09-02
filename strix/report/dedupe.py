"""SDK-native vulnerability-report deduplication."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from agents.models.interface import ModelTracing
from openai.types.responses import ResponseOutputMessage

from strix.config import load_settings
from strix.config.models import (
    StrixProvider,
    configure_sdk_model_defaults,
)
from strix.core.inputs import make_model_settings
from strix.report.state import get_global_report_state


if TYPE_CHECKING:
    from agents.items import ModelResponse
    from agents.model_settings import ModelSettings
    from agents.models.interface import Model

    from strix.config.settings import DedupeSettings


logger = logging.getLogger(__name__)


def _dedupe_model_settings(
    dedupe: DedupeSettings, model_name: str, request_timeout: float | None
) -> ModelSettings:
    llm = load_settings().llm
    return make_model_settings(
        dedupe.reasoning_effort,
        model_name=model_name,
        force_required_tool_choice=False,
        request_timeout=request_timeout,
        # The main model's headers apply only when dedupe falls back to the main
        # model; a dedicated dedupe model may route to another provider, which
        # must never receive the main endpoint's credentials. A dedicated model
        # gets its own DEDUPE_LLM_EXTRA_HEADERS instead.
        extra_headers=dedupe.extra_headers if dedupe.model else llm.extra_headers,
        has_tools=False,
    )


def resolve_dedupe_model(dedupe: DedupeSettings, model_name: str) -> Model:
    """Resolve the dedupe model, bound to its own endpoint when it has one.

    Credentials can't ride on the request: every model implementation already
    passes its own ``api_key``/``base_url``, so the same keys in ``extra_args``
    collide with them and raise before anything is sent. A provider bound to the
    dedupe endpoint keeps it apart from the main model's process-wide defaults.
    """
    api_key = (dedupe.api_key or "").strip() if dedupe.model else ""
    api_base = (dedupe.api_base or "").strip() if dedupe.model else ""
    if not (api_key or api_base):
        return StrixProvider().get_model(model_name)
    return StrixProvider(api_key=api_key or None, base_url=api_base or None).get_model(model_name)


DEDUPE_SYSTEM_PROMPT = """You are an expert vulnerability report deduplication judge.
Your task is to determine if a candidate vulnerability report describes the SAME vulnerability
as any existing report.

CRITICAL DEDUPLICATION RULES:

1. SAME VULNERABILITY means:
   - Same root cause (e.g., "missing input validation" not just "SQL injection")
   - Same affected component/endpoint/file (exact match or clear overlap)
   - Same exploitation method or attack vector
   - Would be fixed by the same code change/patch

2. NOT DUPLICATES if:
   - Different endpoints even with same vulnerability type (e.g., SQLi in /login vs /search)
   - Different parameters in same endpoint (e.g., XSS in 'name' vs 'comment' field)
   - Different root causes (e.g., stored XSS vs reflected XSS in same field)
   - Different severity levels due to different impact
   - One is authenticated, other is unauthenticated

3. ARE DUPLICATES even if:
   - Titles are worded differently
   - Descriptions have different level of detail
   - PoC uses different payloads but exploits same issue
   - One report is more thorough than another
   - Minor variations in technical analysis

4. DEPENDENCY-CVE reports use package identity:
   - Same CVE and same package/ecosystem is a duplicate
   - Same CVE but different package/ecosystem is NOT a duplicate
   - Same package/ecosystem but different CVE is NOT a duplicate

COMPARISON GUIDELINES:
- Focus on the technical root cause, not surface-level similarities
- Same vulnerability type (SQLi, XSS) doesn't mean duplicate - location matters
- Consider the fix: would fixing one also fix the other?
- When uncertain, lean towards NOT duplicate

FIELDS TO ANALYZE:
- title, description: General vulnerability info
- target, endpoint, method: Exact location of vulnerability
- technical_analysis: Root cause details
- poc_description: How it's exploited
- impact: What damage it can cause

Respond with a single JSON object and nothing else:

{
  "is_duplicate": true,
  "duplicate_id": "vuln-0001",
  "confidence": 0.95,
  "reason": "Both reports describe SQL injection in /api/login via the username parameter"
}

Or, if not a duplicate:

{
  "is_duplicate": false,
  "duplicate_id": "",
  "confidence": 0.90,
  "reason": "Different endpoints: candidate is /api/search, existing is /api/login"
}

Rules:
- ``is_duplicate`` is a boolean.
- ``duplicate_id`` is the exact id from existing reports, or "" if not a duplicate.
- ``confidence`` is a number between 0 and 1.
- ``reason`` is a specific explanation mentioning endpoint/parameter/root cause.
- Output ONLY the JSON object — no surrounding prose, no code fences."""


def _prepare_report_for_comparison(report: dict[str, Any]) -> dict[str, Any]:
    relevant_fields = [
        "id",
        "title",
        "description",
        "impact",
        "target",
        "technical_analysis",
        "poc_description",
        "endpoint",
        "method",
        "cve",
        "dependency_metadata",
    ]

    cleaned = {}
    for field in relevant_fields:
        if report.get(field):
            value = report[field]
            if isinstance(value, str) and len(value) > 8000:
                value = value[:8000] + "...[truncated]"
            cleaned[field] = value

    return cleaned


def _dependency_identity(report: dict[str, Any]) -> tuple[str, str, str] | None:
    metadata = report.get("dependency_metadata")
    if not isinstance(metadata, dict):
        return None

    raw_cve = report.get("cve")
    raw_package = metadata.get("package_name")
    if not raw_cve or not raw_package:
        return None

    cve = str(raw_cve).strip().upper()
    ecosystem = str(metadata.get("package_ecosystem") or "").strip().lower()
    package_name = str(raw_package).strip().lower()
    if not cve or not package_name:
        return None
    return cve, ecosystem, package_name


def _manifest_path(report: dict[str, Any]) -> str:
    metadata = report.get("dependency_metadata")
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("manifest_path") or "").strip()


def _distinct_manifest_paths(candidate: dict[str, Any], report: dict[str, Any]) -> bool:
    """Same CVE/package observed in two different manifests is two findings.

    Only applies when both sides carry a manifest_path; a missing path keeps
    the legacy CVE/package/ecosystem identity.
    """
    candidate_path = _manifest_path(candidate)
    report_path = _manifest_path(report)
    return bool(candidate_path and report_path and candidate_path != report_path)


def _report_cve(report: dict[str, Any]) -> str:
    return str(report.get("cve") or "").strip().upper()


def _legacy_report_mentions_package(
    report: dict[str, Any],
    *,
    ecosystem: str,
    package_name: str,
) -> bool:
    fields = [
        "title",
        "description",
        "impact",
        "target",
        "technical_analysis",
        "poc_description",
        "evidence",
    ]
    haystack = " ".join(str(report.get(field) or "") for field in fields).lower()
    package_pattern = rf"(?<![\w@./-]){re.escape(package_name)}(?![\w@./-])"
    if re.search(package_pattern, haystack) is None:
        return False
    if not ecosystem:
        return True
    ecosystem_pattern = rf"(?<![\w@./-]){re.escape(ecosystem)}(?![\w@./-])"
    return re.search(ecosystem_pattern, haystack) is not None


def _check_dependency_duplicate(
    candidate: dict[str, Any],
    existing_reports: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidate_identity = _dependency_identity(candidate)
    if candidate_identity is None:
        return None

    cve, ecosystem, package_name = candidate_identity
    found_legacy_same_cve = False
    for report in existing_reports:
        report_identity = _dependency_identity(report)
        if report_identity is not None:
            report_cve, report_ecosystem, report_package_name = report_identity
            if (report_cve, report_package_name) != (cve, package_name):
                continue
            if _distinct_manifest_paths(candidate, report):
                continue
            if report_ecosystem == ecosystem:
                return {
                    "is_duplicate": True,
                    "duplicate_id": str(report.get("id") or "")[:64],
                    "confidence": 1.0,
                    "reason": "Same dependency CVE/package identity",
                }
            if not report_ecosystem or not ecosystem:
                return {
                    "is_duplicate": True,
                    "duplicate_id": str(report.get("id") or "")[:64],
                    "confidence": 1.0,
                    "reason": "Same dependency CVE/package identity with missing ecosystem",
                }
            continue

        if _report_cve(report) != cve:
            continue
        found_legacy_same_cve = True
        if _legacy_report_mentions_package(
            report,
            ecosystem=ecosystem,
            package_name=package_name,
        ):
            return {
                "is_duplicate": True,
                "duplicate_id": str(report.get("id") or "")[:64],
                "confidence": 1.0,
                "reason": "Same dependency CVE/package identity in legacy report",
            }

    if found_legacy_same_cve:
        return None

    package_label = f"{ecosystem}/{package_name}" if ecosystem else package_name
    return {
        "is_duplicate": False,
        "duplicate_id": "",
        "confidence": 1.0,
        "reason": f"No existing dependency report for {cve} in {package_label}",
    }


def _norm_token_text(value: Any) -> str:
    """Lowercase alphanumeric tokens of a value, sorted and space-joined.

    Rewording ("SQL injection in /login" vs "Login endpoint: SQL injection")
    normalises to the same token bag; word order does not matter, but the
    token multiset does, so distinct identifiers stay distinct.
    """
    if value is None:
        return ""
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    return " ".join(sorted(tokens))


def _norm_lower(value: Any) -> str:
    """Lowercase, whitespace-collapsed value (punctuation preserved)."""
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _norm_cwe(value: Any) -> str:
    """``CWE-79``, ``cwe: 79``, ``79`` → ``CWE-79`` (mirrors the SARIF rule id)."""
    digits = "".join(c for c in str(value or "") if c.isdigit())
    return f"CWE-{digits}" if digits else ""


def _report_location(report: dict[str, Any]) -> str:
    """The finding's location: ``endpoint`` when present, else the first code
    location's file/path. Deliberately file-granular (no line): the identity
    must survive a line shift between two concurrent filings of the same sink.
    """
    endpoint = str(report.get("endpoint") or "").strip()
    if endpoint:
        return _norm_lower(endpoint)
    locations = report.get("code_locations")
    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict):
                continue
            path = location.get("file") or location.get("path")
            if isinstance(path, str) and path.strip():
                return _norm_lower(path)
    return ""


def _dependency_key(report: dict[str, Any]) -> str:
    """Deterministic SCA identity: CVE + ecosystem + package + manifest path."""
    metadata = report.get("dependency_metadata")
    if not isinstance(metadata, dict):
        return ""
    cve = str(report.get("cve") or "").strip().upper()
    package_name = str(metadata.get("package_name") or "").strip().lower()
    if not cve or not package_name:
        return ""
    ecosystem = str(metadata.get("package_ecosystem") or "").strip().lower()
    manifest_path = str(metadata.get("manifest_path") or "").strip()
    return f"{cve}|{ecosystem}|{package_name}|{manifest_path}"


def finding_fingerprint(report: dict[str, Any]) -> str | None:
    """Deterministic identity fingerprint for the concurrent-filing re-check.

    LLM dedupe compares a candidate against the snapshot of reports taken when
    the agent started filing; two agents filing the same finding concurrently
    can both pass that check against the same stale snapshot. The fingerprint
    closes that window with a deterministic, no-LLM comparison at commit time:
    ``finding_class + target + location(endpoint | code file) + method + CWE +
    normalised-title tokens`` (dependency findings use CVE/package identity).

    Exact-equality only: the composite is only compared against reports that
    landed after the caller's snapshot, i.e. reports filed within the same
    race window that describe the same underlying finding. Different endpoints,
    methods, CWEs, or targets fingerprint differently.

    Returns ``None`` when the report carries no identity-bearing field, so the
    caller can skip the guard instead of risking a spurious match on bare text.
    """
    dep_key = _dependency_key(report)
    if dep_key:
        return f"dependency\x1f{dep_key}"

    fields = {
        "class": str(report.get("finding_class") or "dynamic").strip().lower(),
        "target": _norm_lower(report.get("target")),
        "location": _report_location(report),
        "method": _norm_lower(report.get("method")),
        "cwe": _norm_cwe(report.get("cwe")),
        "title": _norm_token_text(report.get("title")),
    }
    if not any(fields[key] for key in ("target", "location", "method", "cwe", "title")):
        return None
    # \x1f cannot appear in any normalised value, so string equality is
    # unambiguous even when some fields are empty on both sides.
    return "\x1f".join(f"{key}:{fields[key]}" for key in fields)


def _parse_dedupe_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in dedupe response: {content[:500]}")
    parsed = json.loads(text[start : end + 1])

    duplicate_id = str(parsed.get("duplicate_id") or "")[:64]
    reason = str(parsed.get("reason") or "")[:500]
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "is_duplicate": bool(parsed.get("is_duplicate", False)),
        "duplicate_id": duplicate_id,
        "confidence": confidence,
        "reason": reason,
    }


def _extract_text(response: ModelResponse) -> str:
    parts: list[str] = []
    for item in response.output:
        if not isinstance(item, ResponseOutputMessage):
            continue
        for chunk in item.content:
            text = getattr(chunk, "text", None)
            if text:
                parts.append(text)
    return "".join(parts)


async def check_duplicate(
    candidate: dict[str, Any], existing_reports: list[dict[str, Any]]
) -> dict[str, Any]:
    if not existing_reports:
        return {
            "is_duplicate": False,
            "duplicate_id": "",
            "confidence": 1.0,
            "reason": "No existing reports to compare against",
        }

    dependency_duplicate = _check_dependency_duplicate(candidate, existing_reports)
    if dependency_duplicate is not None:
        return dependency_duplicate

    try:
        settings = load_settings()
        dedupe = settings.dedupe
        model_name = (dedupe.model or "").strip() or settings.llm.model
        if not model_name:
            return {
                "is_duplicate": False,
                "duplicate_id": "",
                "confidence": 0.0,
                "reason": "No LLM model configured; skipping dedupe check",
            }

        candidate_cleaned = _prepare_report_for_comparison(candidate)
        existing_cleaned = [_prepare_report_for_comparison(r) for r in existing_reports]
        comparison_data = {"candidate": candidate_cleaned, "existing_reports": existing_cleaned}

        user_msg = (
            f"Compare this candidate vulnerability against existing reports:\n\n"
            f"{json.dumps(comparison_data, indent=2)}\n\n"
            f"Respond with ONLY the JSON object described in the system prompt."
        )

        configure_sdk_model_defaults(settings)
        resolved_model = model_name.strip()
        model = resolve_dedupe_model(dedupe, resolved_model)
        response = await model.get_response(
            system_instructions=DEDUPE_SYSTEM_PROMPT,
            input=user_msg,
            model_settings=_dedupe_model_settings(dedupe, resolved_model, settings.llm.timeout),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
        report_state = get_global_report_state()
        if report_state is not None:
            report_state.record_sdk_usage(
                agent_id="dedupe",
                agent_name="dedupe",
                model=resolved_model,
                usage=response.usage,
            )
        content = _extract_text(response)
        if not content:
            return {
                "is_duplicate": False,
                "duplicate_id": "",
                "confidence": 0.0,
                "reason": "Empty response from LLM",
            }

        result = _parse_dedupe_response(content)

        logger.info(
            "Deduplication check: is_duplicate=%s, confidence=%.2f, reason=%s",
            result["is_duplicate"],
            result["confidence"],
            result["reason"][:100],
        )

    except Exception as e:
        logger.exception("Error during vulnerability deduplication check")
        return {
            "is_duplicate": False,
            "duplicate_id": "",
            "confidence": 0.0,
            "reason": f"Deduplication check failed: {e}",
            "error": str(e),
        }
    else:
        return result
