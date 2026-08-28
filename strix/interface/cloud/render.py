"""Output rendering for `strix cloud` commands."""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeGuard

from rich.markup import escape
from rich.table import Table

from strix.interface.terminal_text import sanitize_terminal_text


if TYPE_CHECKING:
    from collections.abc import Iterable

    from rich.console import Console


_MAX_TABLE_COLUMNS = 8
_MAX_CELL_LENGTH = 60
_MAX_DETAIL_FIELDS = 36
_MAX_NESTED_PREVIEW = 5
_NARROW_TABLE_WIDTH = 120
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_INTERNAL_COLUMNS = frozenset(
    {
        "organization_id",
        "user_id",
        "userId",
        "installation_id",
        "added_by",
        "created_by",
        "connected_by",
        "invited_by",
        "uploaded_by",
        "avatarUrl",
    }
)
_LOSSLESS_DETAIL_KEYS = frozenset(
    {
        "api_token",
        "command",
        "docker_command",
        "enrollment_command",
        "secret",
        "signing_secret",
        "token",
        "webhook_secret",
    }
)

_PREFERRED_KEYS = (
    "name",
    "title",
    "repository_full_name",
    "pr_number",
    "pr_title",
    "head_branch",
    "base_branch",
    "verdict",
    "domain",
    "target",
    "default_branch",
    "branch",
    "display_number",
    "status",
    "state",
    "severity",
    "cve",
    "cvss",
    "finding_type",
    "findings_count",
    "open_findings_count",
    "role",
    "email",
    "firstName",
    "lastName",
    "url",
    "provider",
    "secret_prefix",
    "events",
    "action",
    "resource_type",
    "response_status",
    "attempts",
    "scan_type",
    "engagement_type",
    "estimated_credits",
    "cron_expression",
    "timezone",
    "next_run_at",
    "is_active",
    "created_at",
    "updated_at",
    "expires_at",
    "last_used_at",
    "id",
)

_LIST_ENVELOPE_KEYS = frozenset(
    {
        "items",
        "data",
        "scans",
        "vulnerabilities",
        "findings",
        "domains",
        "repositories",
        "repos",
        "schedules",
        "reviews",
        "pr_reviews",
        "workspaces",
        "members",
        "invitations",
        "integrations",
        "connectors",
        "webhooks",
        "deliveries",
        "entries",
        "documents",
        "docs",
        "policies",
        "tokens",
        "uploads",
        "events",
        "audit_logs",
        "logs",
    }
)
_ENVELOPE_METADATA_KEYS = frozenset(
    {
        "total",
        "total_count",
        "totalCount",
        "count",
        "page",
        "limit",
        "page_size",
        "pageSize",
        "has_more",
        "hasMore",
        "next_cursor",
        "nextCursor",
        "meta",
        "pagination",
        "summary",
        "stats",
        "scansThisMonth",
        "organization_id",
    }
)

_VIEW_COLUMNS: dict[str, tuple[str, ...]] = {
    "GET /scans": (
        "title",
        "target",
        "engagement_type",
        "scan_type",
        "status",
        "findings_count",
        "created_at",
        "id",
    ),
    "GET /vulnerabilities": (
        "display_number",
        "title",
        "severity",
        "status",
        "target",
        "cvss",
        "finding_type",
        "id",
    ),
    "GET /pr-reviews": (
        "repository",
        "pull_request",
        "branches",
        "status",
        "verdict",
        "findings",
        "updated_at",
        "id",
    ),
    "GET /integrations": (
        "provider",
        "account_login",
        "installation_id",
        "instance_url",
        "status",
        "repository_selection",
        "default_collection_name",
        "connected_at",
    ),
    "GET /domains": (
        "domain",
        "asset_type",
        "verified",
        "last_scan_at",
        "context",
        "tags",
        "business_unit",
        "id",
    ),
    "GET /repositories": (
        "full_name",
        "provider",
        "default_branch",
        "pr_review_enabled",
        "last_scan_at",
        "tags",
        "business_unit",
        "id",
    ),
    "GET /knowledge": (
        "title",
        "source_type",
        "source_id",
        "tags",
        "severity",
        "status",
        "updated_at",
        "id",
    ),
    "GET /knowledge/repos/{repo}/entries": (
        "title",
        "source_type",
        "source_id",
        "tags",
        "severity",
        "status",
        "updated_at",
        "id",
    ),
    "GET /knowledge/policies": (
        "policy_key",
        "policy_type",
        "is_active",
        "policy_value",
        "updated_at",
        "created_at",
        "id",
    ),
    "GET /domains/{domainId}/test-users": (
        "label",
        "username",
        "mfa_method",
        "mfa_email",
        "has_password",
        "login_url",
        "updated_at",
        "id",
    ),
    "GET /tokens": (
        "name",
        "type",
        "status",
        "scopes",
        "secret_prefix",
        "expires_at",
        "last_used_at",
        "id",
    ),
    "GET /webhooks": ("url", "events", "is_active", "last_delivery_at", "created_at", "id"),
    "GET /webhooks/{webhookId}/deliveries": (
        "event",
        "status",
        "response_status",
        "attempts",
        "created_at",
        "delivered_at",
        "id",
    ),
}


def _is_record(value: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict)


def _is_list(value: object) -> TypeGuard[list[Any]]:
    return isinstance(value, list)


def json_mode(*, flag: bool) -> bool:
    """JSON output is on when the flag is set or when stdout is not a terminal."""
    return flag or not sys.stdout.isatty()


def emit(  # noqa: PLR0911, PLR0912
    console: Console,
    data: Any,
    *,
    as_json: bool,
    row_numbers: bool = False,
    omit_columns: frozenset[str] = frozenset(),
    hint: str | None = None,
    view: str | None = None,
    warning: str | None = None,
) -> None:
    if as_json:
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
        return
    if warning:
        console.print(f"[bold yellow]Save this now:[/] {escape(sanitize_terminal_text(warning))}")
    if view == "source_manifest" and _is_record(data):
        _print_source_manifest(console, data)
        return
    if view == "GET /analytics/scan-frequency":
        _print_scan_frequency(console, data)
        return
    if view in {"GET /analytics/overview", "GET /analytics/stats"} and _is_record(data):
        _print_analytics(console, data)
        return
    if view == "GET /integrations":
        integration_rows = _integration_rows(data)
        if integration_rows is not None:
            _print_table(
                console,
                integration_rows,
                row_numbers=row_numbers,
                omit_columns=omit_columns,
                hint=hint,
                view=view,
            )
            return
    if view == "GET /tokens":
        token_rows = _token_rows(data)
        if token_rows is not None:
            _print_table(
                console,
                token_rows,
                row_numbers=row_numbers,
                omit_columns=omit_columns,
                hint=hint,
                view=view,
            )
            return
    if view == "GET /scans":
        scan_rows = _scan_rows(data)
        if scan_rows is not None:
            _print_table(
                console,
                scan_rows,
                row_numbers=False,
                omit_columns=omit_columns,
                hint="Inspect one scan with `strix cloud scans get ID`.",
                view=view,
            )
            return
    if view == "GET /vulnerabilities":
        vulnerability_rows = _list_of_dicts(data)
        if vulnerability_rows is not None:
            _print_table(
                console,
                vulnerability_rows,
                row_numbers=False,
                omit_columns=omit_columns | frozenset({"scan_id"}),
                hint="Inspect one finding with `strix cloud vulns get ID`.",
                view=view,
            )
            return
    if view == "GET /pr-reviews":
        review_rows = _pr_review_rows(data)
        if review_rows is not None:
            _print_table(
                console,
                review_rows,
                row_numbers=False,
                omit_columns=omit_columns,
                hint="Use `strix cloud pr-reviews get ID` for one review.",
                view=view,
            )
            return
    rows = _list_of_dicts(data)
    if rows is not None:
        _print_table(
            console,
            rows,
            row_numbers=row_numbers,
            omit_columns=omit_columns,
            hint=hint,
            view=view,
        )
        return
    if isinstance(data, str):
        console.print(sanitize_terminal_text(data), markup=False)
        return
    if _is_record(data):
        _print_detail(console, data)
        return
    console.print_json(json.dumps(data, default=str))


def _list_of_dicts(data: Any) -> list[dict[str, Any]] | None:
    """Extract a record list from a raw list or a common paginated envelope."""
    if _is_record(data):
        # Some endpoints wrap the actual envelope in a top-level ``data`` or
        # ``result`` object. Only recurse through an object wrapper; a list in
        # ``data`` is handled with the other named envelope keys below.
        for wrapper in ("data", "result"):
            nested = data.get(wrapper)
            if _is_record(nested):
                nested_rows = _list_of_dicts(nested)
                if nested_rows is not None:
                    return nested_rows
        candidates = [
            (key, value)
            for key, value in data.items()
            if key in _LIST_ENVELOPE_KEYS
            and _is_list(value)
            and all(_is_record(item) for item in value)
        ]
        if len(candidates) == 1:
            list_key, records = candidates[0]
            other_keys = set(data) - {list_key}
            if list_key in {"items", "data"} or other_keys <= _ENVELOPE_METADATA_KEYS:
                data = records
    if not _is_list(data):
        return None
    if not data:
        return []
    records = [item for item in data if _is_record(item)]
    if len(records) != len(data):
        return None
    return records


def _integration_rows(data: Any) -> list[dict[str, Any]] | None:
    """Flatten the two integration collections into one compact human view."""
    if not _is_record(data):
        return _list_of_dicts(data)
    rows: list[dict[str, Any]] = []
    found_collection = False
    for key in ("integrations", "merge_accounts"):
        collection = data.get(key)
        if not _is_list(collection):
            continue
        found_collection = True
        for item in collection:
            if not _is_record(item):
                continue
            row = dict(item)
            if not row.get("account_login") and row.get("account_email"):
                row["account_login"] = row["account_email"]
            rows.append(row)
    return rows if found_collection else None


def _token_rows(data: Any) -> list[dict[str, Any]] | None:
    """Add an explicit lifecycle state to token rows for the human view."""
    records = _list_of_dicts(data)
    if records is None:
        return None
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        if record.get("revoked_at"):
            row["status"] = "revoked"
        elif _timestamp_has_passed(record.get("expires_at")):
            row["status"] = "expired"
        else:
            row["status"] = "active"
        rows.append(row)
    return rows


def _timestamp_has_passed(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed <= datetime.now(UTC)


def _scan_rows(data: Any) -> list[dict[str, Any]] | None:
    """Flatten the nested target and finding summaries returned by scan lists."""
    records = _list_of_dicts(data)
    if records is None:
        return None
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        if not row.get("title") and row.get("name"):
            row["title"] = row["name"]
        if not row.get("id") and isinstance(row.get("scan_id"), str):
            row["id"] = row["scan_id"]

        targets: list[str] = []
        urls = record.get("urls")
        if _is_list(urls):
            targets.extend(url.strip() for url in urls if isinstance(url, str) and url.strip())
        repositories = record.get("repositories")
        if _is_list(repositories):
            for repository in repositories:
                if not _is_record(repository):
                    continue
                identifier = str(
                    repository.get("full_name")
                    or repository.get("name")
                    or repository.get("url")
                    or ""
                ).strip()
                branch = str(repository.get("branch") or "").strip()
                if identifier:
                    targets.append(f"{identifier} @ {branch}" if branch else identifier)
        if targets:
            visible_targets = targets[:2]
            summary = " | ".join(visible_targets)
            if len(targets) > len(visible_targets):
                summary += f" (+{len(targets) - len(visible_targets)} more)"
            row["target"] = summary

        findings = record.get("findings")
        if _is_record(findings) and findings.get("total") is not None:
            row["findings_count"] = findings["total"]
        rows.append(row)
    return rows


def _pr_review_rows(data: Any) -> list[dict[str, Any]] | None:
    """Collapse related PR fields into an eight-column, action-oriented human view."""
    records = _list_of_dicts(data)
    if records is None:
        return None
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        number = record.get("pr_number")
        title = str(record.get("pr_title") or "").strip()
        row["repository"] = record.get("repository_full_name") or record.get("repository")
        row["pull_request"] = " ".join(
            part for part in (f"#{number}" if number is not None else "", title) if part
        )
        head = str(record.get("head_branch") or "").strip()
        base = str(record.get("base_branch") or "").strip()
        row["branches"] = f"{head} → {base}" if head and base else head or base
        total = record.get("findings_count")
        opened = record.get("open_findings_count")
        if isinstance(total, int) and isinstance(opened, int):
            row["findings"] = f"{opened} open / {total} total"
        elif isinstance(total, int):
            row["findings"] = total
        rows.append(row)
    return rows


def _print_table(
    console: Console,
    rows: list[dict[str, Any]],
    *,
    row_numbers: bool = False,
    omit_columns: frozenset[str] = frozenset(),
    hint: str | None = None,
    view: str | None = None,
) -> None:
    if not rows:
        console.print("[dim]No items.[/]")
        if hint:
            console.print(f"[dim]{escape(sanitize_terminal_text(hint))}[/]")
        return
    integration_view = view == "GET /integrations"
    visible_internal: set[str] = {"installation_id"} if integration_view else set()
    view_omissions: set[str] = {"id"} if integration_view else set()
    omit_columns = omit_columns | (_INTERNAL_COLUMNS - visible_internal) | view_omissions
    preferred = _VIEW_COLUMNS.get(view or "", _PREFERRED_KEYS)
    columns: list[str] = [
        key for key in preferred if key not in omit_columns and any(key in row for row in rows)
    ]
    for row in rows:
        for key in row:
            if (
                key not in columns
                and key not in omit_columns
                and len(columns) < _MAX_TABLE_COLUMNS
                and not isinstance(row[key], dict | list)
            ):
                columns.append(key)
    columns = columns[:_MAX_TABLE_COLUMNS]
    if console.width < _NARROW_TABLE_WIDTH:
        _print_cards(console, rows, columns, row_numbers=row_numbers)
        _print_long_identifiers(console, rows)
        console.print(f"[dim]{len(rows)} item(s). Use --json for the full records.[/]")
        if hint:
            console.print(f"[dim]{escape(sanitize_terminal_text(hint))}[/]")
        return
    table = Table(show_lines=False)
    if row_numbers:
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
    for column in columns:
        table.add_column(escape(_human_label(column)))
    for index, row in enumerate(rows, start=1):
        cells = [escape(_cell(row.get(column))) for column in columns]
        if row_numbers:
            cells.insert(0, str(index))
        table.add_row(*cells)
    console.print(table)
    _print_long_identifiers(console, rows)
    console.print(f"[dim]{len(rows)} item(s). Use --json for the full records.[/]")
    if hint:
        console.print(f"[dim]{escape(sanitize_terminal_text(hint))}[/]")


def _print_cards(
    console: Console,
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    row_numbers: bool,
) -> None:
    """Render list rows legibly when a terminal is too narrow for a table."""
    for index, row in enumerate(rows, start=1):
        parts = [
            f"[bold]{escape(_human_label(column))}:[/] {escape(_cell(row.get(column)))}"
            for column in columns
            if row.get(column) is not None
        ]
        prefix = f"[cyan]{index}.[/] " if row_numbers else "[cyan]•[/] "
        if not parts:
            console.print(prefix.rstrip())
            continue
        console.print(prefix + parts[0], soft_wrap=True)
        continuation = "   " if row_numbers else "  "
        for part in parts[1:]:
            console.print(continuation + part, soft_wrap=True)


def _print_long_identifiers(console: Console, rows: list[dict[str, Any]]) -> None:
    """Print opaque IDs losslessly when the compact cell view shortens them."""
    identifiers = [
        (index, row, str(row["id"]))
        for index, row in enumerate(rows, start=1)
        if row.get("id") is not None and len(str(row["id"])) > _MAX_CELL_LENGTH
    ]
    if not identifiers:
        return
    console.print("[dim]Copyable IDs:[/]")
    for index, row, identifier in identifiers:
        label = next(
            (
                str(row[key])
                for key in ("title", "name", "label", "domain", "full_name")
                if row.get(key)
            ),
            f"item {index}",
        )
        console.print(
            f"  {index}. {sanitize_terminal_text(label)}: {sanitize_terminal_text(identifier)}",
            markup=False,
            soft_wrap=True,
        )


def _print_detail(console: Console, data: dict[str, Any]) -> None:
    """Render one API record as a readable field/value view."""
    keys = [key for key in _PREFERRED_KEYS if key in data and key not in _INTERNAL_COLUMNS]
    keys.extend(key for key in data if key not in keys and key not in _INTERNAL_COLUMNS)
    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2))
    table.add_column("field", style="bold cyan", no_wrap=True)
    table.add_column("value", overflow="fold")
    populated_keys = [key for key in keys if data.get(key) is not None]
    visible_keys = populated_keys[:_MAX_DETAIL_FIELDS]
    lossless_fields: list[tuple[str, Any]] = []
    for key in visible_keys:
        value = data.get(key)
        if _is_lossless_detail(key, value):
            lossless_fields.append((key, value))
            continue
        rendered = _nested_summary(value) if _is_record(value) or _is_list(value) else _cell(value)
        table.add_row(escape(_human_label(key)), escape(rendered))
    if table.row_count:
        console.print(table)
    for key, value in lossless_fields:
        console.print(f"{_human_label(key)}:", style="bold cyan", markup=False)
        console.print(_lossless_detail_value(value), markup=False, soft_wrap=True)
    if len(populated_keys) > len(visible_keys):
        console.print(
            f"[dim]{len(populated_keys) - len(visible_keys)} additional field(s) omitted from "
            "this view.[/]"
        )
    console.print("[dim]Use --json for the lossless machine-readable record.[/]")


def _is_lossless_detail(key: str, value: Any) -> bool:
    """Keep one-time credentials and enrollment commands complete and copyable."""
    sensitive_key = key in _LOSSLESS_DETAIL_KEYS or key.endswith(("_token", "_secret"))
    return sensitive_key and not _is_record(value) and not _is_list(value)


def _lossless_detail_value(value: Any) -> str:
    """Preserve structural newlines while making every other control byte visible."""
    return "\n".join(sanitize_terminal_text(line) for line in str(value).split("\n"))


def _nested_summary(value: dict[str, Any] | list[Any]) -> str:
    """Bound nested records so one detail response cannot flood a terminal."""
    if _is_record(value):
        scalar_items = [
            (nested_key, nested_value)
            for nested_key, nested_value in value.items()
            if not isinstance(nested_value, dict | list) and nested_value is not None
        ]
        lines = [
            f"{_human_label(str(nested_key))}: {_cell(nested_value)}"
            for nested_key, nested_value in scalar_items[:_MAX_NESTED_PREVIEW]
        ]
        omitted = len(value) - len(lines)
        if omitted > 0:
            lines.append(f"… {omitted} more field(s)")
        return "\n".join(lines) if lines else f"{len(value)} nested field(s)"
    if not _is_list(value):
        return "none"
    if not value:
        return "none"
    if all(not isinstance(item, dict | list) for item in value):
        preview = ", ".join(_cell(item) for item in value[:12])
        if len(value) > 12:
            preview += f", … {len(value) - 12} more"
        return preview
    records = [item for item in value if _is_record(item)]
    lines = [f"{len(value)} item(s)"]
    for record in records[:_MAX_NESTED_PREVIEW]:
        label = record.get("title") or record.get("name") or record.get("message")
        severity = record.get("severity")
        status = record.get("status") or record.get("state")
        prefix = " / ".join(_cell(part) for part in (severity, status) if part)
        summary = str(label or record.get("id") or "record")
        lines.append(f"- {prefix + ': ' if prefix else ''}{_cell(summary)}")
    if len(value) > len(records[:_MAX_NESTED_PREVIEW]):
        lines.append(f"… {len(value) - len(records[:_MAX_NESTED_PREVIEW])} more; use --json")
    return "\n".join(lines)


def _print_source_manifest(console: Console, data: dict[str, Any]) -> None:
    source = data.get("source")
    manifest = source if _is_record(source) else data
    files = manifest.get("files")
    summary = {key: value for key, value in manifest.items() if key != "files"}
    _print_detail(console, summary)
    if _is_list(files):
        console.print(f"\n[bold]Selected files ({len(files):,})[/]")
        for path in files:
            console.print(f"  {escape(sanitize_terminal_text(path))}", soft_wrap=True)


def _print_analytics(console: Console, data: dict[str, Any]) -> None:
    rows = list(_flatten_summary(data))
    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2))
    table.add_column("metric", style="bold cyan")
    table.add_column("value", overflow="fold")
    for label, value in rows[:_MAX_DETAIL_FIELDS]:
        table.add_row(escape(label), escape(value))
    console.print(table)
    if len(rows) > _MAX_DETAIL_FIELDS:
        console.print(
            f"[dim]Showing {_MAX_DETAIL_FIELDS} of {len(rows)} summary metrics. "
            "Use --json for all data.[/]"
        )
    else:
        console.print("[dim]Use --json for the complete analytics record.[/]")


def _flatten_summary(value: Any, prefix: str = "", depth: int = 0) -> Iterable[tuple[str, str]]:
    if _is_record(value) and depth < 4:
        for key, nested in value.items():
            label = f"{prefix} / {_human_label(key)}" if prefix else _human_label(key)
            yield from _flatten_summary(nested, label, depth + 1)
        return
    if _is_list(value):
        if all(not isinstance(item, dict | list) for item in value):
            yield prefix, _nested_summary(value)
        else:
            yield prefix, f"{len(value)} data point(s)"
        return
    yield prefix or "value", _cell(value)


def _print_scan_frequency(console: Console, data: Any) -> None:
    rows = _find_record_series(data)
    if rows is None:
        if _is_record(data):
            _print_analytics(console, data)
        else:
            console.print_json(json.dumps(data, default=str))
        return
    nonzero = [row for row in rows if _row_has_activity(row)]
    selected = (nonzero[-30:] if nonzero else rows[-14:]) if rows else []
    _print_table(console, selected, view="GET /analytics/scan-frequency")
    if rows:
        qualifier = "non-zero" if nonzero else "most recent"
        console.print(
            f"[dim]Showing {len(selected)} {qualifier} point(s) from {len(rows)} total. "
            "Use --json for the full series.[/]"
        )


def _find_record_series(data: Any) -> list[dict[str, Any]] | None:
    direct = _list_of_dicts(data)
    if direct is not None:
        return direct
    if _is_record(data):
        candidates = [
            series for value in data.values() if (series := _find_record_series(value)) is not None
        ]
        if candidates:
            return max(candidates, key=len)
    return None


def _row_has_activity(row: dict[str, Any]) -> bool:
    count_keys = ("count", "scans", "scan_count", "total", "value")
    return any(isinstance(row.get(key), int | float) and row[key] > 0 for key in count_keys)


def _human_label(column: str) -> str:
    column = sanitize_terminal_text(column)
    if column == "secret_prefix":
        return "prefix"
    labels = {
        "repository_full_name": "repo",
        "pr_number": "PR",
        "pr_title": "title",
        "head_branch": "head",
        "base_branch": "base",
        "findings_count": "findings",
        "open_findings_count": "open",
        "display_number": "finding",
        "created_at": "created",
        "updated_at": "updated",
        "expires_at": "expires",
        "last_used_at": "last used",
    }
    return labels.get(column, _CAMEL_BOUNDARY.sub(" ", column).replace("_", " ").lower())


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if _is_list(value) and all(not _is_record(item) and not _is_list(item) for item in value):
        text = ", ".join(str(item) for item in value)
    elif _is_record(value) or _is_list(value):
        text = f"{len(value)} item(s)"
    else:
        text = str(value)
    text = sanitize_terminal_text(text)
    if len(text) > _MAX_CELL_LENGTH:
        return text[: _MAX_CELL_LENGTH - 1] + "…"
    return text
