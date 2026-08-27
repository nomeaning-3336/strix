"""Output rendering for `strix cloud` commands."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

from rich.markup import escape
from rich.table import Table


if TYPE_CHECKING:
    from rich.console import Console


_MAX_TABLE_COLUMNS = 8
_MAX_CELL_LENGTH = 60
_NARROW_TABLE_WIDTH = 120
_INTERNAL_COLUMNS = frozenset(
    {
        "organization_id",
        "user_id",
        "userId",
        "installation_id",
        "created_by",
        "avatarUrl",
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


def json_mode(*, flag: bool) -> bool:
    """JSON output is on when the flag is set or when stdout is not a terminal."""
    return flag or not sys.stdout.isatty()


def emit(
    console: Console,
    data: Any,
    *,
    as_json: bool,
    row_numbers: bool = False,
    omit_columns: frozenset[str] = frozenset(),
    hint: str | None = None,
) -> None:
    if as_json:
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
        return
    rows = _list_of_dicts(data)
    if rows is not None:
        _print_table(
            console,
            rows,
            row_numbers=row_numbers,
            omit_columns=omit_columns,
            hint=hint,
        )
        return
    if isinstance(data, str):
        console.print(data)
        return
    if isinstance(data, dict):
        _print_detail(console, data)
        return
    console.print_json(json.dumps(data, default=str))


def _list_of_dicts(data: Any) -> list[dict[str, Any]] | None:
    if isinstance(data, dict) and len(data) >= 1:
        lists = [v for v in data.values() if isinstance(v, list)]
        scalars = [v for v in data.values() if not isinstance(v, list | dict)]
        if len(lists) == 1 and not scalars:
            data = lists[0]
    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(item, dict) for item in data):
        return None
    return data


def _print_table(
    console: Console,
    rows: list[dict[str, Any]],
    *,
    row_numbers: bool = False,
    omit_columns: frozenset[str] = frozenset(),
    hint: str | None = None,
) -> None:
    omit_columns = omit_columns | _INTERNAL_COLUMNS
    columns: list[str] = [
        key
        for key in _PREFERRED_KEYS
        if key not in omit_columns and any(key in row for row in rows)
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
        console.print(f"[dim]{len(rows)} item(s). Use --json for the full records.[/]")
        if hint:
            console.print(f"[dim]{hint}[/]")
        return
    table = Table(show_lines=False)
    if row_numbers:
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
    for column in columns:
        table.add_column(column)
    for index, row in enumerate(rows, start=1):
        cells = [_cell(row.get(column)) for column in columns]
        if row_numbers:
            cells.insert(0, str(index))
        table.add_row(*cells)
    console.print(table)
    console.print(f"[dim]{len(rows)} item(s). Use --json for the full records.[/]")
    if hint:
        console.print(f"[dim]{hint}[/]")


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
        console.print(prefix + "  [dim]·[/]  ".join(parts), soft_wrap=False)


def _print_detail(console: Console, data: dict[str, Any]) -> None:
    """Render one API record as a readable field/value view."""
    keys = [key for key in _PREFERRED_KEYS if key in data and key not in _INTERNAL_COLUMNS]
    keys.extend(key for key in data if key not in keys and key not in _INTERNAL_COLUMNS)
    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2))
    table.add_column("field", style="bold cyan", no_wrap=True)
    table.add_column("value", overflow="fold")
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, indent=2, default=str)
        else:
            rendered = _cell(value)
        table.add_row(_human_label(key), rendered)
    console.print(table)
    console.print("[dim]Use --json for the lossless machine-readable record.[/]")


def _human_label(column: str) -> str:
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
        "secret_prefix": "prefix",
    }
    return labels.get(column, column.replace("_", " "))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value)
    if len(text) > _MAX_CELL_LENGTH:
        return text[: _MAX_CELL_LENGTH - 1] + "…"
    return text
