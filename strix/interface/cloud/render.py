"""Output rendering for `strix cloud` commands."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

from rich.table import Table


if TYPE_CHECKING:
    from rich.console import Console


_MAX_TABLE_COLUMNS = 8
_MAX_CELL_LENGTH = 60

_PREFERRED_KEYS = (
    "id",
    "name",
    "title",
    "domain",
    "status",
    "state",
    "severity",
    "role",
    "email",
    "url",
    "scan_type",
    "engagement_type",
    "created_at",
    "updated_at",
)


def json_mode(*, flag: bool) -> bool:
    """JSON output is on when the flag is set or when stdout is not a terminal."""
    return flag or not sys.stdout.isatty()


def emit(console: Console, data: Any, *, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
        return
    rows = _list_of_dicts(data)
    if rows is not None:
        _print_table(console, rows)
        return
    if isinstance(data, str):
        console.print(data)
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


def _print_table(console: Console, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = [key for key in _PREFERRED_KEYS if any(key in row for row in rows)]
    for row in rows:
        for key in row:
            if (
                key not in columns
                and len(columns) < _MAX_TABLE_COLUMNS
                and not isinstance(row[key], dict | list)
            ):
                columns.append(key)
    table = Table(show_lines=False)
    for column in columns[:_MAX_TABLE_COLUMNS]:
        table.add_column(column)
    for row in rows:
        table.add_row(*[_cell(row.get(column)) for column in columns[:_MAX_TABLE_COLUMNS]])
    console.print(table)
    console.print(f"[dim]{len(rows)} item(s). Use --json for the full records.[/]")


def _cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) > _MAX_CELL_LENGTH:
        return text[: _MAX_CELL_LENGTH - 1] + "…"
    return text
