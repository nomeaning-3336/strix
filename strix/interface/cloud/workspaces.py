"""`strix cloud workspaces use` — switch the stored token to another workspace.

The command lists the workspaces of the account, finds the requested one by
ID or by exact name, asks the platform for a token in that workspace, and
stores the token in the credential file. The role of the account in the new
workspace limits the granted scopes.
"""

from __future__ import annotations

import argparse
from typing import Any, cast

from rich.console import Console

from strix.interface.cloud import http
from strix.interface.cloud.render import emit, json_mode
from strix.interface.platform_cli import AUTH_PATH, read_record, save_record


def run_workspace_use(argv: list[str]) -> int:
    """Entry point for ``strix cloud workspaces use``. Returns an exit code."""
    console = Console()
    parser = argparse.ArgumentParser(
        prog="strix cloud workspaces use",
        description="Switch the stored API token to another workspace.",
    )
    parser.add_argument("workspace", metavar="WORKSPACE", help="Workspace ID or exact name.")
    parser.add_argument(
        "--scopes",
        nargs="+",
        metavar="SCOPE",
        default=None,
        help="API scopes for the new token. Without this option the server grants the defaults.",
    )
    parser.add_argument("--json", action="store_true", help="Print the raw JSON response.")
    parser.add_argument("--token", default=None, help="API token override.")
    parser.add_argument("--app-url", default=None, metavar="URL", help="Platform URL override.")
    parser.add_argument(
        "--timeout", default=None, type=float, metavar="SECONDS", help="Request timeout in seconds."
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    as_json = json_mode(flag=bool(args.json))
    http.configure(base_url=args.app_url, timeout=args.timeout)
    try:
        return _use(console, args, as_json=as_json)
    except http.CloudError as exc:
        if as_json:
            emit(console, {"error": str(exc)}, as_json=True)
        else:
            console.print(f"[red]Error:[/] {exc}")
        return exc.exit_code


def _use(console: Console, args: argparse.Namespace, *, as_json: bool) -> int:
    workspace = _find_workspace(args.workspace, token=args.token)
    body: dict[str, Any] = {}
    if args.scopes:
        body["scopes"] = args.scopes
    minted = http.check(
        http.request(
            "POST",
            f"/workspaces/{workspace['id']}/token",
            token=args.token,
            body=body or None,
        )
    )
    if not isinstance(minted, dict) or not minted.get("api_token"):
        raise http.CloudError("the platform did not return a token.")
    minted_record = cast("dict[str, Any]", minted)

    record = read_record() or {}
    record.update(
        {
            "api_token": minted_record["api_token"],
            "organization_id": minted_record.get("organization_id", workspace["id"]),
            "organization_name": minted_record.get("organization_name", workspace.get("name", "")),
            "expires_at": minted_record.get("expires_at"),
            "scopes": minted_record.get("scopes", []),
        }
    )
    if minted_record.get("email"):
        record["email"] = minted_record["email"]
    save_record(record)

    result = {
        "workspace_id": record["organization_id"],
        "workspace_name": record["organization_name"],
        "scopes": record["scopes"],
    }
    if as_json:
        emit(console, result, as_json=True)
        return http.EXIT_OK
    console.print(f"[green]✓ Switched to workspace [bold]{record['organization_name']}[/].[/]")
    scopes = record.get("scopes")
    if isinstance(scopes, list) and scopes:
        console.print(f"  Scopes: [dim]{' '.join(str(s) for s in scopes)}[/]")
    console.print(f"  Token:  stored in [dim]{AUTH_PATH}[/]")
    return http.EXIT_OK


def _find_workspace(selector: str, *, token: str | None) -> dict[str, Any]:
    listed = http.check(http.request("GET", "/workspaces", token=token))
    items = listed.get("workspaces") if isinstance(listed, dict) else None
    workspaces = [cast("dict[str, Any]", item) for item in (items or []) if isinstance(item, dict)]
    if not workspaces:
        raise http.CloudError("no workspaces found for this account.")
    wanted = selector.strip()
    by_id = [w for w in workspaces if w.get("id") == wanted]
    if by_id:
        return by_id[0]
    by_name = [w for w in workspaces if str(w.get("name", "")).casefold() == wanted.casefold()]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        ids = ", ".join(str(w.get("id")) for w in by_name)
        raise http.CloudError(f"multiple workspaces are named {wanted!r}. Use an ID: {ids}")
    names = ", ".join(str(w.get("name")) for w in workspaces)
    raise http.CloudError(f"no workspace matches {wanted!r}. Your workspaces: {names}")
