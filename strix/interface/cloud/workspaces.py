"""`strix cloud workspaces use` — switch the stored token to another workspace.

The command lists the workspaces of the account, finds the requested one by
ID or by exact name, asks the platform to rotate that token in place, and
stores the returned workspace metadata. The bearer secret and expiry stay the
same; the account's role in the target workspace limits the granted scopes.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

from rich.console import Console
from rich.markup import escape

import strix.interface.cloud.http as http  # noqa: PLR0402
from strix.interface.cloud.arguments import CloudArgumentParser
from strix.interface.cloud.render import emit, json_mode
from strix.interface.platform_cli import AUTH_PATH, read_record, save_record
from strix.interface.terminal_text import sanitize_terminal_text


if TYPE_CHECKING:
    import argparse


def run_workspace_use(argv: list[str]) -> int:
    """Entry point for ``strix cloud workspaces use``. Returns an exit code."""
    console = Console()
    parser = CloudArgumentParser(
        prog="strix cloud workspaces use",
        description="Switch the stored API token to another workspace.",
    )
    parser.add_argument(
        "workspace",
        metavar="WORKSPACE",
        help="Workspace number from `workspaces list`, ID, or exact name.",
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        metavar="SCOPE",
        default=None,
        help=(
            "API scopes after switching. Without this option, preserve the stored request scopes."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print the raw JSON response.")
    parser.add_argument("--token", default=None, help="API token override.")
    parser.add_argument("--app-url", default=None, metavar="URL", help="Platform URL override.")
    parser.add_argument(
        "--timeout", default=None, type=float, metavar="SECONDS", help="Request timeout in seconds."
    )
    as_json = json_mode(flag="--json" in argv)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    except http.CloudError as exc:
        _emit_cloud_error(console, exc, as_json=as_json)
        return exc.exit_code

    as_json = json_mode(flag=bool(args.json))
    try:
        http.configure(
            base_url=args.app_url,
            timeout=args.timeout,
            token_override=bool(args.token),
        )
        return _use(console, args, as_json=as_json)
    except http.CloudError as exc:
        _emit_cloud_error(console, exc, as_json=as_json)
        return exc.exit_code


def _use(console: Console, args: argparse.Namespace, *, as_json: bool) -> int:
    workspace = _find_workspace(args.workspace, token=args.token)
    stored_record: dict[str, Any] = read_record() or {}
    # An override token may belong to a different account. Never mix its new
    # workspace state with identity or scope preferences from the stored sign-in.
    external_token = args.token is not None or bool(os.environ.get("STRIX_API_TOKEN", "").strip())
    record: dict[str, Any] = {} if external_token else dict(stored_record)
    body: dict[str, Any] = {}
    if args.scopes:
        body["scopes"] = args.scopes
    elif not external_token:
        stored_scopes = stored_record.get("requested_scopes", stored_record.get("scopes"))
        if (
            isinstance(stored_scopes, list)
            and stored_scopes
            and all(isinstance(scope, str) for scope in stored_scopes)
        ):
            body["scopes"] = [scope for scope in stored_scopes if isinstance(scope, str)]
    switched = _switch_workspace_token(
        str(workspace["id"]),
        token=args.token,
        body=body or None,
    )
    if not isinstance(switched, dict):
        raise _workspace_switch_unknown("the platform returned an invalid response")
    switched_record = cast("dict[str, Any]", switched)
    switched_token = switched_record.get("api_token")
    if not isinstance(switched_token, str) or not switched_token.strip():
        raise _workspace_switch_unknown("the platform response omitted the token")
    switched_scopes = switched_record.get("scopes")
    if not isinstance(switched_scopes, list) or not all(
        isinstance(scope, str) for scope in switched_scopes
    ):
        raise _workspace_switch_unknown("the platform response contained invalid scopes")

    record.update(
        {
            "api_token": switched_token,
            "organization_id": switched_record.get("organization_id", workspace["id"]),
            "organization_name": switched_record.get(
                "organization_name", workspace.get("name", "")
            ),
            "expires_at": switched_record.get("expires_at"),
            "scopes": switched_scopes,
            "requested_scopes": (
                list(args.scopes)
                if args.scopes
                else (
                    stored_record.get(
                        "requested_scopes",
                        stored_record.get("scopes", []),
                    )
                    if not external_token
                    else switched_scopes
                )
            ),
            "app_url": http.app_url(),
        }
    )
    if switched_record.get("email"):
        record["email"] = switched_record["email"]
    try:
        save_record(record)
    except OSError as exc:
        raise http.CloudError(
            "the platform switched the token, but the local workspace metadata could not be "
            f"stored in {AUTH_PATH}: {exc}. The bearer is still valid; fix the file and safely "
            "rerun the same workspace use command.",
            payload={
                "workspace_switched": True,
                "local_record_updated": False,
                "retry_safe": True,
            },
        ) from exc

    result = {
        "workspace_id": record["organization_id"],
        "workspace_name": record["organization_name"],
        "scopes": record["scopes"],
    }
    if as_json:
        emit(console, result, as_json=True)
        return http.EXIT_OK
    workspace_name = escape(sanitize_terminal_text(record["organization_name"]))
    console.print(f"[green]✓ Switched to workspace [bold]{workspace_name}[/].[/]")
    scopes = record.get("scopes")
    if isinstance(scopes, list) and scopes:
        scope_names = [scope for scope in scopes if isinstance(scope, str)]
        if scope_names:
            rendered_scopes = escape(sanitize_terminal_text(" ".join(scope_names)))
            console.print(f"  Scopes: [dim]{rendered_scopes}[/]")
    console.print(f"  Token:  stored in [dim]{escape(sanitize_terminal_text(AUTH_PATH))}[/]")
    return http.EXIT_OK


def _switch_workspace_token(
    workspace_id: str,
    *,
    token: str | None,
    body: dict[str, Any] | None,
) -> Any:
    """Switch in place, distinguishing definitive rejections from lost outcomes."""
    try:
        response = http.request(
            "POST",
            f"/workspaces/{workspace_id}/token",
            token=token,
            body=body,
        )
    except http.CloudError as exc:
        raise _workspace_switch_unknown(str(exc)) from exc

    # Client/auth/conflict responses prove the rotation did not return success.
    # A 5xx or malformed success may arrive after the database commit, but the
    # server preserves the bearer so replaying this exact command is safe.
    if response.status_code in {400, 401, 403, 404, 409, 422}:
        return http.check(response)
    try:
        return http.check(response)
    except http.CloudError as exc:
        raise _workspace_switch_unknown(str(exc)) from exc


def _workspace_switch_unknown(detail: str) -> http.CloudError:
    return http.CloudError(
        "workspace switch outcome is unknown: "
        f"{sanitize_terminal_text(detail)}. The bearer secret is unchanged; safely rerun the "
        "same workspace use command, or list workspaces to check the current one.",
        payload={
            "switch_outcome_unknown": True,
            "retry_safe": True,
        },
    )


def _emit_cloud_error(console: Console, error: http.CloudError, *, as_json: bool) -> None:
    if as_json:
        payload = dict(error.payload) if isinstance(error.payload, dict) else {}
        payload["error"] = str(error)
        emit(console, payload, as_json=True)
        return
    console.print(f"[red]Error:[/] {escape(sanitize_terminal_text(error))}")


def _find_workspace(selector: str, *, token: str | None) -> dict[str, Any]:
    listed = http.check(http.request("GET", "/workspaces", token=token))
    listed_record = cast("dict[str, Any]", listed) if isinstance(listed, dict) else {}
    items = listed_record.get("workspaces")
    workspaces = [cast("dict[str, Any]", item) for item in (items or []) if isinstance(item, dict)]
    if not workspaces:
        raise http.CloudError("no workspaces found for this account.")
    wanted = selector.strip()
    if wanted.isdigit():
        index = int(wanted)
        if 1 <= index <= len(workspaces):
            return workspaces[index - 1]
        raise http.CloudError(
            f"workspace number must be between 1 and {len(workspaces)}. "
            "Run `strix cloud workspaces` to see the numbered list."
        )
    by_id = [w for w in workspaces if w.get("id") == wanted]
    if by_id:
        return by_id[0]
    by_name = [w for w in workspaces if str(w.get("name", "")).casefold() == wanted.casefold()]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        numbers = ", ".join(
            str(index)
            for index, workspace in enumerate(workspaces, start=1)
            if workspace in by_name
        )
        raise http.CloudError(
            f"multiple workspaces are named {wanted!r}. Use its list number: {numbers}"
        )
    names = ", ".join(
        f"{index}: {workspace.get('name')}" for index, workspace in enumerate(workspaces, start=1)
    )
    raise http.CloudError(f"no workspace matches {wanted!r}. Your workspaces: {names}")
