"""`strix cloud login` — managed platform sign-in (app.strix.ai).

Signing in runs an OAuth 2.0 device authorization flow in the browser, creates
the Strix account and workspace when they do not exist yet, and stores a
personal API token in ``~/.strix/platform-auth.json``. The token drives the
managed REST API (scans, credits, top-ups) without a dashboard visit.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import requests
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from strix.config import load_settings
from strix.utils.secret_files import write_secret_text


AUTH_PATH = Path.home() / ".strix" / "platform-auth.json"

_HTTP_TIMEOUT_S = 30
_DEFAULT_POLL_INTERVAL_S = 5
_MAX_POLL_INTERVAL_S = 60
_MAX_EXPIRES_IN_S = 30 * 60

_LOGIN_USAGE = (
    "Usage:\n"
    "  strix cloud login [--no-browser] [--scopes SCOPE ...] [--workspace WORKSPACE]\n"
    "  strix cloud whoami\n  strix cloud logout"
)

_ROLE_RANK = {"viewer": 0, "analyst": 1, "admin": 2}


class PlatformAuthError(Exception):
    """Raised when the device authorization flow fails."""


def _app_url() -> str:
    return load_settings().viewer.app_url.rstrip("/")


def read_record() -> dict[str, Any] | None:
    try:
        data = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    record = cast("dict[str, Any]", data)
    if not record.get("api_token"):
        return None
    return record


def save_record(record: dict[str, Any]) -> None:
    write_secret_text(AUTH_PATH, json.dumps(record, indent=2))


def logout() -> bool:
    try:
        AUTH_PATH.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def run_login(argv: list[str]) -> int:
    """Entry point for ``strix cloud login``. Returns a process exit code."""
    console = Console()
    subcommand = argv[0] if argv else None

    if subcommand in ("-h", "--help", "help"):
        console.print(_LOGIN_USAGE)
        return 0
    if subcommand == "status":
        return _status(console, argv[1:])
    if subcommand == "logout":
        return _logout(console)
    return _login(console, argv)


def _login(console: Console, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="strix cloud login", add_help=True)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser. Print the verification URL instead.",
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        metavar="SCOPE",
        default=None,
        help=(
            "API scopes for the token, for example scans:read billing:write. "
            "The server always includes a minimum scope set. "
            "Without this option, an interactive picker opens after the browser step."
        ),
    )
    parser.add_argument(
        "--workspace",
        metavar="WORKSPACE",
        default=None,
        help=(
            "Workspace that receives the token, by ID or by exact name. "
            "Without this option, an interactive picker opens when you have "
            "more than one workspace."
        ),
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse already printed the message
        return exc.code if isinstance(exc.code, int) else 2

    console.print()
    host = urlparse(_app_url()).netloc or _app_url()
    console.print(f"[bold]Signing in to the Strix platform[/] [dim]({host})[/]")
    console.print(
        "[dim]This creates your account and workspace when needed, and stores an API token.[/]"
    )
    console.print()

    try:
        record = _run_device_flow(
            console,
            open_browser=not args.no_browser,
            scopes=args.scopes,
            workspace=args.workspace,
        )
    except PlatformAuthError as exc:
        console.print(f"[red]Sign-in failed:[/] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Sign-in cancelled.[/]")
        return 130

    try:
        save_record(record)
    except OSError as exc:
        console.print(f"[red]Sign-in succeeded, but the token could not be stored:[/] {exc}")
        console.print(
            f"[dim]Check that {AUTH_PATH.parent} is writable, "
            "then run `strix cloud login` again.[/]"
        )
        return 1
    _print_success(console, record)
    return 0


def _run_device_flow(
    console: Console,
    *,
    open_browser: bool,
    scopes: list[str] | None = None,
    workspace: str | None = None,
) -> dict[str, Any]:
    app_url = _app_url()
    interactive = workspace is not None or (sys.stdin.isatty() and scopes is None)

    try:
        response = requests.post(f"{app_url}/api/v1/cli/login", timeout=_HTTP_TIMEOUT_S)
    except requests.RequestException as exc:
        raise PlatformAuthError(f"could not reach {app_url}: {exc}") from exc
    if not response.ok:
        raise PlatformAuthError(_error_detail(response))
    authorization = _json_object(response)

    user_code = str(authorization.get("user_code") or "")
    verification_uri = str(
        authorization.get("verification_uri_complete")
        or authorization.get("verification_uri")
        or ""
    )
    device_code = str(authorization.get("device_code") or "")
    expires_in = _as_positive_int(
        authorization.get("expires_in"), default=300, maximum=_MAX_EXPIRES_IN_S
    )
    interval = _as_positive_int(
        authorization.get("interval"),
        default=_DEFAULT_POLL_INTERVAL_S,
        maximum=_MAX_POLL_INTERVAL_S,
    )
    if not device_code or not verification_uri:
        raise PlatformAuthError("the server returned an incomplete device authorization")

    console.print(
        Panel.fit(
            Text.assemble(
                ("Confirmation code: ", "dim"),
                (user_code, "bold cyan"),
                ("\n\nOpen this URL in your browser and confirm the code:\n", "dim"),
                (verification_uri, "underline"),
            ),
            title="Verify this device",
        )
    )

    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(verification_uri)

    console.print("[dim]Waiting for browser confirmation…[/]")

    poll_body: dict[str, Any] = {"device_code": device_code}
    if interactive:
        poll_body["interactive"] = True
    elif scopes:
        poll_body["scopes"] = scopes

    deadline = time.monotonic() + expires_in
    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            poll = requests.post(
                f"{app_url}/api/v1/cli/login/poll",
                json=poll_body,
                timeout=_HTTP_TIMEOUT_S,
            )
        except requests.RequestException:
            continue
        if poll.ok:
            return _finish_login(console, app_url, poll, scopes=scopes, workspace=workspace)
        delta = _handle_poll_error(poll)
        if delta is None:
            break
        interval += delta

    raise PlatformAuthError("the sign-in request expired. Run `strix cloud login` again.")


def _handle_poll_error(poll: requests.Response) -> int | None:
    """Return the interval increase, or None when the device code expired."""
    error = ""
    with contextlib.suppress(ValueError, AttributeError):
        error = str(poll.json().get("error", ""))
    if error == "authorization_pending":
        return 0
    if error == "slow_down":
        return 5
    if error == "access_denied":
        raise PlatformAuthError("the sign-in request was denied in the browser")
    if error == "expired_token":
        return None
    raise PlatformAuthError(_error_detail(poll))


def _finish_login(
    console: Console,
    app_url: str,
    poll: requests.Response,
    *,
    scopes: list[str] | None,
    workspace: str | None,
) -> dict[str, Any]:
    result = _json_object(poll)
    if result.get("selection_required"):
        return _complete_selection(console, app_url, result, scopes=scopes, workspace=workspace)
    return _require_api_token(result)


def _signed_in_record(response: requests.Response) -> dict[str, Any]:
    return _require_api_token(_json_object(response))


def _require_api_token(record: dict[str, Any]) -> dict[str, Any]:
    api_token = record.get("api_token")
    if not isinstance(api_token, str) or not api_token.strip():
        raise PlatformAuthError("the server returned a sign-in response without an API token")
    return record


def _complete_selection(
    console: Console,
    app_url: str,
    selection: dict[str, Any],
    *,
    scopes: list[str] | None,
    workspace: str | None,
) -> dict[str, Any]:
    organizations = _dict_items(selection.get("organizations"))
    catalog = _dict_items(selection.get("scopes"))
    selection_token = str(selection.get("selection_token") or "")
    if not selection_token or not organizations:
        raise PlatformAuthError("the server returned an incomplete selection response")

    chosen_org = _choose_workspace(console, organizations, workspace)
    role = str(chosen_org.get("role") or "admin")
    chosen_scopes = scopes
    if chosen_scopes is None and sys.stdin.isatty():
        chosen_scopes = _choose_scopes(console, catalog, role)

    body: dict[str, Any] = {
        "selection_token": selection_token,
        "organization_id": chosen_org.get("id"),
    }
    if chosen_scopes is not None:
        body["scopes"] = chosen_scopes
    try:
        response = requests.post(
            f"{app_url}/api/v1/cli/login/complete",
            json=body,
            timeout=_HTTP_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        raise PlatformAuthError(f"could not reach {app_url}: {exc}") from exc
    if not response.ok:
        raise PlatformAuthError(_error_detail(response))
    return _signed_in_record(response)


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _choose_workspace(
    console: Console, organizations: list[dict[str, Any]], workspace: str | None
) -> dict[str, Any]:
    if workspace is not None:
        wanted = workspace.strip().lower()
        for org in organizations:
            if wanted in (str(org.get("id", "")).lower(), str(org.get("name", "")).lower()):
                return org
        names = ", ".join(str(org.get("name", "")) for org in organizations)
        raise PlatformAuthError(f"no workspace matches {workspace!r}. Your workspaces: {names}")
    if len(organizations) == 1:
        return organizations[0]

    console.print()
    console.print("[bold]Select a workspace for the API token:[/]")
    for index, org in enumerate(organizations, start=1):
        console.print(f"  [cyan]{index}[/]. {org.get('name', '')} [dim]({org.get('role', '')})[/]")
    while True:
        answer = console.input(f"Workspace [1-{len(organizations)}] (1): ").strip() or "1"
        if answer.isdigit() and 1 <= int(answer) <= len(organizations):
            return organizations[int(answer) - 1]
        console.print("[yellow]Enter a number from the list.[/]")


def _choose_scopes(console: Console, catalog: list[dict[str, Any]], role: str) -> list[str] | None:
    """Prompt for token scopes. Returns None to accept the server defaults."""
    rank = _ROLE_RANK.get(role, 2)
    allowed = [
        item for item in catalog if _ROLE_RANK.get(str(item.get("min_role", "viewer")), 0) <= rank
    ]
    if not allowed:
        return None

    console.print()
    console.print("[bold]Select token scopes:[/]")
    console.print(
        "  [cyan]1[/]. Recommended [dim](scans, vulnerabilities, schedules, assets, billing)[/]"
    )
    console.print("  [cyan]2[/]. Full access [dim](every scope your role allows)[/]")
    console.print("  [cyan]3[/]. Minimal [dim](scans and billing read only)[/]")
    console.print("  [cyan]4[/]. Custom [dim](pick individual scopes)[/]")
    while True:
        answer = console.input("Scopes [1-4] (1): ").strip() or "1"
        if answer == "1":
            return None
        if answer == "2":
            return [str(item["scope"]) for item in allowed if item.get("scope")]
        if answer == "3":
            return [
                str(item["scope"]) for item in allowed if item.get("scope") and item.get("minimum")
            ]
        if answer == "4":
            return _choose_custom_scopes(console, allowed)
        console.print("[yellow]Enter a number from 1 to 4.[/]")


def _choose_custom_scopes(console: Console, allowed: list[dict[str, Any]]) -> list[str]:
    selected = {
        str(item["scope"])
        for item in allowed
        if item.get("scope") and (item.get("default") or item.get("minimum"))
    }
    while True:
        console.print()
        for index, item in enumerate(allowed, start=1):
            scope = str(item.get("scope", ""))
            mark = "[green]x[/]" if scope in selected else " "
            required = " [dim](always included)[/]" if item.get("minimum") else ""
            console.print(
                f"  [{mark}] [cyan]{index:>2}[/]. {scope}{required}"
                f"\n         [dim]{item.get('description', '')}[/]"
            )
        answer = console.input(
            "Toggle scopes by number (comma separated), or press Enter to confirm: "
        ).strip()
        if not answer:
            return sorted(selected)
        for part in answer.replace(",", " ").split():
            if not part.isdigit() or not 1 <= int(part) <= len(allowed):
                console.print(f"[yellow]Ignored {part!r}: not a number from the list.[/]")
                continue
            item = allowed[int(part) - 1]
            scope = str(item.get("scope", ""))
            if item.get("minimum"):
                console.print(f"[yellow]{scope} is always included.[/]")
                continue
            if scope in selected:
                selected.discard(scope)
            else:
                selected.add(scope)


def _json_object(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise PlatformAuthError("the server returned a response that is not JSON") from exc
    if not isinstance(data, dict):
        raise PlatformAuthError("the server returned an unexpected response shape")
    return cast("dict[str, Any]", data)


def _as_positive_int(value: Any, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if parsed <= 0:
        return default
    return min(parsed, maximum)


def _error_detail(response: requests.Response) -> str:
    with contextlib.suppress(ValueError, AttributeError):
        detail = response.json().get("detail")
        if detail:
            return str(detail)
    return f"HTTP {response.status_code}"


def _print_success(console: Console, record: dict[str, Any]) -> None:
    email = record.get("email", "")
    organization = record.get("organization_name") or record.get("organization_id", "")
    console.print()
    console.print("[green]✓ Signed in to the Strix platform.[/]")
    if email:
        console.print(f"  Account:   [bold]{email}[/]")
    if organization:
        console.print(f"  Workspace: [bold]{organization}[/]")
    scopes = record.get("scopes")
    if isinstance(scopes, list) and scopes:
        console.print(f"  Scopes:    [dim]{' '.join(str(s) for s in scopes)}[/]")
    console.print(f"  Token:     stored in [dim]{AUTH_PATH}[/]")
    console.print()
    console.print(
        "[dim]The managed platform is ready. Run `strix cloud` to list the commands. "
        "See https://docs.app.strix.ai for the API reference.[/]"
    )


def _status(console: Console, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="strix cloud whoami")
    parser.add_argument("--json", action="store_true", help="Print the session as JSON.")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    record = read_record()
    if record is None:
        if args.json:
            sys.stdout.write(json.dumps({"signed_in": False, "error": "Not signed in"}) + "\n")
            return 1
        console.print("[yellow]Not signed in.[/] Run [bold]strix cloud login[/] to sign in.")
        return 1
    email = record.get("email", "unknown")
    organization = record.get("organization_name") or record.get("organization_id", "")
    expires_at = record.get("expires_at", "")
    if args.json:
        payload = {
            "signed_in": True,
            "email": email,
            "organization_id": record.get("organization_id"),
            "organization_name": record.get("organization_name"),
            "scopes": record.get("scopes", []),
            "expires_at": expires_at or None,
        }
        sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
        return 0
    console.print(f"[green]Signed in[/] as [bold]{email}[/]")
    if organization:
        console.print(f"  Workspace: {organization}")
    if expires_at:
        console.print(f"  Token expires: {expires_at}")
    return 0


def _logout(console: Console) -> int:
    if read_record() is None and not AUTH_PATH.exists():
        console.print("[yellow]Not signed in.[/]")
        return 0
    if not logout():
        console.print(
            f"[red]Could not remove the stored API token.[/] Delete {AUTH_PATH} manually."
        )
        return 1
    console.print("[green]Signed out.[/] The stored API token was removed from this machine.")
    return 0
