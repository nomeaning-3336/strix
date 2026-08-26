"""`strix login` — managed platform sign-in (app.strix.ai).

Signing in runs an OAuth 2.0 device authorization flow in the browser, creates
the Strix account and workspace when they do not exist yet, and stores a
personal API token in ``~/.strix/platform-auth.json``. The token drives the
managed REST API (scans, credits, top-ups) without a dashboard visit.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import webbrowser
from pathlib import Path
from typing import Any, cast

import requests
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from strix.config import load_settings
from strix.utils.secret_files import write_secret_text


AUTH_PATH = Path.home() / ".strix" / "platform-auth.json"

_HTTP_TIMEOUT_S = 30
_DEFAULT_POLL_INTERVAL_S = 5

_LOGIN_USAGE = "Usage:\n  strix login [--no-browser]\n  strix login status\n  strix login logout"


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


def logout() -> None:
    with contextlib.suppress(OSError):
        AUTH_PATH.unlink()


def run_login(argv: list[str]) -> int:
    """Entry point for ``strix login …``. Returns a process exit code."""
    console = Console()
    subcommand = argv[0] if argv else None

    if subcommand in ("-h", "--help", "help"):
        console.print(_LOGIN_USAGE)
        return 0
    if subcommand == "status":
        return _status(console)
    if subcommand == "logout":
        return _logout(console)
    return _login(console, argv)


def _login(console: Console, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="strix login", add_help=True)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser. Print the verification URL instead.",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse already printed the message
        return int(exc.code or 2)

    console.print()
    console.print("[bold]Signing in to the Strix platform[/] [dim](app.strix.ai)[/]")
    console.print(
        "[dim]This creates your account and workspace when needed, and stores an API token.[/]"
    )
    console.print()

    try:
        record = _run_device_flow(console, open_browser=not args.no_browser)
    except PlatformAuthError as exc:
        console.print(f"[red]Sign-in failed:[/] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Sign-in cancelled.[/]")
        return 130

    save_record(record)
    _print_success(console, record)
    return 0


def _run_device_flow(console: Console, *, open_browser: bool) -> dict[str, Any]:
    app_url = _app_url()

    try:
        response = requests.post(f"{app_url}/api/v1/cli/login", timeout=_HTTP_TIMEOUT_S)
    except requests.RequestException as exc:
        raise PlatformAuthError(f"could not reach {app_url}: {exc}") from exc
    if not response.ok:
        raise PlatformAuthError(_error_detail(response))
    authorization = response.json()

    user_code = authorization.get("user_code", "")
    verification_uri = authorization.get("verification_uri_complete") or authorization.get(
        "verification_uri", ""
    )
    device_code = authorization.get("device_code", "")
    expires_in = int(authorization.get("expires_in") or 300)
    interval = int(authorization.get("interval") or _DEFAULT_POLL_INTERVAL_S)
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

    deadline = time.monotonic() + expires_in
    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            poll = requests.post(
                f"{app_url}/api/v1/cli/login/poll",
                json={"device_code": device_code},
                timeout=_HTTP_TIMEOUT_S,
            )
        except requests.RequestException:
            continue
        if poll.ok:
            return dict(poll.json())
        error = ""
        with contextlib.suppress(ValueError):
            error = str(poll.json().get("error", ""))
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error == "access_denied":
            raise PlatformAuthError("the sign-in request was denied in the browser")
        if error == "expired_token":
            break
        raise PlatformAuthError(_error_detail(poll))

    raise PlatformAuthError("the sign-in request expired. Run `strix login` again.")


def _error_detail(response: requests.Response) -> str:
    with contextlib.suppress(ValueError):
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
    console.print(f"  Token:     stored in [dim]{AUTH_PATH}[/]")
    console.print()
    console.print(
        "[dim]The managed API is ready. "
        "See https://docs.app.strix.ai for scans, credits, and top-ups.[/]"
    )


def _status(console: Console) -> int:
    record = read_record()
    if record is None:
        console.print("[yellow]Not signed in.[/] Run [bold]strix login[/] to sign in.")
        return 1
    email = record.get("email", "unknown")
    organization = record.get("organization_name") or record.get("organization_id", "")
    expires_at = record.get("expires_at", "")
    console.print(f"[green]Signed in[/] as [bold]{email}[/]")
    if organization:
        console.print(f"  Workspace: {organization}")
    if expires_at:
        console.print(f"  Token expires: {expires_at}")
    return 0


def _logout(console: Console) -> int:
    if read_record() is None:
        console.print("[yellow]Not signed in.[/]")
        return 0
    logout()
    console.print("[green]Signed out.[/] The stored API token was removed from this machine.")
    return 0
