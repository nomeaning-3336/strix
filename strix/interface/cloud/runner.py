"""Generic command runner for `strix cloud`.

The runner turns one entry of the command table into an argument parser,
sends the HTTP request, renders the result, and returns the exit code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import webbrowser
from contextlib import suppress
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

from rich.console import Console

from strix.interface.cloud import http
from strix.interface.cloud.render import emit, json_mode
from strix.interface.cloud.source_upload import SourceBundle, prepare_source, remove_bundle
from strix.interface.cloud.spec import DEFAULT_VERBS, SPEC, Cmd, P


_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WAIT_POLL_S = 15
_TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "failed",
        "cancelled",
        "canceled",
        "stopped",
        "error",
        "expired",
        "succeeded",
    }
)


def _dest(name: str) -> str:
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def _metavar(name: str) -> str:
    return _CAMEL_BOUNDARY.sub("_", name).upper()


def resolve(group: str, tokens: list[str]) -> tuple[Cmd, list[str]] | None:
    """Find the command for a verb. Two-word verbs match before one-word verbs."""
    commands = SPEC.get(group)
    if commands is None:
        return None
    if len(tokens) >= 2:
        two = f"{tokens[0]} {tokens[1]}"
        if two in commands:
            return commands[two], tokens[2:]
    if tokens and tokens[0] in commands:
        return commands[tokens[0]], tokens[1:]
    default = DEFAULT_VERBS.get(group)
    if default is not None and (not tokens or tokens[0].startswith("-")):
        return commands[default], tokens
    return None


def run(group: str, verb_label: str, cmd: Cmd, argv: list[str]) -> int:
    console = Console()
    parser = _build_parser(group, verb_label, cmd)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    path = cmd.path
    for name in _PLACEHOLDER.findall(cmd.path):
        value = quote(str(getattr(args, _dest(name))), safe="")
        path = path.replace("{" + name + "}", value)

    as_json = json_mode(flag=bool(getattr(args, "json", False)))
    token = getattr(args, "token", None)
    http.configure(base_url=getattr(args, "app_url", None), timeout=getattr(args, "timeout", None))

    try:
        query = _collect(args, cmd.query)
        body = _collect(args, cmd.body)
        data = getattr(args, "data", None)
        if data:
            body.update(_load_data(data))
        if getattr(args, "no_monthly_cap", False):
            body["monthly_cap_credits"] = None
        return _execute(console, cmd, args, path, query, body, as_json=as_json, token=token)
    except http.CloudError as exc:
        _emit_error(console, exc, as_json=as_json)
        return exc.exit_code


def _execute(  # noqa: PLR0912
    console: Console,
    cmd: Cmd,
    args: argparse.Namespace,
    path: str,
    query: dict[str, Any],
    body: dict[str, Any],
    *,
    as_json: bool,
    token: str | None,
) -> int:
    if cmd.path == "/billing/topup":
        return _topup(console, args, body, as_json=as_json, token=token)
    source_bundle: SourceBundle | None = None
    source_upload_id: str | None = None
    try:
        if cmd.path == "/scans" and cmd.method == "POST":
            _set_default_scan_engagement(
                body,
                has_local_source=getattr(args, "source", None) is not None,
            )
            source_bundle = _prepare_scan_source(console, args, as_json=as_json)
            if source_bundle is not None and getattr(args, "dry_run", False):
                emit(
                    console,
                    {
                        "source": source_bundle.summary(
                            show_files=getattr(args, "show_files", False)
                        )
                    },
                    as_json=as_json,
                )
                return http.EXIT_OK
            if source_bundle is not None:
                source_upload_id = _upload_scan_source(source_bundle, token=token)
                existing = body.get("upload_ids")
                body["upload_ids"] = [
                    *(existing if isinstance(existing, list) else []),
                    source_upload_id,
                ]

        response = http.request(
            cmd.method,
            path,
            token=token,
            query=query or None,
            body=body if cmd.method in ("POST", "PUT", "PATCH") else None,
        )
    except BaseException:
        if source_upload_id is not None:
            _delete_upload(source_upload_id, token=token)
        raise
    finally:
        if source_bundle is not None:
            remove_bundle(source_bundle)
    if cmd.binary:
        return _emit_binary(console, response, getattr(args, "output", None))
    try:
        result = http.check(response)
    except BaseException:
        if source_upload_id is not None:
            _delete_upload(source_upload_id, token=token)
        raise
    if getattr(args, "wait", False):
        if cmd.wait_self:
            result = _poll(console, path, token=token, as_json=as_json)
        elif cmd.wait_path:
            result = _wait(console, cmd, result, token=token, as_json=as_json)
    if source_bundle is not None:
        result = {
            "source": source_bundle.summary(show_files=getattr(args, "show_files", False)),
            "upload_id": source_upload_id,
            "scan": result,
        }
    if cmd.link:
        return _handoff_link(console, cmd, args, result, as_json=as_json)
    workspace_list = cmd.method == "GET" and cmd.path == "/workspaces"
    emit(
        console,
        result,
        as_json=as_json,
        row_numbers=workspace_list,
        omit_columns=frozenset({"id"}) if workspace_list else frozenset(),
        hint=("Switch with `strix cloud workspaces use NUMBER`." if workspace_list else None),
    )
    return http.EXIT_OK


def _set_default_scan_engagement(
    body: dict[str, Any], *, has_local_source: bool = False
) -> None:
    """Infer the scan type from its targets when the caller did not choose one."""
    if body.get("engagement_type"):
        return
    if body.get("internal_targets"):
        body["engagement_type"] = "internal_infra"
    elif body.get("domain_ids"):
        body["engagement_type"] = "live_test"
    elif has_local_source or body.get("repository_ids") or body.get("upload_ids"):
        body["engagement_type"] = "code_review"


def _handoff_link(
    console: Console, cmd: Cmd, args: argparse.Namespace, result: Any, *, as_json: bool
) -> int:
    """Print a hosted URL a person must open, and open the browser when interactive."""
    fields = cast("dict[str, Any]", result) if isinstance(result, dict) else {}
    url = fields.get(cmd.link) if cmd.link else None
    if not isinstance(url, str) or not url:
        emit(console, result, as_json=as_json)
        return http.EXIT_OK
    interactive = sys.stdout.isatty() and not getattr(args, "no_browser", False)
    if as_json:
        emit(console, result, as_json=True)
    else:
        console.print(f"Open this URL to continue:\n  [bold]{url}[/]")
    if interactive:
        webbrowser.open(url)
    return http.EXIT_OK


def _load_data(value: str) -> dict[str, Any]:
    """Read a JSON object from a literal string, a `@file` path, or `-` for stdin."""
    if value == "-":
        text = sys.stdin.read()
    elif value.startswith("@"):
        path = Path(value[1:]).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise http.CloudError(f"could not read {path}: {exc}") from exc
    else:
        text = value
    try:
        parsed_value = json.loads(text)
    except ValueError as exc:
        raise http.CloudError("--data must be a JSON object.") from exc
    if not isinstance(parsed_value, dict):
        raise http.CloudError("--data must be a JSON object.")
    return cast("dict[str, Any]", parsed_value)


def _build_parser(group: str, verb_label: str, cmd: Cmd) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"strix cloud {group} {verb_label}", description=cmd.help)
    for name in _PLACEHOLDER.findall(cmd.path):
        parser.add_argument(_dest(name), metavar=_metavar(name))
    for param in cmd.query + cmd.body:
        _add_option(parser, param)
    parser.add_argument("--json", action="store_true", help="Print the raw JSON response.")
    parser.add_argument("--token", default=None, help="API token override.")
    parser.add_argument("--app-url", default=None, metavar="URL", help="Platform URL override.")
    parser.add_argument(
        "--timeout",
        default=None,
        type=float,
        metavar="SECONDS",
        help="Request timeout in seconds.",
    )
    if cmd.method in ("POST", "PUT", "PATCH"):
        parser.add_argument(
            "--data",
            default=None,
            metavar="JSON",
            help="JSON object with extra request fields. Use @file to read a file, or - for stdin.",
        )
    if cmd.path == "/billing/auto-topup" and cmd.method == "PUT":
        parser.add_argument(
            "--no-monthly-cap",
            action="store_true",
            help="Remove the monthly cap. Omit this flag to keep the stored cap.",
        )
    if cmd.binary:
        parser.add_argument("--output", default=None, metavar="FILE", help="Write to this file.")
    if cmd.link:
        parser.add_argument(
            "--no-browser",
            action="store_true",
            help="Do not open the browser. Print the URL only.",
        )
    if cmd.wait_path or cmd.wait_self:
        parser.add_argument(
            "--wait", action="store_true", help="Wait until the operation reaches a final state."
        )
    if cmd.path == "/billing/topup":
        parser.add_argument(
            "--yes", action="store_true", help="Do not ask for confirmation before payment."
        )
        parser.add_argument(
            "--no-pay",
            action="store_true",
            help="Print the payment challenge instead of paying it.",
        )
        parser.add_argument(
            "--payment-method",
            default=None,
            metavar="PM_ID",
            help=(
                "Stripe payment method for the card payment. "
                "Defaults to MPPX_STRIPE_PAYMENT_METHOD."
            ),
        )
    if cmd.path == "/scans" and cmd.method == "POST":
        parser.add_argument(
            "--source",
            default=None,
            metavar="DIRECTORY",
            help="Package a local directory, upload it, and attach it to this scan.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Build and print the source manifest without uploading or starting a scan.",
        )
        parser.add_argument(
            "--yes",
            action="store_true",
            help="Approve the displayed source upload without an interactive prompt.",
        )
        parser.add_argument(
            "--show-files",
            action="store_true",
            help="Include every selected relative path in the source manifest.",
        )
        parser.add_argument(
            "--exclude",
            action="append",
            default=[],
            metavar="GLOB",
            help="Exclude a path glob from the upload. May be repeated.",
        )
        parser.add_argument(
            "--include-hidden",
            action="store_true",
            help="Include hidden files except .git and secret-like filenames.",
        )
        parser.add_argument(
            "--include-sensitive",
            action="store_true",
            help="Include files with secret-like names. Use only after reviewing --dry-run.",
        )
        parser.add_argument(
            "--include-archives",
            action="store_true",
            help="Include nested archives. Use only when they are required source inputs.",
        )
    return parser


def _prepare_scan_source(
    console: Console, args: argparse.Namespace, *, as_json: bool
) -> SourceBundle | None:
    source = getattr(args, "source", None)
    source_flags = (
        "dry_run",
        "show_files",
        "include_hidden",
        "include_sensitive",
        "include_archives",
    )
    if source is None:
        if any(getattr(args, name, False) for name in source_flags) or getattr(args, "exclude", []):
            raise http.CloudError("source upload options require --source DIRECTORY.")
        return None
    bundle = prepare_source(
        source,
        include_hidden=bool(getattr(args, "include_hidden", False)),
        include_sensitive=bool(getattr(args, "include_sensitive", False)),
        include_archives=bool(getattr(args, "include_archives", False)),
        exclude=cast("list[str]", getattr(args, "exclude", [])),
    )
    if getattr(args, "dry_run", False):
        return bundle
    if getattr(args, "yes", False):
        return bundle
    if as_json or not (sys.stdin.isatty() and sys.stdout.isatty()):
        remove_bundle(bundle)
        raise http.CloudError(
            "source upload requires explicit approval in non-interactive mode. "
            "Review with --dry-run --show-files, then rerun with --yes."
        )
    console.print(
        "[bold]Local source upload[/]\n"
        f"  {len(bundle.manifest.files):,} file(s), "
        f"{_format_bytes(bundle.manifest.total_bytes)} "
        f"({_format_bytes(bundle.archive_bytes)} compressed)\n"
        f"  {sum(bundle.manifest.excluded.values()):,} path(s) excluded\n"
        "  Only the selected files will be sent to Strix Cloud."
    )
    answer = console.input("Upload this source and start the scan? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        remove_bundle(bundle)
        raise http.CloudError("source upload cancelled.")
    return bundle


def _upload_scan_source(bundle: SourceBundle, *, token: str | None) -> str:
    file_name = f"strix-source-{bundle.archive_sha256[:12]}.zip"
    requested = http.check(
        http.request(
            "POST",
            "/uploads/request",
            token=token,
            body={
                "file_name": file_name,
                "file_size": bundle.archive_bytes,
                "category": "repository",
            },
        )
    )
    if not isinstance(requested, dict):
        raise http.CloudError("the platform returned an invalid source upload response.")
    fields = cast("dict[str, Any]", requested)
    upload_id = fields.get("upload_id")
    signed_url = fields.get("signed_url")
    upload_token = fields.get("token")
    if not all(isinstance(value, str) and value for value in (upload_id, signed_url, upload_token)):
        raise http.CloudError("the platform did not return complete source upload credentials.")
    try:
        http.upload_file(cast("str", signed_url), cast("str", upload_token), bundle.archive_path)
        http.check(
            http.request(
                "POST",
                "/uploads/complete",
                token=token,
                body={"upload_id": upload_id},
            )
        )
    except BaseException:
        _delete_upload(cast("str", upload_id), token=token)
        raise
    return cast("str", upload_id)


def _delete_upload(upload_id: str, *, token: str | None) -> None:
    with suppress(http.CloudError):
        http.request("DELETE", f"/uploads/{quote(upload_id, safe='')}", token=token)


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value / (1024 * 1024):.1f} MB"


def _add_option(parser: argparse.ArgumentParser, param: P) -> None:
    flag = "--" + (param.flag or param.name.replace("_", "-"))
    if param.kind == "bool":
        parser.add_argument(
            flag,
            dest=param.name,
            action=argparse.BooleanOptionalAction,
            default=None,
            required=param.required,
            help=param.help,
        )
    elif param.kind == "list":
        parser.add_argument(
            flag,
            dest=param.name,
            nargs="+",
            default=None,
            required=param.required,
            help=param.help,
        )
    elif param.kind in ("int", "float"):
        parser.add_argument(
            flag,
            dest=param.name,
            type=int if param.kind == "int" else float,
            default=None,
            required=param.required,
            help=param.help,
        )
    else:
        parser.add_argument(
            flag, dest=param.name, default=None, required=param.required, help=param.help
        )


def _collect(args: argparse.Namespace, params: tuple[P, ...]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for param in params:
        value = getattr(args, param.name, None)
        if value is None:
            continue
        if param.kind == "json" and isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise http.CloudError(f"--{param.name.replace('_', '-')} must be JSON") from exc
        values[param.name] = value
    return values


def _emit_binary(console: Console, response: Any, output: str | None) -> int:
    if not response.ok:
        http.check(response)
    if output:
        Path(output).write_bytes(response.content)
        console.print(f"Saved to [bold]{output}[/]")
        return http.EXIT_OK
    sys.stdout.buffer.write(response.content)
    return http.EXIT_OK


def _emit_error(console: Console, exc: http.CloudError, *, as_json: bool) -> None:
    if as_json:
        payload = {"error": str(exc)}
        if exc.payload is not None:
            payload["detail"] = exc.payload
        sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
        return
    console.print(f"[red]Error:[/] {exc}")


def _created_id(created: Any) -> str | None:
    """Read the identifier of a created item. The API names it `id` or `<resource>_id`."""
    if not isinstance(created, dict):
        return None
    fields = cast("dict[str, Any]", created)
    for key, value in fields.items():
        if (key == "id" or key.endswith("_id")) and isinstance(value, str):
            return value
    return None


def _wait(console: Console, cmd: Cmd, created: Any, *, token: str | None, as_json: bool) -> Any:
    item_id = _created_id(created)
    if not item_id or not cmd.wait_path:
        return created
    path = cmd.wait_path.replace("{id}", str(item_id))
    if not as_json:
        console.print(f"[dim]Waiting for {item_id} to reach a final state…[/]")
    return _poll(console, path, token=token, as_json=as_json)


def _poll(console: Console, path: str, *, token: str | None, as_json: bool) -> Any:
    """Poll a GET path until its status is final. Returns the last response."""
    while True:
        time.sleep(_WAIT_POLL_S)
        current: Any = http.check(http.request("GET", path, token=token))
        fields = cast("dict[str, Any]", current) if isinstance(current, dict) else {}
        status = str(fields.get("status", ""))
        if status.lower() in _TERMINAL_STATUSES:
            return fields if isinstance(current, dict) else current
        if not as_json:
            console.print(f"[dim]  status: {status or 'unknown'}[/]")


def _topup(
    console: Console,
    args: argparse.Namespace,
    body: dict[str, Any],
    *,
    as_json: bool,
    token: str | None,
) -> int:
    response = http.request("POST", "/billing/topup", token=token, body=body)
    if response.status_code != 402:
        emit(console, http.check(response), as_json=as_json)
        return http.EXIT_OK

    challenge = http.parsed(response)
    if getattr(args, "no_pay", False):
        emit(
            console,
            {"error": "Payment required", "challenge": challenge},
            as_json=as_json,
        )
        return http.EXIT_PAYMENT

    npx = shutil.which("npx")
    if npx is None:
        emit(console, challenge, as_json=as_json)
        console.print(
            "[yellow]Payment required.[/] Install Node.js and run the command again, "
            "or pay the challenge above with an MPP wallet client."
        )
        return http.EXIT_PAYMENT

    credit_count = body.get("credits")
    if not getattr(args, "yes", False) and sys.stdin.isatty():
        answer = console.input(f"Buy {credit_count} credit(s) now? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            console.print("[yellow]Payment cancelled.[/]")
            return http.EXIT_PAYMENT

    url = f"{http.app_url()}/api/v1/billing/topup"
    auth_header = f"X-Strix-Authorization: Bearer {http.api_token(token)}"
    command = [
        npx,
        "--yes",
        "mppx",
        url,
        "-J",
        json.dumps(body),
        "-H",
        auth_header,
    ]
    payment_method = getattr(args, "payment_method", None) or os.environ.get(
        "MPPX_STRIPE_PAYMENT_METHOD"
    )
    if payment_method:
        command += ["-M", f"paymentMethod={payment_method}"]
    elif not os.environ.get("MPPX_ACCOUNT") and not os.environ.get("MPPX_STRIPE_SECRET_KEY"):
        console.print(
            "[dim]Tip: payments need a wallet. Set up a Stripe agent wallet at "
            "https://link.com/agents, and the user approves each payment in the Link app. "
            "If the user does not want a wallet, run "
            "`strix cloud billing subscribe --plan strix_top_up` for a hosted checkout link.[/]"
        )
    result = subprocess.run(command, check=False)  # noqa: S603
    return http.EXIT_OK if result.returncode == 0 else http.EXIT_PAYMENT
