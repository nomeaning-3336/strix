"""Billing top-up and agent-wallet execution for ``strix cloud``."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import strix.interface.cloud.http as http  # noqa: PLR0402
from strix.interface.cloud.payment_proxy import WalletUpstreamResponse, wallet_payment_bridge
from strix.interface.cloud.render import emit
from strix.interface.terminal_text import sanitize_terminal_text


if TYPE_CHECKING:
    import argparse

    from rich.console import Console


_MAX_WALLET_DETAIL_CHARS = 2_000
# Keep the wallet client on the exact protocol implementation used by the
# platform. This version is also old enough to remain installable in npm
# environments that apply a short package-publication safety window.
_MPPX_PACKAGE = "mppx@0.8.17"
_NPM_REGISTRY = "https://registry.npmjs.org"
_WALLET_ENV_NAMES = frozenset(
    {
        "ALL_PROXY",
        "APPDATA",
        "COLORTERM",
        "COMSPEC",
        "FORCE_COLOR",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOCALAPPDATA",
        "NO_COLOR",
        "NO_PROXY",
        "PATH",
        "PATHEXT",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "XDG_CONFIG_HOME",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)
_AUTHORIZATION_SECRET = re.compile(r"(?i)((?:bearer|payment)\s+)[^\s\"']+")
_LOOPBACK_NO_PROXY = ("127.0.0.1", "localhost", "::1")


@dataclass(frozen=True)
class _WalletClientResult:
    process: subprocess.CompletedProcess[str]
    upstream_responses: tuple[WalletUpstreamResponse, ...]


def run_topup(  # noqa: PLR0911, PLR0912
    console: Console,
    args: argparse.Namespace,
    body: dict[str, Any],
    *,
    as_json: bool,
    token: str | None,
) -> int:
    """Handle the HTTP 402 challenge and optional agent-wallet payment."""
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

    credit_count = body.get("credits")
    if not getattr(args, "yes", False):
        if as_json or not (sys.stdin.isatty() and sys.stdout.isatty()):
            emit(
                console,
                {
                    "error": (
                        "Payment requires explicit approval in non-interactive mode. "
                        "Review the challenge, then re-run with --yes to authorize payment."
                    ),
                    "challenge": challenge,
                },
                as_json=as_json,
            )
            return http.EXIT_PAYMENT
        answer = console.input(f"Buy {credit_count} credit(s) now? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            console.print("[yellow]Payment cancelled.[/]")
            return http.EXIT_PAYMENT

    npx = shutil.which("npx")
    if npx is None:
        message = (
            "Payment requires a wallet client. Install Node.js and run the command again, "
            "or pay the challenge with an MPP wallet client."
        )
        if as_json:
            emit(
                console,
                {"error": message, "challenge": challenge},
                as_json=True,
            )
        else:
            emit(console, challenge, as_json=False)
            console.print(f"[yellow]Payment required.[/] {message}")
        return http.EXIT_PAYMENT

    payment_method = getattr(args, "payment_method", None) or os.environ.get(
        "MPPX_STRIPE_PAYMENT_METHOD"
    )
    if not payment_method and (
        not as_json
        and not os.environ.get("MPPX_ACCOUNT")
        and not os.environ.get("MPPX_STRIPE_SECRET_KEY")
    ):
        console.print(
            "[dim]Tip: payments need a wallet. Set up a Stripe agent wallet at "
            "https://link.com/agents, and the user approves each payment in the Link app. "
            "If the user does not want a wallet, run "
            "`strix cloud billing subscribe --plan strix_top_up` for a hosted checkout link.[/]"
        )
    try:
        wallet_result = _run_wallet_client(
            npx,
            args,
            body,
            token=token,
            payment_method=payment_method,
            capture_output=as_json,
        )
    except KeyboardInterrupt:
        emit(
            console,
            {
                "error": (
                    "Payment was interrupted after the wallet started. The outcome is unknown; "
                    "run `strix cloud billing credits` and check the balance before retrying."
                ),
                "interrupted": True,
                "payment_outcome_unknown": True,
            },
            as_json=as_json,
        )
        return 130
    except OSError:
        emit(
            console,
            {
                "error": "Could not start the wallet client securely.",
                "challenge": challenge,
            },
            as_json=as_json,
        )
        return http.EXIT_PAYMENT

    result = wallet_result.process
    confirmed_receipt = _confirmed_topup_receipt(wallet_result.upstream_responses)
    if confirmed_receipt is not None:
        if as_json:
            emit(console, confirmed_receipt, as_json=True)
        return http.EXIT_OK

    if not as_json:
        console.print(
            "[yellow]The wallet exited without a confirmed receipt. The payment outcome is "
            "unknown; run `strix cloud billing credits` before retrying.[/]"
        )
        return http.EXIT_PAYMENT

    stdout = str(getattr(result, "stdout", "") or "").strip()
    stderr = str(getattr(result, "stderr", "") or "").strip()
    if result.returncode == 0:
        try:
            receipt = json.loads(stdout)
        except (TypeError, ValueError):
            emit(
                console,
                {
                    "error": (
                        "The wallet reported success but did not return JSON. Check the credit "
                        "balance before retrying payment."
                    ),
                    "detail": _wallet_detail(stdout or stderr or "No wallet output was returned."),
                    "payment_outcome_unknown": True,
                },
                as_json=True,
            )
            return http.EXIT_PAYMENT
        if not _valid_topup_receipt(receipt):
            emit(
                console,
                {
                    "error": (
                        "The wallet returned an invalid top-up receipt. Check the credit balance "
                        "before retrying payment."
                    ),
                    "detail": _wallet_detail(stdout),
                    "payment_outcome_unknown": True,
                },
                as_json=True,
            )
            return http.EXIT_PAYMENT
        emit(
            console,
            {
                "error": (
                    "The wallet returned a receipt, but the Strix billing endpoint did not "
                    "confirm it. Check the credit balance before retrying payment."
                ),
                "detail": _wallet_detail(stdout),
                "payment_outcome_unknown": True,
            },
            as_json=True,
        )
        return http.EXIT_PAYMENT

    emit(
        console,
        {
            "error": (
                "The wallet exited without a confirmed receipt. The payment outcome is unknown; "
                "run `strix cloud billing credits` and check the balance before retrying."
            ),
            "detail": _wallet_detail(
                stderr or stdout or f"Wallet client exited with status {result.returncode}."
            ),
            "wallet_exit_code": result.returncode,
            "payment_outcome_unknown": True,
        },
        as_json=True,
    )
    return http.EXIT_PAYMENT


def _run_wallet_client(
    npx: str,
    args: argparse.Namespace,
    body: dict[str, Any],
    *,
    token: str | None,
    payment_method: str | None,
    capture_output: bool,
) -> _WalletClientResult:
    """Run mppx through the loopback bridge without exposing the API token to it."""
    upstream_url = f"{http.app_url()}/api/v1/billing/topup"
    body_json = json.dumps(body)
    wallet_env = _wallet_environment()
    upstream_responses: list[WalletUpstreamResponse] = []
    with tempfile.TemporaryDirectory(prefix="strix-wallet-") as wallet_cwd:
        wallet_root = Path(wallet_cwd)
        user_config = wallet_root / "user.npmrc"
        global_config = wallet_root / "global.npmrc"
        user_config.touch(mode=0o600)
        global_config.touch(mode=0o600)
        with wallet_payment_bridge(
            upstream_url=upstream_url,
            api_token=http.api_token(token),
            expected_body=body_json.encode(),
            timeout=getattr(args, "timeout", None),
            response_observer=upstream_responses.append,
        ) as wallet_url:
            command = [
                npx,
                "--yes",
                f"--registry={_NPM_REGISTRY}",
                "--ignore-scripts",
                f"--userconfig={user_config}",
                f"--globalconfig={global_config}",
                f"--cache={wallet_root / 'npm-cache'}",
                _MPPX_PACKAGE,
                wallet_url,
                "--fail",
                "-J",
                body_json,
            ]
            if payment_method:
                command += ["-M", f"paymentMethod={payment_method}"]
            process = subprocess.run(  # noqa: S603
                command,
                check=False,
                capture_output=capture_output,
                text=True,
                env=wallet_env,
                cwd=wallet_root,
            )
    return _WalletClientResult(process=process, upstream_responses=tuple(upstream_responses))


def _wallet_environment() -> dict[str, str]:
    """Pass only platform essentials and explicit wallet variables to npm/mppx."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in _WALLET_ENV_NAMES or name.startswith("MPPX_")
    }
    for name in ("NO_PROXY", "no_proxy"):
        entries = [entry.strip() for entry in environment.get(name, "").split(",") if entry.strip()]
        normalized = {entry.lower().strip("[]") for entry in entries}
        entries.extend(host for host in _LOOPBACK_NO_PROXY if host not in normalized)
        environment[name] = ",".join(entries)
    return environment


def _wallet_detail(value: str) -> str:
    """Bound and redact third-party wallet diagnostics before returning JSON."""
    redacted = _AUTHORIZATION_SECRET.sub(r"\1[redacted]", sanitize_terminal_text(value))
    if len(redacted) <= _MAX_WALLET_DETAIL_CHARS:
        return redacted
    return redacted[: _MAX_WALLET_DETAIL_CHARS - 1] + "…"


def _valid_topup_receipt(value: Any) -> bool:
    """Require the documented success shape before reporting a paid top-up."""
    if not isinstance(value, dict):
        return False
    fields = cast("dict[str, Any]", value)
    credits_granted = fields.get("credits_granted")
    balance = fields.get("balance")
    return (
        isinstance(credits_granted, int)
        and not isinstance(credits_granted, bool)
        and credits_granted >= 0
        and isinstance(fields.get("duplicate"), bool)
        and isinstance(fields.get("reference"), str)
        and bool(fields["reference"])
        and isinstance(balance, int)
        and not isinstance(balance, bool)
        and balance >= 0
    )


def _confirmed_topup_receipt(
    responses: tuple[WalletUpstreamResponse, ...],
) -> dict[str, Any] | None:
    """Return a receipt only when the trusted bridge observed its successful response."""
    for response in reversed(responses):
        if not 200 <= response.status_code < 300:
            continue
        try:
            receipt = json.loads(response.body)
        except (TypeError, ValueError):
            continue
        if _valid_topup_receipt(receipt):
            return cast("dict[str, Any]", receipt)
    return None
