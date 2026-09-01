"""Tests for the Stripe Link wallet setup path of `strix cloud billing topup`."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

from rich.console import Console

from strix.interface.cloud import billing


if TYPE_CHECKING:
    import pytest


_MIN_LINK_CONTEXT_CHARS = 100


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["link-cli"], returncode=0, stdout=stdout, stderr="")


def test_payment_context_is_long_enough_for_link_approval() -> None:
    context = billing._payment_context({"credits": 5})
    assert len(context) >= _MIN_LINK_CONTEXT_CHARS
    assert "5" in context


def test_mppx_wallet_configured_follows_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MPPX_ACCOUNT", raising=False)
    monkeypatch.delenv("MPPX_STRIPE_SECRET_KEY", raising=False)
    assert billing._mppx_wallet_configured() is False
    monkeypatch.setenv("MPPX_ACCOUNT", "agent")
    assert billing._mppx_wallet_configured() is True


def test_link_wallet_authenticated_reads_status_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        billing,
        "_run_link_cli",
        lambda *_args, **_kwargs: _completed('[{"authenticated": true}]'),
    )
    assert billing._link_wallet_authenticated("npx") is True


def test_link_wallet_authenticated_handles_unusable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(billing, "_run_link_cli", lambda *_args, **_kwargs: _completed("not json"))
    assert billing._link_wallet_authenticated("npx") is False


def test_link_wallet_authenticated_handles_launch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError

    monkeypatch.setattr(billing, "_run_link_cli", explode)
    assert billing._link_wallet_authenticated("npx") is False


def test_prepare_link_wallet_skips_login_when_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(billing, "_link_wallet_authenticated", lambda _npx: True)
    assert billing._prepare_link_wallet(Console(), "npx", as_json=True) is None


def test_prepare_link_wallet_explains_setup_without_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(billing, "_link_wallet_authenticated", lambda _npx: False)
    message = billing._prepare_link_wallet(Console(), "npx", as_json=True)
    assert message is not None
    assert "https://link.com/agents" in message
