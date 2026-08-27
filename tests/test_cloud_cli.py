"""Tests for the `strix cloud` CLI: routing, request building, and output."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import webbrowser
from typing import TYPE_CHECKING, Any

import pytest
import requests

from strix.interface import cloud, platform_cli
from strix.interface.cloud import http, render, runner, workspaces
from strix.interface.cloud.spec import SPEC


if TYPE_CHECKING:
    from pathlib import Path


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text if payload is None else json.dumps(payload)
        self.content = content
        self.ok = 200 <= status_code < 400
        self.headers = {"content-type": "application/json" if payload is not None else "text/plain"}

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


@pytest.fixture(autouse=True)
def _token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIX_API_TOKEN", "test-token")


def test_help_returns_zero() -> None:
    assert cloud.run_cloud([]) == 0
    assert cloud.run_cloud(["--help"]) == 0


def test_unknown_group_returns_usage_error() -> None:
    assert cloud.run_cloud(["bogus"]) == 2


def test_unknown_verb_returns_usage_error() -> None:
    assert cloud.run_cloud(["scans", "bogus"]) == 2


def test_group_without_verb_lists_verbs() -> None:
    assert cloud.run_cloud(["scans"]) == 0


def test_resolve_prefers_two_word_verbs() -> None:
    resolved = runner.resolve("billing", ["auto-topup", "update", "--enabled"])
    assert resolved is not None
    cmd, remaining = resolved
    assert cmd.path == "/billing/auto-topup"
    assert cmd.method == "PUT"
    assert remaining == ["--enabled"]


def test_resolve_default_verb() -> None:
    resolved = runner.resolve("audit", [])
    assert resolved is not None
    cmd, remaining = resolved
    assert cmd.method == "GET"
    assert remaining == []


def test_dest_converts_camel_case() -> None:
    assert runner._dest("scanId") == "scan_id"
    assert runner._dest("chatId") == "chat_id"
    assert runner._metavar("findingId") == "FINDING_ID"


def test_placeholder_substitution(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    seen: dict[str, Any] = {}

    def fake_request(method: str, path: str, **kwargs: Any) -> FakeResponse:
        seen.update(method=method, path=path, query=kwargs.get("query"))
        return FakeResponse(payload={"id": "abc"})

    monkeypatch.setattr(http, "request", fake_request)
    code = cloud.run_cloud(["scans", "get", "abc-123", "--json"])
    assert code == 0
    assert seen["method"] == "GET"
    assert seen["path"] == "/scans/abc-123"
    assert json.loads(capsys.readouterr().out) == {"id": "abc"}


def test_placeholder_substitution_percent_encodes_path_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, path: str, **_kwargs: Any) -> FakeResponse:
        seen["path"] = path
        return FakeResponse(payload={"entries": []})

    monkeypatch.setattr(http, "request", fake_request)
    code = cloud.run_cloud(["knowledge", "repos", "entries", "usestrix/.github", "--json"])
    assert code == 0
    assert seen["path"] == "/knowledge/repos/usestrix%2F.github/entries"


def test_query_and_body_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen.update(query=kwargs.get("query"), body=kwargs.get("body"))
        return FakeResponse(payload={"ok": True})

    monkeypatch.setattr(http, "request", fake_request)
    assert cloud.run_cloud(["scans", "list", "--status", "running", "--json"]) == 0
    assert seen["query"] == {"status": "running"}

    assert (
        cloud.run_cloud(
            [
                "scans",
                "start",
                "--engagement-type",
                "live_test",
                "--domain-ids",
                "d1",
                "d2",
                "--json",
            ]
        )
        == 0
    )
    assert seen["body"] == {"engagement_type": "live_test", "domain_ids": ["d1", "d2"]}


def test_data_merges_extra_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"ok": True})

    monkeypatch.setattr(http, "request", fake_request)
    code = cloud.run_cloud(
        ["scans", "start", "--data", '{"engagement_type": "code_review"}', "--json"]
    )
    assert code == 0
    assert seen["body"] == {"engagement_type": "code_review"}


def test_data_reads_a_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"ok": True})

    monkeypatch.setattr(http, "request", fake_request)
    request_file = tmp_path / "request.json"
    request_file.write_text('{"focus": "IDOR"}', encoding="utf-8")
    assert cloud.run_cloud(["scans", "start", "--data", f"@{request_file}", "--json"]) == 0
    assert seen["body"] == {"focus": "IDOR"}


def test_data_reads_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"ok": True})

    monkeypatch.setattr(http, "request", fake_request)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"context": "staging"}'))
    assert cloud.run_cloud(["scans", "start", "--data", "-", "--json"]) == 0
    assert seen["body"] == {"context": "staging"}


def test_data_reports_a_missing_file(tmp_path: Path) -> None:
    assert cloud.run_cloud(["scans", "start", "--data", f"@{tmp_path / 'nope.json'}"]) == 1


def test_auto_topup_removes_the_monthly_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"ok": True})

    monkeypatch.setattr(http, "request", fake_request)
    code = cloud.run_cloud(
        [
            "billing",
            "auto-topup",
            "update",
            "--enabled",
            "--topup-credits",
            "20",
            "--no-monthly-cap",
            "--json",
        ]
    )
    assert code == 0
    assert seen["body"] == {
        "enabled": True,
        "topup_credits": 20,
        "monthly_cap_credits": None,
    }


def test_costs_default_verb_is_the_overview(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, path: str, **_kwargs: Any) -> FakeResponse:
        seen["path"] = path
        return FakeResponse(payload={"total_cost": 1})

    monkeypatch.setattr(http, "request", fake_request)
    assert cloud.run_cloud(["costs", "--json"]) == 0
    assert seen["path"] == "/llm-costs"


def test_binary_download_writes_a_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=200, content=b"%PDF-1.7")
    )
    target = tmp_path / "report.pdf"
    assert cloud.run_cloud(["scans", "report", "scan-1", "--output", str(target)]) == 0
    assert target.read_bytes() == b"%PDF-1.7"


def test_wait_polls_until_the_status_is_final(monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = iter(["running", "completed"])

    def fake_request(method: str, _path: str, **_kwargs: Any) -> FakeResponse:
        if method == "POST":
            return FakeResponse(payload={"id": "scan-1", "status": "pending"})
        return FakeResponse(payload={"id": "scan-1", "status": next(statuses)})

    monkeypatch.setattr(http, "request", fake_request)
    monkeypatch.setattr(runner, "_WAIT_POLL_S", 0)
    assert cloud.run_cloud(["scans", "start", "--domain-ids", "d1", "--wait", "--json"]) == 0
    assert next(statuses, None) is None


def test_insufficient_credits_exits_with_payment_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=402, payload={})
    )
    assert cloud.run_cloud(["scans", "start", "--domain-ids", "d1"]) == http.EXIT_PAYMENT


def test_data_rejects_non_object() -> None:
    assert cloud.run_cloud(["scans", "start", "--data", "[1,2]"]) == 1
    assert cloud.run_cloud(["scans", "start", "--data", "not json"]) == 1


def test_missing_token_exits_with_auth_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("STRIX_API_TOKEN", raising=False)
    monkeypatch.setattr(platform_cli, "AUTH_PATH", tmp_path / "platform-auth.json")
    assert cloud.run_cloud(["credits"]) == http.EXIT_AUTH


def test_http_error_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    for status, expected in ((401, http.EXIT_AUTH), (403, http.EXIT_AUTH), (500, http.EXIT_ERROR)):
        monkeypatch.setattr(
            http,
            "request",
            lambda *_a, _s=status, **_k: FakeResponse(status_code=_s, payload={"error": "x"}),
        )
        assert cloud.run_cloud(["scans", "list"]) == expected


def test_credits_alias_routes_to_billing(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, path: str, **_kwargs: Any) -> FakeResponse:
        seen["path"] = path
        return FakeResponse(payload={"balance": 3})

    monkeypatch.setattr(http, "request", fake_request)
    assert cloud.run_cloud(["credits", "--json"]) == 0
    assert seen["path"] == "/billing/credits"


def test_whoami_json_is_machine_readable_and_omits_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    monkeypatch.delenv("STRIX_API_TOKEN", raising=False)
    monkeypatch.setattr(platform_cli, "AUTH_PATH", tmp_path / "platform-auth.json")
    platform_cli.save_record(
        {
            "api_token": "strix_pat_secret",
            "email": "agent@example.test",
            "organization_id": "org_1",
            "organization_name": "Example",
            "scopes": ["scans:read"],
            "expires_at": "2026-09-01T00:00:00Z",
        }
    )

    assert cloud.run_cloud(["whoami", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "signed_in": True,
        "email": "agent@example.test",
        "organization_id": "org_1",
        "organization_name": "Example",
        "scopes": ["scans:read"],
        "expires_at": "2026-09-01T00:00:00Z",
    }
    assert "api_token" not in payload


def test_topup_no_pay_prints_challenge(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    challenge = {"payment_requirements": [{"amount": 500}]}
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=402, payload=challenge)
    )
    code = cloud.run_cloud(["billing", "topup", "--credits", "5", "--no-pay", "--json"])
    assert code == http.EXIT_PAYMENT
    assert json.loads(capsys.readouterr().out) == {
        "error": "Payment required",
        "challenge": challenge,
    }


def test_topup_success_without_payment(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    receipt = {"credits_granted": 5, "duplicate": False, "balance": 5}
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=200, payload=receipt)
    )
    code = cloud.run_cloud(["billing", "topup", "--credits", "5", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out) == receipt


def test_topup_passes_payment_method_to_wallet(monkeypatch: pytest.MonkeyPatch) -> None:
    challenge = {"payment_requirements": [{"amount": 500}]}
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=402, payload=challenge)
    )
    monkeypatch.setattr(http, "api_token", lambda *_a, **_k: "tok")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/npx")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        commands.append(command)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(subprocess, "run", fake_run)
    code = cloud.run_cloud(
        ["billing", "topup", "--credits", "5", "--yes", "--payment-method", "pm_card_visa"]
    )
    assert code == 0
    assert "X-Strix-Authorization: Bearer tok" in commands[0]
    assert "Authorization: Bearer tok" not in commands[0]
    assert "-M" in commands[0]
    assert "paymentMethod=pm_card_visa" in commands[0]


def test_render_json_mode_when_not_a_tty() -> None:
    assert render.json_mode(flag=True) is True
    # Under pytest, stdout is captured and is not a terminal.
    assert render.json_mode(flag=False) is True


def test_render_list_extraction() -> None:
    rows = render._list_of_dicts({"scans": [{"id": "a"}, {"id": "b"}]})
    assert rows == [{"id": "a"}, {"id": "b"}]
    assert render._list_of_dicts({"scans": [], "total": 1}) is None
    assert render._list_of_dicts([{"id": "a"}, "x"]) is None


def test_spec_paths_are_well_formed() -> None:
    for group, commands in SPEC.items():
        for verb, cmd in commands.items():
            assert cmd.path.startswith("/"), f"{group} {verb}"
            assert cmd.method in ("GET", "POST", "PUT", "PATCH", "DELETE"), f"{group} {verb}"
            assert cmd.help, f"{group} {verb} has no help text"
            for param in cmd.query + cmd.body:
                assert param.kind in ("str", "int", "float", "bool", "list", "json"), (
                    f"{group} {verb} {param.name}"
                )


def test_every_command_builds_a_parser() -> None:
    for group, commands in SPEC.items():
        for verb, cmd in commands.items():
            parser = runner._build_parser(group, verb, cmd)
            assert parser.prog == f"strix cloud {group} {verb}"


def test_app_url_and_timeout_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, url: str, **kwargs: Any) -> FakeResponse:
        seen["url"] = url
        seen["timeout"] = kwargs.get("timeout")
        return FakeResponse(status_code=200, payload={"balance": 1})

    monkeypatch.setattr(http, "api_token", lambda _override=None: "t")
    monkeypatch.setattr(requests, "request", fake_request)
    code = cloud.run_cloud(
        ["credits", "--app-url", "https://example.test/", "--timeout", "7", "--json"]
    )
    assert code == 0
    assert seen["url"] == "https://example.test/api/v1/billing/credits"
    assert seen["timeout"] == 7


def test_created_id_reads_resource_id() -> None:
    assert runner._created_id({"scan_id": "abc", "status": "pending"}) == "abc"
    assert runner._created_id({"id": "xyz"}) == "xyz"
    assert runner._created_id({"status": "pending"}) is None


def test_billing_subscribe_prints_checkout_url(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(method: str, path: str, **kwargs: Any) -> FakeResponse:
        seen["method"], seen["path"] = method, path
        seen["body"] = kwargs.get("body")
        return FakeResponse(status_code=200, payload={"checkout_url": "https://pay.test/session"})

    monkeypatch.setattr(http, "request", fake_request)
    code = cloud.run_cloud(["billing", "subscribe", "--plan", "strix_cloud", "--json"])
    assert code == 0
    assert seen["method"] == "POST"
    assert seen["path"] == "/billing/checkout"
    assert seen["body"] == {"product": "strix_cloud"}
    assert "https://pay.test/session" in capsys.readouterr().out


def test_knowledge_policy_flags_use_the_api_field_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"success": True})

    monkeypatch.setattr(http, "request", fake_request)
    code = cloud.run_cloud(
        [
            "knowledge",
            "policies",
            "add",
            "--key",
            "no-production-data",
            "--content",
            "Never test production data.",
            "--policy-type",
            "constraint",
            "--no-enabled",
            "--metadata",
            '{"owner":"security"}',
            "--json",
        ]
    )
    assert code == 0
    assert seen["body"] == {
        "policy_key": "no-production-data",
        "policy_value": "Never test production data.",
        "policy_type": "constraint",
        "is_active": False,
        "metadata": {"owner": "security"},
    }


def test_pr_review_start_sends_provider_installation_and_pull_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"review_id": "review-1", "status": "pending"})

    monkeypatch.setattr(http, "request", fake_request)
    code = cloud.run_cloud(
        [
            "pr-reviews",
            "start",
            "--provider",
            "github",
            "--installation-id",
            "123",
            "--repository-full-name",
            "org/app",
            "--pr-number",
            "42",
            "--json",
        ]
    )
    assert code == 0
    assert seen["body"] == {
        "provider": "github",
        "installation_id": 123,
        "repository_full_name": "org/app",
        "pr_number": 42,
    }


def test_llm_settings_uses_kebab_case_flag_for_camel_case_api_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"ok": True})

    monkeypatch.setattr(http, "request", fake_request)
    code = cloud.run_cloud(
        [
            "llm-settings",
            "update",
            "--model-configs",
            "[]",
            "--assignments",
            "{}",
            "--json",
        ]
    )
    assert code == 0
    assert seen["body"] == {"modelConfigs": [], "assignments": {}}


def test_integration_install_url_does_not_open_browser(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(status_code=200, payload={"url": "https://github.test/app"}),
    )

    def fake_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(webbrowser, "open", fake_open)
    code = cloud.run_cloud(["integrations", "install", "github", "--json"])
    assert code == 0
    assert opened == []
    assert "https://github.test/app" in capsys.readouterr().out


def test_workspaces_use_switches_stored_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    auth_path = tmp_path / "platform-auth.json"
    monkeypatch.setattr(platform_cli, "AUTH_PATH", auth_path)
    monkeypatch.setattr(workspaces, "AUTH_PATH", auth_path)
    platform_cli.save_record(
        {
            "api_token": "old",
            "email": "a@b.test",
            "scopes": ["scans:read", "organizations:read", "tokens:write"],
        }
    )

    calls: list[tuple[str, str]] = []
    token_body: dict[str, Any] | None = None

    def fake_request(method: str, path: str, **kwargs: Any) -> FakeResponse:
        nonlocal token_body
        calls.append((method, path))
        if path == "/workspaces":
            return FakeResponse(
                status_code=200,
                payload={"workspaces": [{"id": "org_1", "name": "Team One", "role": "admin"}]},
            )
        token_body = kwargs.get("body")
        return FakeResponse(
            status_code=201,
            payload={
                "api_token": "new-token",
                "organization_id": "org_1",
                "organization_name": "Team One",
                "scopes": ["scans:read"],
            },
        )

    monkeypatch.setattr(http, "request", fake_request)
    code = cloud.run_cloud(["workspaces", "use", "team one", "--json"])
    assert code == 0
    assert calls == [("GET", "/workspaces"), ("POST", "/workspaces/org_1/token")]
    assert token_body == {
        "scopes": ["scans:read", "organizations:read", "tokens:write"]
    }
    record = platform_cli.read_record()
    assert record is not None
    assert record["api_token"] == "new-token"
    assert record["organization_name"] == "Team One"
    assert record["email"] == "a@b.test"
    assert "org_1" in capsys.readouterr().out


def test_workspaces_use_reports_unknown_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(
            status_code=200, payload={"workspaces": [{"id": "org_1", "name": "Team One"}]}
        ),
    )
    assert cloud.run_cloud(["workspaces", "use", "missing", "--json"]) == 1


def test_group_help_lists_all_verbs_instead_of_default_verb_help(capsys: Any) -> None:
    assert cloud.run_cloud(["workspaces", "-h"]) == 0
    output = capsys.readouterr().out
    assert "workspaces verbs" in output
    assert "list" in output
    assert "create" in output
    assert "use" in output
