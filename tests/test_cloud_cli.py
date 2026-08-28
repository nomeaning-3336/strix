"""Tests for the `strix cloud` CLI: routing, request building, and output."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

import pytest
import requests

from strix.interface import cloud, platform_cli
from strix.interface.cloud import billing, http, payment_proxy, render, runner, workspaces
from strix.interface.cloud.spec import GROUP_HELP, SPEC


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

    def iter_content(self, chunk_size: int) -> Any:
        for index in range(0, len(self.content), chunk_size):
            yield self.content[index : index + chunk_size]

    def close(self) -> None:
        pass


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


def test_successful_html_response_is_reported_without_dumping_html(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(text="<!DOCTYPE html><html>preview gate</html>"),
    )

    assert cloud.run_cloud(["workspaces", "list", "--json"]) == 1
    output = capsys.readouterr().out
    assert "non-JSON response" in output
    assert "STRIX_APP_URL" in output
    assert "<!DOCTYPE" not in output


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


def test_token_create_accepts_expiry_and_rbac_scope_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"id": "token-1", "token": "strix_pat_once"})

    monkeypatch.setattr(http, "request", fake_request)
    code = cloud.run_cloud(
        [
            "tokens",
            "create",
            "--type",
            "service",
            "--name",
            "ci",
            "--expires-at",
            "2026-09-30T12:00:00Z",
            "--rbac-scopes",
            '[{"type":"tag","value":"staging"}]',
            "--json",
        ]
    )

    assert code == 0
    assert seen["body"] == {
        "type": "service",
        "name": "ci",
        "expires_at": "2026-09-30T12:00:00Z",
        "rbac_scopes": [{"type": "tag", "value": "staging"}],
    }


def test_token_create_rejects_non_array_rbac_scopes(capsys: Any) -> None:
    code = cloud.run_cloud(
        [
            "tokens",
            "create",
            "--type",
            "service",
            "--name",
            "ci",
            "--rbac-scopes",
            '{"type":"tag","value":"staging"}',
            "--json",
        ]
    )

    assert code == http.EXIT_USAGE
    assert json.loads(capsys.readouterr().out)["error"] == ("--rbac-scopes must be a JSON array")


def test_token_create_rejects_two_expiration_modes(capsys: Any) -> None:
    code = cloud.run_cloud(
        [
            "tokens",
            "create",
            "--type",
            "personal",
            "--name",
            "local",
            "--expires-at",
            "2026-09-30T12:00:00Z",
            "--expires-in-days",
            "30",
            "--json",
        ]
    )

    assert code == http.EXIT_USAGE
    assert json.loads(capsys.readouterr().out)["error"] == (
        "--expires-at and --expires-in-days are mutually exclusive."
    )


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


def test_required_secret_body_field_can_come_from_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"ok": True})

    monkeypatch.setattr(http, "request", fake_request)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"token":"provider-secret"}'))

    assert cloud.run_cloud(["integrations", "connect", "gitlab", "--data", "-", "--json"]) == 0
    assert seen["body"] == {"token": "provider-secret"}


def test_required_body_field_is_validated_after_data_merge(capsys: Any) -> None:
    assert cloud.run_cloud(["integrations", "connect", "gitlab", "--json"]) == http.EXIT_USAGE
    assert "--provider-token" in json.loads(capsys.readouterr().out)["error"]


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


def test_wait_failure_keeps_the_created_operation_id(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    def fake_request(method: str, _path: str, **_kwargs: Any) -> FakeResponse:
        if method == "POST":
            return FakeResponse(payload={"id": "scan-created", "status": "running"})
        return FakeResponse(status_code=503, payload={"detail": "temporarily unavailable"})

    monkeypatch.setattr(http, "request", fake_request)
    assert cloud.run_cloud(["scans", "start", "--domain-ids", "d1", "--wait", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation_id"] == "scan-created"
    assert payload["status_unknown"] is True


def test_ambiguous_scan_request_warns_before_retry(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(
        http,
        "request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(http.CloudError("connection reset")),
    )

    assert cloud.run_cloud(["scans", "start", "--domain-ids", "d1", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["launch_outcome_unknown"] is True
    assert "scans list" in payload["error"]


def test_insufficient_credits_exits_with_payment_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=402, payload={})
    )
    assert cloud.run_cloud(["scans", "start", "--domain-ids", "d1"]) == http.EXIT_PAYMENT


def test_data_rejects_non_object() -> None:
    assert cloud.run_cloud(["scans", "start", "--data", "[1,2]"]) == http.EXIT_USAGE
    assert cloud.run_cloud(["scans", "start", "--data", "not json"]) == http.EXIT_USAGE


def test_typed_json_flag_parse_error_is_usage_error() -> None:
    assert cloud.run_cloud(["scans", "start", "--domain-paths", "not-json"]) == http.EXIT_USAGE


def test_missing_token_exits_with_auth_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("STRIX_API_TOKEN", raising=False)
    monkeypatch.setattr(platform_cli, "AUTH_PATH", tmp_path / "platform-auth.json")
    assert cloud.run_cloud(["credits"]) == http.EXIT_AUTH


def test_stored_token_is_never_sent_to_a_different_platform_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STRIX_API_TOKEN", raising=False)
    monkeypatch.setattr(http, "_app_url_override", "https://attacker.example")
    monkeypatch.setattr(
        http,
        "read_record",
        lambda: {"api_token": "stored-secret", "app_url": "https://app.strix.ai"},
    )
    monkeypatch.setattr(
        http.requests,
        "request",
        lambda *_args, **_kwargs: pytest.fail("a mismatched origin must not receive the token"),
    )

    with pytest.raises(http.CloudError, match="different platform") as raised:
        http.request("GET", "/billing/credits")
    assert raised.value.exit_code == http.EXIT_AUTH


def test_stored_token_requires_an_issuer_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRIX_API_TOKEN", raising=False)
    monkeypatch.setattr(http, "_app_url_override", "https://app.strix.ai")
    monkeypatch.setattr(http, "read_record", lambda: {"api_token": "legacy-secret"})
    monkeypatch.setattr(
        http.requests,
        "request",
        lambda *_args, **_kwargs: pytest.fail("an unbound token must not be sent"),
    )

    with pytest.raises(http.CloudError, match="not bound") as raised:
        http.request("GET", "/billing/credits")
    assert raised.value.exit_code == http.EXIT_AUTH


def test_stored_token_is_sent_only_to_its_bound_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STRIX_API_TOKEN", raising=False)
    monkeypatch.setattr(http, "_app_url_override", "https://preview.strix.ai")
    monkeypatch.setattr(
        http,
        "read_record",
        lambda: {"api_token": "stored-secret", "app_url": "https://preview.strix.ai"},
    )
    seen: dict[str, Any] = {}

    def request(_method: str, url: str, **kwargs: Any) -> FakeResponse:
        seen.update(url=url, headers=kwargs["headers"])
        return FakeResponse(payload={"balance": 1})

    monkeypatch.setattr(http.requests, "request", request)
    response = http.request("GET", "/billing/credits")

    assert response.status_code == 200
    assert seen["url"] == "https://preview.strix.ai/api/v1/billing/credits"
    assert seen["headers"]["Authorization"] == "Bearer stored-secret"


def test_explicit_token_can_target_an_explicit_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STRIX_API_TOKEN", raising=False)
    monkeypatch.setattr(http, "_app_url_override", "https://preview.strix.ai")
    monkeypatch.setattr(
        http,
        "read_record",
        lambda: {"api_token": "stored-secret", "app_url": "https://app.strix.ai"},
    )
    seen: dict[str, Any] = {}

    def request(_method: str, url: str, **kwargs: Any) -> FakeResponse:
        seen.update(url=url, headers=kwargs["headers"])
        return FakeResponse(payload={"balance": 1})

    monkeypatch.setattr(http.requests, "request", request)
    override_value = "explicit-preview-" + str(1)
    response = http.request("GET", "/billing/credits", token=override_value)

    assert response.status_code == 200
    assert seen["url"] == "https://preview.strix.ai/api/v1/billing/credits"
    assert seen["headers"]["Authorization"] == f"Bearer {override_value}"


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


def test_topup_noninteractive_requires_explicit_payment_approval(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    challenge = {"payment_requirements": [{"amount": 500}]}
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=402, payload=challenge)
    )
    monkeypatch.setattr(runner.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        billing.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("wallet must not run without --yes"),
    )

    code = cloud.run_cloud(["billing", "topup", "--credits", "5", "--json"])

    assert code == http.EXIT_PAYMENT
    payload = json.loads(capsys.readouterr().out)
    assert "requires explicit approval" in payload["error"]
    assert payload["challenge"] == challenge


@pytest.mark.parametrize("explicit_json,stdout_tty", [(True, True), (False, False)])
def test_topup_machine_output_never_prompts_even_with_terminal_stdin(
    explicit_json: bool,
    stdout_tty: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    challenge = {"payment_requirements": [{"amount": 500}]}
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=402, payload=challenge)
    )
    monkeypatch.setattr(runner.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(runner.sys.stdout, "isatty", lambda: stdout_tty)
    monkeypatch.setattr(
        runner.Console,
        "input",
        lambda *_a, **_k: pytest.fail("machine-readable top-up must not prompt"),
    )

    argv = ["billing", "topup", "--credits", "5"]
    if explicit_json:
        argv.append("--json")
    assert cloud.run_cloud(argv) == http.EXIT_PAYMENT

    payload = json.loads(capsys.readouterr().out)
    assert "requires explicit approval" in payload["error"]
    assert payload["challenge"] == challenge


def test_topup_payment_flags_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(http, "request", lambda *_a, **_k: pytest.fail("must not request"))
    assert (
        cloud.run_cloud(["billing", "topup", "--credits", "5", "--yes", "--no-pay", "--json"])
        == http.EXIT_USAGE
    )


def test_data_cannot_override_an_explicit_payment_amount(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(http, "request", lambda *_a, **_k: pytest.fail("must not request"))

    assert (
        cloud.run_cloud(
            [
                "billing",
                "topup",
                "--credits",
                "5",
                "--data",
                '{"credits": 500}',
                "--yes",
                "--json",
            ]
        )
        == http.EXIT_USAGE
    )
    assert "cannot override explicit" in json.loads(capsys.readouterr().out)["error"]


def test_topup_missing_wallet_keeps_json_machine_readable(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    challenge = {"payment_requirements": [{"amount": 500}]}
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=402, payload=challenge)
    )
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    code = cloud.run_cloud(["billing", "topup", "--credits", "5", "--yes", "--json"])

    assert code == http.EXIT_PAYMENT
    payload = json.loads(capsys.readouterr().out)
    assert "wallet client" in payload["error"]
    assert payload["challenge"] == challenge


def test_topup_success_without_payment(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    receipt = {"credits_granted": 5, "duplicate": False, "balance": 5}
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=200, payload=receipt)
    )
    code = cloud.run_cloud(["billing", "topup", "--credits", "5", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out) == receipt


def test_topup_keeps_token_out_of_wallet_process_and_forwards_payment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    challenge = {"payment_requirements": [{"amount": 500}]}
    receipt = {
        "credits_granted": 5,
        "duplicate": False,
        "reference": "pay_test_1",
        "balance": 10,
    }
    api_credential = "opaque-test-api-credential-value"
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".npmrc").write_text("registry=https://malicious.invalid\n", encoding="utf-8")
    monkeypatch.setenv("UNRELATED_CODING_AGENT_SECRET", "must-not-reach-wallet")
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=402, payload=challenge)
    )
    monkeypatch.setattr(http, "api_token", lambda *_a, **_k: api_credential)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/npx")
    commands: list[list[str]] = []
    child_envs: list[dict[str, str]] = []
    child_cwds: list[Path] = []
    upstream: dict[str, Any] = {}

    def fake_upstream_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        upstream.update(method=method, url=url, **kwargs)
        return FakeResponse(payload=receipt, content=json.dumps(receipt).encode())

    monkeypatch.setattr(payment_proxy.requests, "request", fake_upstream_request)

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        commands.append(command)
        child_envs.append(kwargs["env"])
        child_cwds.append(Path(kwargs["cwd"]))
        wallet_url = next(
            argument for argument in command if argument.startswith("http://127.0.0.1:")
        )
        request = urllib.request.Request(  # noqa: S310
            wallet_url,
            data=json.dumps({"credits": 5}).encode(),
            headers={
                "Authorization": "Payment wallet-credential",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310
            stdout = response.read().decode()
        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": stdout,
                "stderr": "",
            },
        )()

    monkeypatch.setattr(subprocess, "run", fake_run)
    code = cloud.run_cloud(
        ["billing", "topup", "--credits", "5", "--yes", "--payment-method", "pm_card_visa"]
    )
    assert code == 0
    assert all(api_credential not in argument for argument in commands[0])
    assert "mppx@0.8.17" in commands[0]
    assert "--registry=https://registry.npmjs.org" in commands[0]
    assert "--ignore-scripts" in commands[0]
    assert "-H" not in commands[0]
    assert "--fail" in commands[0]
    assert child_envs[0].get("STRIX_API_TOKEN") is None
    assert child_envs[0].get("UNRELATED_CODING_AGENT_SECRET") is None
    for name in ("NO_PROXY", "no_proxy"):
        bypasses = child_envs[0][name].split(",")
        assert "127.0.0.1" in bypasses
        assert "localhost" in bypasses
        assert "::1" in bypasses
    assert child_cwds[0] != tmp_path
    assert upstream["method"] == "POST"
    assert upstream["url"].endswith("/api/v1/billing/topup")
    assert upstream["headers"]["X-Strix-Authorization"] == f"Bearer {api_credential}"
    assert upstream["headers"]["Authorization"] == "Payment wallet-credential"
    assert upstream["data"] == json.dumps({"credits": 5}).encode()
    assert "-M" in commands[0]
    assert "paymentMethod=pm_card_visa" in commands[0]


def test_topup_wallet_failure_is_one_redacted_json_object(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    challenge = {"payment_requirements": [{"amount": 500}]}
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=402, payload=challenge)
    )
    monkeypatch.setattr(http, "api_token", lambda *_a, **_k: "tok")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/npx")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: type(
            "Result",
            (),
            {
                "returncode": 9,
                "stdout": "",
                "stderr": (
                    "failed with Bearer super-secret and "
                    "Authorization: Payment wallet-super-secret\x1b[2J"
                ),
            },
        )(),
    )

    assert cloud.run_cloud(["billing", "topup", "--credits", "5", "--yes", "--json"]) == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["wallet_exit_code"] == 9
    assert payload["payment_outcome_unknown"] is True
    assert "billing credits" in payload["error"]
    assert "super-secret" not in payload["detail"]
    assert "Bearer [redacted]" in payload["detail"]
    assert "Payment [redacted]" in payload["detail"]
    assert "\x1b" not in payload["detail"]


def test_topup_wallet_interruption_reports_unknown_payment_outcome(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(
            status_code=402,
            payload={"payment_requirements": [{"amount": 500}]},
        ),
    )
    monkeypatch.setattr(http, "api_token", lambda *_a, **_k: "tok")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/npx")
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    assert cloud.run_cloud(["billing", "topup", "--credits", "5", "--yes", "--json"]) == 130
    payload = json.loads(capsys.readouterr().out)
    assert payload["interrupted"] is True
    assert payload["payment_outcome_unknown"] is True
    assert "billing credits" in payload["error"]
    assert "before retrying" in payload["error"]


def test_topup_non_json_wallet_success_requires_balance_verification(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(
            status_code=402,
            payload={"payment_requirements": [{"amount": 500}]},
        ),
    )
    monkeypatch.setattr(http, "api_token", lambda *_a, **_k: "tok")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/npx")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: type("Result", (), {"returncode": 0, "stdout": "paid", "stderr": ""})(),
    )

    assert cloud.run_cloud(["billing", "topup", "--credits", "5", "--yes", "--json"]) == 5
    payload = json.loads(capsys.readouterr().out)
    assert "did not return JSON" in payload["error"]
    assert "before retrying" in payload["error"]
    assert payload["payment_outcome_unknown"] is True


def test_topup_rejects_parseable_wallet_error_as_a_success(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(
            status_code=402,
            payload={"payment_requirements": [{"amount": 500}]},
        ),
    )
    monkeypatch.setattr(http, "api_token", lambda *_a, **_k: "tok")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/npx")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": '{"detail":"Failed to process the top-up payment"}',
                "stderr": "",
            },
        )(),
    )

    assert cloud.run_cloud(["billing", "topup", "--credits", "5", "--yes", "--json"]) == 5
    payload = json.loads(capsys.readouterr().out)
    assert "invalid top-up receipt" in payload["error"]
    assert payload["payment_outcome_unknown"] is True


def test_topup_does_not_trust_an_unobserved_wallet_receipt(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    receipt = {
        "credits_granted": 5,
        "duplicate": False,
        "reference": "untrusted-wallet-output",
        "balance": 10,
    }
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(
            status_code=402,
            payload={"payment_requirements": [{"amount": 500}]},
        ),
    )
    monkeypatch.setattr(http, "api_token", lambda *_a, **_k: "tok")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/npx")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: type(
            "Result",
            (),
            {"returncode": 0, "stdout": json.dumps(receipt), "stderr": ""},
        )(),
    )

    assert cloud.run_cloud(["billing", "topup", "--credits", "5", "--yes", "--json"]) == 5
    payload = json.loads(capsys.readouterr().out)
    assert "did not confirm" in payload["error"]
    assert payload["payment_outcome_unknown"] is True


def test_topup_human_mode_requires_a_bridge_confirmed_receipt(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(render.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(
            status_code=402,
            payload={"payment_requirements": [{"amount": 500}]},
        ),
    )
    monkeypatch.setattr(http, "api_token", lambda *_a, **_k: "tok")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/npx")
    monkeypatch.setattr(
        payment_proxy.requests,
        "request",
        lambda *_a, **_k: FakeResponse(status_code=200, content=b"<html>not a receipt</html>"),
    )

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        wallet_url = next(
            argument for argument in command if argument.startswith("http://127.0.0.1:")
        )
        request = urllib.request.Request(  # noqa: S310
            wallet_url,
            data=json.dumps({"credits": 5}).encode(),
            headers={"Authorization": "Payment wallet-credential"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310
            response.read()
        return type("Result", (), {"returncode": 0, "stdout": None, "stderr": None})()

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert cloud.run_cloud(["billing", "topup", "--credits", "5", "--yes"]) == 5
    output = capsys.readouterr().out
    assert "without a confirmed receipt" in output
    assert "outcome is unknown" in output
    assert "billing credits" in output


def test_render_json_mode_when_not_a_tty() -> None:
    assert render.json_mode(flag=True) is True
    # Under pytest, stdout is captured and is not a terminal.
    assert render.json_mode(flag=False) is True


def test_render_list_extraction() -> None:
    rows = render._list_of_dicts({"scans": [{"id": "a"}, {"id": "b"}]})
    assert rows == [{"id": "a"}, {"id": "b"}]
    assert render._list_of_dicts({"scans": [], "total": 1}) == []
    assert render._list_of_dicts([{"id": "a"}, "x"]) is None


def test_spec_paths_are_well_formed() -> None:
    for group, commands in SPEC.items():
        for verb, cmd in commands.items():
            assert cmd.path.startswith("/"), f"{group} {verb}"
            assert cmd.method in ("GET", "POST", "PUT", "PATCH", "DELETE"), f"{group} {verb}"
            assert cmd.help, f"{group} {verb} has no help text"
            for param in cmd.query + cmd.body:
                assert param.kind in (
                    "str",
                    "int",
                    "float",
                    "bool",
                    "list",
                    "json",
                    "json-list",
                ), f"{group} {verb} {param.name}"


@pytest.mark.parametrize(
    ("group", "verb"),
    [
        ("scans", "list"),
        ("vulns", "list"),
        ("domains", "list"),
        ("repos", "list"),
        ("pr-reviews", "list"),
        ("pr-reviews", "findings"),
        ("webhooks", "deliveries"),
        ("audit", "list"),
    ],
)
def test_paginated_commands_expose_integer_page_and_limit(group: str, verb: str) -> None:
    params = {param.name: param for param in SPEC[group][verb].query}
    assert params["page"].kind == "int"
    assert params["limit"].kind == "int"


def test_list_query_types_match_the_api_contract() -> None:
    scans = {param.name: param for param in SPEC["scans"]["list"].query}
    assert scans["include_retests"].kind == "bool"
    assert {"sort_by", "sort_order"} <= scans.keys()

    vulnerabilities = {param.name: param for param in SPEC["vulns"]["list"].query}
    assert "sort_order" in vulnerabilities

    for group in ("domains", "repos"):
        params = {param.name: param for param in SPEC[group]["list"].query}
        assert params["limit"].kind == "int"
        assert "sort_order" in params

    reviews = {param.name: param for param in SPEC["pr-reviews"]["list"].query}
    findings = {param.name: param for param in SPEC["pr-reviews"]["findings"].query}
    audit = {param.name: param for param in SPEC["audit"]["list"].query}
    assert reviews["include_counts"].kind == "bool"
    assert findings["include_stats"].kind == "bool"
    assert audit["all"].kind == "bool"

    components = {param.name: param for param in SPEC["repos"]["supply-chain components"].query}
    knowledge = {param.name: param for param in SPEC["knowledge"]["list"].query}
    assert components["limit"].kind == "int"
    assert components["offset"].kind == "int"
    assert knowledge["limit"].kind == "int"


def test_scan_creating_replay_commands_support_bounded_waits() -> None:
    assert SPEC["scans"]["rerun"].wait_path == "/scans/{id}"
    assert SPEC["vulns"]["retest"].wait_path == "/scans/{id}"


def test_scan_start_parameter_contract_and_help() -> None:
    params = {param.name: param for param in SPEC["scans"]["start"].body}
    assert params["headers"].kind == "json"
    assert "array" in params["headers"].help.lower()
    assert params["concerns"].kind == "str"
    assert all(tier in params["scan_tier"].help for tier in ("lite", "standard", "ultra"))
    assert "pro" not in params["scan_tier"].help
    assert "max" not in params["scan_tier"].help
    assert "self-hosted" in params["model_config_id"].help.lower()
    assert "self-hosted" in params["max_budget_usd"].help.lower()


def test_report_branding_flags_preserve_the_api_query_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["query"] = kwargs.get("query")
        return FakeResponse(content=b"report")

    monkeypatch.setattr(http, "request", fake_request)
    output = tmp_path / "report.pdf"
    assert (
        cloud.run_cloud(
            [
                "scans",
                "report",
                "scan-1",
                "--provider-name",
                "Strix Partner",
                "--member-name-0",
                "Alex",
                "--member-email-0",
                "alex@example.test",
                "--member-name-1",
                "Sam",
                "--member-email-1",
                "sam@example.test",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_bytes() == b"report"
    assert seen["query"] == {
        "providerName": "Strix Partner",
        "memberName0": "Alex",
        "memberEmail0": "alex@example.test",
        "memberName1": "Sam",
        "memberEmail1": "sam@example.test",
    }


def test_scan_start_collects_header_array_and_string_concerns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"id": "scan-1"})

    monkeypatch.setattr(http, "request", fake_request)
    assert (
        cloud.run_cloud(
            [
                "scans",
                "start",
                "--headers",
                '[{"name":"X-Test","value":"one"}]',
                "--concerns",
                "authorization boundaries",
                "--scan-tier",
                "standard",
                "--json",
            ]
        )
        == 0
    )
    assert seen["body"] == {
        "headers": [{"name": "X-Test", "value": "one"}],
        "concerns": "authorization boundaries",
        "scan_tier": "standard",
    }


def test_control_only_scan_and_chat_messages_do_not_require_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def fake_request(_method: str, path: str, **kwargs: Any) -> FakeResponse:
        calls.append((path, kwargs.get("body")))
        return FakeResponse(payload={"success": True})

    monkeypatch.setattr(http, "request", fake_request)
    assert cloud.run_cloud(["scans", "message", "scan-1", "--cancel-current", "--json"]) == 0
    assert (
        cloud.run_cloud(
            ["chat", "send", "chat-1", "--stop-agent", "--agent-id", "agent-1", "--json"]
        )
        == 0
    )
    assert calls == [
        ("/scans/scan-1/message", {"cancel_current": True}),
        ("/chat/chat-1/message", {"stop_agent": True, "agent_id": "agent-1"}),
    ]


def test_chat_repositories_use_the_api_object_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"id": "chat-1"})

    monkeypatch.setattr(http, "request", fake_request)
    assert (
        cloud.run_cloud(
            [
                "chat",
                "start",
                "--message",
                "Review this repository",
                "--repos",
                '[{"repoId":"repo-1","branch":"main"}]',
                "--json",
            ]
        )
        == 0
    )
    assert seen["body"] == {
        "message": "Review this repository",
        "repos": [{"repoId": "repo-1", "branch": "main"}],
    }


def test_schedule_budget_accepts_fractional_usd(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["body"] = kwargs.get("body")
        return FakeResponse(payload={"id": "schedule-1"})

    monkeypatch.setattr(http, "request", fake_request)
    assert (
        cloud.run_cloud(["schedules", "update", "schedule-1", "--max-budget-usd", "1.5", "--json"])
        == 0
    )
    assert seen["body"] == {"max_budget_usd": 1.5}


def test_integration_disconnect_sends_installation_id_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_request(method: str, path: str, **kwargs: Any) -> FakeResponse:
        seen.update(method=method, path=path, query=kwargs.get("query"))
        return FakeResponse(payload={"success": True})

    monkeypatch.setattr(http, "request", fake_request)
    assert (
        cloud.run_cloud(
            ["integrations", "disconnect", "github", "--installation-id", "42", "--json"]
        )
        == 0
    )
    assert seen == {
        "method": "DELETE",
        "path": "/integrations/github",
        "query": {"installation_id": 42},
    }


def test_connector_command_flag_is_boolean_and_warns_that_it_is_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    param = next(
        param for param in SPEC["connectors"]["get"].query if param.name == "include_command"
    )
    assert param.kind == "bool"
    assert "sensitive" in param.help.lower()

    seen: dict[str, Any] = {}

    def fake_request(_method: str, _path: str, **kwargs: Any) -> FakeResponse:
        seen["query"] = kwargs.get("query")
        return FakeResponse(payload={"id": "connector-1"})

    monkeypatch.setattr(http, "request", fake_request)
    assert cloud.run_cloud(["connectors", "get", "connector-1", "--include-command", "--json"]) == 0
    assert seen["query"] == {"include_command": True}


def test_corrected_help_distinguishes_inboxes_reports_and_self_hosted_commands() -> None:
    inbox = SPEC["domains"]["test-users provision-inbox"]
    assert "does not create a test user" in inbox.help

    report = {param.name: param.help for param in SPEC["scans"]["report"].query}
    assert "Report content" in report["format"]
    assert "file type" in report["type"]

    for command in (*SPEC["costs"].values(), *SPEC["llm-settings"].values()):
        assert "self-hosted only" in command.help.lower()
    assert "self-hosted only" in GROUP_HELP["costs"].lower()
    assert "self-hosted only" in GROUP_HELP["llm-settings"].lower()


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


@pytest.mark.parametrize(
    "argv,payload",
    [
        (
            ["billing", "subscribe", "--plan", "strix_cloud"],
            {"checkout_url": "file:///tmp/not-a-checkout"},
        ),
        (
            ["integrations", "install", "github"],
            {"url": "javascript:alert(1)"},
        ),
    ],
)
def test_handoff_links_reject_non_http_schemes(
    argv: list[str],
    payload: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    monkeypatch.setattr(runner.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(status_code=200, payload=payload),
    )
    monkeypatch.setattr(
        webbrowser,
        "open",
        lambda _url: pytest.fail("an untrusted URL must never be opened"),
    )

    assert cloud.run_cloud(argv) == http.EXIT_ERROR
    output = capsys.readouterr().out
    assert "invalid continuation URL" in output
    assert next(iter(payload.values())) not in output


def test_handoff_missing_expected_url_is_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(status_code=200, payload={"status": "created"}),
    )
    assert cloud.run_cloud(["integrations", "install", "github", "--json"]) == 1
    assert "expected url URL" in json.loads(capsys.readouterr().out)["error"]


def test_workspaces_use_switches_stored_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    monkeypatch.delenv("STRIX_API_TOKEN", raising=False)
    auth_path = tmp_path / "platform-auth.json"
    monkeypatch.setattr(platform_cli, "AUTH_PATH", auth_path)
    monkeypatch.setattr(workspaces, "AUTH_PATH", auth_path)
    platform_cli.save_record(
        {
            "api_token": "old",
            "email": "a@b.test",
            "scopes": ["scans:read", "organizations:read", "tokens:write"],
            "requested_scopes": [
                "scans:read",
                "scans:write",
                "organizations:read",
                "tokens:write",
            ],
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
            status_code=200,
            payload={
                "api_token": "old",
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
        "scopes": ["scans:read", "scans:write", "organizations:read", "tokens:write"]
    }
    record = platform_cli.read_record()
    assert record is not None
    assert record["api_token"] == "old"
    assert record["organization_name"] == "Team One"
    assert record["email"] == "a@b.test"
    assert "org_1" in capsys.readouterr().out


def test_workspace_use_explicit_token_starts_with_fresh_account_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth_path = tmp_path / "platform-auth.json"
    monkeypatch.setattr(platform_cli, "AUTH_PATH", auth_path)
    monkeypatch.setattr(workspaces, "AUTH_PATH", auth_path)
    platform_cli.save_record(
        {
            "api_token": "account-a-token",
            "email": "account-a@example.test",
            "organization_id": "org_a",
            "organization_name": "Account A",
            "scopes": ["scans:read"],
            "requested_scopes": ["scans:read", "tokens:write"],
        }
    )
    switch_body: dict[str, Any] | None = None

    def fake_request(method: str, path: str, **kwargs: Any) -> FakeResponse:
        nonlocal switch_body
        assert kwargs.get("token") == "account-b-token"
        if method == "GET":
            return FakeResponse(payload={"workspaces": [{"id": "org_b", "name": "Account B"}]})
        assert path == "/workspaces/org_b/token"
        switch_body = kwargs.get("body")
        return FakeResponse(
            payload={
                "api_token": "account-b-token",
                "organization_id": "org_b",
                "organization_name": "Account B",
                "scopes": ["scans:read", "organizations:read"],
            }
        )

    monkeypatch.setattr(http, "request", fake_request)
    assert (
        cloud.run_cloud(["workspaces", "use", "Account B", "--token", "account-b-token", "--json"])
        == 0
    )
    assert switch_body is None
    record = platform_cli.read_record()
    assert record is not None
    assert record["api_token"] == "account-b-token"
    assert record["organization_id"] == "org_b"
    assert record["requested_scopes"] == ["scans:read", "organizations:read"]
    assert "email" not in record
    assert "account-a@example.test" not in auth_path.read_text(encoding="utf-8")


def test_workspace_use_environment_token_starts_with_fresh_account_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth_path = tmp_path / "platform-auth.json"
    monkeypatch.setattr(platform_cli, "AUTH_PATH", auth_path)
    monkeypatch.setattr(workspaces, "AUTH_PATH", auth_path)
    monkeypatch.setenv("STRIX_API_TOKEN", "account-b-token")
    platform_cli.save_record(
        {
            "api_token": "account-a-token",
            "email": "account-a@example.test",
            "organization_id": "org_a",
            "organization_name": "Account A",
            "scopes": ["scans:read"],
            "requested_scopes": ["scans:read", "tokens:write"],
        }
    )
    switch_body: dict[str, Any] | None = None

    def fake_request(method: str, path: str, **kwargs: Any) -> FakeResponse:
        nonlocal switch_body
        assert kwargs.get("token") is None
        if method == "GET":
            return FakeResponse(payload={"workspaces": [{"id": "org_b", "name": "Account B"}]})
        assert path == "/workspaces/org_b/token"
        switch_body = kwargs.get("body")
        return FakeResponse(
            payload={
                "api_token": "account-b-token",
                "organization_id": "org_b",
                "organization_name": "Account B",
                "email": "account-b@example.test",
                "scopes": ["scans:read", "organizations:read"],
            }
        )

    monkeypatch.setattr(http, "request", fake_request)
    assert cloud.run_cloud(["workspaces", "use", "Account B", "--json"]) == 0
    assert switch_body is None
    record = platform_cli.read_record()
    assert record is not None
    assert record["api_token"] == "account-b-token"
    assert record["organization_id"] == "org_b"
    assert record["email"] == "account-b@example.test"
    assert record["requested_scopes"] == ["scans:read", "organizations:read"]
    assert "account-a@example.test" not in auth_path.read_text(encoding="utf-8")


def test_workspaces_use_reports_unknown_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(
            status_code=200, payload={"workspaces": [{"id": "org_1", "name": "Team One"}]}
        ),
    )
    assert cloud.run_cloud(["workspaces", "use", "missing", "--json"]) == 1


def test_workspaces_use_reports_auth_storage_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    def fake_request(method: str, path: str, **_kwargs: Any) -> FakeResponse:
        if method == "GET":
            return FakeResponse(payload={"workspaces": [{"id": "org_1", "name": "Team One"}]})
        assert path == "/workspaces/org_1/token"
        return FakeResponse(
            payload={
                "api_token": "test-token",
                "organization_id": "org_1",
                "organization_name": "Team One",
                "scopes": ["scans:read"],
            }
        )

    monkeypatch.setattr(http, "request", fake_request)
    monkeypatch.setattr(
        workspaces, "save_record", lambda _record: (_ for _ in ()).throw(OSError("disk full"))
    )

    assert cloud.run_cloud(["workspaces", "use", "1", "--json"]) == http.EXIT_ERROR
    payload = json.loads(capsys.readouterr().out)
    assert "could not be stored" in payload["error"]
    assert payload["workspace_switched"] is True
    assert payload["local_record_updated"] is False
    assert payload["retry_safe"] is True


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("connection reset"),
        FakeResponse(status_code=503, text="temporarily unavailable"),
        FakeResponse(status_code=200, text="not JSON"),
        FakeResponse(status_code=200, payload={"organization_id": "org_1"}),
    ],
)
def test_workspace_use_reports_retry_safe_unknown_outcomes(
    failure: Exception | FakeResponse,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    def fake_request(method: str, path: str, **_kwargs: Any) -> FakeResponse:
        if method == "GET":
            return FakeResponse(payload={"workspaces": [{"id": "org_1", "name": "Team One"}]})
        assert path == "/workspaces/org_1/token"
        if isinstance(failure, Exception):
            raise http.CloudError(str(failure)) from failure
        return failure

    monkeypatch.setattr(http, "request", fake_request)

    assert cloud.run_cloud(["workspaces", "use", "1", "--json"]) == http.EXIT_ERROR
    payload = json.loads(capsys.readouterr().out)
    assert payload["switch_outcome_unknown"] is True
    assert payload["retry_safe"] is True
    assert "safely rerun" in payload["error"]


def test_workspace_use_preserves_definitive_conflict(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    def fake_request(method: str, path: str, **_kwargs: Any) -> FakeResponse:
        if method == "GET":
            return FakeResponse(payload={"workspaces": [{"id": "org_1", "name": "Team One"}]})
        return FakeResponse(
            status_code=409,
            payload={"error": {"code": "token_conflict", "message": "token changed"}},
        )

    monkeypatch.setattr(http, "request", fake_request)

    assert cloud.run_cloud(["workspaces", "use", "1", "--json"]) == http.EXIT_ERROR
    payload = json.loads(capsys.readouterr().out)
    assert "token changed" in payload["error"]
    assert "switch_outcome_unknown" not in payload


def test_group_help_lists_all_verbs_instead_of_default_verb_help(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(render.sys.stdout, "isatty", lambda: True)
    assert cloud.run_cloud(["workspaces", "-h"]) == 0
    output = capsys.readouterr().out
    assert "workspaces verbs" in output
    assert "list" in output
    assert "create" in output
    assert "use" in output


def test_workspace_alias_routes_to_workspaces(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def fake_request(method: str, path: str, **_kwargs: Any) -> FakeResponse:
        seen.update(method=method, path=path)
        return FakeResponse(payload={"workspaces": []})

    monkeypatch.setattr(http, "request", fake_request)
    assert cloud.run_cloud(["workspace", "list", "--json"]) == 0
    assert seen == {"method": "GET", "path": "/workspaces"}


def test_workspace_human_list_is_numbered_and_hides_ids(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(render.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(
            payload={
                "workspaces": [
                    {"id": "org_secret", "name": "Team One", "role": "admin", "current": True}
                ]
            }
        ),
    )

    assert cloud.run_cloud(["workspaces", "list"]) == 0
    output = capsys.readouterr().out
    assert "1." in output
    assert "Team One" in output
    assert "yes" in output
    assert "org_secret" not in output
    assert "workspaces use NUMBER" in output


def test_integrations_human_list_exposes_installation_id_and_json_stays_full(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    payload = {
        "integrations": [
            {
                "id": "integration-uuid",
                "organization_id": "org-secret",
                "connected_by": "user-secret",
                "provider": "github",
                "installation_id": 154419799,
                "account_login": "usestrix",
                "repository_selection": "selected",
                "connected_at": "2026-08-27T12:00:00Z",
            }
        ],
        "merge_accounts": [
            {
                "id": "merge-uuid",
                "provider": "jira",
                "status": "linked",
                "default_collection_name": "Security",
            }
        ],
        "bitbucket_oauth_enabled": True,
    }
    monkeypatch.setattr(render.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(http, "request", lambda *_a, **_k: FakeResponse(payload=payload))

    assert cloud.run_cloud(["integrations", "list"]) == 0
    output = capsys.readouterr().out
    for value in ("1.", "2.", "github", "usestrix", "154419799", "jira", "Security"):
        assert value in output
    for value in ("integration-uuid", "merge-uuid", "org-secret", "user-secret"):
        assert value not in output
    assert "--installation-id INSTALLATION_ID" in output

    assert cloud.run_cloud(["integrations", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == payload


def test_pr_review_human_list_prioritizes_actionable_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(render.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(
            payload={
                "items": [
                    {
                        "id": "review-id",
                        "organization_id": "org-id",
                        "user_id": "user-id",
                        "installation_id": 42,
                        "repository_full_name": "usestrix/strix",
                        "pr_number": 1177,
                        "pr_title": "Improve cloud CLI",
                        "head_branch": "feature",
                        "base_branch": "main",
                        "verdict": "pass",
                        "status": "posted",
                        "findings_count": 0,
                        "open_findings_count": 0,
                    }
                ],
                "meta": {"total": 1},
            }
        ),
    )

    assert cloud.run_cloud(["pr-reviews", "list"]) == 0
    output = capsys.readouterr().out
    for value in (
        "usestrix/strix",
        "1177",
        "Improve cloud CLI",
        "feature",
        "main",
        "posted",
        "pass",
        "0 open / 0 total",
        "review-id",
    ):
        assert value in output
    for value in ("org-id", "user-id", "installation_id"):
        assert value not in output


def test_human_get_prioritizes_details_and_hides_internal_identity_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(render.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        http,
        "request",
        lambda *_a, **_k: FakeResponse(
            payload={
                "id": "review-id",
                "organization_id": "org-id",
                "user_id": "user-id",
                "repository_full_name": "usestrix/strix",
                "pr_number": 1177,
                "pr_title": "Improve cloud CLI",
                "verdict": "pass",
                "findings": [{"severity": "high", "title": "Example"}],
            }
        ),
    )

    assert cloud.run_cloud(["pr-reviews", "get", "review-id"]) == 0
    output = capsys.readouterr().out
    for value in ("usestrix/strix", "1177", "Improve cloud CLI", "pass", "Example"):
        assert value in output
    assert "org-id" not in output
    assert "user-id" not in output
    assert "lossless machine-readable" in output


def test_workspace_use_accepts_list_number(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    auth_path = tmp_path / "platform-auth.json"
    monkeypatch.setattr(platform_cli, "AUTH_PATH", auth_path)
    monkeypatch.setattr(workspaces, "AUTH_PATH", auth_path)
    platform_cli.save_record({"api_token": "old", "scopes": ["organizations:read", "tokens:write"]})
    called_paths: list[str] = []

    def fake_request(_method: str, path: str, **_kwargs: Any) -> FakeResponse:
        called_paths.append(path)
        if path == "/workspaces":
            return FakeResponse(
                payload={
                    "workspaces": [
                        {"id": "org_1", "name": "One"},
                        {"id": "org_2", "name": "Two"},
                    ]
                }
            )
        return FakeResponse(
            status_code=200,
            payload={
                "api_token": "old",
                "organization_id": "org_2",
                "organization_name": "Two",
                "scopes": ["organizations:read", "tokens:write"],
            },
        )

    monkeypatch.setattr(http, "request", fake_request)
    assert cloud.run_cloud(["workspaces", "use", "2", "--json"]) == 0
    assert called_paths == ["/workspaces", "/workspaces/org_2/token"]
    record = platform_cli.read_record()
    assert record is not None
    assert record["api_token"] == "old"


def test_logout_help_does_not_remove_stored_auth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    auth_path = tmp_path / "platform-auth.json"
    monkeypatch.setattr(platform_cli, "AUTH_PATH", auth_path)
    platform_cli.save_record({"api_token": "keep-me"})

    assert cloud.run_cloud(["logout", "--help"]) == 0
    assert platform_cli.read_record() == {"api_token": "keep-me"}
    assert "usage: strix cloud logout" in capsys.readouterr().out


def test_logout_rejects_unknown_arguments_without_removing_stored_auth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    auth_path = tmp_path / "platform-auth.json"
    monkeypatch.setattr(platform_cli, "AUTH_PATH", auth_path)
    platform_cli.save_record({"api_token": "keep-me"})

    assert cloud.run_cloud(["logout", "--bogus"]) == 2
    assert platform_cli.read_record() == {"api_token": "keep-me"}
