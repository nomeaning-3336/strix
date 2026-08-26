"""Tests for the `strix cloud` CLI: routing, request building, and output."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any

import pytest
import requests

from strix.interface import cloud, platform_cli
from strix.interface.cloud import http, render, runner
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


def test_topup_no_pay_prints_challenge(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    challenge = {"payment_requirements": [{"amount": 500}]}
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=402, payload=challenge)
    )
    code = cloud.run_cloud(["billing", "topup", "--credits", "5", "--no-pay", "--json"])
    assert code == http.EXIT_PAYMENT
    assert json.loads(capsys.readouterr().out) == challenge


def test_topup_success_without_payment(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    receipt = {"credits_granted": 5, "duplicate": False, "balance": 5}
    monkeypatch.setattr(
        http, "request", lambda *_a, **_k: FakeResponse(status_code=200, payload=receipt)
    )
    code = cloud.run_cloud(["billing", "topup", "--credits", "5", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out) == receipt


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
