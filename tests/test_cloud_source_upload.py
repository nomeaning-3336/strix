"""Local-source packaging and scan upload tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from typing import TYPE_CHECKING, Any

import pytest

from strix.interface import cloud
from strix.interface.cloud import http, source_upload


if TYPE_CHECKING:
    from pathlib import Path


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)
        self.content = b""
        self.ok = 200 <= status_code < 400
        self.headers = {"content-type": "application/json"}

    def json(self) -> Any:
        return self._payload


@pytest.fixture(autouse=True)
def _token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIX_API_TOKEN", "test-token")


def _git_source(tmp_path: Path) -> Path:
    git = shutil.which("git")
    assert git is not None
    subprocess.run([git, "init", "-q", str(tmp_path)], check=True)  # noqa: S603
    (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.log\n", encoding="utf-8")
    (tmp_path / "ignored.log").write_text("ignored\n", encoding="utf-8")
    subprocess.run(  # noqa: S603
        [git, "-C", str(tmp_path), "add", "app.py", ".gitignore"], check=True
    )
    return tmp_path


def test_source_defaults_are_private_and_git_aware(tmp_path: Path) -> None:
    source = _git_source(tmp_path)
    (source / ".hidden.py").write_text("hidden\n", encoding="utf-8")
    (source / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (source / "private.pem").write_text("secret\n", encoding="utf-8")
    (source / "fixture.zip").write_bytes(b"not really a zip")
    (source / "node_modules").mkdir()
    (source / "node_modules" / "dep.js").write_text("dep\n", encoding="utf-8")
    (source / "linked.py").symlink_to(source / "app.py")

    bundle = source_upload.prepare_source(
        str(source),
        include_hidden=False,
        include_sensitive=False,
        include_archives=False,
        exclude=[],
    )
    try:
        names = [item.archive_name for item in bundle.manifest.files]
        assert names == ["README.md", "app.py"]
        assert bundle.manifest.total_bytes > 0
        assert bundle.archive_bytes <= source_upload.MAX_ARCHIVE_BYTES
        assert bundle.manifest.excluded["hidden"] == 3
        assert bundle.manifest.excluded["sensitive_filename"] == 1
        assert bundle.manifest.excluded["nested_archive"] == 1
        assert bundle.manifest.excluded["dependency_or_build_output"] == 1
        assert bundle.manifest.excluded["symlink_or_non_file"] == 1
        with zipfile.ZipFile(bundle.archive_path) as archive:
            assert archive.namelist() == names
    finally:
        source_upload.remove_bundle(bundle)


def test_hidden_and_sensitive_files_need_separate_opt_ins(tmp_path: Path) -> None:
    source = _git_source(tmp_path)
    (source / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (source / ".github").mkdir()
    (source / ".github" / "workflow.yml").write_text("name: test\n", encoding="utf-8")

    hidden = source_upload.select_source(source, include_hidden=True)
    hidden_names = {item.archive_name for item in hidden.files}
    assert ".github/workflow.yml" in hidden_names
    assert ".env" not in hidden_names

    sensitive = source_upload.select_source(source, include_hidden=True, include_sensitive=True)
    assert ".env" in {item.archive_name for item in sensitive.files}
    assert all(
        not name.startswith(".git/") for name in (item.archive_name for item in sensitive.files)
    )


def test_strixignore_and_cli_excludes_are_applied(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("keep\n", encoding="utf-8")
    (tmp_path / "generated.py").write_text("generated\n", encoding="utf-8")
    (tmp_path / "test_app.py").write_text("test\n", encoding="utf-8")
    (tmp_path / ".strixignore").write_text("generated.py\n", encoding="utf-8")

    manifest = source_upload.select_source(tmp_path, exclude=["test_*.py"])
    assert [item.archive_name for item in manifest.files] == ["keep.py"]
    assert manifest.excluded["user_pattern"] == 2


def test_source_limits_expanded_bytes_before_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(source_upload, "MAX_TOTAL_BYTES", 5)
    (tmp_path / "large.py").write_bytes(b"a" * 6)
    with pytest.raises(http.CloudError, match="expanded-size limit"):
        source_upload.select_source(tmp_path)


def test_source_dry_run_never_calls_the_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    (tmp_path / "app.py").write_text("print('safe')\n", encoding="utf-8")

    def fail_request(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("dry-run must not make an API request")

    monkeypatch.setattr(http, "request", fail_request)
    assert (
        cloud.run_cloud(
            ["scans", "start", "--source", str(tmp_path), "--dry-run", "--show-files", "--json"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"]["files"] == ["app.py"]
    assert payload["source"]["archive_sha256"]


def test_noninteractive_source_upload_requires_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    (tmp_path / "app.py").write_text("print('safe')\n", encoding="utf-8")
    monkeypatch.setattr(
        http,
        "request",
        lambda *_args, **_kwargs: pytest.fail("approval must happen before any API request"),
    )
    assert cloud.run_cloud(["scans", "start", "--source", str(tmp_path), "--json"]) == 1
    assert "requires explicit approval" in capsys.readouterr().out


def test_source_upload_is_completed_and_attached_to_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    (tmp_path / "app.py").write_text("print('safe')\n", encoding="utf-8")
    calls: list[tuple[str, str, dict[str, Any]]] = []
    uploaded_path: Path | None = None

    def fake_request(method: str, path: str, **kwargs: Any) -> FakeResponse:
        calls.append((method, path, kwargs))
        if path == "/uploads/request":
            return FakeResponse(
                {
                    "upload_id": "upload-1",
                    "signed_url": "https://storage.test/object",
                    "token": "signed",
                }
            )
        if path == "/uploads/complete":
            return FakeResponse({"id": "upload-1"})
        if path == "/scans":
            return FakeResponse({"scan_id": "scan-1", "status": "pending"})
        raise AssertionError(path)

    def fake_upload(_url: str, _token: str, path: Path) -> None:
        nonlocal uploaded_path
        uploaded_path = path
        assert path.exists()

    monkeypatch.setattr(http, "request", fake_request)
    monkeypatch.setattr(http, "upload_file", fake_upload)

    assert (
        cloud.run_cloud(
            ["scans", "start", "--source", str(tmp_path), "--yes", "--show-files", "--json"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["upload_id"] == "upload-1"
    assert payload["scan"]["scan_id"] == "scan-1"
    assert payload["source"]["files"] == ["app.py"]
    scan_call = next(call for call in calls if call[1] == "/scans")
    assert scan_call[2]["body"] == {
        "engagement_type": "code_review",
        "upload_ids": ["upload-1"],
    }
    assert uploaded_path is not None and not uploaded_path.exists()


def test_source_upload_with_domain_is_a_live_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    (tmp_path / "app.py").write_text("print('safe')\n", encoding="utf-8")
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake_request(method: str, path: str, **kwargs: Any) -> FakeResponse:
        calls.append((method, path, kwargs))
        if path == "/uploads/request":
            return FakeResponse(
                {
                    "upload_id": "upload-1",
                    "signed_url": "https://storage.test/object",
                    "token": "signed",
                }
            )
        if path == "/uploads/complete":
            return FakeResponse({"id": "upload-1"})
        if path == "/scans":
            return FakeResponse({"scan_id": "scan-1", "status": "pending"})
        raise AssertionError(path)

    monkeypatch.setattr(http, "request", fake_request)
    monkeypatch.setattr(http, "upload_file", lambda *_args, **_kwargs: None)

    assert (
        cloud.run_cloud(
            [
                "scans",
                "start",
                "--source",
                str(tmp_path),
                "--domain-ids",
                "domain-1",
                "--yes",
                "--json",
            ]
        )
        == 0
    )
    scan_call = next(call for call in calls if call[1] == "/scans")
    assert scan_call[2]["body"] == {
        "engagement_type": "live_test",
        "domain_ids": ["domain-1"],
        "upload_ids": ["upload-1"],
    }
    assert json.loads(capsys.readouterr().out)["scan"]["scan_id"] == "scan-1"


def test_failed_scan_deletes_completed_source_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.py").write_text("print('safe')\n", encoding="utf-8")
    paths: list[tuple[str, str]] = []

    def fake_request(method: str, path: str, **_kwargs: Any) -> FakeResponse:
        paths.append((method, path))
        if path == "/uploads/request":
            return FakeResponse(
                {
                    "upload_id": "upload-1",
                    "signed_url": "https://storage.test/object",
                    "token": "signed",
                }
            )
        if path == "/uploads/complete":
            return FakeResponse({"id": "upload-1"})
        if path == "/scans":
            return FakeResponse({"detail": "not enough credits"}, status_code=402)
        if path == "/uploads/upload-1":
            return FakeResponse({"ok": True})
        raise AssertionError(path)

    monkeypatch.setattr(http, "request", fake_request)
    monkeypatch.setattr(http, "upload_file", lambda *_args, **_kwargs: None)
    assert cloud.run_cloud(["scans", "start", "--source", str(tmp_path), "--yes", "--json"]) == 5
    assert ("DELETE", "/uploads/upload-1") in paths
