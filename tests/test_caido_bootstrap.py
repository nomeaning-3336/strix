"""A bootstrap that dies mid-setup must not leave its transport behind.

The bootstrap now runs concurrently with the scan start, so teardown can
cancel it at any await — including inside ``Client.connect()``, where the
client exists but no caller will ever see it to close it.
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from strix.runtime.caido_bootstrap import bootstrap_caido


class _FakeExecResult:
    stderr = b""
    exit_code = 0

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout

    def ok(self) -> bool:
        return True


class _FakeSession:
    async def exec(self, *_args: Any, **_kwargs: Any) -> _FakeExecResult:
        return _FakeExecResult('{"data":{"loginAsGuest":{"token":{"accessToken":"t"}}}}')


@dataclass
class _FakeProject:
    id: str
    name: str = "sandbox"


class _FakeProjectSDK:
    def __init__(
        self,
        *,
        create_errors: list[BaseException] | None = None,
        select_errors: list[BaseException] | None = None,
        projects: list[_FakeProject] | None = None,
    ) -> None:
        self.create_errors = list(create_errors or [])
        self.select_errors = list(select_errors or [])
        self.projects = projects or []
        self.create_calls = 0
        self.selected_ids: list[str] = []
        self.list_calls = 0

    async def create(self, _options: Any) -> _FakeProject:
        self.create_calls += 1
        if self.create_errors:
            raise self.create_errors.pop(0)
        return _FakeProject("created")

    async def select(self, project_id: str) -> _FakeProject:
        self.selected_ids.append(project_id)
        if self.select_errors:
            raise self.select_errors.pop(0)
        return _FakeProject(project_id)

    async def list(self) -> list[_FakeProject]:
        self.list_calls += 1
        return self.projects


class _FakeClient:
    def __init__(
        self,
        connect_error: BaseException | None = None,
        *,
        project: _FakeProjectSDK | None = None,
    ) -> None:
        self.connect_error = connect_error
        self.project = project or _FakeProjectSDK()
        self.closed = False

    async def connect(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error

    async def aclose(self) -> None:
        self.closed = True


class _FakeClientFactory:
    def __init__(self, clients: list[_FakeClient]) -> None:
        self.clients = iter(clients)
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> _FakeClient:
        self.calls.append((args, kwargs))
        return next(self.clients)


def _install_sdk(
    monkeypatch: pytest.MonkeyPatch,
    clients: list[_FakeClient],
) -> _FakeClientFactory:
    factory = _FakeClientFactory(clients)
    sdk = types.ModuleType("caido_sdk_client")
    sdk.Client = factory  # type: ignore[attr-defined]
    sdk.TokenAuthOptions = lambda token: token  # type: ignore[attr-defined]
    sdk_types = types.ModuleType("caido_sdk_client.types")
    sdk_types.CreateProjectOptions = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "caido_sdk_client", sdk)
    monkeypatch.setitem(sys.modules, "caido_sdk_client.types", sdk_types)
    return factory


async def _bootstrap_expecting(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> _FakeClient:
    """Run a bootstrap whose ``connect()`` fails with ``error``."""
    setup_client = _FakeClient()
    client = _FakeClient(error)
    _install_sdk(monkeypatch, [setup_client, client])

    with pytest.raises(type(error)):
        await bootstrap_caido(
            _FakeSession(),  # type: ignore[arg-type]
            host_url="http://host",
            container_url="http://container",
        )
    assert setup_client.closed
    return client


async def test_cancellation_during_connect_closes_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = await _bootstrap_expecting(monkeypatch, asyncio.CancelledError())
    assert client.closed


async def test_failed_connect_closes_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = await _bootstrap_expecting(monkeypatch, RuntimeError("no listener"))
    assert client.closed


async def test_select_retries_without_creating_another_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_project = _FakeProjectSDK(select_errors=[RuntimeError("not ready")])
    setup_client = _FakeClient(project=setup_project)
    returned_client = _FakeClient()
    factory = _install_sdk(monkeypatch, [setup_client, returned_client])
    sleep_calls: list[float] = []

    async def _sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(
        "strix.runtime.caido_bootstrap.asyncio.sleep",
        _sleep,
    )

    result = await bootstrap_caido(
        _FakeSession(),  # type: ignore[arg-type]
        host_url="http://host",
        container_url="http://container",
    )

    assert result is returned_client
    assert setup_project.create_calls == 1
    assert setup_project.selected_ids == ["created", "created"]
    assert setup_client.closed
    assert factory.calls[0][1]["timeout_ms"] == 45_000
    assert factory.calls[1][1].get("timeout_ms") is None
    assert sleep_calls == [2.0]


async def test_create_failure_reuses_the_most_recent_sandbox_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_project = _FakeProjectSDK(
        create_errors=[RuntimeError("create timed out")],
        projects=[
            _FakeProject("project-1"),
            _FakeProject("project-2"),
            _FakeProject("other", name="other"),
        ],
    )
    setup_client = _FakeClient(project=setup_project)
    returned_client = _FakeClient()
    _install_sdk(monkeypatch, [setup_client, returned_client])

    async def _sleep(_delay: float) -> None:
        pass

    monkeypatch.setattr("strix.runtime.caido_bootstrap.asyncio.sleep", _sleep)

    result = await bootstrap_caido(
        _FakeSession(),  # type: ignore[arg-type]
        host_url="http://host",
        container_url="http://container",
    )

    assert result is returned_client
    assert setup_project.create_calls == 1
    assert setup_project.list_calls == 1
    assert setup_project.selected_ids == ["project-2"]
    assert setup_client.closed


async def test_project_setup_failure_chains_last_error_and_closes_setup_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[BaseException] = [
        RuntimeError("first"),
        RuntimeError("second"),
        RuntimeError("last"),
    ]
    setup_project = _FakeProjectSDK(select_errors=errors)
    setup_client = _FakeClient(project=setup_project)
    _install_sdk(monkeypatch, [setup_client])

    async def _sleep(_delay: float) -> None:
        pass

    monkeypatch.setattr("strix.runtime.caido_bootstrap.asyncio.sleep", _sleep)

    with pytest.raises(
        RuntimeError,
        match="Caido project setup failed after 3 attempts",
    ) as exc_info:
        await bootstrap_caido(
            _FakeSession(),  # type: ignore[arg-type]
            host_url="http://host",
            container_url="http://container",
        )

    assert exc_info.value.__cause__ is errors[-1]
    assert setup_client.closed
    assert setup_project.create_calls == 1
    assert setup_project.selected_ids == ["created", "created", "created"]


async def test_cancelled_project_setup_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_project = _FakeProjectSDK(select_errors=[asyncio.CancelledError()])
    setup_client = _FakeClient(project=setup_project)
    _install_sdk(monkeypatch, [setup_client])
    sleep_calls: list[float] = []

    async def _sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(
        "strix.runtime.caido_bootstrap.asyncio.sleep",
        _sleep,
    )

    with pytest.raises(asyncio.CancelledError):
        await bootstrap_caido(
            _FakeSession(),  # type: ignore[arg-type]
            host_url="http://host",
            container_url="http://container",
        )

    assert setup_project.create_calls == 1
    assert setup_project.selected_ids == ["created"]
    assert sleep_calls == []
    assert setup_client.closed
