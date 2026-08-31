"""Caido client bootstrap.

The Caido CLI runs as an in-container sidecar listening on
``127.0.0.1:48080`` *inside* the sandbox. We grab a guest token by
``session.exec()``-ing curl from inside the container, then construct
a host-side :class:`caido_sdk_client.Client` against the runtime's
exposed-port URL for all subsequent SDK calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agents.sandbox.session import BaseSandboxSession
    from caido_sdk_client import Client


logger = logging.getLogger(__name__)


_LOGIN_AS_GUEST_BODY = (
    '{"query":"mutation LoginAsGuest { loginAsGuest { token { accessToken } } }"}'
)
_PROJECT_SETUP_TIMEOUT_MS = 45_000
_PROJECT_SETUP_ATTEMPTS = 3


async def _login_as_guest(
    session: BaseSandboxSession,
    *,
    container_url: str,
    attempts: int = 10,
) -> str:
    """``session.exec`` curl to fetch a guest token; retry until ready.

    Caido's GraphQL listener may not be up the instant the container
    starts. The retry loop also doubles as the Caido readiness probe —
    no separate TCP healthcheck needed.
    """
    last_err: str | None = None
    for i in range(1, attempts + 1):
        result = await session.exec(
            "curl",
            "-fsS",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-d",
            _LOGIN_AS_GUEST_BODY,
            f"{container_url}/graphql",
            timeout=15,
        )
        if result.ok():
            try:
                payload = json.loads(result.stdout)
                token = (
                    payload.get("data", {})
                    .get("loginAsGuest", {})
                    .get("token", {})
                    .get("accessToken")
                )
                if token:
                    return str(token)
                last_err = f"loginAsGuest returned no token: {payload}"
            except json.JSONDecodeError as exc:
                last_err = f"unparseable response: {exc}: {result.stdout!r}"
        else:
            stderr = result.stderr.decode("utf-8", errors="replace")[:200]
            last_err = f"curl exit {result.exit_code}: {stderr}"
        logger.debug("loginAsGuest attempt %d/%d failed: %s", i, attempts, last_err)
        await asyncio.sleep(min(2.0 * i, 8.0))

    raise RuntimeError(f"loginAsGuest failed after {attempts} attempts: {last_err}")


async def _setup_project(host_url: str, access_token: str) -> None:
    """Connect with a longer deadline and select the sandbox project."""
    from caido_sdk_client import Client, TokenAuthOptions
    from caido_sdk_client.types import CreateProjectOptions

    client = Client(
        host_url,
        auth=TokenAuthOptions(token=access_token),
        timeout_ms=_PROJECT_SETUP_TIMEOUT_MS,
    )
    try:
        await client.connect()
        project = None
        last_exc: Exception | None = None
        for i in range(1, _PROJECT_SETUP_ATTEMPTS + 1):
            try:
                if project is None:
                    project = await client.project.create(
                        CreateProjectOptions(name="sandbox", temporary=True),
                    )
                await client.project.select(project.id)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if project is None:
                    with contextlib.suppress(Exception):
                        projects = await client.project.list()
                        sandbox_projects = [item for item in projects if item.name == "sandbox"]
                        if sandbox_projects:
                            project = max(sandbox_projects, key=lambda item: item.id)
                logger.warning(
                    "Caido project setup attempt %d/%d failed: %s",
                    i,
                    _PROJECT_SETUP_ATTEMPTS,
                    exc,
                )
                if i < _PROJECT_SETUP_ATTEMPTS:
                    await asyncio.sleep(min(2.0 * i, 8.0))
            else:
                logger.info("Caido project selected: %s", project.id)
                return
        raise RuntimeError(
            f"Caido project setup failed after {_PROJECT_SETUP_ATTEMPTS} attempts"
        ) from last_exc
    finally:
        with contextlib.suppress(Exception):
            await client.aclose()


async def bootstrap_caido(
    session: BaseSandboxSession,
    *,
    host_url: str,
    container_url: str,
) -> Client:
    """Connect to the in-container Caido sidecar and select a fresh project."""
    # The Caido SDK (and its generated GraphQL schema) is slow to import and is
    # only needed once a sandbox is actually being bootstrapped, so it is
    # imported here rather than at module scope.
    from caido_sdk_client import Client, TokenAuthOptions

    logger.info("Bootstrapping Caido client (host=%s, container=%s)", host_url, container_url)

    access_token = await _login_as_guest(session, container_url=container_url)

    await _setup_project(host_url, access_token)

    client = Client(host_url, auth=TokenAuthOptions(token=access_token))
    try:
        # connect() is inside the guard as well: a cancellation there (scan
        # teardown while the bootstrap is still in flight) would otherwise
        # leave the half-connected transport behind.
        await client.connect()
    except BaseException:
        # The client never reaches the session bundle if connect or project
        # setup fails, so close it here to avoid leaking the transport.
        with contextlib.suppress(Exception):
            await client.aclose()
        raise
    return client
