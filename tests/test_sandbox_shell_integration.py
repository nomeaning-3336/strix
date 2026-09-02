"""Docker-backed regression tests for the sandbox shell stdout reliability.

These exercise the real ``ExecCommandTool`` / ``WriteStdinTool`` against a live
strix sandbox container. They are the integration half of the guard implemented
in ``strix.agents.factory._wrap_exec_command`` (default ``tty=True``), which
works around the SDK's intermittent empty-stdout path on plain-pipe (``tty=False``)
exec under concurrency.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from agents.sandbox.capabilities.tools.shell_tool import (
    ExecCommandArgs,
    ExecCommandTool,
    WriteStdinArgs,
    WriteStdinTool,
)

from strix.config import load_settings
from strix.runtime import session_manager


def _docker_available() -> bool:
    try:
        import docker  # noqa: PLC0415

        docker.from_env().ping()
        return True
    except Exception:  # noqa: BLE001 - any docker/env failure means "skip"
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="Docker sandbox not available"
)


def _output_text(result: str) -> str:
    return result.split("Output:", 1)[1] if "Output:" in result else ""


@pytest.mark.asyncio
async def test_concurrent_tty_exec_and_persistent_process_keep_stdout() -> None:
    scan_id = "test-shell-stdout"
    bundle = await session_manager.create_or_reuse(
        scan_id,
        image=load_settings().runtime.image,
        local_sources=[],
    )
    session = bundle["session"]
    try:
        exec_tool = ExecCommandTool(session=session)

        async def run(cmd: str) -> str:
            return await exec_tool.run(ExecCommandArgs(cmd=cmd, tty=True))

        # Two concurrent "agents" running quick commands both receive stdout.
        first, second = await asyncio.gather(
            run("echo marker-one"),
            run("echo marker-two"),
        )
        assert "marker-one" in _output_text(first)
        assert "marker-two" in _output_text(second)

        # A persistent interactive process survives, and a subsequent exec still
        # returns its stdout (the failure mode that stranded the OpenFront mapper).
        started = await exec_tool.run(ExecCommandArgs(cmd="python3 -i", tty=True))
        match = re.search(r"session ID (\d+)", started)
        assert match, started
        session_id = int(match.group(1))

        stdin_tool = WriteStdinTool(session=session)
        reply = await stdin_tool.run(
            WriteStdinArgs(session_id=session_id, chars="print('HELLO-FROM-PTY')\n")
        )
        assert "HELLO-FROM-PTY" in _output_text(reply)

        after = await run("echo after-marker")
        assert "after-marker" in _output_text(after)
    finally:
        await session_manager.cleanup(scan_id)
