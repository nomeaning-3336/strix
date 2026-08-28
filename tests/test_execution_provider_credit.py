"""Provider-billing failures: classification, fallout, and fleet revival.

Modelled on a real run in which an Anthropic balance ran dry mid-scan and took
eighteen agents down in thirteen seconds.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from agents import RunConfig, Runner
from openai import BadRequestError, InternalServerError

from strix.core import execution
from strix.core.agents import AgentCoordinator


# The body Anthropic actually returns, as it reached the logs.
_CREDIT_BODY = (
    'b\'{"type":"error","error":{"type":"invalid_request_error","message":"Your credit '
    "balance is too low to access the Anthropic API. Please go to Plans & Billing to "
    'upgrade or purchase credits."},"request_id":"req_011CeVWdCD5de48p3k3MFHc1"}\''
)


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _credit_bad_request() -> BadRequestError:
    """The pre-stream shape: a clean 400 straight from the provider."""
    return BadRequestError(
        f"litellm.BadRequestError: AnthropicException - {_CREDIT_BODY}",
        response=httpx.Response(400, request=_request()),
        body=None,
    )


def _credit_midstream() -> InternalServerError:
    """The mid-stream shape: LiteLLM re-wraps the same 400 body as a 500.

    Nothing on the surface of this one says "billing": the status code is
    retryable and the message names a fallback. Only the body text gives it away.
    """
    inner = Exception(f"AnthropicError: {_CREDIT_BODY}")
    outer = InternalServerError(
        "litellm.MidStreamFallbackError: litellm.InternalServerError: AnthropicException - "
        "Your credit balance is too low to access the Anthropic API.",
        response=httpx.Response(500, request=_request()),
        body=None,
    )
    outer.__cause__ = inner
    return outer


def test_credit_error_detected_in_both_wire_shapes() -> None:
    assert execution._is_provider_credit_error(_credit_bad_request()) is True
    assert execution._is_provider_credit_error(_credit_midstream()) is True


def test_credit_error_detected_through_a_wrapped_chain() -> None:
    root_cause = Exception(_CREDIT_BODY)
    middle = RuntimeError("during handling of the above exception")
    middle.__cause__ = root_cause
    top = RuntimeError("agent run failed")
    top.__context__ = middle
    assert execution._is_provider_credit_error(top) is True


def test_unrelated_errors_are_not_credit_errors() -> None:
    assert execution._is_provider_credit_error(ValueError("nope")) is False
    assert (
        execution._is_provider_credit_error(
            BadRequestError("bad", response=httpx.Response(400, request=_request()), body=None)
        )
        is False
    )


def test_credit_error_is_never_retried_as_transient() -> None:
    """A 500-shaped billing rejection must not read as a server blip."""
    assert execution._is_transient_model_error(_credit_midstream()) is False
    assert execution._is_transient_model_error(_credit_bad_request()) is False


def test_a_cyclic_exception_chain_terminates() -> None:
    first = RuntimeError("a")
    second = RuntimeError("b")
    first.__cause__ = second
    second.__cause__ = first
    assert execution._is_provider_credit_error(first) is False


class _FakeStream:
    def __init__(self, exc: BaseException | None = None) -> None:
        self._exc = exc
        self._events: list[Any] = []
        self.run_loop_exception: BaseException | None = None

    async def stream_events(self) -> Any:
        if self._exc is not None:
            raise self._exc
        for event in self._events:
            yield event


async def _run_cycle(
    monkeypatch: pytest.MonkeyPatch,
    coordinator: AgentCoordinator,
    agent_id: str,
    streams: list[_FakeStream],
) -> tuple[Any, int]:
    monkeypatch.setattr(execution, "_TRANSIENT_MODEL_RETRY_BASE_DELAY_S", 0.0)
    monkeypatch.setattr(execution, "_TRANSIENT_MODEL_RETRY_MAX_DELAY_S", 0.0)
    calls = {"n": 0}

    def _fake_run_streamed(*_args: Any, **_kwargs: Any) -> _FakeStream:
        stream = streams[calls["n"]]
        calls["n"] += 1
        return stream

    monkeypatch.setattr(Runner, "run_streamed", _fake_run_streamed)
    result = await execution._run_cycle(
        object(),
        coordinator,
        agent_id,
        input_data="task",
        run_config=cast("RunConfig", object()),
        context={},
        max_turns=5,
        session=None,
        interactive=True,
        event_sink=None,
        hooks=None,
    )
    return result, calls["n"]


@pytest.mark.asyncio
async def test_credit_error_fails_once_without_burning_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)

    result, attempts = await _run_cycle(
        monkeypatch, coordinator, "root", [_FakeStream(exc=_credit_midstream())]
    )

    assert result is None
    assert attempts == 1, "a billing rejection must not be replayed"
    assert coordinator.statuses["root"] == "failed"
    assert coordinator.errors["root"] == execution.PROVIDER_CREDIT_MESSAGE
    assert coordinator.provider_outage_active is True


@pytest.mark.asyncio
async def test_a_successful_call_revives_the_agents_the_outage_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scenario the user hit: credits are topped up and only the root is reachable."""
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    for index in range(3):
        await coordinator.register(f"child{index}", f"hunter {index}", parent_id="root")

    # The outage takes the whole fleet down within seconds of itself.
    for agent_id in ("root", "child0", "child1", "child2"):
        await _run_cycle(
            monkeypatch, coordinator, agent_id, [_FakeStream(exc=_credit_bad_request())]
        )
    assert all(
        coordinator.statuses[agent_id] == "failed"
        for agent_id in ("root", "child0", "child1", "child2")
    )
    # Every one of them is gated behind a user message it will never get.
    assert all(
        coordinator.runtimes[agent_id].user_wake_required
        for agent_id in ("child0", "child1", "child2")
    )

    # The user adds credits and messages the root, which is all they can reach.
    await _run_cycle(monkeypatch, coordinator, "root", [_FakeStream()])

    for child in ("child0", "child1", "child2"):
        assert coordinator.statuses[child] == "waiting", f"{child} was left for dead"
        assert coordinator.runtimes[child].user_wake_required is False
        assert coordinator.pending_counts[child] == 1
        assert child not in coordinator.errors
        assert coordinator.runtimes[child].mailbox[0]["content"] == (
            execution.PROVIDER_RECOVERED_MESSAGE
        )
    assert coordinator.provider_outage_active is False


@pytest.mark.asyncio
async def test_revival_leaves_agents_that_died_for_other_reasons_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    await coordinator.register("child", "hunter", parent_id="root")

    await _run_cycle(monkeypatch, coordinator, "root", [_FakeStream(exc=_credit_bad_request())])
    await coordinator.set_status("child", "failed", error="something else broke")

    await _run_cycle(monkeypatch, coordinator, "root", [_FakeStream()])

    assert coordinator.statuses["child"] == "failed"
    assert coordinator.errors["child"] == "something else broke"


@pytest.mark.asyncio
async def test_revival_is_a_no_op_without_an_outage() -> None:
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    await coordinator.set_status("root", "failed", error="unrelated")

    assert coordinator.provider_outage_active is False
    assert await coordinator.revive_after_provider_outage("recovered") == []
    assert coordinator.statuses["root"] == "failed"
