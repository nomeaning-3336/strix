"""Deterministic compact child context handoff (Child Context v1).

No model calls, no real coordinator, no real network. Pure-function
tests plus an integration test that drives ``create_agent`` end-to-end
with fakes for the runner closure, the spawn primitive, and the
``RunContextWrapper`` carrying ``scan_targets`` / ``turn_input``.

The point of this commit is to stop focused arbitrary subagents from
requiring a complete serialized copy of the parent's current trajectory
when the parent opts into ``inherit_context=False``.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from strix.core.agents import AgentCoordinator
from strix.core.child_context import (
    ChildContextPacket,
    build_packet_from_task,
    render_packet,
)
from strix.core.execution import spawn_child_agent
from strix.core.sessions import scrub_images_from_items
from strix.tools.agents_graph.tools import _create_agent_impl, create_agent


OBJECTIVE = "Audit the login flow for IDOR and CSRF."


# ---------------------------------------------------------------------------
# 1. Pure dataclass purity
# ---------------------------------------------------------------------------


def test_child_context_packet_is_frozen_and_pure() -> None:
    packet = ChildContextPacket(
        objective=OBJECTIVE,
        scope=("/workspace/api",),
    )
    with pytest.raises(FrozenInstanceError):
        packet.objective = "tampered"  # type: ignore[misc]
    # No agent-runtime / partition / report imports.
    module_dict = vars(ChildContextPacket)
    assert "scope" in module_dict["__dataclass_fields__"]
    assert set(module_dict["__dataclass_fields__"]) == {
        "objective",
        "scope",
        "known_facts",
        "open_questions",
        "evidence_refs",
        "do_not_repeat",
    }


def test_optional_packet_fields_default_to_empty_tuples() -> None:
    packet = ChildContextPacket(objective=OBJECTIVE, scope=())
    assert packet.known_facts == ()
    assert packet.open_questions == ()
    assert packet.evidence_refs == ()
    assert packet.do_not_repeat == ()


# ---------------------------------------------------------------------------
# 2. Pure builder + renderer determinism
# ---------------------------------------------------------------------------


def test_build_packet_from_task_populates_objective_scope_and_defaults() -> None:
    packet = build_packet_from_task(
        task=OBJECTIVE,
        scan_targets=("/workspace/api", "https://app.example.com"),
    )
    assert packet.objective == OBJECTIVE
    assert packet.scope == ("/workspace/api", "https://app.example.com")
    assert packet.known_facts == ()
    assert packet.open_questions == ()
    assert packet.evidence_refs == ()
    assert len(packet.do_not_repeat) >= 1


def test_build_packet_dedupes_scope_order_preserving() -> None:
    packet = build_packet_from_task(
        task=OBJECTIVE,
        scan_targets=("/a", "/b", "/a", "/c", "/b"),
    )
    assert packet.scope == ("/a", "/b", "/c")


def test_build_packet_handles_absent_or_non_string_scan_targets() -> None:
    packet = build_packet_from_task(task=OBJECTIVE)
    assert packet.scope == ()
    packet = build_packet_from_task(task=OBJECTIVE, scan_targets=None)
    assert packet.scope == ()
    packet = build_packet_from_task(task=OBJECTIVE, scan_targets=("", 123, "/x", None))
    assert packet.scope == ("/x",)


def test_render_packet_is_byte_deterministic() -> None:
    packet = build_packet_from_task(
        task=OBJECTIVE,
        scan_targets=("/workspace/api",),
    )
    assert render_packet(packet) == render_packet(packet)
    assert render_packet(packet) == render_packet(packet)
    # Canonical JSON with sort_keys=True -> parseable, stable key order.
    payload_line = render_packet(packet).split("Task packet:\n", 1)[1]
    payload_dict = json.loads(payload_line)
    assert payload_dict["objective"] == OBJECTIVE
    assert payload_dict["scope"] == ["/workspace/api"]


def test_render_packet_preserves_original_task_text_verbatim() -> None:
    packet = build_packet_from_task(
        task=OBJECTIVE,
        scan_targets=("/workspace/api",),
    )
    rendered = render_packet(packet)
    assert OBJECTIVE in rendered


def test_render_packet_canonical_json_key_order() -> None:
    packet = build_packet_from_task(
        task=OBJECTIVE,
        scan_targets=("/workspace/api",),
    )
    payload_line = render_packet(packet).split("Task packet:\n", 1)[1]
    parsed = json.loads(payload_line)
    assert list(parsed.keys()) == [
        "do_not_repeat",
        "evidence_refs",
        "known_facts",
        "objective",
        "open_questions",
        "scope",
    ]


def test_render_packet_header_is_stable() -> None:
    packet = build_packet_from_task(task=OBJECTIVE)
    rendered = render_packet(packet)
    assert rendered.startswith(
        "Compact parent handoff.\n"
        "\n"
        "Do not repeat broad discovery already completed by the parent.\n"
        "Work from this packet and inspect additional evidence only as your task requires.\n"
        "\n"
        "Task packet:\n"
    )


# ---------------------------------------------------------------------------
# 3. create_agent integration: full vs compact
# ---------------------------------------------------------------------------


class _RecordingSpawner:
    """Captures every spawn call so we can assert kwargs deterministically."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def spawn(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"success": True, "agent_id": "agent-xyz", "name": kwargs.get("name")}


class _StubCtx:
    """Minimal stand-in for ``agents.RunContextWrapper``.

    The ``create_agent`` tool only reads ``ctx.context`` (via ``_ctx``) and
    ``ctx.turn_input``. Both are exposed as plain attributes here.
    """

    def __init__(
        self,
        *,
        turn_input: list[Any] | None = None,
        scan_targets: list[str] | None = None,
    ) -> None:
        self.context: dict[str, Any] = {}
        self.turn_input: list[Any] = turn_input or []

        self.spawner = _RecordingSpawner()
        # Populate the inner context dict the same way the runner does.
        self.context["agent_id"] = "parent-1"
        self.context["spawn_child_agent"] = self.spawner.spawn
        self.context["coordinator"] = AgentCoordinator()
        if scan_targets is not None:
            self.context["scan_targets"] = list(scan_targets)

    def as_wrapper(self) -> Any:
        return self


@pytest.mark.asyncio  # type: ignore[misc]
async def test_create_agent_full_mode_preserves_existing_parent_history() -> None:
    parent_history = [
        {"role": "user", "content": "prior user message"},
        {"role": "assistant", "content": "prior assistant reply"},
    ]
    ctx = _StubCtx(turn_input=parent_history, scan_targets=["/workspace/api"])
    await _create_agent_impl(
        ctx=ctx.as_wrapper(),
        name="Auth Specialist",
        task=OBJECTIVE,
        inherit_context=True,
    )
    assert len(ctx.spawner.calls) == 1
    call = ctx.spawner.calls[0]
    assert call["parent_history"] == parent_history
    assert "initial_input" not in call
    # Full-mode task length is the original task length.
    assert call["task"] == OBJECTIVE


@pytest.mark.asyncio  # type: ignore[misc]
async def test_create_agent_compact_mode_passes_parent_history_empty_list() -> None:
    parent_history = [
        {"role": "user", "content": "should NOT be sent"},
        {"role": "assistant", "content": "should NOT be sent"},
    ]
    ctx = _StubCtx(turn_input=parent_history, scan_targets=["/workspace/api"])
    await _create_agent_impl(
        ctx=ctx.as_wrapper(),
        name="SQLi Validator",
        task=OBJECTIVE,
        inherit_context=False,
    )
    assert len(ctx.spawner.calls) == 1
    call = ctx.spawner.calls[0]
    assert call["parent_history"] == []  # exactly [], not omitted
    assert "initial_input" in call
    initial_input = call["initial_input"]
    assert isinstance(initial_input, list) and len(initial_input) == 1
    assert initial_input[0]["role"] == "user"
    # Parent transcript strings must not appear in the compact child input.
    assert "should NOT be sent" not in initial_input[0]["content"]
    # Original task is preserved verbatim inside the packet.
    assert OBJECTIVE in initial_input[0]["content"]
    # Scan scope propagated from context.
    assert "/workspace/api" in initial_input[0]["content"]


@pytest.mark.asyncio  # type: ignore[misc]
async def test_create_agent_compact_mode_works_without_scan_targets() -> None:
    ctx = _StubCtx(turn_input=[], scan_targets=None)
    await _create_agent_impl(
        ctx=ctx.as_wrapper(),
        name="Test Specialist",
        task=OBJECTIVE,
        inherit_context=False,
    )
    call = ctx.spawner.calls[0]
    assert call["parent_history"] == []
    assert "initial_input" in call
    # Empty scope renders as an empty list in the packet.
    payload = json.loads(call["initial_input"][0]["content"].split("Task packet:\n", 1)[1])
    assert payload["scope"] == []


@pytest.mark.asyncio  # type: ignore[misc]
async def test_create_agent_compact_mode_dedupes_scan_targets() -> None:
    ctx = _StubCtx(
        turn_input=[],
        scan_targets=["/a", "/b", "/a", "/c"],
    )
    await _create_agent_impl(
        ctx=ctx.as_wrapper(),
        name="Test",
        task=OBJECTIVE,
        inherit_context=False,
    )
    payload = json.loads(
        ctx.spawner.calls[0]["initial_input"][0]["content"].split("Task packet:\n", 1)[1]
    )
    assert payload["scope"] == ["/a", "/b", "/c"]


# ---------------------------------------------------------------------------
# 4. create_agent signature + schema preservation (regression)
# ---------------------------------------------------------------------------


def test_create_agent_public_signature_unchanged() -> None:
    """The model-facing ``create_agent`` tool schema gains no new parameters."""
    # The SDK wrapper derives the JSON schema from the function signature, so
    # asserting the impl's parameters (minus ctx, which is the wrapper's context)
    # pins the model-facing contract.
    params = list(inspect.signature(_create_agent_impl).parameters)
    # ctx + the four model-facing arguments.
    assert params == ["ctx", "name", "task", "inherit_context", "skills"]


def test_create_agent_json_schema_unchanged() -> None:
    """The SDK-derived JSON schema exposes exactly the four parameters."""
    schema = create_agent.params_json_schema
    properties = schema.get("properties", {})
    assert set(properties) == {"name", "task", "inherit_context", "skills"}


def test_spawn_child_agent_keeps_existing_kwargs() -> None:
    """Adding ``initial_input`` must not drop any existing required kwarg."""
    params = inspect.signature(spawn_child_agent).parameters
    for keyword in ("name", "task", "skills", "parent_history", "parent_ctx"):
        assert keyword in params
    # The new optional kwarg is present with a default of None.
    assert "initial_input" in params
    assert params["initial_input"].default is None


# ---------------------------------------------------------------------------
# 5. Metrics surfaced via logger (smoke check)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio  # type: ignore[misc]
async def test_create_agent_compact_mode_log_emits_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    parent_history = [{"role": "user", "content": "x" * 5000}]
    ctx = _StubCtx(turn_input=parent_history, scan_targets=["/workspace/api"])
    with caplog.at_level("INFO", logger="strix.tools.agents_graph.tools"):
        await _create_agent_impl(
            ctx=ctx.as_wrapper(),
            name="XSS Specialist",
            task=OBJECTIVE,
            inherit_context=False,
        )
    record_text = " ".join(record.getMessage() for record in caplog.records)
    assert "context_mode=compact" in record_text
    assert "inherited_items=0" in record_text
    assert "inherited_chars=0" in record_text
    # compact_packet_chars measures ONLY the deterministic packet - the
    # identity/termination framing around it is excluded.
    expected_packet = render_packet(
        build_packet_from_task(task=OBJECTIVE, scan_targets=["/workspace/api"])
    )
    assert f"compact_packet_chars={len(expected_packet)}" in record_text
    assert f"task_chars={len(OBJECTIVE)}" in record_text


@pytest.mark.asyncio  # type: ignore[misc]
async def test_create_agent_full_mode_log_emits_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    parent_history = [{"role": "user", "content": "y" * 123}]
    ctx = _StubCtx(turn_input=parent_history)
    with caplog.at_level("INFO", logger="strix.tools.agents_graph.tools"):
        await _create_agent_impl(
            ctx=ctx.as_wrapper(),
            name="Test",
            task=OBJECTIVE,
            inherit_context=True,
        )
    record_text = " ".join(record.getMessage() for record in caplog.records)
    assert "context_mode=full" in record_text
    assert "inherited_items=1" in record_text
    # inherited_chars measures the SCRUBBED serialized parent history - the
    # same representation child_initial_input transmits (images replaced by
    # text), including message wrapper structure.
    expected_serialized = len(
        json.dumps(
            scrub_images_from_items(parent_history),
            ensure_ascii=False,
            default=str,
        )
    )
    assert f"inherited_chars={expected_serialized}" in record_text
    assert "compact_packet_chars=0" in record_text


@pytest.mark.asyncio  # type: ignore[misc]
async def test_inherited_chars_uses_scrubbed_history(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Images are measured as their scrubbed text, matching transmitted bytes."""
    parent_history = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look at this"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + "A" * 4000,
                },
            ],
        }
    ]
    ctx = _StubCtx(turn_input=parent_history)
    with caplog.at_level("INFO", logger="strix.tools.agents_graph.tools"):
        await _create_agent_impl(
            ctx=ctx.as_wrapper(),
            name="Test",
            task=OBJECTIVE,
            inherit_context=True,
        )
    record_text = " ".join(record.getMessage() for record in caplog.records)
    scrubbed = scrub_images_from_items(parent_history)
    assert scrubbed != parent_history  # image replaced
    expected_serialized = len(json.dumps(scrubbed, ensure_ascii=False, default=str))
    assert f"inherited_chars={expected_serialized}" in record_text
    # The raw image payload must NOT be part of the measured size.
    raw_serialized = len(json.dumps(parent_history, ensure_ascii=False, default=str))
    assert expected_serialized < raw_serialized
