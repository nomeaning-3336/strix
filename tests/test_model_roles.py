"""Tests for role-aware model routing (root vs subagent models)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strix.config.settings import LlmSettings
from strix.core.hooks import ReportUsageHooks


def _make_report_state() -> MagicMock:
    state = MagicMock()
    state.get_total_llm_cost.return_value = 0.0
    state.record_sdk_usage = MagicMock()
    return state


# --- settings surface ---


def test_llm_settings_read_per_role_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIX_ROOT_LLM", "openrouter/deepseek/deepseek-v4-pro")
    monkeypatch.setenv("STRIX_SUBAGENT_LLM", "openrouter/z-ai/glm-5.3-flash")
    settings = LlmSettings()
    assert settings.root_model == "openrouter/deepseek/deepseek-v4-pro"
    assert settings.subagent_model == "openrouter/z-ai/glm-5.3-flash"
    assert settings.model is None


def test_llm_settings_default_strix_llm_still_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIX_LLM", "openrouter/z-ai/glm-5.3-flash")
    monkeypatch.delenv("STRIX_ROOT_LLM", raising=False)
    monkeypatch.delenv("STRIX_SUBAGENT_LLM", raising=False)
    settings = LlmSettings()
    assert settings.model == "openrouter/z-ai/glm-5.3-flash"
    assert settings.root_model is None
    assert settings.subagent_model is None


# --- per-agent usage attribution ---


@pytest.mark.asyncio
async def test_usage_attributed_to_child_model_from_context() -> None:
    hooks = ReportUsageHooks(model="root-model")
    state = _make_report_state()
    child_ctx: MagicMock = MagicMock()
    child_ctx.context = {
        "agent_id": "child-1",
        "parent_id": "root-1",
        "model": "worker-model",
    }
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_end(child_ctx, MagicMock(), MagicMock())
    args = state.record_sdk_usage.call_args.kwargs
    assert args["model"] == "worker-model"
    assert args["agent_id"] == "child-1"


@pytest.mark.asyncio
async def test_usage_falls_back_to_hook_model_without_context_model() -> None:
    hooks = ReportUsageHooks(model="root-model")
    state = _make_report_state()
    ctx: MagicMock = MagicMock()
    ctx.context = {"agent_id": "root-1", "parent_id": None}
    with patch("strix.core.hooks.get_global_report_state", return_value=state):
        await hooks.on_llm_end(ctx, MagicMock(), MagicMock())
    args = state.record_sdk_usage.call_args.kwargs
    assert args["model"] == "root-model"
