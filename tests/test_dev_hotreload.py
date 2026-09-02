"""Tests for the opt-in hot-reload manager (strix.dev.hotreload)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from strix.dev.hotreload import HotReloadManager, hot_reload_enabled


if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_hot_reload_enabled_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRIX_HOT_RELOAD", raising=False)
    assert hot_reload_enabled() is False
    monkeypatch.setenv("STRIX_HOT_RELOAD", "1")
    assert hot_reload_enabled() is True
    monkeypatch.setenv("STRIX_HOT_RELOAD", "false")
    assert hot_reload_enabled() is False


@pytest.mark.asyncio
async def test_watcher_publishes_epoch_on_change(tmp_path: Path) -> None:
    (tmp_path / "prompts").mkdir()
    skill = tmp_path / "prompts" / "root.md"
    _write(skill, "old guidance\n")
    manager = HotReloadManager([tmp_path], debounce_s=0.0, poll_interval_s=0.05)
    manager.start_watch()
    assert manager.current_epoch == 0

    _write(skill, "new guidance for the skill with extra words now\n")
    time.sleep(0.01)  # let the fs mtime tick advance past the baseline
    await manager.poll_once()  # first poll only seeds the debounce clock
    assert manager.current_epoch == 0
    await manager.poll_once()  # second poll publishes the epoch
    assert manager.current_epoch == 1
    assert manager.changed_files[0].kind == "render"
    assert str(manager.changed_files[0].path) == str(skill)

    # No change -> no new epoch.
    await manager.poll_once()
    assert manager.current_epoch == 1


@pytest.mark.asyncio
async def test_maybe_apply_adopts_fresh_surfaces(tmp_path: Path) -> None:
    (tmp_path / "prompts").mkdir()
    skill = tmp_path / "prompts" / "root.md"
    _write(skill, "old")
    manager = HotReloadManager([tmp_path], debounce_s=0.0)
    manager.start_watch()

    builder = lambda: SimpleNamespace(  # noqa: E731
        instructions="new instructions",
        tools=["tool-a"],
        tool_use_behavior="run_llm_again",
    )
    manager.register("agent-1", builder)
    agent = SimpleNamespace(
        instructions="old instructions", tools=[], tool_use_behavior=None
    )

    assert await manager.maybe_apply("agent-1", agent) is False  # no epoch yet
    _write(skill, "new content for skill file")
    time.sleep(0.01)  # let the fs mtime tick advance past the baseline
    await manager.poll_once()  # seed debounce clock
    await manager.poll_once()  # publish epoch
    assert await manager.maybe_apply("agent-1", agent) is True
    assert agent.instructions == "new instructions"
    assert agent.tools == ["tool-a"]
    assert agent.tool_use_behavior == "run_llm_again"
    assert manager.adopted_epoch("agent-1") == manager.current_epoch
    # Already adopted for this epoch.
    assert await manager.maybe_apply("agent-1", agent) is False


@dataclass
class StubRunConfig:
    model: str
    model_settings: Any = field(default=None)


@pytest.mark.asyncio
async def test_apply_model_switches_root_role(tmp_path: Path) -> None:
    models = tmp_path / "agent-models.json"
    manager = HotReloadManager([tmp_path], models_file=models, debounce_s=0.0)
    cfg = StubRunConfig(model="openrouter/z-ai/glm-5.3-flash")
    _write(
        models,
        json.dumps({"root": "openrouter/deepseek/deepseek-v4-pro-0813"}),
    )
    out = await manager.apply_model("root-id", cfg, is_root=True)
    assert out.model == "openrouter/deepseek/deepseek-v4-pro-0813"
    assert out.model_settings is not None
    # Unchanged when the file does not name the role/agent.
    cfg2 = StubRunConfig(model="openrouter/z-ai/glm-5.3-flash")
    out2 = await manager.apply_model("child-1", cfg2, is_root=False)
    assert out2 is cfg2


@pytest.mark.asyncio
async def test_apply_model_prefers_agent_id_override(tmp_path: Path) -> None:
    models = tmp_path / "agent-models.json"
    manager = HotReloadManager([tmp_path], models_file=models, debounce_s=0.0)
    cfg = StubRunConfig(model="openrouter/z-ai/glm-5.3-flash")
    _write(
        models,
        json.dumps(
            {
                "subagent": "openrouter/deepseek/deepseek-v4-flash-0731",
                "special-1": "openrouter/deepseek/deepseek-v4-pro-0813",
            }
        ),
    )
    out = await manager.apply_model("special-1", cfg, is_root=False)
    assert out.model == "openrouter/deepseek/deepseek-v4-pro-0813"
