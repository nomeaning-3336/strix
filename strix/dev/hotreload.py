"""Opt-in live development reload for a running Strix scan (v1).

A scan normally loads agent code once at process start, so editing the repo has
no effect on a run already in progress. With ``STRIX_HOT_RELOAD=1`` (or
``--hot-reload``) a :class:`HotReloadManager` watches the Strix source tree for
changes and publishes numbered *reload epochs*. At the safe boundary between an
agent's turns the manager swaps the *reloadable configuration* of the live
agent — its system instructions, loaded skills, and freshly rebuilt tool
objects — so the next model turn uses the current repo state.

What v1 can refresh (rendered at agent-build time from disk):

- system prompts (``system_prompt.jinja``) and coordination/skill markdown;
- tool objects and schemas, via re-running the agent builder;
- skills a child loads from disk.

What v1 deliberately does NOT do (logged, not applied):

- Python *function bodies* of already-imported modules: the running process
  still executes the old bytecode. In-place patching is v2 (Jurigged-style
  grafting or a stable execution trampoline).

The manager is strictly opt-in; when disabled nothing changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


logger = logging.getLogger(__name__)

# File suffixes that feed agent construction at render/build time (fully
# refreshable in v1) versus Python sources (detected but not hot-applied yet).
_RENDER_AFFECTING_SUFFIXES = frozenset({".jinja", ".md", ".yaml", ".yml", ".json"})
_PYTHON_SUFFIXES = frozenset({".py"})
_WATCH_SUFFIXES = _RENDER_AFFECTING_SUFFIXES | _PYTHON_SUFFIXES
_SKIPPED_DIR_PARTS = frozenset({".git", ".venv", "__pycache__", "node_modules", ".next"})


def hot_reload_enabled() -> bool:
    """Opt-in gate. Reads STRIX_HOT_RELOAD (1/true/yes)."""
    value = os.environ.get("STRIX_HOT_RELOAD", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


@dataclass
class ReloadChange:
    """One detected working-tree change since the last epoch."""

    path: Path
    kind: str = "render"  # "render" (refreshable in v1) | "python" (log only in v1)


class HotReloadManager:
    """Watch the Strix source tree, publish epochs, and adopt them at safe points.

    Thread-safe usage: the watcher runs in the scan's event loop; agents call
    :meth:`maybe_apply` at turn boundaries on the same loop, so an asyncio lock
    serialises epoch publication and adoption.
    """

    def __init__(
        self,
        watch_roots: Iterable[Path],
        *,
        models_file: Path | None = None,
        poll_interval_s: float = 0.5,
        debounce_s: float = 0.3,
    ) -> None:
        self.watch_roots = [Path(r) for r in watch_roots]
        self.models_file = Path(models_file) if models_file else None
        self.poll_interval_s = poll_interval_s
        self.debounce_s = debounce_s
        self.current_epoch = 0
        self.changed_files: list[ReloadChange] = []
        self._baseline: dict[str, tuple[int, int]] = {}
        self._pending_epoch = 0
        self._pending_files: list[ReloadChange] = []
        self._builders: dict[str, Callable[[], Any]] = {}
        self._adopted_epoch: dict[str, int] = {}
        self._model_overrides: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._last_change_at: float | None = None

    # -- live model switching (DSH-style) ---------------------------------

    def _reload_model_overrides(self) -> None:
        """Read the per-run agent-models file, if any.

        Shape (flat dict): ``{"root": model, "subagent": model, "<agent_id>": model}``.
        ``root``/``subagent`` are role defaults; any other key is an agent id
        override. Editing this file takes effect at the agent's next turn
        boundary.
        """
        if self.models_file is None or not self.models_file.is_file():
            self._model_overrides = {}
            return
        try:
            raw = json.loads(self.models_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._model_overrides = {
                    str(k): str(v) for k, v in raw.items() if isinstance(v, str) and str(v).strip()
                }
                return
        except (OSError, json.JSONDecodeError, ValueError):
            logger.warning("unreadable hot-reload models file %s", self.models_file, exc_info=True)
        self._model_overrides = {}

    def model_for(self, agent_id: str, *, is_root: bool) -> str | None:
        """The configured model override for this agent (agent id, then role)."""
        if not self._model_overrides:
            return None
        direct = self._model_overrides.get(agent_id)
        if direct:
            return direct
        return self._model_overrides.get("root" if is_root else "subagent")

    async def apply_model(self, agent_id: str, run_config: Any, *, is_root: bool) -> Any:
        """Point an agent's live run config at a newly configured model.

        Called at every turn boundary (cheap): reads the models file fresh, so
        an operator edit applies on the next model request without restarting.
        Recomputes the model settings for the new model the same way the runner
        does for fresh roles. Returns the (possibly replaced) run config.
        """
        self._reload_model_overrides()
        target = self.model_for(agent_id, is_root=is_root)
        if not target:
            return run_config
        current = run_config.model if isinstance(run_config.model, str) else None
        if target == current:
            return run_config
        import dataclasses  # noqa: PLC0415

        from strix.config import load_settings  # noqa: PLC0415
        from strix.core.inputs import make_model_settings  # noqa: PLC0415

        llm = load_settings().llm
        model_settings = make_model_settings(
            llm.reasoning_effort,
            model_name=target,
            force_required_tool_choice=llm.force_required_tool_choice,
            request_timeout=llm.timeout,
            prompt_cache=llm.prompt_cache,
            extra_headers=llm.extra_headers,
        )
        logger.info("[hot-reload] agent %s model -> %s", agent_id, target)
        return dataclasses.replace(run_config, model=target, model_settings=model_settings)

    def telemetry(self) -> dict[str, Any]:
        """Per-agent adopted epochs (for the viewer / run log)."""
        return {
            "epoch": self.current_epoch,
            "agents": dict(self._adopted_epoch),
        }

    # -- watcher ----------------------------------------------------------

    def _iter_watch_files(self) -> list[Path]:
        files: list[Path] = []
        for root in self.watch_roots:
            if not root.is_dir():
                continue
            for p in root.rglob("*"):
                if not p.is_file() or p.suffix not in _WATCH_SUFFIXES:
                    continue
                if any(part in _SKIPPED_DIR_PARTS for part in p.parts):
                    continue
                files.append(p)
        return files

    def snapshot(self) -> dict[str, tuple[int, int]]:
        snap: dict[str, tuple[int, int]] = {}
        for p in self._iter_watch_files():
            try:
                st = p.stat()
            except OSError:
                continue
            snap[str(p)] = (st.st_mtime_ns, st.st_size)
        return snap

    def _detect(self) -> list[ReloadChange]:
        current = self.snapshot()
        changed: list[ReloadChange] = []
        for key in current:
            if self._baseline.get(key) != current[key]:
                path = Path(key)
                kind = "python" if path.suffix in _PYTHON_SUFFIXES else "render"
                changed.append(ReloadChange(path=path, kind=kind))
        missing = [key for key in self._baseline if key not in current]
        changed.extend(ReloadChange(path=Path(key), kind="render") for key in missing)
        return changed

    def start_watch(self) -> None:
        """Seed the baseline from the current tree (call before running the scan)."""
        self._baseline = self.snapshot()
        self._last_change_at = None

    async def run_watcher(self) -> None:
        """Background task: poll mtimes and publish a debounced new epoch."""
        while not self._stop.is_set():
            try:
                await self.poll_once()
            except Exception:
                logger.exception("hot-reload watcher poll failed")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval_s)

    async def poll_once(self) -> None:
        changed = self._detect()
        if not changed:
            self._last_change_at = None
            return
        now = time.monotonic()
        if self._last_change_at is None:
            self._last_change_at = now
            return
        if now - self._last_change_at < self.debounce_s:
            return
        async with self._lock:
            self.current_epoch += 1
            self._pending_epoch = self.current_epoch
            self._pending_files = changed
            self.changed_files = changed
            self._baseline = self.snapshot()
        kinds = {c.kind for c in changed}
        logger.info(
            "[hot-reload] epoch %d detected (%s): %s",
            self.current_epoch,
            ",".join(sorted(kinds)),
            ", ".join(str(c.path) for c in changed[:8]),
        )
        self._last_change_at = None

    async def stop_watcher(self) -> None:
        self._stop.set()

    # -- adoption ---------------------------------------------------------

    def register(self, agent_id: str, builder: Callable[[], Any]) -> None:
        """Bind an agent id to a builder that reconstructs it from current code."""
        self._builders[agent_id] = builder
        self._adopted_epoch[agent_id] = self._pending_epoch or self.current_epoch

    def adopted_epoch(self, agent_id: str) -> int:
        return self._adopted_epoch.get(agent_id, self.current_epoch)

    async def maybe_apply(self, agent_id: str, agent: Any) -> bool:
        """Adopt a pending epoch onto the live agent (call between turns).

        Re-runs the registered builder and copies the reloadable surfaces
        (instructions, tools, tool_use_behavior) onto the existing agent so its
        conversation, session, and runtime state stay untouched.
        """
        async with self._lock:
            pending = self._pending_epoch
            adopted = self._adopted_epoch.get(agent_id)
            if pending <= (adopted or 0):
                return False
            builder = self._builders.get(agent_id)
            changed = self._pending_files
            if builder is None:
                self._adopted_epoch[agent_id] = pending
                return False

        fresh = builder()
        applied: list[str] = []
        for attr, fresh_value in (
            ("instructions", getattr(fresh, "instructions", None)),
            ("tools", getattr(fresh, "tools", None)),
            ("tool_use_behavior", getattr(fresh, "tool_use_behavior", None)),
        ):
            if fresh_value is not None:
                setattr(agent, attr, fresh_value)
                applied.append(attr)
        async with self._lock:
            self._adopted_epoch[agent_id] = pending
        render_changes = [str(c.path) for c in changed if c.kind == "render"]
        py_changes = [str(c.path) for c in changed if c.kind == "python"]
        note = ""
        if py_changes:
            note = (
                " (python bodies unchanged until the v2 in-place patcher; "
                "next turn still runs old bytecode)"
            )
        logger.info(
            "[hot-reload] agent %s adopted epoch %d [%s]%s%s",
            agent_id,
            pending,
            ", ".join(applied),
            f": {', '.join(render_changes[:6])}" if render_changes else "",
            note,
        )
        return True
