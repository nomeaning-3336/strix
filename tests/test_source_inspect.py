"""source_inspect_many — batched, deterministic, authorized source inspection."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING

from strix.tools.source_inspect.tool import (
    do_source_inspect_many,
    map_operation_path,
    resolve_authorized_roots,
)


if TYPE_CHECKING:
    from pathlib import Path


def _tree(root: Path) -> Path:
    (root / "src" / "server").mkdir(parents=True)
    (root / "src" / "core").mkdir(parents=True)
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "dist").mkdir(parents=True)
    (root / "src" / "server" / "GameServer.ts").write_text(
        "class GameServer {\n  playerID: string;\n  sharesBorderWith: true;\n}",
        encoding="utf-8",
    )
    (root / "src" / "core" / "Attack.ts").write_text(
        "export const canAttack = (t) => t !== null;\n",
        encoding="utf-8",
    )
    (root / "src" / "core" / "notes.txt").write_text(
        "sharesBorderWith not here\n", encoding="utf-8"
    )
    # Noise that searches must skip.
    (root / "node_modules" / "pkg" / "index.js").write_text(
        "sharesBorderWith noise\n", encoding="utf-8"
    )
    (root / "dist" / "bundle.min.js").write_text(
        "sharesBorderWith minified\n", encoding="utf-8"
    )
    return root


async def test_batched_reads_are_ordered_and_sliced(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo")
    result = await do_source_inspect_many(
        reads=[
            {"path": "src/server/GameServer.ts", "start_line": 1, "end_line": 2},
            {"path": "src/core/Attack.ts"},
        ],
        searches=[],
        roots=[root],
    )
    assert result["success"] is True
    assert result["reads"][0]["ok"] is True
    assert result["reads"][0]["content"] == "class GameServer {\n  playerID: string;"
    assert result["reads"][0]["lines_total"] == 4
    assert result["reads"][1]["ok"] is True
    assert "canAttack" in result["reads"][1]["content"]


async def test_read_output_cap_and_truncation_flag(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo")
    result = await do_source_inspect_many(
        reads=[{"path": "src/core/Attack.ts", "max_chars": 10}],
        searches=[],
        roots=[root],
    )
    entry = result["reads"][0]
    assert entry["ok"] is True
    assert entry["truncated"] is True
    assert len(entry["content"]) <= 10 + len("...[truncated]") + 1


async def test_independent_errors_do_not_abort_the_batch(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo")
    result = await do_source_inspect_many(
        reads=[
            {"path": "src/missing/File.ts"},
            {"path": "src/server/GameServer.ts"},
        ],
        searches=[{"pattern": "canAttack", "path": "src/core"}],
        roots=[root],
    )
    assert result["reads"][0]["ok"] is False
    assert "cannot read" in result["reads"][0]["error"]
    assert result["reads"][1]["ok"] is True
    assert result["searches"][0]["ok"] is True


async def test_search_is_deterministic_and_skips_noise(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo")
    result = await do_source_inspect_many(
        reads=[],
        searches=[{"pattern": "sharesBorderWith", "path": "."}],
        roots=[root],
    )
    entry = result["searches"][0]
    assert entry["ok"] is True
    files = [m["file"] for m in entry["matches"]]
    # Deterministic sorted walk order; dependency/build noise is skipped.
    assert files == ["src/core/notes.txt", "src/server/GameServer.ts"]
    assert all("node_modules" not in f and "dist" not in f for f in files)


async def test_search_single_file_and_include_filter(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo")
    by_file = await do_source_inspect_many(
        reads=[],
        searches=[{"pattern": "sharesBorderWith", "path": "src/server/GameServer.ts"}],
        roots=[root],
    )
    assert len(by_file["searches"][0]["matches"]) == 1

    included = await do_source_inspect_many(
        reads=[],
        searches=[{"pattern": "sharesBorderWith", "path": ".", "include": r".*\.ts$"}],
        roots=[root],
    )
    files = [m["file"] for m in included["searches"][0]["matches"]]
    assert files == ["src/server/GameServer.ts"]


async def test_search_validation_errors_are_per_op(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo")
    result = await do_source_inspect_many(
        reads=[],
        searches=[
            {"pattern": "[unclosed", "path": "."},
            {"pattern": "canAttack", "path": "src/nowhere"},
            {"pattern": "canAttack", "path": "src/core"},
        ],
        roots=[root],
    )
    assert "invalid regex" in result["searches"][0]["error"]
    assert "search path not found" in result["searches"][1]["error"]
    assert result["searches"][2]["ok"] is True


async def test_max_matches_truncates(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo")
    (root / "many.ts").write_text("needle\n" * 20, encoding="utf-8")
    result = await do_source_inspect_many(
        reads=[],
        searches=[{"pattern": "needle", "path": ".", "max_matches": 5}],
        roots=[root],
    )
    entry = result["searches"][0]
    assert len(entry["matches"]) == 5
    assert entry["truncated"] is True


async def test_path_traversal_and_out_of_root_rejected(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo")
    secret = tmp_path / "secret.txt"
    secret.write_text("do not leak", encoding="utf-8")

    result = await do_source_inspect_many(
        reads=[{"path": "../secret.txt"}],
        searches=[{"pattern": "leak", "path": "../../"}],
        roots=[root],
    )
    assert result["reads"][0]["ok"] is False
    assert "outside the authorized source roots" in result["reads"][0]["error"]
    assert result["searches"][0]["ok"] is False

    # map_operation_path alone never resolves outside the root.
    assert map_operation_path("../secret.txt", [root]) is None
    assert map_operation_path("/workspace/repo/../secret.txt", [root]) is None
    assert map_operation_path("/etc/passwd", [root]) is None


async def test_workspace_style_paths_and_no_roots(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo")
    result = await do_source_inspect_many(
        reads=[{"path": f"/workspace/{root.name}/src/core/Attack.ts"}],
        searches=[],
        roots=[root],
    )
    assert result["reads"][0]["ok"] is True

    none_result = await do_source_inspect_many(reads=[{"path": "a.ts"}], searches=[], roots=[])
    assert none_result["success"] is False
    assert "use exec_command" in none_result["error"]


def test_resolve_authorized_roots_from_report_state(tmp_path: Path) -> None:
    local_root = _tree(tmp_path / "local")
    repo_root = _tree(tmp_path / "clone")
    report_state = SimpleNamespace(
        run_record={
            "local_sources": [{"source_path": str(local_root)}],
            "targets_info": [
                {"type": "repository", "details": {"cloned_repo_path": str(repo_root)}},
                {"type": "web_application", "details": {}},
            ],
        }
    )
    roots = resolve_authorized_roots(report_state)
    assert len(roots) == 2
    # Deterministic order and no duplicates when the clone repeats a source.
    dup = SimpleNamespace(
        run_record={
            "local_sources": [{"source_path": str(local_root)}, {"source_path": str(local_root)}],
            "targets_info": [],
        }
    )
    assert [str(r) for r in resolve_authorized_roots(dup)] == [str(local_root.resolve())]


async def test_root_detection_for_reads(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo")
    # A file that exists in two roots resolves against the first root listed.
    other = _tree(tmp_path / "other")
    (other / "src" / "core" / "Attack.ts").write_text("other content\n", encoding="utf-8")
    result = await do_source_inspect_many(
        reads=[{"path": "src/core/Attack.ts"}],
        searches=[],
        roots=[root, other],
    )
    assert "canAttack" in result["reads"][0]["content"]


def test_empty_batch_rejected(tmp_path: Path) -> None:
    result = asyncio.run(do_source_inspect_many(reads=[], searches=[], roots=[tmp_path]))
    assert result["success"] is False
    assert "at least one read or one search" in result["error"]
