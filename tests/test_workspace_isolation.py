"""Safety-mode local workspace isolation."""

from __future__ import annotations

from pathlib import Path

from strix.runtime.local_dir_staging import materialize_isolated_sources
from strix.runtime.session_manager import build_bind_mounts


def test_isolated_copy_does_not_modify_original(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    original = source / "app.py"
    original.write_text("before\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "scan"

    [staged] = materialize_isolated_sources(
        [
            {
                "source_path": str(source),
                "workspace_subdir": "source",
                "protect_metadata": True,
            }
        ],
        run_dir=run_dir,
    )
    staged_file = Path(staged["source_path"]) / "app.py"
    staged_file.write_text("after\n", encoding="utf-8")

    assert original.read_text(encoding="utf-8") == "before\n"
    assert staged_file.read_text(encoding="utf-8") == "after\n"
    assert staged["original_source_path"] == str(source.resolve())
    assert staged["workspace_mode"] == "isolated_copy"


def test_isolated_copy_keeps_metadata_read_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    (source / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (source / ".agents").mkdir()
    (source / ".agents" / "rules.md").write_text("instructions\n", encoding="utf-8")

    [staged] = materialize_isolated_sources(
        [
            {
                "source_path": str(source),
                "workspace_subdir": "source",
                "protect_metadata": True,
            }
        ],
        run_dir=tmp_path / "runs" / "scan",
    )

    assert staged["protect_metadata"] is True
    read_only = {mount["target"] for mount in build_bind_mounts([staged]) if mount.get("read_only")}
    assert "/workspace/source/.git" in read_only
    assert "/workspace/source/.agents" in read_only


def test_isolated_copy_drops_out_of_tree_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    (source / "escape").symlink_to(secret)

    [staged] = materialize_isolated_sources(
        [
            {
                "source_path": str(source),
                "workspace_subdir": "source",
                "protect_metadata": True,
            }
        ],
        run_dir=tmp_path / "runs" / "scan",
    )

    assert not (Path(staged["source_path"]) / "escape").exists()
