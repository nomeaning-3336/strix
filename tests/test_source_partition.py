"""Deterministic source partitioning — contract tests.

Covers the 14 required cases from the v1 spec plus unit coverage for the
gitignore matcher, LOC counting, classification and weight policy.  Every test
is deterministic across reruns and platforms: no live repos, no network, no
LLM, pure local fixtures (``tmp_path`` + the committed synthetic tree under
``tests/fixtures/source_partition/basic_repo``).
"""

from __future__ import annotations

import json
import unicodedata
from fractions import Fraction
from pathlib import Path

import pytest

import strix.tools.source_partition.inventory as inventory_module
from strix.tools.source_partition import (
    FileKind,
    PartitionConfig,
    PartitionUnit,
    SourceInventory,
    build_partition_units,
    inventory_source,
    partition_source,
    partition_units,
)
from strix.tools.source_partition.classify import classify_file
from strix.tools.source_partition.gitignore import chain_decides, parse_gitignore
from strix.tools.source_partition.loc import count_loc, language_for
from strix.tools.source_partition.normalize import (
    display_root_names,
    normalize_rel_path,
    path_sort_key,
)
from strix.tools.source_partition.units import effective_weight


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "source_partition" / "basic_repo"

# kinds excluded from partition units by default (never reach a shard)
_EXCLUDED_KINDS = frozenset({FileKind.GENERATED, FileKind.VENDOR})

_EXCLUDED_PREFIXES = ("node_modules/", "dist/", ".git/", ".hg/", ".svn/", "src/legacy/")
_EXCLUDED_FILES = {"package-lock.json"}


def _write_code(path: Path, lines: int) -> None:
    """A source file with exactly ``lines`` meaningful lines (no comments)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"v{i} = {i}\n" for i in range(lines)), encoding="utf-8")


def _write_text(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"plain line {i}\n" for i in range(lines)), encoding="utf-8")


def _rel_entries(inventory: SourceInventory) -> set[str]:
    return {entry.rel for entry in inventory.entries}


# ---------------------------------------------------------------------------
# 1. Same input -> byte-identical manifest
# ---------------------------------------------------------------------------


def test_same_input_byte_identical_manifest() -> None:
    manifest_a = partition_source([FIXTURE], workers=3)
    manifest_b = partition_source([FIXTURE], workers=3)
    assert manifest_a.to_json() == manifest_b.to_json()
    assert manifest_a.to_dict() == manifest_b.to_dict()

    inventory_a = inventory_source([FIXTURE])
    inventory_b = inventory_source([FIXTURE])
    assert inventory_a.entries == inventory_b.entries
    assert inventory_a.notes == inventory_b.notes


# ---------------------------------------------------------------------------
# 2/3. No duplicates; every included file in exactly one shard
# ---------------------------------------------------------------------------


def test_no_file_in_two_shards_and_mapping_is_consistent() -> None:
    manifest = partition_source([FIXTURE], workers=3)
    all_files = [file for shard in manifest.shards for file in shard.files]
    assert len(all_files) == len(set(all_files))
    assert set(manifest.file_to_shard) == set(all_files)
    # Shard ids are contiguous and files within a shard are case-folded-sorted.
    assert [shard.shard_id for shard in manifest.shards] == list(range(manifest.effective_workers))
    for shard in manifest.shards:
        assert shard.files == tuple(sorted(shard.files, key=path_sort_key))


def test_every_included_file_in_exactly_one_shard() -> None:
    inventory = inventory_source([FIXTURE])
    units, unit_notes = build_partition_units(inventory)
    included = {unit.display for unit in units}
    assert unit_notes == ()  # clean fixture produces no diagnostics

    manifest = partition_source([FIXTURE], workers=3)
    shard_files = {file for shard in manifest.shards for file in shard.files}

    assert shard_files == included
    assert set(manifest.file_to_shard) == included
    # The wrapper surfaced the (empty) inventory/unit notes on the manifest.
    assert manifest.notes == inventory.notes == ()

    # Against the inventory: only GENERATED/VENDOR (weight-policy exclusions)
    # are missing from the manifest; every other file appears exactly once.
    inventory_expected = {
        entry.rel for entry in inventory.entries if entry.kind not in _EXCLUDED_KINDS
    }
    assert shard_files == inventory_expected
    assert len(shard_files) == len(inventory_expected)


# ---------------------------------------------------------------------------
# 4. Reasonable balance
# ---------------------------------------------------------------------------


def test_reasonable_balance(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    for index in range(6):
        _write_code(root / f"d{index}" / "a.py", lines=100)

    manifest = partition_source([root], workers=3)
    total = manifest.total_weight
    assert total == 600
    assert manifest.effective_workers == 3

    # cap = ceil(total / E * tolerance) with tolerance 1.25 = 5/4
    cap = (total * 5 + 4 * 3 - 1) // (4 * 3)
    weights = sorted(shard.weight for shard in manifest.shards)
    assert all(weight <= cap for weight in weights)
    # LPT on six equal 100-weight subtrees with 3 workers lands 200/200/200.
    assert weights == [200, 200, 200]
    # Every top-level directory stayed whole (locality side effect).
    for shard in manifest.shards:
        assert len(shard.files) == 2


# ---------------------------------------------------------------------------
# 5. Locality preservation
# ---------------------------------------------------------------------------


def test_locality_subtrees_stay_together_when_they_fit(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_code(root / "src" / "server" / "auth" / "auth.py", lines=40)
    _write_code(root / "src" / "server" / "session" / "session.py", lines=40)
    _write_code(root / "src" / "core" / "engine.py", lines=30)
    _write_code(root / "scripts" / "build.py", lines=60)
    _write_code(root / "scripts" / "run.py", lines=50)
    _write_text(root / "docs" / "guide.md", lines=40)

    manifest = partition_source([root], workers=2)
    mapping = manifest.file_to_shard

    assert manifest.effective_workers == 2
    # src/server/auth + src/server/session share a shard (parent kept whole).
    assert mapping["src/server/auth/auth.py"] == mapping["src/server/session/session.py"]
    # The whole src subtree stayed together...
    src_shards = {
        mapping[file]
        for shard in manifest.shards
        for file in shard.files
        if file.startswith("src/")
    }
    assert src_shards == {mapping["src/server/auth/auth.py"]}
    # ... and every top-level dir is in exactly one shard.
    for prefix in ("src/", "scripts/", "docs/"):
        dir_files = [file for file in mapping if file.startswith(prefix)]
        assert len({mapping[file] for file in dir_files}) == 1


def test_locality_splits_only_at_necessary_boundary(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_code(root / "src" / "server" / "auth" / "auth.py", lines=60)
    _write_code(root / "src" / "server" / "session" / "session.py", lines=60)
    _write_code(root / "src" / "core" / "core.py", lines=60)
    _write_code(root / "lib" / "x.py", lines=60)
    _write_code(root / "lib" / "y.py", lines=60)

    manifest = partition_source([root], workers=3)
    mapping = manifest.file_to_shard
    assert manifest.effective_workers == 3

    # src (180) exceeds cap and must split, but the split happens at the
    # deepest boundary that still fits: src/server stays whole.
    server_shards = {mapping[file] for file in mapping if file.startswith("src/server/")}
    assert len(server_shards) == 1
    assert mapping["src/server/auth/auth.py"] == mapping["src/server/session/session.py"]
    # lib and src/core each stayed whole too, and no shard is empty.
    assert len({mapping[file] for file in mapping if file.startswith("lib/")}) == 1
    assert len({mapping[file] for file in mapping if file.startswith("src/core/")}) == 1
    assert len(manifest.shards) == 3
    assert all(shard.files for shard in manifest.shards)


# ---------------------------------------------------------------------------
# 6. .gitignore honoured (nested, negation, anchored, dir, **)
# ---------------------------------------------------------------------------


def _gitignore_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text(
        "/bld/\n*.log\n!keep.log\n**/scratch/**\nvendor-ish/\n",
        encoding="utf-8",
    )
    (root / "src" / ".gitignore").write_text("legacy/*\n!legacy/keep.ts\n", encoding="utf-8")

    _write_code(root / "bld" / "x.ts", 3)  # anchored /bld/ -> ignored
    _write_code(root / "a" / "bld" / "y.ts", 3)  # anchored only at root -> kept
    _write_code(root / "notes.log", 3)  # *.log -> ignored
    _write_code(root / "keep.log", 3)  # !keep.log -> kept
    _write_code(root / "scratch" / "z.ts", 3)  # **/scratch/** -> ignored
    _write_code(root / "a" / "scratch" / "deep" / "z.ts", 3)  # ignored
    _write_code(root / "vendor-ish" / "w.ts", 3)  # unanchored dir rule -> ignored
    _write_code(root / "src" / "legacy" / "old.ts", 3)  # nested legacy/* -> ignored
    _write_code(root / "src" / "legacy" / "keep.ts", 3)  # nested negation -> kept
    _write_code(root / "src" / "legacy" / "sub" / "deep.ts", 3)  # dir pruned
    _write_code(root / "src" / "other.ts", 3)  # kept
    _write_code(root / "deep" / "other.ts", 3)  # kept


def test_gitignore_semantics_honoured(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _gitignore_tree(root)

    inventory = inventory_source([root])
    rels = _rel_entries(inventory)
    for ignored in (
        "bld/x.ts",
        "notes.log",
        "scratch/z.ts",
        "a/scratch/deep/z.ts",
        "vendor-ish/w.ts",
        "src/legacy/old.ts",
        "src/legacy/sub/deep.ts",
    ):
        assert ignored not in rels, ignored
    for kept in (
        "a/bld/y.ts",
        "keep.log",
        "src/legacy/keep.ts",
        "src/other.ts",
        "deep/other.ts",
    ):
        assert kept in rels, kept


def test_gitignore_matcher_rules() -> None:
    scope = parse_gitignore("*.log\n!keep.log\n/build/\nfoo/bar\n**/tmp/**\n", base_rel="")
    assert scope.decide("a.log", is_dir=False) is True
    assert scope.decide("x/y/z.log", is_dir=False) is True
    assert scope.decide("keep.log", is_dir=False) is False  # negation wins
    assert scope.decide("build", is_dir=True) is True  # anchored dir
    assert scope.decide("x/build", is_dir=True) is None  # anchored: root only
    assert scope.decide("foo/bar", is_dir=False) is True  # mid-slash anchored
    assert scope.decide("x/foo/bar", is_dir=False) is None
    assert scope.decide("a/tmp/x.ts", is_dir=False) is True  # ** glob
    assert scope.decide("tmp/x", is_dir=False) is True
    assert scope.decide("x/y/tmp/z/w", is_dir=True) is True
    assert scope.decide("atmp", is_dir=True) is None  # not a match


def test_gitignore_dir_only_and_scope_precedence() -> None:
    scope = parse_gitignore("vendor/\n", base_rel="")
    assert scope.decide("deep/vendor", is_dir=True) is True
    assert scope.decide("vendor/file.ts", is_dir=False) is None  # dir-only skips files
    assert scope.relative("deep/vendor") == "deep/vendor"

    root_rules = parse_gitignore("*.ts\n", base_rel="")
    nested_rules = parse_gitignore("!keep.ts\n", base_rel="src")
    # A deeper .gitignore overrides a shallower one.
    assert chain_decides((root_rules, nested_rules), "src/keep.ts", is_dir=False) is False
    assert chain_decides((root_rules, nested_rules), "src/drop.ts", is_dir=False) is True
    # Outside the nested scope, the root rule stands.
    assert chain_decides((root_rules, nested_rules), "top.ts", is_dir=False) is True
    # Nested scope applies only below its own directory.
    assert nested_rules.relative("src/keep.ts") == "keep.ts"
    assert nested_rules.relative("other.ts") is None


# ---------------------------------------------------------------------------
# 7. Generated / vendor / build excluded
# ---------------------------------------------------------------------------


def test_generated_vendor_build_absent_from_inventory_and_manifest() -> None:
    inventory = inventory_source([FIXTURE])
    rels = _rel_entries(inventory)

    for rel in rels:
        assert not rel.startswith(_EXCLUDED_PREFIXES), rel
    assert not (rels & _EXCLUDED_FILES)

    # Classification still sees vendor/generated (inventory distinguishes
    # classes) but the unit/weight layer excludes them entirely.
    kinds = {entry.rel: entry.kind for entry in inventory.entries}
    assert kinds["vendor/vendor.js"] is FileKind.VENDOR
    assert kinds["generated/api_pb2.py"] is FileKind.GENERATED

    manifest = partition_source([FIXTURE], workers=3)
    shard_files = {file for shard in manifest.shards for file in shard.files}
    assert not any(
        file.startswith(("generated/", "vendor/", "node_modules/", "dist/"))
        or file in _EXCLUDED_FILES
        or file.endswith((".min.js", ".lock"))
        for file in shard_files
    )
    # And the nested legacy/ dir is gone from the inventory already.
    assert not any(rel.startswith("src/legacy/") for rel in rels)


# ---------------------------------------------------------------------------
# 8. More workers than useful units
# ---------------------------------------------------------------------------


def test_more_workers_than_units(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_code(root / "a.py", lines=5)
    _write_code(root / "b.py", lines=5)

    manifest = partition_source([root], workers=5)
    assert manifest.requested_workers == 5
    assert manifest.effective_workers == 2
    assert len(manifest.shards) == 2
    assert all(shard.files for shard in manifest.shards)  # no empty shards


# ---------------------------------------------------------------------------
# 9. Windows vs POSIX normalization
# ---------------------------------------------------------------------------


def test_path_normalization_rules() -> None:
    assert normalize_rel_path(r"src\server\GameServer.ts") == "src/server/GameServer.ts"
    assert normalize_rel_path("src//server/../GameServer.ts") is None
    assert normalize_rel_path("") is None
    assert normalize_rel_path("./a/./b") == "a/b"
    # NFC: decomposed 'e' + combining acute collapses to composed 'é'.
    decomposed = "cafe\u0301.ts"
    assert normalize_rel_path(decomposed) == "caf\u00e9.ts"
    assert unicodedata.is_normalized("NFC", normalize_rel_path(decomposed) or "")
    # Ordering is case-insensitive (case-folded), with the raw string as
    # tiebreak for exact-case ties only.
    assert path_sort_key("a.py") < path_sort_key("B.py")
    assert path_sort_key("b.py") == path_sort_key("b.py")


def test_identical_trees_produce_identical_manifest(tmp_path: Path) -> None:
    def build(base: Path) -> Path:
        root = base / "repo"
        _write_code(root / "src" / "server" / "auth.py", lines=40)
        _write_code(root / "src" / "core.py", lines=60)
        _write_text(root / "docs" / "guide.md", lines=10)
        return root

    first = build(tmp_path / "one")
    second = build(tmp_path / "two")
    assert (
        partition_source([first], workers=2).to_json()
        == partition_source([second], workers=2).to_json()
    )


# ---------------------------------------------------------------------------
# 10. Bulk distribution / single oversized file
# ---------------------------------------------------------------------------


def test_bulk_distributed_across_shards_when_under_one_dir(tmp_path: Path) -> None:
    # Nearly everything lives under src/: with workers=4 the recursive expand
    # rule must distribute the bulk across multiple shards instead of dumping
    # it onto one shard.
    root = tmp_path / "repo"
    for index in range(16):
        _write_code(root / "src" / f"pkg{index:02d}" / "code.py", lines=100)
    _write_text(root / "README.md", lines=4)

    manifest = partition_source([root], workers=4)
    assert manifest.effective_workers == 4
    total = manifest.total_weight
    assert total > 1600
    weights = sorted(shard.weight for shard in manifest.shards)
    # LPT on sixteen equal 100-weight subtrees: every shard carries 400-401.
    assert all(weight >= 400 for weight in weights)
    # No single shard holds >= 75% of the total.
    assert weights[-1] / total < 0.75
    # The bulk under src/ really is spread: src files live in every shard.
    src_shards = {
        manifest.file_to_shard[file] for file in manifest.file_to_shard if file.startswith("src/")
    }
    assert src_shards == {0, 1, 2, 3}
    assert all(shard.files for shard in manifest.shards)


def test_single_oversized_file_may_hold_the_bulk(tmp_path: Path) -> None:
    # The only unit allowed to stay whole above cap is an individual file.
    root = tmp_path / "repo"
    _write_code(root / "huge.py", lines=2000)
    _write_code(root / "a.py", lines=100)
    _write_code(root / "b.py", lines=100)
    _write_code(root / "c.py", lines=100)

    manifest = partition_source([root], workers=4)
    total = manifest.total_weight
    assert total == 2300
    assert manifest.effective_workers == 4
    heaviest = max(manifest.shards, key=lambda shard: shard.weight)
    assert heaviest.weight / total >= 0.75
    # ...and that shard contains exactly the one unsplittable file.
    assert heaviest.files == ("huge.py",)
    assert heaviest.weight == 2000
    # No file is duplicated anywhere.
    all_files = [file for shard in manifest.shards for file in shard.files]
    assert len(all_files) == len(set(all_files)) == 4


# ---------------------------------------------------------------------------
# 11. Empty repository
# ---------------------------------------------------------------------------


def test_empty_repo_returns_empty_manifest(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir(parents=True)
    manifest = partition_source([root], workers=4)
    assert manifest.requested_workers == 4
    assert manifest.effective_workers == 0
    assert manifest.shards == ()
    assert manifest.file_to_shard == {}
    assert manifest.total_weight == 0
    assert manifest.total_loc == 0
    assert manifest.to_json() == manifest.to_json()  # stable, no exception

    # A non-directory / missing root is skipped with a note, not an error.
    missing = tmp_path / "does-not-exist"
    inventory = inventory_source([root, missing])
    assert any("non-directory root" in note for note in inventory.notes)


# ---------------------------------------------------------------------------
# 12. Symlink cycle / duplicates
# ---------------------------------------------------------------------------


def test_symlink_cycle_not_followed_and_no_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "real").mkdir(parents=True)
    _write_code(root / "real" / "a.py", lines=5)
    try:
        (root / "loop").symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    inventory = inventory_source([root])
    rels = _rel_entries(inventory)
    # The self-referential directory symlink is never descended.
    assert not any("loop/" in rel or rel.startswith("loop") for rel in rels)
    assert "real/a.py" in rels

    # A symlinked file pointing at the same inode is counted once (first path
    # in sorted order wins) — deterministic across runs.
    alias = root / "alias.py"
    try:
        alias.symlink_to(root / "real" / "a.py")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    first = inventory_source([root])
    second = inventory_source([root])
    assert _rel_entries(first) == _rel_entries(second)
    assert len(_rel_entries(first) & {"alias.py", "real/a.py"}) == 1
    manifest = partition_source([root], workers=1)
    shard_files = {file for shard in manifest.shards for file in shard.files}
    assert len(shard_files & {"alias.py", "real/a.py"}) == 1


# ---------------------------------------------------------------------------
# 13. Unreadable file
# ---------------------------------------------------------------------------


def test_unreadable_file_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    _write_code(root / "locked.py", lines=5)
    _write_code(root / "open.py", lines=5)

    real_read = inventory_module.read_bytes

    def failing_read(path: Path, limit: int | None = None) -> bytes:
        if Path(path).name == "locked.py":
            raise PermissionError(13, "Permission denied", str(path))
        return real_read(path, limit)

    monkeypatch.setattr(inventory_module, "read_bytes", failing_read)
    inventory = inventory_source([root])

    rels = _rel_entries(inventory)
    assert "locked.py" not in rels
    assert "open.py" in rels
    assert any(
        "skipping unreadable file" in note and "locked.py" in note for note in inventory.notes
    )

    # The manifest still completes for the readable remainder.
    manifest = partition_source([root], workers=2)
    assert manifest.effective_workers == 1
    assert manifest.shards[0].files == ("open.py",)


# ---------------------------------------------------------------------------
# 14. Stability under root reordering + multi-root display
# ---------------------------------------------------------------------------


def test_root_order_permutation_and_multi_root_prefix(tmp_path: Path) -> None:
    first = tmp_path / "repo-a"
    second = tmp_path / "repo-b"
    _write_code(first / "app" / "a.py", lines=20)
    _write_code(second / "lib" / "b.py", lines=30)

    forward = partition_source([first, second], workers=2)
    backward = partition_source([second, first], workers=2)
    assert forward.to_json() == backward.to_json()

    shard_files = {file for shard in forward.shards for file in shard.files}
    assert "repo-a/app/a.py" in shard_files
    assert "repo-b/lib/b.py" in shard_files

    # A single root keeps plain repo-relative paths (source_inspect spelling).
    single = partition_source([first], workers=1)
    assert single.shards[0].files == ("app/a.py",)


def test_same_root_name_disambiguation(tmp_path: Path) -> None:
    checkout_one = tmp_path / "x" / "checkout"
    checkout_two = tmp_path / "y" / "checkout"
    _write_code(checkout_one / "a.py", lines=5)
    _write_code(checkout_two / "b.py", lines=5)

    manifest = partition_source([checkout_one, checkout_two], workers=2)
    shard_files = {file for shard in manifest.shards for file in shard.files}
    # Canonical root order decides who keeps the bare name and who gets -2.
    assert len(shard_files) == 2
    assert sum(file.startswith("checkout/") for file in shard_files) == 1
    assert sum(file.startswith("checkout-2/") for file in shard_files) == 1


# ---------------------------------------------------------------------------
# Unicode / auto-workers / config toggles / ordering
# ---------------------------------------------------------------------------


def test_nfc_manifest_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    decomposed_name = "cafe\u0301.py"
    target = root / decomposed_name
    try:
        _write_code(target, lines=5)
        on_disk = next(child.name for child in root.iterdir())
    except OSError:
        pytest.skip("filesystem refused the decomposed name")

    manifest = partition_source([root], workers=1)
    shard_files = [file for shard in manifest.shards for file in shard.files]
    assert len(shard_files) == 1
    manifest_name = shard_files[0]
    assert unicodedata.is_normalized("NFC", manifest_name)
    # Whether or not the fs kept the decomposed spelling, the manifest spells
    # the NFC form of the actual on-disk name.
    assert manifest_name == unicodedata.normalize("NFC", on_disk)


def test_auto_workers_and_config_default(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_code(root / "a.py", lines=5)
    _write_code(root / "b.py", lines=5)
    _write_code(root / "c.py", lines=5)
    config = PartitionConfig(default_workers=2)
    manifest = partition_source([root], workers=0, config=config)
    assert manifest.requested_workers == 2
    assert manifest.effective_workers == 2


def test_exclude_tests_config(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_code(root / "app.py", lines=10)
    (root / "tests").mkdir()
    _write_code(root / "tests" / "test_app.py", lines=10)

    included = partition_source([root], workers=2)
    assert any(file.startswith("tests/") for shard in included.shards for file in shard.files)

    excluded = partition_source([root], workers=2, config=PartitionConfig(exclude_tests=True))
    shard_files = {file for shard in excluded.shards for file in shard.files}
    assert not any(file.startswith("tests/") for file in shard_files)
    assert "app.py" in shard_files


def test_shard_files_sorted_casefolded(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    for name in ("Zebra.ts", "apple.py", "Banana.py"):
        _write_code(root / name, lines=2)
    manifest = partition_source([root], workers=1)
    files = manifest.shards[0].files
    assert files == tuple(sorted(files, key=path_sort_key))
    assert files[0] == "apple.py"  # casefold order, not raw ASCII order


def test_json_serialization_shape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_code(root / "a.py", lines=10)
    manifest = partition_source([root], workers=1)
    payload = json.loads(manifest.to_json())
    assert payload["requested_workers"] == 1
    assert payload["effective_workers"] == 1
    assert payload["total_weight"] == 10
    assert payload["total_loc"] == 10
    assert payload["shards"][0]["files"] == ["a.py"]
    assert payload["shards"][0]["weight"] == 10
    assert payload["file_to_shard"] == {"a.py": 0}
    assert payload["notes"] == []


# ---------------------------------------------------------------------------
# Hardening: oversized files, notes plumbing, composability, prefix registry
# ---------------------------------------------------------------------------


def test_oversized_source_file_still_partitioned_with_note(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    # ~2 KiB of real Python: far above the deliberately tiny 1 KiB threshold.
    _write_code(root / "server.py", lines=220)
    _write_code(root / "small.py", lines=5)

    config = PartitionConfig(max_file_bytes=1024)
    manifest = partition_source([root], workers=2, config=config)
    assert manifest.effective_workers == 2

    shard_files = {file for shard in manifest.shards for file in shard.files}
    assert "server.py" in shard_files
    assert "small.py" in shard_files
    # The oversized file sits in exactly one shard with its streamed weight.
    assert manifest.total_weight == 225
    assert manifest.total_loc == 225
    assert len({manifest.file_to_shard["server.py"]}) == 1
    assert any(
        "oversized file counted by streaming LOC" in note and "server.py" in note
        for note in manifest.notes
    )
    # Deterministic across reruns, note included.
    assert manifest.to_json() == partition_source([root], workers=2, config=config).to_json()


def test_streaming_loc_parity_with_full_read(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    content = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(90))
    (root / "big.py").write_text(content, encoding="utf-8")

    default_manifest = partition_source([root], workers=1)
    streamed_manifest = partition_source(
        [root], workers=1, config=PartitionConfig(max_file_bytes=1024)
    )
    # Same content must produce the same LOC/weight whether read whole or
    # streamed line-by-line.
    assert streamed_manifest.total_loc == default_manifest.total_loc == 180
    assert streamed_manifest.total_weight == default_manifest.total_weight == 180
    assert streamed_manifest.shards[0].files == default_manifest.shards[0].files
    assert any("oversized" in note for note in streamed_manifest.notes)
    assert default_manifest.notes == ()


def test_oversized_data_artifact_skipped_with_note(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "dump.csv").write_text("\n".join(f"row{i}" for i in range(400)), encoding="utf-8")
    _write_code(root / "app.py", lines=5)

    manifest = partition_source([root], workers=1, config=PartitionConfig(max_file_bytes=512))
    shard_files = [file for shard in manifest.shards for file in shard.files]
    assert "dump.csv" not in shard_files  # oversized DATA: conservative limit
    assert "app.py" in shard_files
    assert any("oversized data artifact" in note and "dump.csv" in note for note in manifest.notes)


def test_partition_units_is_pure_over_units() -> None:
    # The assignment stage never touches the filesystem: hand-built units only.
    units = [
        PartitionUnit(
            root_index=0, rel="a.py", display="a.py", kind=FileKind.SOURCE, loc=40, weight=40
        ),
        PartitionUnit(
            root_index=0, rel="b.py", display="b.py", kind=FileKind.SOURCE, loc=40, weight=40
        ),
        PartitionUnit(
            root_index=0, rel="c.py", display="c.py", kind=FileKind.SOURCE, loc=40, weight=40
        ),
    ]
    first = partition_units(units, workers=2)
    second = partition_units(list(reversed(units)), workers=2)
    assert first.to_json() == second.to_json()
    assert first.effective_workers == 2
    assert {file for shard in first.shards for file in shard.files} == {"a.py", "b.py", "c.py"}
    assert first.notes == ()
    # Caller-supplied notes are surfaced on the manifest verbatim.
    with_note = partition_units(units, workers=2, notes=("note-1",))
    assert with_note.notes == ("note-1",)


def test_display_root_names_never_collide_with_generated_suffixes(tmp_path: Path) -> None:
    first = tmp_path / "a" / "checkout"
    second = tmp_path / "b" / "checkout"
    third = tmp_path / "c" / "checkout-2"
    for root in (first, second, third):
        _write_code(root / "a.py", lines=5)

    # Unit-level guarantee: the generated prefix never collides with another
    # root's natural name (the real checkout-2) or another generated prefix.
    names = display_root_names([first, second, third])
    assert len(names) == 3
    assert len(set(names)) == 3

    manifest = partition_source([first, second, third], workers=3)
    display_paths = [file for shard in manifest.shards for file in shard.files]
    assert len(display_paths) == len(set(display_paths)) == 3
    prefixes = {path.split("/", 1)[0] for path in display_paths}
    assert prefixes == set(names)
    # Every included file maps to exactly one unique display path.
    assert set(manifest.file_to_shard) == set(display_paths)


# ---------------------------------------------------------------------------
# Unit coverage: LOC, classification, weights
# ---------------------------------------------------------------------------


def test_count_loc_python() -> None:
    text = "x = 1\n# comment\n\ny = 2\n\ndef f():\n    return 3\n"
    assert count_loc(text, "python") == 4
    # Docstrings count as block comments.
    docstring = '"""A docstring.\nMore text.\n"""\nvalue = 1\n'
    assert count_loc(docstring, "python") == 1


def test_count_loc_c_style() -> None:
    text = "// header comment\nconst a = 1; // trailing\n/* block\nstill block\n*/\nconst b = 2;\n"
    assert count_loc(text, "c_style") == 2


def test_count_loc_blank_only_for_other_languages() -> None:
    text = "# a shell comment\n\nreal=1\n"
    assert language_for(".sh") == "text"
    assert count_loc(text, "text") == 2  # comments count when unlisted


def test_classification() -> None:
    cases = {
        "src/app.py": FileKind.SOURCE,
        "src/foo.ts": FileKind.SOURCE,
        "tests/unit/test_auth.py": FileKind.TEST,
        "src/auth_test.go": FileKind.TEST,
        "src/auth.test.ts": FileKind.TEST,
        "src/conftest.py": FileKind.TEST,
        "vendor/lib.js": FileKind.VENDOR,
        "third_party/x.c": FileKind.VENDOR,
        "generated/api.pb.go": FileKind.GENERATED,
        "src/gen/model_pb.ts": FileKind.GENERATED,
        "docs/openapi.yaml": FileKind.SCHEMA,
        "api/schema.sql": FileKind.SCHEMA,
        "protos/auth.proto": FileKind.SCHEMA,
        "data/dump.csv": FileKind.DATA,
        "migrations/001.sql": FileKind.DATA,
        "README.md": FileKind.TEXT,
        "src/notes.txt": FileKind.TEXT,
    }
    for rel, expected in cases.items():
        assert classify_file(rel) is expected, rel


def test_effective_weight_rounding() -> None:
    assert effective_weight(100, Fraction(1, 1)) == 100
    assert effective_weight(10, Fraction(1, 5)) == 2
    assert effective_weight(3, Fraction(1, 5)) == 1  # floor of 1
    assert effective_weight(6, Fraction(1, 4)) == 2
    assert effective_weight(40, Fraction(1, 20)) == 2
    assert effective_weight(0, Fraction(1, 1)) == 1
