from __future__ import annotations

from typing import Any

from strix.interface.completions import completion_candidates, run_completions


def test_root_completion_candidates() -> None:
    assert completion_candidates(["cl"]) == ["cloud"]
    assert "completions" in completion_candidates([""])


def test_cloud_group_and_alias_candidates() -> None:
    candidates = completion_candidates(["cloud", "work"])
    assert candidates == ["workspace", "workspaces"]


def test_cloud_verb_candidates_include_multiword_prefixes() -> None:
    assert "test-users" in completion_candidates(["cloud", "domains", ""])
    assert completion_candidates(["cloud", "domains", "test-users", "in"]) == [
        "inbox",
        "inbox-message",
    ]


def test_cloud_leaf_flag_candidates_come_from_command_spec() -> None:
    candidates = completion_candidates(["cloud", "scans", "start", "--"])
    assert "--domain-ids" in candidates
    assert "--json" in candidates
    assert "--wait" in candidates


def test_boolean_completion_includes_positive_and_negative_flags() -> None:
    candidates = completion_candidates(["cloud", "billing", "auto-topup", "update", "--"])
    assert "--enabled" in candidates
    assert "--no-enabled" in candidates
    assert "--no-monthly-cap" in candidates


def test_completion_scripts_cover_supported_shells(capsys: Any) -> None:
    for shell in ("zsh", "bash", "fish"):
        assert run_completions([shell]) == 0
        output = capsys.readouterr().out
        assert "completions --candidates" in output


def test_completion_rejects_unknown_shell(capsys: Any) -> None:
    assert run_completions(["powershell"]) == 2
    assert "Choose zsh, bash, or fish" in capsys.readouterr().err
