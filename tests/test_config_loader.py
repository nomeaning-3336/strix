"""Tests for strix.config.loader: JSON overrides, alias resolution, persistence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import AliasChoices, Field, ValidationError
from pydantic.fields import FieldInfo

from strix.config import loader
from strix.config.settings import ContextSettings


if TYPE_CHECKING:
    from pathlib import Path


_LLM_ENV_KEYS = [
    "STRIX_LLM",
    # Role-aware routing: part of the linked LLM connection tuple.
    "STRIX_ROOT_LLM",
    "STRIX_SUBAGENT_LLM",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "LLM_API_BASE",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "LITELLM_BASE_URL",
    "OLLAMA_API_BASE",
    "STRIX_REASONING_EFFORT",
    "STRIX_FORCE_REQUIRED_TOOL_CHOICE",
    "LLM_TIMEOUT",
    "PERPLEXITY_API_KEY",
    # RuntimeSettings
    "STRIX_IMAGE",
    "STRIX_RUNTIME_BACKEND",
    # TelemetrySettings
    "STRIX_TELEMETRY",
]


@pytest.fixture(autouse=True)
def _reset_loader_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module globals and clear known env vars for deterministic runs."""
    for key in _LLM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(loader, "_cached", None)
    monkeypatch.setattr(loader, "_override", None)


# --------------------------------------------------------------------------- #
# _read_json_overrides
# --------------------------------------------------------------------------- #


def test_read_json_overrides_missing_file(tmp_path: Path) -> None:
    assert loader._read_json_overrides(tmp_path / "nope.json") == {}


def test_read_json_overrides_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "cli-config.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert loader._read_json_overrides(path) == {}


def test_read_json_overrides_non_dict_env(tmp_path: Path) -> None:
    path = tmp_path / "cli-config.json"
    path.write_text(json.dumps({"env": ["not", "a", "dict"]}), encoding="utf-8")
    assert loader._read_json_overrides(path) == {}


def test_read_json_overrides_maps_to_nested_settings(tmp_path: Path) -> None:
    path = tmp_path / "cli-config.json"
    path.write_text(
        json.dumps({"env": {"STRIX_LLM": "my-model", "PERPLEXITY_API_KEY": "pk"}}),
        encoding="utf-8",
    )
    assert loader._read_json_overrides(path) == {
        "llm": {"model": "my-model"},
        "integrations": {"perplexity_api_key": "pk"},
    }


def test_read_json_overrides_skips_keys_already_in_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_LLM", "from-env")
    path = tmp_path / "cli-config.json"
    path.write_text(json.dumps({"env": {"STRIX_LLM": "from-file"}}), encoding="utf-8")
    # env wins -> the JSON value is not surfaced as an init kwarg.
    assert loader._read_json_overrides(path) == {}


def test_read_json_overrides_env_wins_across_field_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # api_key resolves from AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"). The env
    # sets one alias while the persisted file holds another. Env must still win, so
    # the stale file value must not be surfaced as an init kwarg (which outranks env).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    path = tmp_path / "cli-config.json"
    path.write_text(json.dumps({"env": {"LLM_API_KEY": "sk-file"}}), encoding="utf-8")
    assert loader._read_json_overrides(path) == {}


def test_read_json_overrides_env_wins_case_insensitively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Settings use case_sensitive=False, so a lowercase env var also counts as set.
    monkeypatch.setenv("strix_llm", "from-env")
    path = tmp_path / "cli-config.json"
    path.write_text(json.dumps({"env": {"STRIX_LLM": "from-file"}}), encoding="utf-8")
    assert loader._read_json_overrides(path) == {}


def test_read_json_overrides_uses_json_when_no_alias_in_environ(tmp_path: Path) -> None:
    # No alias of api_key is set in the environment -> the file value is used, even
    # when it is stored under a non-first alias.
    path = tmp_path / "cli-config.json"
    path.write_text(json.dumps({"env": {"OPENAI_API_KEY": "sk-file"}}), encoding="utf-8")
    assert loader._read_json_overrides(path) == {"llm": {"api_key": "sk-file"}}


def test_tool_output_max_bytes_rejects_sub_notice_values() -> None:
    with pytest.raises(ValidationError):
        ContextSettings(STRIX_TOOL_OUTPUT_MAX_BYTES=64)


def test_tool_output_max_bytes_accepts_floor() -> None:
    assert ContextSettings(STRIX_TOOL_OUTPUT_MAX_BYTES=1024).tool_output_max_bytes == 1024


# --------------------------------------------------------------------------- #
# _aliases_for
# --------------------------------------------------------------------------- #


def test_aliases_for_simple_alias() -> None:
    finfo = FieldInfo(alias="SIMPLE_ALIAS")
    assert loader._aliases_for(finfo) == ["SIMPLE_ALIAS"]


def test_aliases_for_alias_choices() -> None:
    finfo: FieldInfo = Field(  # type: ignore[assignment]
        default=None,
        validation_alias=AliasChoices("FIRST", "SECOND"),
    )
    assert loader._aliases_for(finfo) == ["FIRST", "SECOND"]


def test_aliases_for_string_validation_alias() -> None:
    finfo: FieldInfo = Field(default=None, validation_alias="STR_ALIAS")  # type: ignore[assignment]
    assert loader._aliases_for(finfo) == ["STR_ALIAS"]


def test_aliases_for_no_alias() -> None:
    assert loader._aliases_for(FieldInfo()) == []


# --------------------------------------------------------------------------- #
# apply_config_override + load_settings round-trip
# --------------------------------------------------------------------------- #


def test_apply_override_and_load_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cli-config.json"
    path.write_text(
        json.dumps({"env": {"STRIX_LLM": "round-trip-model", "PERPLEXITY_API_KEY": "pk"}}),
        encoding="utf-8",
    )

    loader.apply_config_override(path)
    settings = loader.load_settings()

    assert settings.llm.model == "round-trip-model"
    assert settings.integrations.perplexity_api_key == "pk"
    # Second call is memoized -> same object.
    assert loader.load_settings() is settings


def test_apply_config_override_invalidates_cache(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    first.write_text(json.dumps({"env": {"STRIX_LLM": "first-model"}}), encoding="utf-8")
    second = tmp_path / "second.json"
    second.write_text(json.dumps({"env": {"STRIX_LLM": "second-model"}}), encoding="utf-8")

    loader.apply_config_override(first)
    assert loader.load_settings().llm.model == "first-model"

    loader.apply_config_override(second)
    assert loader.load_settings().llm.model == "second-model"


# --------------------------------------------------------------------------- #
# persist_current
# --------------------------------------------------------------------------- #


def test_persist_current_writes_env_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIX_LLM", "persisted-model")
    target = tmp_path / "sub" / "cli-config.json"
    loader.apply_config_override(target)

    loader.persist_current()

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "env": {"STRIX_LLM": "persisted-model"}
    }


def test_persist_current_sets_0600_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIX_LLM", "persisted-model")
    target = tmp_path / "cli-config.json"
    loader.apply_config_override(target)

    loader.persist_current()

    assert target.stat().st_mode & 0o777 == 0o600


# --------------------------------------------------------------------------- #
# persist_current merges into the stored env block instead of overwriting it
# --------------------------------------------------------------------------- #


def test_persist_current_keeps_file_values_when_env_is_unset(tmp_path: Path) -> None:
    """A run configured from the file must not erase it on the way out."""
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps({"env": {"STRIX_LLM": "file-model", "LLM_API_KEY": "file-key"}}),
        encoding="utf-8",
    )
    loader.apply_config_override(target)
    assert loader.load_settings().llm.model == "file-model"

    loader.persist_current()

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "env": {"STRIX_LLM": "file-model", "LLM_API_KEY": "file-key"}
    }


def test_persist_current_env_overrides_file_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps({"env": {"STRIX_LLM": "file-model", "PERPLEXITY_API_KEY": "file-pplx"}}),
        encoding="utf-8",
    )
    loader.apply_config_override(target)
    monkeypatch.setenv("PERPLEXITY_API_KEY", "env-pplx")

    loader.persist_current()

    # The untouched model survives; the overridden unrelated key is updated.
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "env": {"STRIX_LLM": "file-model", "PERPLEXITY_API_KEY": "env-pplx"}
    }


def test_persist_current_empty_env_clears_file_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cli-config.json"
    target.write_text(json.dumps({"env": {"STRIX_LLM": "file-model"}}), encoding="utf-8")
    loader.apply_config_override(target)
    monkeypatch.setenv("STRIX_LLM", "")

    loader.persist_current()

    assert json.loads(target.read_text(encoding="utf-8")) == {"env": {}}


def test_persist_current_empty_primary_alias_does_not_save_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty LLM_API_KEY must not persist a non-empty OPENAI_API_KEY sibling."""
    target = tmp_path / "cli-config.json"
    target.write_text(json.dumps({"env": {"PERPLEXITY_API_KEY": "pplx"}}), encoding="utf-8")
    loader.apply_config_override(target)
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sibling-key")

    assert loader.load_settings().llm.api_key == ""

    loader.persist_current()

    assert json.loads(target.read_text(encoding="utf-8")) == {"env": {"PERPLEXITY_API_KEY": "pplx"}}


def test_persist_current_replaces_corrupt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cli-config.json"
    target.write_text("{not json", encoding="utf-8")
    loader.apply_config_override(target)
    monkeypatch.setenv("STRIX_LLM", "recovered-model")

    loader.persist_current()

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "env": {"STRIX_LLM": "recovered-model"}
    }


# --------------------------------------------------------------------------- #
# Linked LLM connection: a shell change drops the whole stored connection
# --------------------------------------------------------------------------- #


def test_linked_llm_model_change_drops_stored_key_and_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps(
            {
                "env": {
                    "STRIX_LLM": "file-model",
                    "LLM_API_KEY": "file-key",
                    "LLM_API_BASE": "http://file-base",
                }
            }
        ),
        encoding="utf-8",
    )
    loader.apply_config_override(target)
    monkeypatch.setenv("STRIX_LLM", "shell-model")

    assert loader.load_settings().llm.model == "shell-model"
    # The stored key/base belong to the previous model; they must not load.
    assert loader.load_settings().llm.api_key is None
    assert loader.load_settings().llm.api_base is None

    loader.persist_current()

    assert json.loads(target.read_text(encoding="utf-8")) == {"env": {"STRIX_LLM": "shell-model"}}


def test_linked_llm_key_change_drops_stored_model_and_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps(
            {
                "env": {
                    "STRIX_LLM": "file-model",
                    "LLM_API_KEY": "file-key",
                    "LLM_API_BASE": "http://file-base",
                }
            }
        ),
        encoding="utf-8",
    )
    loader.apply_config_override(target)
    monkeypatch.setenv("LLM_API_KEY", "shell-key")

    loader.persist_current()

    assert json.loads(target.read_text(encoding="utf-8")) == {"env": {"LLM_API_KEY": "shell-key"}}


def test_linked_llm_base_change_drops_stored_model_and_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps({"env": {"STRIX_LLM": "file-model", "LLM_API_KEY": "file-key"}}),
        encoding="utf-8",
    )
    loader.apply_config_override(target)
    monkeypatch.setenv("LLM_API_BASE", "http://shell-base")

    loader.persist_current()

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "env": {"LLM_API_BASE": "http://shell-base"}
    }


def test_linked_llm_unchanged_keeps_stored_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-exporting the same value is not a change: the connection survives."""
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps({"env": {"STRIX_LLM": "same-model", "LLM_API_KEY": "file-key"}}),
        encoding="utf-8",
    )
    loader.apply_config_override(target)
    monkeypatch.setenv("STRIX_LLM", "same-model")

    loader.persist_current()

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "env": {"STRIX_LLM": "same-model", "LLM_API_KEY": "file-key"}
    }


def test_unrelated_stored_settings_survive_a_connection_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps(
            {
                "env": {
                    "STRIX_LLM": "file-model",
                    "LLM_API_KEY": "file-key",
                    "PERPLEXITY_API_KEY": "pplx",
                }
            }
        ),
        encoding="utf-8",
    )
    loader.apply_config_override(target)
    monkeypatch.setenv("STRIX_LLM", "shell-model")

    loader.persist_current()

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "env": {"STRIX_LLM": "shell-model", "PERPLEXITY_API_KEY": "pplx"}
    }


# --------------------------------------------------------------------------- #
# Role-aware routing: all three model vars are part of the connection tuple
# --------------------------------------------------------------------------- #


def test_root_model_change_drops_stored_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps({"env": {"STRIX_ROOT_LLM": "old-root", "LLM_API_KEY": "old-key"}}),
        encoding="utf-8",
    )
    loader.apply_config_override(target)
    monkeypatch.setenv("STRIX_ROOT_LLM", "new-root")

    loader.persist_current()

    assert json.loads(target.read_text(encoding="utf-8")) == {"env": {"STRIX_ROOT_LLM": "new-root"}}


def test_subagent_model_change_does_not_resurrect_stored_key_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repointing workers at another endpoint must not reuse the old credentials."""
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps(
            {
                "env": {
                    "STRIX_ROOT_LLM": "gpt-5.6-sol",
                    "STRIX_SUBAGENT_LLM": "deepseek-v4-pro",
                    "LLM_API_KEY": "old-key",
                    "LLM_API_BASE": "http://old-base",
                }
            }
        ),
        encoding="utf-8",
    )
    loader.apply_config_override(target)
    monkeypatch.setenv("STRIX_SUBAGENT_LLM", "another-endpoint-model")

    loader.persist_current()

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "env": {"STRIX_SUBAGENT_LLM": "another-endpoint-model"}
    }


def test_role_models_load_independently_when_connection_is_unchanged(
    tmp_path: Path,
) -> None:
    """Without a shell change both role models resolve from the file."""
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps(
            {
                "env": {
                    "STRIX_ROOT_LLM": "gpt-5.6-sol",
                    "STRIX_SUBAGENT_LLM": "deepseek-v4-pro",
                    "LLM_API_KEY": "file-key",
                }
            }
        ),
        encoding="utf-8",
    )
    loader.apply_config_override(target)

    settings = loader.load_settings()

    assert settings.llm.root_model == "gpt-5.6-sol"
    assert settings.llm.subagent_model == "deepseek-v4-pro"
    assert settings.llm.api_key == "file-key"


def test_read_json_overrides_drops_stale_connection_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading must not mix a shell model with the file's stored key/base."""
    target = tmp_path / "cli-config.json"
    target.write_text(
        json.dumps({"env": {"STRIX_LLM": "old-model", "LLM_API_KEY": "old-key"}}),
        encoding="utf-8",
    )
    loader.apply_config_override(target)
    monkeypatch.setenv("STRIX_LLM", "new-model")

    settings = loader.load_settings()

    assert settings.llm.model == "new-model"
    assert settings.llm.api_key is None
