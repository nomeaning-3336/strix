"""Action-safety guidance reaches the agent only when a safety mode is active."""

from __future__ import annotations

import pytest

from strix.agents.prompt import render_system_prompt


# Phrased as prohibitions, so they misdescribe the tools an `off`-mode agent actually has.
_SAFETY_ONLY_PHRASES = [
    "ACTION SAFETY POLICY",
    "do not override",
    "blocked as stale",
    "must be split into a creation call",
]


@pytest.mark.parametrize("phrase", _SAFETY_ONLY_PHRASES)
@pytest.mark.parametrize("context", [None, {}, {"safety_mode": "off"}])
def test_safety_guidance_is_absent_without_a_safety_mode(
    phrase: str,
    context: dict[str, str] | None,
) -> None:
    assert phrase not in render_system_prompt(system_prompt_context=context)


@pytest.mark.parametrize("phrase", _SAFETY_ONLY_PHRASES)
@pytest.mark.parametrize("mode", ["guarded", "observe"])
def test_safety_guidance_is_present_in_a_safety_mode(phrase: str, mode: str) -> None:
    assert phrase in render_system_prompt(system_prompt_context={"safety_mode": mode})


def test_browser_skill_carries_no_safety_prohibitions() -> None:
    """The browser skill is always loaded, so mode-specific rules do not belong in it."""
    prompt = render_system_prompt(skills=["agent_browser"], system_prompt_context={})

    assert "agent-browser snapshot" in prompt
    for phrase in _SAFETY_ONLY_PHRASES:
        assert phrase not in prompt


def test_observe_mode_states_its_passive_only_contract() -> None:
    prompt = render_system_prompt(system_prompt_context={"safety_mode": "observe"})

    assert "passive target interaction only" in prompt
    assert "Guarded mode permits" not in prompt
