from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import litellm
from agents.usage import Usage

from strix.report.pricing import (
    canonical_deepseek_model,
    configured_api_base,
    is_deepseek_endpoint,
    resolve_litellm_model,
    resolve_priced_model,
)
from strix.report.usage import LLMUsageLedger


def test_resolves_common_bare_model_names() -> None:
    resolve_litellm_model.cache_clear()
    assert resolve_litellm_model("deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    assert resolve_litellm_model("openai/deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    assert resolve_litellm_model("grok-4.5") == "xai/grok-4.5"
    # MiniMax-M3 is sold by several LiteLLM providers at different prices, so
    # the resolver must not guess from its bare name. A provider-qualified
    # model remains deterministic.
    assert resolve_litellm_model("minimax/MiniMax-M3") == "minimax/MiniMax-M3"


def test_resolver_returns_none_for_unresolvable_model() -> None:
    resolve_litellm_model.cache_clear()
    assert resolve_litellm_model("provider/not-a-real-model") is None


def test_ledger_uses_estimate_when_routed_provider_reports_no_cost() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()

    with patch("litellm.completion_cost", return_value=0.42):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")

    assert ledger.total_cost == 0.42


def test_ledger_prefers_observed_cost_over_estimate() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()

    with patch("litellm.completion_cost", return_value=0.42):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")
    ledger.record_observed_cost(0.17)

    assert ledger.total_cost == 0.17


def test_hydrated_estimate_continues_accumulating_new_estimates() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()
    ledger.hydrate({"cost": 0.42})

    with patch("litellm.completion_cost", return_value=0.17):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")

    assert ledger.total_cost == 0.59


def test_zero_cost_disables_both_observed_and_estimated_costs() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()
    ledger.zero_cost = True

    with patch("litellm.completion_cost", return_value=0.42) as estimate:
        ledger.record(agent_id="a", usage=usage, model="deepseek-v4-flash")
        ledger.record_observed_cost(1.0)

    estimate.assert_not_called()
    assert ledger.total_cost == 0.0


def test_resolver_uses_provider_when_bare_entry_has_one() -> None:
    original = litellm.model_cost
    litellm.model_cost = {
        "example": {
            "litellm_provider": "example-provider",
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
        }
    }
    try:
        resolve_litellm_model.cache_clear()
        assert resolve_litellm_model("example") == "example-provider/example"
    finally:
        litellm.model_cost = original
        resolve_litellm_model.cache_clear()


def test_resolver_does_not_guess_between_differently_priced_providers() -> None:
    original = litellm.model_cost
    litellm.model_cost = {
        "provider-a/example": {
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
        },
        "provider-b/example": {
            "input_cost_per_token": 3.0,
            "output_cost_per_token": 4.0,
        },
    }
    try:
        resolve_litellm_model.cache_clear()
        assert resolve_litellm_model("example") is None
    finally:
        litellm.model_cost = original
        resolve_litellm_model.cache_clear()


# --------------------------------------------------------------------------- #
# Direct DeepSeek over the OpenAI-compatible endpoint
#
# A direct DeepSeek configuration is spelled openai/<name> because DeepSeek
# serves the OpenAI wire protocol. LiteLLM's cost map is provider-keyed, so the
# id resolves to nothing and the run reported $0.00 with tokens recorded. The
# endpoint decides the provider for pricing; routing is untouched.
# --------------------------------------------------------------------------- #

_DEEPSEEK_BASE = "https://api.deepseek.com/v1"


def test_is_deepseek_endpoint_matches_deepseek_hosts_only() -> None:
    assert is_deepseek_endpoint("https://api.deepseek.com/v1")
    assert is_deepseek_endpoint("https://API.DeepSeek.com")
    assert is_deepseek_endpoint("https://api.deepseek.com:443/v1/")
    assert not is_deepseek_endpoint("https://openrouter.ai/api/v1")
    assert not is_deepseek_endpoint("https://api.openai.com/v1")
    # A lookalike host must not be attributed to DeepSeek.
    assert not is_deepseek_endpoint("https://api.deepseek.com.evil.example/v1")
    assert not is_deepseek_endpoint(None)
    assert not is_deepseek_endpoint("")


def test_direct_deepseek_endpoint_prices_openai_prefixed_model() -> None:
    resolve_priced_model.cache_clear()
    assert (
        resolve_priced_model("openai/deepseek-flash", _DEEPSEEK_BASE)
        == "deepseek/deepseek-v4-flash"
    )
    # Without the endpoint there is nothing to attribute it to.
    assert resolve_priced_model("openai/deepseek-flash", "https://gateway.internal/v1") is None


def test_deepseek_model_aliases_normalize_to_the_priced_canonical_name() -> None:
    resolve_priced_model.cache_clear()
    for alias in ("deepseek-flash", "deepseek-v4.1-flash", "openai/deepseek-v4.1-flash"):
        assert resolve_priced_model(alias, _DEEPSEEK_BASE) == "deepseek/deepseek-v4-flash", alias
    # Names DeepSeek serves that LiteLLM already keys stay valid.
    assert resolve_priced_model("deepseek-chat", _DEEPSEEK_BASE) == "deepseek/deepseek-chat"
    assert resolve_priced_model("deepseek-reasoner", _DEEPSEEK_BASE) == "deepseek/deepseek-reasoner"


def test_unknown_deepseek_model_is_not_guessed_at_a_neighbouring_rate() -> None:
    """An unpriced model must stay unpriced rather than borrow a price."""
    resolve_priced_model.cache_clear()
    assert resolve_priced_model("openai/deepseek-brand-new-9", _DEEPSEEK_BASE) is None
    assert canonical_deepseek_model("brand-new-model") is None


def test_openrouter_and_ordinary_endpoints_keep_their_resolution() -> None:
    resolve_priced_model.cache_clear()
    openrouter = "https://openrouter.ai/api/v1"
    assert (
        resolve_priced_model("openrouter/deepseek/deepseek-v4-flash", openrouter)
        == "openrouter/deepseek/deepseek-v4-flash"
    )
    assert (
        resolve_priced_model("openrouter/anthropic/claude-3.5-sonnet", openrouter)
        == "openrouter/anthropic/claude-3.5-sonnet"
    )
    # An ordinary OpenAI-compatible endpoint is not assumed to be DeepSeek.
    assert resolve_priced_model("openai/gpt-4o", "https://api.openai.com/v1") == "openai/gpt-4o"
    # Nor is every openai/-prefixed model OpenAI billing.
    assert resolve_priced_model("openai/gpt-4o", _DEEPSEEK_BASE) == "openai/gpt-4o"


def _entry(*, cached: int, uncached: int, output: int) -> Usage:
    return Usage(
        requests=1,
        input_tokens=cached + uncached,
        output_tokens=output,
        total_tokens=cached + uncached + output,
        input_tokens_details={"cached_tokens": cached, "cache_write_tokens": 0},
    )


def _ledger_usage(*, cached: int, uncached: int, output: int) -> Usage:
    usage = _entry(cached=cached, uncached=uncached, output=output)
    usage.request_usage_entries = [usage]
    return usage


def test_direct_deepseek_ledger_prices_with_every_usage_bucket() -> None:
    """All three buckets reach the pricer under the attributed model id."""
    ledger = LLMUsageLedger()
    usage = _ledger_usage(cached=800, uncached=200, output=50)
    captured: list[dict[str, object]] = []

    def _fake_cost(*, completion_response: dict[str, object], model: str) -> float:
        captured.append({"model": model, "usage": completion_response["usage"]})
        return 0.25

    with (
        patch("strix.report.usage.configured_api_base", return_value=_DEEPSEEK_BASE),
        patch("litellm.completion_cost", side_effect=_fake_cost),
    ):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-flash")

    assert ledger.total_cost == 0.25
    assert captured, "the pricer was never reached"
    assert captured[0]["model"] == "deepseek/deepseek-v4-flash"
    payload = cast("dict[str, object]", captured[0]["usage"])
    assert payload["prompt_tokens"] == 1000
    assert payload["completion_tokens"] == 50
    # The cache bucket must survive: pricing without it overstates cost.
    details = cast("dict[str, object]", payload["prompt_tokens_details"])
    assert details["cached_tokens"] == 800


def test_direct_deepseek_cost_is_positive_and_cache_aware() -> None:
    """With real prices, cache hits must be cheaper than uncached input."""
    ledger = LLMUsageLedger()
    with patch("strix.report.usage.configured_api_base", return_value=_DEEPSEEK_BASE):
        ledger.record(
            agent_id="a",
            usage=_ledger_usage(cached=1_000_000, uncached=0, output=0),
            model="openai/deepseek-flash",
        )
    cached_cost = ledger.total_cost

    uncached_ledger = LLMUsageLedger()
    with patch("strix.report.usage.configured_api_base", return_value=_DEEPSEEK_BASE):
        uncached_ledger.record(
            agent_id="a",
            usage=_ledger_usage(cached=0, uncached=1_000_000, output=0),
            model="openai/deepseek-flash",
        )

    assert cached_cost > 0
    assert uncached_ledger.total_cost > cached_cost


def test_unknown_custom_endpoint_reports_no_cost_rather_than_inventing_one() -> None:
    ledger = LLMUsageLedger()
    with patch(
        "strix.report.usage.configured_api_base", return_value="https://gateway.internal/v1"
    ):
        ledger.record(
            agent_id="a",
            usage=_ledger_usage(cached=800, uncached=200, output=50),
            model="openai/some-internal-model",
        )
    assert ledger.total_cost == 0.0
    # Tokens are still recorded — only the cost is unknown.
    assert ledger.to_record()["total_tokens"] == 1050


def test_configured_api_base_returns_the_endpoint_and_never_a_key() -> None:
    settings = SimpleNamespace(
        llm=SimpleNamespace(api_base=_DEEPSEEK_BASE, api_key="sk-secret-value")
    )
    with patch("strix.config.loader.load_settings", return_value=settings):
        assert configured_api_base() == _DEEPSEEK_BASE
        assert "sk-secret-value" not in str(configured_api_base())
