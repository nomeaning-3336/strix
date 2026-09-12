"""LiteLLM model-name resolution for local cost estimates.

A model id only names a model; it does not say who bills for it. A direct
DeepSeek endpoint reached over the OpenAI-compatible API is configured as
``openai/<name>`` (or a bare custom name), and LiteLLM's cost map is keyed by
provider, so nothing in the map matches and the run silently reports $0.00.
Resolution therefore takes the configured endpoint into account: when the base
URL is DeepSeek's own API, the request is attributed to DeepSeek for *pricing
only* — the request routing is untouched.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast


# Prefixes that describe the wire protocol or a local gateway rather than the
# billing provider. Stripped before matching a provider-keyed price entry.
_PROTOCOL_PREFIXES = ("litellm/", "any-llm/", "openai/")

# DeepSeek's own API. Reached over the OpenAI-compatible protocol, so the
# model id arrives as ``openai/deepseek-flash`` and cannot be priced by name.
_DEEPSEEK_HOST_SUFFIX = ".deepseek.com"

# Model ids DeepSeek accepts that are not (or not yet) keys in LiteLLM's cost
# map, mapped to the canonical name it does carry prices for. Only aliases that
# resolve to a real price entry are honoured; anything else stays unpriced
# rather than being guessed at.
_DEEPSEEK_MODEL_ALIASES: dict[str, str] = {
    "deepseek-flash": "deepseek-v4-flash",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4.1-flash": "deepseek-v4-flash",
    "deepseek-pro": "deepseek-v4-pro",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "deepseek-v4.1-pro": "deepseek-v4-pro",
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
    "deepseek-coder": "deepseek-coder",
    "deepseek-v3": "deepseek-v3",
    "deepseek-v3.2": "deepseek-v3.2",
}


def _model_cost() -> dict[str, dict[str, Any]]:
    import litellm

    return cast("dict[str, dict[str, Any]]", getattr(litellm, "model_cost"))  # noqa: B009


def _strip_protocol_prefix(model: str) -> str:
    normalized = model.strip()
    for prefix in _PROTOCOL_PREFIXES:
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix)
    return normalized


def is_deepseek_endpoint(base_url: str | None) -> bool:
    """True when ``base_url`` points at DeepSeek's own API."""
    if not isinstance(base_url, str) or not base_url.strip():
        return False
    try:
        from urllib.parse import urlsplit

        host = (urlsplit(base_url.strip()).hostname or "").lower()
    except ValueError:
        return False
    return host == _DEEPSEEK_HOST_SUFFIX.lstrip(".") or host.endswith(_DEEPSEEK_HOST_SUFFIX)


def configured_api_base() -> str | None:
    """The configured LLM endpoint, or ``None`` when settings are unavailable.

    Imported lazily: ``strix.config`` pulls in modules that reach back into cost
    reporting, so this must not run at import time. Never returns or logs a key
    — only the endpoint, which is not a credential.
    """
    try:
        from strix.config.loader import load_settings  # breaks a config import cycle

        return load_settings().llm.api_base
    except Exception:  # noqa: BLE001 - pricing must never break a scan
        return None


def canonical_deepseek_model(model: str | None) -> str | None:
    """Return the ``deepseek/<name>`` id this model prices as, or ``None``.

    Only names that LiteLLM actually carries a price for are returned, so an
    unknown DeepSeek-side name stays unpriced instead of being billed at a
    neighbouring model's rate.
    """
    if not model:
        return None
    bare = _strip_protocol_prefix(model)
    if bare.startswith("deepseek/"):
        bare = bare.split("/", 1)[1]
    elif "/" in bare:
        # A provider-qualified id for someone else is not a DeepSeek name.
        return None

    name = _DEEPSEEK_MODEL_ALIASES.get(bare, bare)
    if not name.startswith("deepseek"):
        return None
    candidate = f"deepseek/{name}"
    return candidate if candidate in _model_cost() else None


@lru_cache(maxsize=256)
def resolve_priced_model(model: str, base_url: str | None = None) -> str | None:
    """Return a provider-qualified, priceable name for ``model``, or ``None``.

    ``base_url`` is the endpoint the request actually goes to. When it is
    DeepSeek's API the id is attributed to DeepSeek before the generic lookup,
    which is what makes an ``openai/<deepseek-model>`` configuration priceable.
    Every other endpoint keeps the previous name-based behaviour, so OpenRouter
    and ordinary OpenAI-compatible routes are unaffected.
    """
    if not isinstance(model, str) or not model.strip():
        return None
    if is_deepseek_endpoint(base_url):
        deepseek = canonical_deepseek_model(model)
        if deepseek is not None:
            return deepseek
    return resolve_litellm_model(model)


@lru_cache(maxsize=512)
def resolve_litellm_model(model: str) -> str | None:
    """Return a provider-qualified model name that LiteLLM can price."""
    try:
        normalized = _strip_protocol_prefix(model)
        if not normalized:
            return None

        model_cost = _model_cost()
        bare_entry = model_cost.get(normalized)
        if "/" not in normalized and isinstance(bare_entry, dict):
            provider = bare_entry.get("litellm_provider")
            if isinstance(provider, str) and provider:
                return f"{provider}/{normalized}"
        if "/" in normalized and isinstance(bare_entry, dict):
            return normalized

        names = [normalized]
        if "/" in normalized:
            names.append(normalized.rsplit("/", 1)[-1])
        for name in names:
            matches = sorted(key for key in model_cost if key.endswith(f"/{name}"))
            if not matches:
                continue
            prices = {
                (
                    model_cost[key].get("input_cost_per_token"),
                    model_cost[key].get("output_cost_per_token"),
                )
                for key in matches
                if isinstance(model_cost.get(key), dict)
            }
            if len(matches) == 1 or len(prices) == 1:
                return matches[0]
        return None  # noqa: TRY300
    except Exception:  # noqa: BLE001
        return None
