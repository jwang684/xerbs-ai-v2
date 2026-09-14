"""Model pricing, kept deliberately away from clinical decision code.

This module exists so that a dollar figure never has to be computed inside the
recommendation path. It knows nothing about patients, patterns or formulas; it
turns token counts into an estimate and nothing else.

Pricing is a static registry, not a live lookup. Fetching prices at runtime
would put a network call on the path of every inference, make cost depend on
whether that call succeeded, and make historical numbers irreproducible. A
table that is occasionally stale is easier to reason about than a number whose
provenance changes underneath you.

PRICING VERSION: 2026-09-14
SOURCE: https://developers.openai.com/api/docs/models/gpt-5.4-mini
  gpt-5.4-mini -- input $0.75 / 1M tokens, cached input $0.075 / 1M tokens,
                  output $4.5 / 1M tokens (standard endpoints).
  Regional processing (data residency) endpoints carry a 10% uplift for
  GPT-5.4 Mini; this registry prices the standard endpoint, which is what
  LLM_BASE_URL points at. Re-check when that changes.

An unknown model yields None, never a guess. A wrong price silently applied to
months of usage is a worse outcome than an obvious gap.
"""

from dataclasses import dataclass

PRICING_VERSION = "2026-09-14"
PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.4-mini"

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None


# Keyed by the exact model id sent in the request payload.
MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-5.4-mini": ModelPricing(
        input_per_million=0.75,
        output_per_million=4.5,
        cached_input_per_million=0.075,
    ),
}


def pricing_for(model: str | None) -> ModelPricing | None:
    if not model:
        return None
    return MODEL_PRICING.get(str(model).strip())


def estimate_cost_usd(
    *,
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cached_input_tokens: int | None = None,
) -> float | None:
    """Estimate the cost of one call, or None when it cannot be known.

    Returns None -- never 0.0 as a stand-in -- when the model is unpriced or no
    token counts were reported. Zero is a real answer meaning "this cost
    nothing"; using it for "unknown" would quietly understate spend.

    Cached input is billed at its own lower rate where the provider reports it,
    and those tokens are subtracted from the billable prompt count so they are
    not charged twice.
    """
    pricing = pricing_for(model)
    if pricing is None:
        return None
    if prompt_tokens is None and completion_tokens is None:
        return None

    prompt = int(prompt_tokens or 0)
    completion = int(completion_tokens or 0)
    cached = int(cached_input_tokens or 0)

    # Cached tokens are reported as a subset of prompt_tokens.
    cached = max(0, min(cached, prompt))
    uncached_prompt = prompt - cached

    cost = (uncached_prompt / _PER_MILLION) * pricing.input_per_million
    cost += (completion / _PER_MILLION) * pricing.output_per_million
    if cached and pricing.cached_input_per_million is not None:
        cost += (cached / _PER_MILLION) * pricing.cached_input_per_million
    elif cached:
        # Model priced but no cached rate published: bill at the input rate
        # rather than silently treating those tokens as free.
        cost += (cached / _PER_MILLION) * pricing.input_per_million

    # Sub-cent calls are the norm here; keep enough places to stay meaningful.
    return round(cost, 8)
