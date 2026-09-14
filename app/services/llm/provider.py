from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProviderUsage:
    """Token accounting reported by the provider. Observational only.

    Every field is optional because availability is a property of the provider
    and the model, not something this service can assume. A missing count stays
    None: it is never estimated from text length, because an invented number
    that looks like a measurement is worse than an honest gap -- it would be
    billed against in reports and nobody would know it was fiction.

    Nothing here may influence pattern hypotheses, formula selection, corpus
    resolution, safety, eligibility or recommendation state.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # Present only on providers/models that report them; absent elsewhere.
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in (
            "prompt_tokens", "completion_tokens", "total_tokens",
            "cached_input_tokens", "reasoning_tokens"))


@dataclass
class ProviderResult:
    summary: str
    pattern_hypotheses: list[dict] = field(default_factory=list)
    formula_candidates: list[dict] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)
    model_confidence: float = 0.0
    provider: str = "unknown"
    model: str = "unknown"
    # X1D-TELEMETRY1: operational metadata, carried alongside the clinical
    # result rather than inside it. Defaults keep every existing provider and
    # test valid without change.
    usage: ProviderUsage | None = None
    provider_latency_ms: float | None = None


class LLMProvider(ABC):
    @abstractmethod
    async def generate_recommendation(self, *, text_input: str, symptoms: list[str], goals: list[str], constraints: list[str], image_data: str | None, language: str) -> ProviderResult:
        raise NotImplementedError
