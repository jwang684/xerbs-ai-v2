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
    # X1D-CLARIFY1: raw, unvalidated clarification proposals. They are model
    # output derived from patient text, so they are data to be checked, never
    # instructions to follow. Nothing renders these directly -- the
    # deterministic validator decides what a patient sees.
    clarification_proposals: list[dict] = field(default_factory=list)
    # X1D-CLARIFY3: how many proposals were dropped here for not being JSON
    # objects. Without this count, "the model proposed nothing" and "the model
    # proposed several things of the wrong shape" are the same observation,
    # and they call for opposite fixes.
    clarification_proposals_discarded: int = 0
    # X1D-LEGACYDIAG3.2: which kind of call produced this result. Operational
    # only -- it never reaches the clinical contract, and nothing downstream
    # branches on it except telemetry.
    inference_purpose: str = "FULL_REASONING"
    # X1D-LEGACYDIAG3.2: raw interview block, typed later by the assembler so a
    # validation failure is contained there rather than at the provider. Empty
    # on a full-reasoning call.
    interview: dict = field(default_factory=dict)
    # X1D-LEGACYDIAG2: raw reasoning block, typed later by the assembler so a
    # validation failure is contained there rather than at the provider.
    clinical_reasoning: dict = field(default_factory=dict)


INTERVIEW = "INTERVIEW"
FULL_REASONING = "FULL_REASONING"


class LLMProvider(ABC):
    @abstractmethod
    async def generate_recommendation(self, *, text_input: str, symptoms: list[str], goals: list[str], constraints: list[str], image_data: str | None, language: str) -> ProviderResult:
        raise NotImplementedError

    async def generate_interview(self, *, text_input: str, symptoms: list[str], language: str) -> ProviderResult:
        """The small call: what should we ask next, and why.

        Concrete rather than abstract so an existing provider keeps working
        without change; one that has not implemented it simply cannot be
        routed to interview mode, which the assembler checks before routing.

        The result must never carry formula candidates or a reasoning
        envelope. That is asserted structurally in the tests rather than left
        to each implementation's good behaviour.
        """
        raise NotImplementedError
