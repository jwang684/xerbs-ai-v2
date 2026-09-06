from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProviderResult:
    summary: str
    pattern_hypotheses: list[dict] = field(default_factory=list)
    formula_candidates: list[dict] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)
    model_confidence: float = 0.0
    provider: str = "unknown"
    model: str = "unknown"


class LLMProvider(ABC):
    @abstractmethod
    async def generate_recommendation(self, *, text_input: str, symptoms: list[str], goals: list[str], constraints: list[str], image_data: str | None, language: str) -> ProviderResult:
        raise NotImplementedError
