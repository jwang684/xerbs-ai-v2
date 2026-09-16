from .provider import INTERVIEW, LLMProvider, ProviderResult


class MockProvider(LLMProvider):
    async def generate_recommendation(self, *, text_input: str, symptoms: list[str], goals: list[str], constraints: list[str], image_data: str | None, language: str) -> ProviderResult:
        return ProviderResult(
            summary="Offline contract-test result. No clinical inference was performed.",
            pattern_hypotheses=[],
            formula_candidates=[],
            uncertainty_flags=["MOCK_PROVIDER", "PRACTITIONER_REVIEW_REQUIRED"],
            model_confidence=0.0,
            provider="mock",
            model="mock-v1",
        )

    async def generate_interview(self, *, text_input: str, symptoms: list[str], language: str, on_display_text=None, carry_state: dict | None = None, known_domains: list | None = None) -> ProviderResult:
        """Offline stand-in. Proposes nothing, so the coverage floor supplies
        the questions -- which is the correct behaviour for a provider that
        performs no clinical inference."""
        return ProviderResult(
            summary="Offline contract-test interview. No clinical inference was performed.",
            pattern_hypotheses=[],
            formula_candidates=[],
            uncertainty_flags=["MOCK_PROVIDER", "PRACTITIONER_REVIEW_REQUIRED"],
            model_confidence=0.0,
            provider="mock",
            model="mock-v1",
            inference_purpose=INTERVIEW,
        )
