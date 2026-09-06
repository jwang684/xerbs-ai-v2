from .provider import LLMProvider, ProviderResult


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
