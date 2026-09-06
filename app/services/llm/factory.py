from app.core.config import get_settings
from .mock import MockProvider
from .openai_compatible import OpenAICompatibleProvider
from .provider import LLMProvider


def get_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "mock":
        return MockProvider()
    if settings.llm_provider in {"openai", "openai-compatible"}:
        missing = [name for name, value in {
            "LLM_BASE_URL": settings.llm_base_url,
            "LLM_API_KEY": settings.llm_api_key,
            "LLM_MODEL": settings.llm_model,
        }.items() if not value]
        if missing:
            raise RuntimeError(f"Missing required LLM configuration: {', '.join(missing)}")
        return OpenAICompatibleProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    raise RuntimeError(f"Unsupported LLM provider: {settings.llm_provider}")
