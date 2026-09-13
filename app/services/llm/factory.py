"""Provider selection.

The mock provider exists so the contract tests can run without a model. It
returns a fixed string -- "Offline contract-test result. No clinical inference
was performed." -- and zero confidence.

On staging that string reached a patient. LLM_PROVIDER was set to "mock", the
factory honoured it without comment, and the result travelled the whole governed
pipeline looking exactly like a real one: same response shape, same persistence,
same history entry. Nothing downstream could tell it apart, because nothing
downstream was told.

So provider selection is now a policy decision, not a lookup. A non-clinical
provider is permitted only where non-clinical answers are expected -- local
development and the test suite. Anywhere else, asking for one is a configuration
error and the service says so instead of quietly answering.

Failing here is the safe direction: a caller that gets an explicit error tells
the patient the service is unavailable, which is true. A caller that gets a mock
result tells the patient a diagnosis, which is not.
"""

from app.core.config import get_settings
from .mock import MockProvider
from .openai_compatible import OpenAICompatibleProvider
from .provider import LLMProvider


class ProviderConfigurationError(RuntimeError):
    """The configured provider cannot be used in this environment."""


# Providers that do not perform clinical inference. Keep this list explicit: a
# stub added later must be named here to inherit the guard.
NON_CLINICAL_PROVIDERS = {"mock"}

# Environments where a non-clinical answer is the expected outcome. Anything
# else -- staging, production, or an unset/unknown value -- requires a real
# provider. Unknown counts as "not permitted" deliberately: a typo in
# ENVIRONMENT must not silently re-open the mock path.
NON_CLINICAL_ENVIRONMENTS = {"development", "test", "testing", "local"}

REAL_PROVIDERS = {"openai", "openai-compatible"}


def _normalise(value: str | None) -> str:
    return (value or "").strip().lower()


def configuration_error() -> str | None:
    """Why the current configuration is unusable, or None if it is fine."""
    settings = get_settings()
    provider = _normalise(settings.llm_provider)
    environment = _normalise(settings.environment)

    if provider in NON_CLINICAL_PROVIDERS:
        if environment in NON_CLINICAL_ENVIRONMENTS:
            return None
        return (
            f"LLM_PROVIDER={provider!r} does not perform clinical inference and "
            f"is not permitted in ENVIRONMENT={environment!r}. Configure a real "
            f"provider ({', '.join(sorted(REAL_PROVIDERS))}) with LLM_BASE_URL, "
            f"LLM_API_KEY and LLM_MODEL."
        )

    if provider in REAL_PROVIDERS:
        missing = [name for name, value in {
            "LLM_BASE_URL": settings.llm_base_url,
            "LLM_API_KEY": settings.llm_api_key,
            "LLM_MODEL": settings.llm_model,
        }.items() if not value]
        if missing:
            return f"Missing required LLM configuration: {', '.join(missing)}"
        return None

    return f"Unsupported LLM provider: {settings.llm_provider!r}"


def provider_policy() -> dict:
    """Describe the current selection without constructing anything.

    Health reports this, so a misconfigured deployment is identifiable from
    outside without reading logs and without exposing any credential.
    """
    settings = get_settings()
    provider = _normalise(settings.llm_provider)
    environment = _normalise(settings.environment)
    return {
        "llm_provider": provider,
        "environment": environment,
        "performs_clinical_inference": provider not in NON_CLINICAL_PROVIDERS,
        "non_clinical_provider_permitted_here":
            environment in NON_CLINICAL_ENVIRONMENTS,
        "configuration_valid": configuration_error() is None,
    }


def get_provider() -> LLMProvider:
    problem = configuration_error()
    if problem:
        raise ProviderConfigurationError(problem)

    settings = get_settings()
    if _normalise(settings.llm_provider) in NON_CLINICAL_PROVIDERS:
        return MockProvider()

    return OpenAICompatibleProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
