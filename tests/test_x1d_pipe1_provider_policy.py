"""X1D-PIPE1: a non-clinical provider must not become the clinical provider.

What happened on staging
------------------------
LLM_PROVIDER was "mock". The factory honoured it, and the mock provider's fixed
string -- "Offline contract-test result. No clinical inference was performed."
-- travelled the entire governed pipeline and was shown to a real patient as a
diagnosis, with model confidence 0 and no candidate formula.

Nothing downstream could have caught it. The mock response has the same shape,
the same contract version, and the same persistence path as a real one. The
only place the difference is knowable is here, at selection time.

These tests pin the three properties that make that impossible to repeat:

  1. a non-clinical provider is refused outside development and test;
  2. a real provider with incomplete configuration fails explicitly, and never
     falls back to the mock;
  3. the mock stays available where it is legitimately needed, or the contract
     suite has nothing to run against.

The guard fails closed by design. An explicit error becomes "the service is
unavailable", which is true. A mock result becomes "here is your diagnosis",
which is not.
"""

import pytest

from app.core.config import Settings, get_settings
from app.services.llm import factory
from app.services.llm.mock import MockProvider
from app.services.llm.openai_compatible import OpenAICompatibleProvider


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings is lru_cached; each case needs its own configuration."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def configure(monkeypatch, **values):
    """Install a Settings built from explicit values, bypassing .env."""
    settings = Settings(**values)
    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    return settings


# ----------------------------------------------------------------------
# 1. The mock provider cannot silently become the staging clinical provider
# ----------------------------------------------------------------------

class TestNonClinicalProviderIsRefusedInRealEnvironments:
    @pytest.mark.parametrize("environment", ["staging", "production", "prod"])
    def test_mock_is_refused(self, monkeypatch, environment):
        configure(monkeypatch, llm_provider="mock", environment=environment)
        with pytest.raises(factory.ProviderConfigurationError) as excinfo:
            factory.get_provider()
        message = str(excinfo.value)
        assert "mock" in message
        assert environment in message
        # The message must tell the operator what to do about it.
        assert "LLM_API_KEY" in message

    def test_an_unknown_environment_is_treated_as_real(self, monkeypatch):
        """A typo in ENVIRONMENT must not re-open the mock path."""
        configure(monkeypatch, llm_provider="mock", environment="stagng")
        with pytest.raises(factory.ProviderConfigurationError):
            factory.get_provider()

    def test_an_empty_environment_is_treated_as_real(self, monkeypatch):
        configure(monkeypatch, llm_provider="mock", environment="")
        with pytest.raises(factory.ProviderConfigurationError):
            factory.get_provider()

    def test_the_refusal_does_not_return_a_provider_at_all(self, monkeypatch):
        """Fail closed: no object is handed back that could answer anyway."""
        configure(monkeypatch, llm_provider="mock", environment="production")
        with pytest.raises(factory.ProviderConfigurationError):
            provider = factory.get_provider()
            assert provider is None  # unreachable; stated for intent


# ----------------------------------------------------------------------
# 2. A real provider with missing configuration fails loudly
# ----------------------------------------------------------------------

class TestMissingRealProviderConfigurationFailsExplicitly:
    @pytest.mark.parametrize("omit", ["llm_base_url", "llm_api_key", "llm_model"])
    def test_each_missing_field_is_named(self, monkeypatch, omit):
        values = {
            "llm_provider": "openai",
            "environment": "staging",
            "llm_base_url": "https://api.example.test/v1",
            "llm_api_key": "not-a-real-key",
            "llm_model": "some-model",
        }
        values[omit] = None
        configure(monkeypatch, **values)

        with pytest.raises(factory.ProviderConfigurationError) as excinfo:
            factory.get_provider()
        assert omit.upper() in str(excinfo.value)

    def test_missing_configuration_never_falls_back_to_mock(self, monkeypatch):
        """The dangerous failure mode is substitution, not refusal."""
        configure(monkeypatch, llm_provider="openai", environment="staging")
        with pytest.raises(factory.ProviderConfigurationError):
            factory.get_provider()

    def test_an_unsupported_provider_name_is_refused(self, monkeypatch):
        configure(monkeypatch, llm_provider="definitely-not-a-provider",
                  environment="staging")
        with pytest.raises(factory.ProviderConfigurationError) as excinfo:
            factory.get_provider()
        assert "Unsupported" in str(excinfo.value)

    def test_a_complete_real_configuration_is_accepted(self, monkeypatch):
        configure(monkeypatch, llm_provider="openai", environment="staging",
                  llm_base_url="https://api.example.test/v1",
                  llm_api_key="not-a-real-key", llm_model="some-model")
        provider = factory.get_provider()
        assert isinstance(provider, OpenAICompatibleProvider)


# ----------------------------------------------------------------------
# 3. The mock stays usable where it is meant to be used
# ----------------------------------------------------------------------

class TestMockRemainsAvailableForTests:
    @pytest.mark.parametrize("environment",
                             ["development", "test", "testing", "local"])
    def test_mock_is_permitted(self, monkeypatch, environment):
        configure(monkeypatch, llm_provider="mock", environment=environment)
        assert isinstance(factory.get_provider(), MockProvider)

    def test_the_suite_itself_still_gets_a_mock(self):
        """conftest pins ENVIRONMENT=test; without that, nothing here runs."""
        assert isinstance(factory.get_provider(), MockProvider)


# ----------------------------------------------------------------------
# 4. The policy is inspectable without constructing a provider
# ----------------------------------------------------------------------

class TestPolicyIsReportable:
    def test_policy_describes_a_misconfigured_deployment(self, monkeypatch):
        configure(monkeypatch, llm_provider="mock", environment="staging")
        policy = factory.provider_policy()
        assert policy["performs_clinical_inference"] is False
        assert policy["non_clinical_provider_permitted_here"] is False
        assert policy["configuration_valid"] is False

    def test_policy_describes_a_correct_deployment(self, monkeypatch):
        configure(monkeypatch, llm_provider="openai", environment="staging",
                  llm_base_url="https://api.example.test/v1",
                  llm_api_key="not-a-real-key", llm_model="some-model")
        policy = factory.provider_policy()
        assert policy["performs_clinical_inference"] is True
        assert policy["configuration_valid"] is True

    def test_policy_never_reports_a_credential(self, monkeypatch):
        """Health is unauthenticated; it must not leak configuration values."""
        secret = "sk-this-value-must-never-appear"
        configure(monkeypatch, llm_provider="openai", environment="staging",
                  llm_base_url="https://api.example.test/v1",
                  llm_api_key=secret, llm_model="some-model")
        blob = repr(factory.provider_policy()) + repr(factory.configuration_error())
        assert secret not in blob
        assert "api.example.test" not in blob
        assert "some-model" not in blob


# ----------------------------------------------------------------------
# 5. The consumer-facing endpoint refuses rather than answering with a mock
# ----------------------------------------------------------------------

class TestGenerateEndpointFailsClosed:
    def test_generate_returns_503_when_provider_is_not_clinical(self, monkeypatch):
        """A 503 is relayable as "unavailable". A mock body is not."""
        from fastapi.testclient import TestClient
        from app.main import app
        from app.api import base44

        configure(monkeypatch, llm_provider="mock", environment="staging")
        monkeypatch.setattr(base44, "_service", None)  # force re-resolution

        client = TestClient(app)
        response = client.post(
            "/api/v1/integrations/base44/generate",
            json={
                "organization_id": "xerbs-core",
                "request_id": "req-pipe1-test",
                "intake": {"text_input": "发热、咽痛", "language": "zh"},
            },
            headers={"Idempotency-Key": "pipe1-fail-closed-test"},
        )

        assert response.status_code == 503
        body = response.text
        assert "Offline contract-test result" not in body
        assert "No clinical inference was performed" not in body
