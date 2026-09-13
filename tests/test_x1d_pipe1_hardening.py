"""X1D-PIPE1 pre-switch hardening: make switching off mock operationally safe.

tests/test_x1d_pipe1_provider_policy.py pins *which* provider may be selected.
This file pins what happens around that decision when it goes wrong, because
that is what makes the switch risky rather than the switch itself:

  * the application must still start and stay diagnosable -- the provider used
    to be built at import, so one wrong value meant the app could not be
    imported, uvicorn never started, /health went down with it, and the
    operator got a crash loop carrying no signal at all;

  * the deployment must report itself as not production-ready, so nobody reads
    a 200 from /health as evidence that patient-facing inference is real;

  * a broken real configuration must never yield a mock;

  * provider failures must not carry upstream text. Base44GenerationService
    writes str(exc) onto the generation row and returns it to xerbs-core, so a
    raw body would be persisted and shipped across a service boundary -- and an
    OpenAI 401 body echoes part of the key back. With LLM_PROVIDER=mock there is
    no key and nothing to leak, which is precisely why this is fixed before the
    switch and not after it.
"""

import pytest

from app.core.config import Settings, get_settings
from app.services.llm import factory
from app.services.llm.mock import MockProvider
from app.services.llm.openai_compatible import OpenAICompatibleProvider


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def configure(monkeypatch, **values):
    settings = Settings(**values)
    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    return settings


# ----------------------------------------------------------------------
# Health stays up and tells the truth
# ----------------------------------------------------------------------

class TestHealthSurvivesMisconfiguration:
    @staticmethod
    def _client(monkeypatch, **values):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.api import base44
        from app.services.llm import factory as factory_module

        settings = Settings(**values)
        monkeypatch.setattr(factory_module, "get_settings", lambda: settings)
        monkeypatch.setattr(base44, "_service", None)   # force re-resolution
        return TestClient(app)

    BROKEN = [
        {"llm_provider": "mock", "environment": "staging"},
        {"llm_provider": "mock", "environment": "production"},
        {"llm_provider": "openai", "environment": "staging"},
        {"llm_provider": "openai", "environment": "staging",
         "llm_base_url": "https://api.example.test/v1",
         "llm_model": "some-model"},                               # no key
        {"llm_provider": "openai", "environment": "staging",
         "llm_base_url": "https://api.example.test/v1",
         "llm_api_key": "not-a-real-key"},                         # no model
        {"llm_provider": "nonsense", "environment": "staging"},
    ]

    @pytest.mark.parametrize("values", BROKEN)
    def test_health_is_reachable_under_every_misconfiguration(
            self, monkeypatch, values):
        response = self._client(monkeypatch, **values).get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @pytest.mark.parametrize("values", BROKEN)
    def test_health_reports_not_production_ready(self, monkeypatch, values):
        block = self._client(monkeypatch, **values).get(
            "/health").json()["clinical_inference"]
        assert block["usable"] is False
        assert block["configuration_status"] == "NOT_PRODUCTION_READY"
        assert block["problem"]

    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_readiness_is_503_for_mock_in_a_real_environment(
            self, monkeypatch, environment):
        """The required invariant, expressed as a status code."""
        response = self._client(monkeypatch, llm_provider="mock",
                                environment=environment).get("/health/clinical")
        assert response.status_code == 503
        assert response.json()["ready_for_clinical_use"] is False
        assert response.json()["performs_clinical_inference"] is False

    def test_readiness_is_200_when_correctly_configured(self, monkeypatch):
        response = self._client(
            monkeypatch, llm_provider="openai", environment="staging",
            llm_base_url="https://api.example.test/v1",
            llm_api_key="not-a-real-key",
            llm_model="some-model").get("/health/clinical")
        assert response.status_code == 200
        assert response.json()["ready_for_clinical_use"] is True

    def test_health_never_echoes_configuration_values(self, monkeypatch):
        secret = "sk-must-never-appear-in-health"
        client = self._client(monkeypatch, llm_provider="openai",
                              environment="staging",
                              llm_base_url="https://api.example.test/v1",
                              llm_api_key=secret,
                              llm_model="secret-model-name")
        for path in ("/health", "/health/clinical"):
            body = client.get(path).text
            assert secret not in body
            assert "api.example.test" not in body
            assert "secret-model-name" not in body

    def test_generation_fails_closed_while_health_stays_up(self, monkeypatch):
        """Both halves of the requirement, in one deployment state."""
        client = self._client(monkeypatch, llm_provider="mock",
                              environment="staging")
        assert client.get("/health").status_code == 200

        response = client.post(
            "/api/v1/integrations/base44/generate",
            json={"organization_id": "xerbs-core",
                  "request_id": "req-hardening",
                  "intake": {"text_input": "发热、咽痛", "language": "zh"}},
            headers={"Idempotency-Key": "pipe1-hardening-fail-closed"})

        assert response.status_code == 503
        assert "Offline contract-test result" not in response.text


# ----------------------------------------------------------------------
# Never fall back to mock
# ----------------------------------------------------------------------

class TestNoFallbackToMock:
    @pytest.mark.parametrize("values", [
        {"llm_provider": "openai", "environment": "staging"},
        {"llm_provider": "openai", "environment": "production",
         "llm_base_url": "https://api.example.test/v1"},
        {"llm_provider": "openai-compatible", "environment": "staging",
         "llm_api_key": "not-a-real-key"},
        {"llm_provider": "nonsense", "environment": "staging"},
        {"llm_provider": "", "environment": "staging"},
    ])
    def test_a_broken_real_configuration_never_yields_a_mock(
            self, monkeypatch, values):
        configure(monkeypatch, **values)
        with pytest.raises(factory.ProviderConfigurationError):
            provider = factory.get_provider()
            assert not isinstance(provider, MockProvider)   # unreachable

    def test_the_configuration_contract_names_are_unchanged(self):
        """Renaming any of these silently breaks every deployment."""
        fields = set(Settings.model_fields)
        for name in ("llm_provider", "llm_base_url", "llm_api_key", "llm_model"):
            assert name in fields

    @pytest.mark.parametrize("value", ["openai", "openai-compatible"])
    def test_both_documented_values_resolve_the_real_provider(
            self, monkeypatch, value):
        configure(monkeypatch, llm_provider=value, environment="staging",
                  llm_base_url="https://api.example.test/v1",
                  llm_api_key="not-a-real-key", llm_model="some-model")
        assert isinstance(factory.get_provider(), OpenAICompatibleProvider)

    def test_mock_is_still_constructible_for_development(self, monkeypatch):
        """MockProvider must not be removed; tests and local dev need it."""
        configure(monkeypatch, llm_provider="mock", environment="development")
        assert isinstance(factory.get_provider(), MockProvider)


# ----------------------------------------------------------------------
# Provider failures must be safe to persist and return
# ----------------------------------------------------------------------

class TestProviderErrorsCarryNoUpstreamText:
    KEY = "sk-must-never-appear-in-an-error"

    @pytest.fixture
    def provider(self):
        return OpenAICompatibleProvider(
            base_url="https://api.example.test/v1",
            api_key=self.KEY, model="some-model")

    def _call(self, provider):
        import asyncio
        return asyncio.run(provider.generate_recommendation(
            text_input="发热、咽痛", symptoms=[], goals=[], constraints=[],
            image_data=None, language="zh"))

    @staticmethod
    def _install_client(monkeypatch, post):
        from app.services.llm import openai_compatible as mod

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return await post()

        monkeypatch.setattr(mod.httpx, "AsyncClient", _Client)
        return mod

    def test_an_http_error_body_is_never_returned(self, provider, monkeypatch):
        leaky = ('{"error":{"message":"Incorrect API key provided: '
                 + self.KEY + '. See the docs."}}')

        class _Resp:
            status_code = 401
            is_error = True
            text = leaky

            def json(self):
                import json
                return json.loads(leaky)

        async def post():
            return _Resp()

        mod = self._install_client(monkeypatch, post)

        with pytest.raises(mod.ProviderCallError) as excinfo:
            self._call(provider)

        message = str(excinfo.value)
        assert self.KEY not in message
        assert "Incorrect API key provided" not in message
        assert "401" in message          # the actionable part survives

    def test_a_transport_error_does_not_return_the_endpoint(
            self, provider, monkeypatch):
        import httpx

        async def post():
            raise httpx.ConnectError(
                "connection failed to "
                "https://api.example.test/v1/chat/completions")

        mod = self._install_client(monkeypatch, post)

        with pytest.raises(mod.ProviderCallError) as excinfo:
            self._call(provider)
        message = str(excinfo.value)
        assert "api.example.test" not in message
        assert "ConnectError" in message

    def test_non_json_model_output_is_not_echoed(self, provider, monkeypatch):
        """Model output derives from patient text; it is not error material."""
        payload = {"choices": [{"message": {"content": "not json at all 患者原话"}}]}

        class _Resp:
            status_code = 200
            is_error = False
            text = "{}"

            def json(self):
                return payload

        async def post():
            return _Resp()

        mod = self._install_client(monkeypatch, post)

        with pytest.raises(mod.ProviderCallError) as excinfo:
            self._call(provider)
        assert "患者原话" not in str(excinfo.value)

    def test_the_error_remains_a_runtime_error(self):
        """Existing callers catch RuntimeError and map it to 503."""
        from app.services.llm.openai_compatible import ProviderCallError
        assert issubclass(ProviderCallError, RuntimeError)
