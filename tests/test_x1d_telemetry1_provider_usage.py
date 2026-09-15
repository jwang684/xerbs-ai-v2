"""X1D-TELEMETRY1: what a real inference cost, recorded without touching it.

The provider response has always carried usage accounting; the client read
choices[0].message.content and discarded the rest. So there was no way to say
what a diagnosis cost, how many tokens it took, or how slow the model was --
only how slow the whole request looked from a browser.

Two properties matter more than the numbers themselves:

  * telemetry is observational. Nothing recorded here may reach pattern
    hypotheses, formula selection, corpus resolution, safety, eligibility or
    recommendation state;

  * telemetry is fail-open, alone among the things this service does. A
    diagnosis that is clinically correct and governed must not be turned into a
    failure because an accounting row could not be written.

A third property is easy to lose later: a replayed idempotent request must not
look like a second call. It never reaches the provider, so it must never
produce a second usage record either -- otherwise reported spend drifts above
what was actually billed.
"""

import json

import pytest

from app.services.llm import pricing
from app.services.llm.provider import ProviderResult, ProviderUsage
from app.services.telemetry import provider_usage as telemetry


API_KEY = "sk-must-never-appear-in-telemetry"
PROMPT_TEXT = "发热、咽痛、头痛、口渴3天"


def result_with(usage=None, latency=12.5, model="gpt-5.4-mini"):
    return ProviderResult(
        summary="临床推理摘要", pattern_hypotheses=[{"name": "风热犯表"}],
        formula_candidates=[], uncertainty_flags=[], model_confidence=0.6,
        provider="openai-compatible", model=model,
        usage=usage, provider_latency_ms=latency,
    )


# ======================================================================
# A-E: usage capture from the real response shape
# ======================================================================

class TestUsageCapture:
    @staticmethod
    def extract(body):
        from app.services.llm.openai_compatible import _extract_usage
        return _extract_usage(body)

    def test_the_openai_usage_object_is_captured(self):
        usage = self.extract({"usage": {"prompt_tokens": 412,
                                        "completion_tokens": 233,
                                        "total_tokens": 645}})
        assert usage is not None

    def test_prompt_tokens_preserved(self):
        assert self.extract({"usage": {"prompt_tokens": 412}}).prompt_tokens == 412

    def test_completion_tokens_preserved(self):
        assert self.extract(
            {"usage": {"completion_tokens": 233}}).completion_tokens == 233

    def test_total_tokens_preserved(self):
        assert self.extract({"usage": {"total_tokens": 645}}).total_tokens == 645

    def test_detailed_fields_captured_when_present(self):
        usage = self.extract({"usage": {
            "prompt_tokens": 412, "completion_tokens": 233,
            "prompt_tokens_details": {"cached_tokens": 128},
            "completion_tokens_details": {"reasoning_tokens": 64}}})
        assert usage.cached_input_tokens == 128
        assert usage.reasoning_tokens == 64

    def test_detailed_fields_absent_is_normal(self):
        """Those objects are model-dependent; absence is not an error."""
        usage = self.extract({"usage": {"prompt_tokens": 1, "completion_tokens": 2}})
        assert usage.cached_input_tokens is None
        assert usage.reasoning_tokens is None

    @pytest.mark.parametrize("body", [
        {}, {"usage": None}, {"usage": "nope"}, {"usage": {}},
        {"usage": {"prompt_tokens": None}},
    ])
    def test_missing_usage_stays_unavailable(self, body):
        """Never estimated from text length: a fake number would be billed."""
        assert self.extract(body) is None

    @pytest.mark.parametrize("bad", ["abc", -5, True, {"a": 1}, [1]])
    def test_malformed_counts_are_discarded_not_coerced(self, bad):
        assert self.extract({"usage": {"prompt_tokens": bad}}) is None


# ======================================================================
# F: latency measured on a monotonic clock, around the provider call only
# ======================================================================

class TestLatencyMeasurement:
    def test_the_provider_uses_a_monotonic_clock(self):
        import ast
        import inspect
        import textwrap
        from app.services.llm.openai_compatible import OpenAICompatibleProvider

        code = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(
            OpenAICompatibleProvider))))
        # Wall clock can step backwards under NTP and yield negative durations.
        # X1D-LEGACYDIAG4.1: the timed span moved into _call_blocking /
        # _call_streaming. Both are checked, which is wider than before.
        assert "time.monotonic()" in code
        assert "time.time()" not in code

    def test_latency_travels_on_the_result(self):
        assert result_with(latency=1234.5).provider_latency_ms == 1234.5

    def test_provider_and_generation_latency_are_separate_fields(self):
        payload = telemetry.build_usage_payload(
            generation_id="gen-1", correlation_id="case-1",
            result=result_with(latency=900.0), generation_latency_ms=1500.0)
        # Collapsing them would make the provider look slower than it is.
        assert payload["provider_latency_ms"] == 900.0
        assert payload["generation_latency_ms"] == 1500.0


# ======================================================================
# G-H: cost estimation
# ======================================================================

class TestCostEstimation:
    def test_known_model_costs_are_exact(self):
        # 1M prompt @ $0.75 + 1M completion @ $4.50
        cost = pricing.estimate_cost_usd(
            model="gpt-5.4-mini", prompt_tokens=1_000_000,
            completion_tokens=1_000_000)
        assert cost == pytest.approx(5.25)

    def test_a_realistic_call_is_sub_cent(self):
        cost = pricing.estimate_cost_usd(
            model="gpt-5.4-mini", prompt_tokens=400, completion_tokens=250)
        assert cost == pytest.approx(400 / 1e6 * 0.75 + 250 / 1e6 * 4.5)
        assert 0 < cost < 0.01

    def test_cached_input_is_billed_at_its_own_rate(self):
        cost = pricing.estimate_cost_usd(
            model="gpt-5.4-mini", prompt_tokens=1000,
            completion_tokens=0, cached_input_tokens=400)
        expected = (600 / 1e6 * 0.75) + (400 / 1e6 * 0.075)
        assert cost == pytest.approx(expected)

    def test_cached_tokens_are_not_billed_twice(self):
        both = pricing.estimate_cost_usd(
            model="gpt-5.4-mini", prompt_tokens=1000, completion_tokens=0,
            cached_input_tokens=1000)
        plain = pricing.estimate_cost_usd(
            model="gpt-5.4-mini", prompt_tokens=1000, completion_tokens=0)
        assert both < plain

    def test_unknown_model_yields_none_not_an_exception(self):
        assert pricing.estimate_cost_usd(
            model="some-unpriced-model", prompt_tokens=100,
            completion_tokens=100) is None

    @pytest.mark.parametrize("model", [None, "", "   "])
    def test_missing_model_yields_none(self, model):
        assert pricing.estimate_cost_usd(
            model=model, prompt_tokens=1, completion_tokens=1) is None

    def test_absent_usage_yields_none_not_zero(self):
        """Zero means 'cost nothing'; unknown must not masquerade as free."""
        assert pricing.estimate_cost_usd(
            model="gpt-5.4-mini", prompt_tokens=None,
            completion_tokens=None) is None

    def test_pricing_is_versioned_and_sourced(self):
        assert pricing.PRICING_VERSION
        assert pricing.PRICING_SOURCE.startswith("https://")

    def test_pricing_is_static_not_fetched_at_runtime(self):
        import ast
        import inspect
        source = inspect.getsource(pricing)
        tree = ast.parse(source)
        imported = {n.names[0].name.split(".")[0]
                    for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom))
                    and n.names and getattr(n, "module", None) is not None
                    or isinstance(n, ast.Import)}
        for network in ("httpx", "requests", "urllib", "aiohttp"):
            assert network not in imported


# ======================================================================
# K-L: telemetry is observational, fail-open, and carries no secrets
# ======================================================================

class TestTelemetryIsSafe:
    def test_the_payload_carries_no_secret_or_prompt_material(self):
        payload = telemetry.build_usage_payload(
            generation_id="gen-1", correlation_id="case-1",
            result=result_with(ProviderUsage(prompt_tokens=10,
                                             completion_tokens=5)))
        blob = json.dumps(payload, ensure_ascii=False)
        for forbidden in (API_KEY, PROMPT_TEXT, "Authorization", "Bearer",
                          "sk-", "eyJ", "railway.internal", "password",
                          "@example.com"):
            assert forbidden not in blob

    def test_the_payload_fields_are_an_explicit_allowlist(self):
        payload = telemetry.build_usage_payload(
            generation_id="gen-1", correlation_id="case-1",
            result=result_with(ProviderUsage(prompt_tokens=1)))
        assert set(payload) == {
            "generation_id", "correlation_id", "provider", "model",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "cached_input_tokens", "reasoning_tokens", "provider_latency_ms",
            "generation_latency_ms", "estimated_cost_usd", "pricing_version",
            "pricing_source", "occurred_at",
            # X1D-LEGACYDIAG3.2: which contract produced the call. The set
            # stays closed and exhaustively asserted; this names the one
            # addition rather than loosening the check.
            "inference_purpose"}

    def test_a_storage_failure_is_swallowed(self, monkeypatch):
        """A valid governed diagnosis must not fail because accounting did."""
        def boom():
            raise RuntimeError("database unreachable")
        monkeypatch.setattr(telemetry, "get_session_factory", boom)
        assert telemetry.record_provider_usage(
            generation_id="gen-1", correlation_id="case-1",
            result=result_with(ProviderUsage(prompt_tokens=1))) is False

    def test_a_malformed_result_is_swallowed(self, monkeypatch):
        assert telemetry.record_provider_usage(
            generation_id="gen-1", correlation_id="case-1",
            result=object()) is False          # type: ignore[arg-type]

    def test_absent_usage_still_records_latency_and_model(self):
        payload = telemetry.build_usage_payload(
            generation_id="gen-1", correlation_id="case-1",
            result=result_with(usage=None, latency=777.0))
        assert payload["prompt_tokens"] is None
        assert payload["estimated_cost_usd"] is None
        assert payload["pricing_version"] is None   # no price, no version claim
        assert payload["provider_latency_ms"] == 777.0
        assert payload["model"] == "gpt-5.4-mini"

    def test_telemetry_never_reaches_the_clinical_path(self):
        """The recorder exposes no reader the pipeline could consult."""
        public = {n for n in dir(telemetry) if not n.startswith("_")}
        for reader in ("get_usage", "read_usage", "usage_for", "fetch_usage",
                       "last_usage", "total_cost"):
            assert reader not in public

    def test_the_assembler_ignores_the_observer_return_value(self):
        import ast
        import inspect
        import textwrap
        from app.services.recommendation.assembler import RecommendationAssembler

        code = ast.unparse(ast.parse(textwrap.dedent(
            inspect.getsource(RecommendationAssembler.generate))))
        assert "on_provider_result(result)" in code
        # Assigning from it would let telemetry alter the clinical flow.
        assert "= on_provider_result(" not in code


# ======================================================================
# I-J: replay creates no telemetry; a genuine follow-up creates its own
# ======================================================================

class TestIdempotentReplayIsNotBilledTwice:
    def test_replay_returns_before_the_provider_is_reached(self):
        import ast
        import inspect
        import textwrap
        from app.services.integration.base44 import Base44GenerationService

        code = ast.unparse(ast.parse(textwrap.dedent(
            inspect.getsource(Base44GenerationService.generate))))
        # The existing idempotency check returns the stored response, so
        # _execute -- and with it the provider and the telemetry write -- is
        # never entered on a replay.
        assert "if existing:" in code
        assert "return self._response(existing)" in code

    def test_telemetry_is_written_only_from_the_execute_path(self):
        import ast
        import inspect
        import textwrap
        from app.services.integration.base44 import Base44GenerationService

        replay = ast.unparse(ast.parse(textwrap.dedent(
            inspect.getsource(Base44GenerationService.generate))))
        execute = ast.unparse(ast.parse(textwrap.dedent(
            inspect.getsource(Base44GenerationService._execute))))
        assert "record_provider_usage" not in replay
        assert "record_provider_usage" in execute

    def test_each_generation_gets_its_own_record(self):
        """A follow-up is a separate generation, so its counts stand alone."""
        first = telemetry.build_usage_payload(
            generation_id="gen-1", correlation_id="case-1",
            result=result_with(ProviderUsage(prompt_tokens=400,
                                             completion_tokens=200)))
        second = telemetry.build_usage_payload(
            generation_id="gen-2", correlation_id="case-1",
            result=result_with(ProviderUsage(prompt_tokens=650,
                                             completion_tokens=310)))
        assert first["generation_id"] != second["generation_id"]
        assert first["correlation_id"] == second["correlation_id"]
        assert first["prompt_tokens"] != second["prompt_tokens"]
        assert first["estimated_cost_usd"] != second["estimated_cost_usd"]


# ======================================================================
# N: provider error sanitisation still holds
# ======================================================================

class TestErrorSanitisationIntact:
    def test_provider_call_error_is_still_a_runtime_error(self):
        from app.services.llm.openai_compatible import ProviderCallError
        assert issubclass(ProviderCallError, RuntimeError)

    def test_upstream_bodies_are_still_not_returned(self):
        import ast
        import inspect
        import textwrap
        from app.services.llm.openai_compatible import OpenAICompatibleProvider

        tree = ast.parse(textwrap.dedent(inspect.getsource(
            OpenAICompatibleProvider.generate_recommendation)))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                    and node.exc.args):
                rendered = ast.unparse(node.exc.args[0])
                assert "response.text" not in rendered
                assert "error_text" not in rendered
