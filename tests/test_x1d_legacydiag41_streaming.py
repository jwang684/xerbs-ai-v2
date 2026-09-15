"""X1D-LEGACYDIAG4.1: show the text sooner, change nothing about what is true.

Why this phase exists
---------------------
LEGACYDIAG4 traced the legacy system and found its advantage was not better
reasoning. It streamed: first content in about a second, the document building
in front of the patient. Xerbs generates more and better clinical content and
then shows a blank screen for 4 to 19 seconds before revealing all of it.

So the provider call is now read incrementally and the summary field -- which
consumer_diagnosis_adapter.project_for_consumer already sends to the patient --
is forwarded as it arrives.

What these tests are about
--------------------------
Almost entirely the boundary, not the speed. A streamed fragment is a picture
of an answer being written; it is not an answer. The dangerous failure is not
"streaming broke", it is "something downstream started believing a fragment".
So the tests below mostly assert absence:

  * one inference powers both the stream and the result;
  * `final` exists only on the far side of the same validation as before;
  * no partial text reaches corpus, safety, eligibility or purchase;
  * a failure cannot be dressed up as a successful finish.

The scanner has its own section because it is the one genuinely new piece of
parsing, and parsing patient-derived text is where display bugs turn into
clinical ones. It is built to give up rather than guess.
"""

import ast
import asyncio
import inspect
import json
import textwrap

import pytest

from app.api import base44 as base44_api
from app.services.integration import base44 as base44_service
from app.services.llm import openai_compatible as oc
from app.services.llm.mock import MockProvider
from app.services.llm.provider import ProviderResult
from app.services.llm.streaming import STREAMABLE_KEYS, SummaryStreamScanner
from app.services.recommendation import assembler as assembler_module
from app.services.telemetry import provider_usage


def code(obj):
    """Executable source with docstrings stripped."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
    return ast.unparse(tree)


BODY = json.dumps({"summary": "外感风热，肺失宣降。",
                   "pattern_hypotheses": [{"name": "风热犯表"}],
                   "model_confidence": 0.4}, ensure_ascii=False)


# ======================================================================
# 1-3: one inference, and it reports before it finishes
# ======================================================================

class TestOneInference:
    def test_1_the_streaming_endpoint_delegates_to_the_same_service(self):
        """Not a second path: the same generate(), the same guarantees."""
        source = code(base44_api.generate_stream)
        assert "service.generate(" in source
        for forbidden in ("get_provider()", "generate_recommendation(",
                          "generate_interview(", "RecommendationAssembler"):
            assert forbidden not in source

    def test_2_no_second_call_is_made_to_finalise(self):
        source = code(base44_api.generate_stream)
        assert source.count("service.generate(") == 1

    def test_2b_the_assembler_still_calls_one_contract_per_turn(self):
        source = code(assembler_module.RecommendationAssembler.generate)
        assert source.count("await self.provider.generate_interview(") == 1
        assert source.count("await self.provider.generate_recommendation(") == 1
        assert "if interview_mode:" in source          # exclusive branches

    def test_3_deltas_are_emitted_before_the_final_event(self):
        """The queue is drained as it fills, not after the task completes."""
        source = code(base44_api.generate_stream)
        assert "await queue.get()" in source
        assert source.index("create_task") < source.index("await queue.get()")

    def test_3b_the_provider_reports_during_the_read_loop(self):
        source = code(oc.OpenAICompatibleProvider._call_streaming)
        assert "on_display_text(fresh)" in source
        assert "aiter_lines" in source


# ======================================================================
# 4-6: a fragment is never an answer
# ======================================================================

class TestPartialIsNeverAuthority:
    def test_4_nothing_downstream_reads_a_delta(self):
        """The sink writes to a queue and the queue feeds only the wire."""
        source = code(base44_api.generate_stream)
        assert "queue.put_nowait(('delta'" in source.replace('"', "'")
        for forbidden in ("corpus", "safety", "eligib", "formula",
                          "purchas", "resolve"):
            assert forbidden not in source.lower()

    def test_5_final_comes_only_from_the_returned_response(self):
        source = code(base44_api.generate_stream)
        assert "response = await service.generate(" in source
        assert "queue.put_nowait(('final', response))" in \
            source.replace('"', "'")

    def test_5b_validation_still_happens_in_one_place(self):
        """_decode is the only route from a provider body to an object."""
        for branch in (oc.OpenAICompatibleProvider._call_blocking,
                       oc.OpenAICompatibleProvider._call_streaming):
            assert "self._decode(" in code(branch)
        decode = code(oc.OpenAICompatibleProvider._decode)
        assert "json.loads(raw)" in decode
        assert "isinstance(data, dict)" in decode

    def test_6_a_malformed_completion_cannot_emit_a_successful_final(self):
        """_decode raises, run() turns it into error, final is never queued."""
        provider = oc.OpenAICompatibleProvider(
            base_url="https://example.invalid", api_key="k", model="m")
        for bad in ("", "not json", "[1,2,3]", '{"a": '):
            with pytest.raises(oc.ProviderCallError):
                provider._decode(bad)

    def test_6b_a_failure_queues_an_error_not_a_final(self):
        """No except handler in run() can queue a final event.

        Walked over the AST rather than sliced out of the text: the function
        contains several try blocks and a generator that legitimately mentions
        "final", so any slice either misses a handler or sweeps in code that is
        not a handler at all.
        """
        tree = ast.parse(textwrap.dedent(
            inspect.getsource(base44_api.generate_stream)))
        run = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.AsyncFunctionDef) and n.name == "run")
        handlers = [h for h in ast.walk(run)
                    if isinstance(h, ast.ExceptHandler)]
        assert handlers, "run() must handle failure explicitly"
        for handler in handlers:
            body = ast.unparse(ast.Module(body=handler.body, type_ignores=[]))
            assert "'final'" not in body.replace('"', "'")
        whole = ast.unparse(run)
        assert "GENERATION_FAILED" in whole


# ======================================================================
# 7-8: failure directions unchanged
# ======================================================================

class TestFailureDirections:
    def test_7_provider_failure_remains_fail_closed(self):
        for branch in (oc.OpenAICompatibleProvider._call_blocking,
                       oc.OpenAICompatibleProvider._call_streaming):
            body = code(branch)
            assert "raise ProviderCallError" in body
            assert "return ProviderResult" not in body

    def test_7b_an_http_error_is_raised_not_streamed(self):
        source = code(oc.OpenAICompatibleProvider._call_streaming)
        assert "response.is_error" in source
        assert "raise ProviderCallError" in source

    def test_8_telemetry_remains_fail_open(self):
        source = code(provider_usage.record_provider_usage)
        assert "except Exception" in source
        assert "return False" in source

    def test_8b_usage_is_requested_so_streaming_does_not_lose_accounting(self):
        source = code(oc.OpenAICompatibleProvider._call_streaming)
        assert "stream_options" in source
        assert "include_usage" in source

    def test_8c_missing_usage_is_unknown_not_zero(self):
        source = code(oc.OpenAICompatibleProvider._call_streaming)
        assert "if usage_payload else None" in source


# ======================================================================
# 9-11: replay and disconnect
# ======================================================================

class TestReplayAndDisconnect:
    def test_9_replay_short_circuits_before_any_provider_call(self):
        source = code(base44_service.Base44GenerationService.generate)
        assert "return self._response(existing)" in source
        head = source[:source.index("return self._response(existing)")]
        assert "_execute" not in head

    def test_10_replay_creates_no_second_row(self):
        source = code(base44_service.Base44GenerationService.generate)
        early = source[:source.index("return self._response(existing)")]
        assert "GenerationRequest(" not in early

    def test_10b_the_streaming_endpoint_adds_no_replay_logic_of_its_own(self):
        source = code(base44_api.generate_stream)
        for forbidden in ("GenerationRequest", "request_hash", "select(",
                          "Session"):
            assert forbidden not in source

    def test_11_disconnect_does_not_cancel_the_governed_task(self):
        """Letting it finish is what keeps persistence and idempotency sane."""
        source = code(base44_api.generate_stream)
        assert "task.cancel()" not in source
        assert "finally:" in source

    def test_11b_a_partial_stream_leaves_no_completed_generation(self):
        """Completion is written by _execute, never by the stream loop."""
        source = code(base44_api.generate_stream)
        assert "SUCCEEDED" not in source
        assert "status" not in code(base44_api.generate_stream).replace(
            "'status'", "").replace('"status"', "")


# ======================================================================
# 12-17: authority
# ======================================================================

class TestAuthorityUnchanged:
    FORBIDDEN = ("REVIEWED", "VERIFIED", "VERIFIED_EXTERNAL",
                 "clinical_ranking_eligible", "ready_for_formula_retrieval",
                 "consumer_purchasable", "resolved_product_id",
                 "review_status", "safety_verdict")

    @pytest.mark.parametrize("name", FORBIDDEN)
    def test_12_17_streamed_content_cannot_set_it(self, name):
        for module in (base44_api.generate_stream,
                       oc.OpenAICompatibleProvider._call_streaming):
            assert name not in code(module)

    def test_the_scanner_cannot_name_authority_either(self):
        from app.services.llm import streaming as module

        body = code(module)
        for name in self.FORBIDDEN:
            assert name not in body

    def test_17_safety_engine_is_untouched_by_this_phase(self):
        source = code(assembler_module.RecommendationAssembler.generate)
        assert "SafetyScreenRequest(" in source
        assert "eligible_formula_candidates" in source
        assert "MODEL_FORMULA_CANDIDATES_SUPPRESSED_UNVERIFIED_CORPUS" in source

    def test_the_assembler_assigns_no_readiness(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(
            assembler_module.RecommendationAssembler.generate)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = ast.unparse(ast.Module(
                    body=[ast.Expr(t) for t in node.targets], type_ignores=[]))
                assert "ready_for_formula_retrieval" not in targets


# ======================================================================
# 18-21: what may leave the service
# ======================================================================

class TestWhatIsEmitted:
    def test_20_a_provider_body_is_never_relayed(self):
        """An unexpected failure yields a fixed string, never the exception.

        A provider 401 body echoes part of the API key, so the catch-all must
        not interpolate the exception. The two named handlers above it do relay
        their message, and that is deliberate: both are raised by this service
        with text it wrote itself -- a conflict notice, and a configuration
        error naming env var NAMES, not values -- and the blocking endpoint
        already returns exactly the same strings.
        """
        source = code(base44_api.generate_stream)
        catch_all = source[source.rindex("except Exception"):]
        assert "str(exc)" not in catch_all
        assert "Generation failed." in catch_all

    def test_20d_the_named_handlers_relay_only_this_services_own_text(self):
        from app.services.llm.factory import configuration_error

        source = code(base44_api.generate_stream)
        assert "IdempotencyConflictError" in source
        assert "ProviderConfigurationError" in source
        # configuration_error names variables, never their values
        assert "LLM_API_KEY" in code(configuration_error)
        assert "settings.llm_api_key}" not in code(configuration_error)

    def test_20b_the_api_key_is_redacted_on_every_error_path(self):
        for branch in (oc.OpenAICompatibleProvider._call_blocking,
                       oc.OpenAICompatibleProvider._call_streaming):
            body = code(branch)
            assert "_redact(" in body
            assert "self.api_key" in body

    def test_20c_no_prompt_or_key_appears_in_an_event(self):
        source = code(base44_api.generate_stream)
        for forbidden in ("SYSTEM_PROMPT", "api_key", "Authorization",
                          "messages"):
            assert forbidden not in source

    def test_21_only_the_summary_field_is_ever_streamed(self):
        assert STREAMABLE_KEYS == ("summary", "interview_summary")

    def test_21b_raw_json_is_not_emitted(self):
        """The scanner yields decoded text, never the surrounding syntax."""
        scanner = SummaryStreamScanner()
        out = "".join(scanner.feed(BODY[:i]) for i in range(1, len(BODY) + 1))
        assert out == "外感风热，肺失宣降。"
        for syntax in ("{", "}", '"summary"', "pattern_hypotheses"):
            assert syntax not in out

    def test_21c_a_delta_is_bounded(self):
        assert base44_api.MAX_DELTA_CHARS <= 1000
        assert "MAX_DELTA_CHARS" in code(base44_api.generate_stream)


# ======================================================================
# The scanner: it must give up rather than guess
# ======================================================================

class TestScanner:
    def test_it_reconstructs_across_worst_case_chunk_boundaries(self):
        scanner = SummaryStreamScanner()
        out = "".join(scanner.feed(BODY[:i]) for i in range(1, len(BODY) + 1))
        assert out == "外感风热，肺失宣降。"

    def test_a_nested_key_of_the_same_name_is_ignored(self):
        body = json.dumps({"pattern_hypotheses": [{"summary": "WRONG"}],
                           "summary": "RIGHT"})
        assert SummaryStreamScanner().feed(body) == "RIGHT"

    def test_escapes_are_decoded_not_shown(self):
        body = json.dumps({"summary": 'a"b\nc治'})
        assert SummaryStreamScanner().feed(body) == 'a"b\nc治'

    def test_a_split_escape_waits_rather_than_showing_a_backslash(self):
        scanner = SummaryStreamScanner()
        assert "\\" not in scanner.feed('{"summary": "a\\')

    def test_a_non_string_value_disables_rather_than_guessing(self):
        scanner = SummaryStreamScanner()
        scanner.feed('{"summary": 123}')
        assert scanner.active is False

    def test_it_stops_after_the_field_closes(self):
        scanner = SummaryStreamScanner()
        scanner.feed(BODY)
        assert scanner.active is False

    def test_it_never_raises(self):
        for junk in ("", "{", '{"', "[[[[", '{"summary"', '{"summary":',
                     '{"a": {"b": [1, 2', "\\", '{"summary": "x'):
            SummaryStreamScanner().feed(junk)      # must not raise

    def test_an_absurd_body_disables_the_scanner(self):
        scanner = SummaryStreamScanner()
        scanner.feed("{" * 300_000)
        assert scanner.active is False

    def test_it_emits_each_character_exactly_once(self):
        scanner = SummaryStreamScanner()
        pieces = [scanner.feed(BODY[:i]) for i in range(1, len(BODY) + 1)]
        assert "".join(pieces) == "外感风热，肺失宣降。"


# ======================================================================
# 22-24: the existing contract is unchanged
# ======================================================================

class TestNoRegression:
    def test_22_the_blocking_endpoint_still_exists_unchanged(self):
        source = code(base44_api.generate)
        assert "service.generate(" in source
        assert "StreamingResponse" not in source

    def test_22b_a_blocking_call_streams_nothing(self):
        source = code(oc.OpenAICompatibleProvider._call)
        assert "if on_display_text is None:" in source
        assert "_call_blocking" in source

    def test_22c_the_mock_provider_still_satisfies_the_contract(self):
        result = asyncio.run(MockProvider().generate_interview(
            text_input="x", symptoms=[], language="zh"))
        assert isinstance(result, ProviderResult)
        assert result.formula_candidates == []

    def test_23_question_behaviour_is_unchanged(self):
        from app.services.clarification import coverage as cov
        from app.services.clarification.validator import (
            FIELD_PATTERN, MAX_PROPOSALS_PER_TURN, REJECTION_REASONS)

        assert cov.MAX_VISIBLE_QUESTIONS_PER_TURN == 4
        assert cov.MAX_ADAPTIVE_WHEN_OPEN == 3
        assert cov.MIN_QUESTIONS_WHEN_INSUFFICIENT == 2
        assert MAX_PROPOSALS_PER_TURN == 3
        assert len(REJECTION_REASONS) == 10
        assert FIELD_PATTERN.pattern == r"^[a-z][a-z0-9_]{1,38}[a-z0-9]$"

    def test_23b_the_prompts_are_unchanged_by_this_phase(self):
        assert "at most 3" in oc.INTERVIEW_SYSTEM_PROMPT
        assert "clinical_reasoning" in oc.SYSTEM_PROMPT
        for prompt in (oc.SYSTEM_PROMPT, oc.INTERVIEW_SYSTEM_PROMPT):
            assert "stream" not in prompt.lower()

    def test_24_depth_semantics_are_unchanged(self):
        from app.services.interview.mode import (
            MAX_INTERVIEW_TURNS, REASON_DEPTH_REACHED, decide_mode)

        assert MAX_INTERVIEW_TURNS == 3
        mode, reason = decide_mode(accumulated_text="咳嗽3天。",
                                   missing_information=[],
                                   interview_depth=3)
        assert (mode, reason) == ("FULL_REASONING", REASON_DEPTH_REACHED)

    def test_24b_telemetry_keeps_its_existing_meaning(self):
        """provider_latency_ms is still the whole call, not time-to-first."""
        source = code(oc.OpenAICompatibleProvider._call_streaming)
        assert "provider_latency_ms = (time.monotonic() - started) * 1000.0" \
            in source
        assert "first_delta_ms" in source
        assert ProviderResult(summary="x").first_delta_ms is None
