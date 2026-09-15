"""X1D-LEGACYDIAG3.2: a smaller call for the turns that only need questions.

The measurement that motivated this
-----------------------------------
LEGACYDIAG3.1 decomposed a turn end to end. The provider call is 15.2-19.4s and
everything else in ai-v2 is 27-34ms, so the only place latency can come from is
what the model is asked to write. On an incomplete turn it writes a summary,
pattern hypotheses, formula candidates and the whole ClinicalReasoningEnvelope
-- pathogenesis, treatment principle, formula hypotheses with ingredients,
dosages, administration, contraindications -- roughly 2000-2600 completion
tokens, and the patient is then shown two short questions.

So the interview turns get their own contract, which asks for working
hypotheses, what is missing, and at most three questions.

What these tests are mostly about
---------------------------------
Not the speed -- that is measured on staging. They are about the interview path
having no clinical authority and taking none of the existing governance out of
the loop:

  * the small contract has no field a formula could be written into;
  * routing is deterministic and never asks the model whether to stop;
  * every proposed question still passes CLARIFY1-3.1 and the LEGACYDIAG3.1
    cross-turn layer;
  * the corpus, SafetyEngine and the purchase chokepoint decide exactly what
    they decided before.

One design note worth reading before changing anything here: the first cut also
skipped corpus retrieval on interview turns, on the theory that mid-interview
nobody should see a formula. Four existing tests caught it, because a REVIEWED
sourced formula reaching the assembler is a governance guarantee. The interview
path changes which prompt runs. It does not change what the corpus does.
"""

import ast
import asyncio
import inspect
import json
import textwrap

import pytest

from app.schemas.intake import RecommendationRequest
from app.schemas.reasoning import (
    InterviewHypothesis,
    InterviewReasoning,
)
from app.services.clarification import coverage as cov
from app.services.clarification.validator import validate_proposals
from app.services.interview import mode as interview_mode
from app.services.interview.mode import (
    FULL_REASONING,
    INTERVIEW,
    MAX_INTERVIEW_TURNS,
    REASON_COVERAGE_SUFFICIENT,
    REASON_DEPTH_REACHED,
    REASON_MATERIALLY_MISSING,
    REASON_PROVIDER_UNSUPPORTED,
    decide_mode,
    describe_policy,
    provider_supports_interview,
)
from app.services.llm import openai_compatible as oc
from app.services.llm.mock import MockProvider
from app.services.llm.provider import LLMProvider, ProviderResult
from app.services.telemetry import provider_usage


SPARSE = "咳嗽发热3天。"
RICH = ("发热3天，怕冷无汗，咳嗽有白痰，口不渴，食欲正常，大便正常，"
        "睡眠可，头身酸痛，胸不闷，受凉后起病。")


def missing(*priorities):
    return [type("M", (), {"field": "f%d" % i, "priority": p})()
            for i, p in enumerate(priorities)]


def _executable_code(module_or_obj):
    """Source with docstrings removed.

    These modules explain at length what they refuse to do, so a substring
    search finds the prose rather than the behaviour.
    """
    src = inspect.getsource(module_or_obj)
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
    return ast.unparse(tree)


# ======================================================================
# A: routing
# ======================================================================

class TestScenarioA_Routing:
    def test_an_incomplete_complaint_routes_to_interview(self):
        mode, reason = decide_mode(accumulated_text=SPARSE,
                                   missing_information=missing("MEDIUM"))
        assert mode == INTERVIEW
        assert reason == REASON_MATERIALLY_MISSING

    def test_a_high_priority_gap_routes_to_interview(self):
        mode, _ = decide_mode(accumulated_text=RICH,
                              missing_information=missing("HIGH"))
        assert mode == INTERVIEW

    def test_a_covered_complaint_routes_to_full_reasoning(self):
        mode, reason = decide_mode(accumulated_text=RICH,
                                   missing_information=[])
        assert mode == FULL_REASONING
        assert reason == REASON_COVERAGE_SUFFICIENT

    def test_the_depth_escape_forces_full_reasoning(self):
        """A case must always end up getting full clinical reasoning."""
        mode, reason = decide_mode(accumulated_text=SPARSE,
                                   missing_information=missing("HIGH"),
                                   interview_depth=MAX_INTERVIEW_TURNS)
        assert mode == FULL_REASONING
        assert reason == REASON_DEPTH_REACHED

    def test_a_provider_without_the_small_contract_is_never_routed_to_it(self):
        mode, reason = decide_mode(accumulated_text=SPARSE,
                                   missing_information=missing("HIGH"),
                                   supports_interview=False)
        assert mode == FULL_REASONING
        assert reason == REASON_PROVIDER_UNSUPPORTED

    def test_support_is_detected_not_assumed(self):
        class _Bare(LLMProvider):
            async def generate_recommendation(self, **kw):
                raise NotImplementedError

        assert provider_supports_interview(MockProvider()) is True
        assert provider_supports_interview(_Bare()) is False

    def test_routing_never_consults_the_model(self):
        """The one input that must not exist here.

        provider_supports_interview does mention a provider, but it only asks
        whether a method is implemented -- it never calls one. What must be
        absent is any model output: the advisory information_sufficient flag,
        and any await that could reach an inference.
        """
        code = _executable_code(interview_mode)
        for forbidden in ("information_sufficient", "await ",
                          "ProviderResult", "clarification_proposals"):
            assert forbidden not in code

    def test_routing_takes_no_model_output_as_an_argument(self):
        signature = inspect.signature(decide_mode)
        assert set(signature.parameters) == {
            "accumulated_text", "missing_information", "interview_depth",
            "supports_interview"}

    def test_routing_is_reproducible(self):
        first = decide_mode(accumulated_text=SPARSE,
                            missing_information=missing("MEDIUM"))
        for _ in range(10):
            assert decide_mode(accumulated_text=SPARSE,
                               missing_information=missing("MEDIUM")) == first

    @pytest.mark.parametrize("bad", [None, "two", -1, 0])
    def test_a_nonsense_depth_does_not_crash_routing(self, bad):
        mode, _ = decide_mode(accumulated_text=SPARSE,
                              missing_information=[], interview_depth=bad)
        assert mode in (INTERVIEW, FULL_REASONING)


# ======================================================================
# B-F: the interview path carries no clinical authority
# ======================================================================

class TestScenarioBF_NoAuthority:
    FORBIDDEN_FIELDS = (
        "formula_candidates", "formula_hypotheses", "formula_id", "formula",
        "ingredients", "dosage", "administration", "contraindications",
        "precautions", "treatment_principle", "pathogenesis",
        "consumer_purchasable", "clinical_ranking_eligible", "safety_verdict",
        "review_status", "verification", "purchasable", "eligibility",
    )

    @pytest.mark.parametrize("field", FORBIDDEN_FIELDS)
    def test_b_the_interview_schema_has_no_such_field(self, field):
        assert field not in InterviewReasoning.model_fields
        assert field not in InterviewHypothesis.model_fields

    def test_b2_the_schema_surface_is_exactly_four_fields(self):
        assert set(InterviewReasoning.model_fields) == {
            "interview_summary", "working_hypotheses", "missing_information",
            "information_sufficient"}

    @pytest.mark.parametrize("field", FORBIDDEN_FIELDS)
    def test_b3_the_interview_prompt_never_asks_for_it(self, field):
        """Checked as a JSON key, not as a word.

        The prompt names several of these in its prohibitions -- "no formula,
        no herb, no dosage" -- so a bare substring search finds the very
        instruction that forbids them. What must be absent is a requested
        field, which in this contract is always a quoted key.
        """
        assert '\"%s\"' % field not in oc.INTERVIEW_SYSTEM_PROMPT

    def test_b4_the_interview_prompt_forbids_treatment_content(self):
        for required in ("no formula", "no herb", "no dosage",
                         "no treatment plan", "no final syndrome diagnosis"):
            assert required in oc.INTERVIEW_SYSTEM_PROMPT

    def test_b5_an_interview_result_carries_no_formula_candidates(self):
        result = asyncio.run(MockProvider().generate_interview(
            text_input=SPARSE, symptoms=[], language="zh"))
        assert result.formula_candidates == []
        assert result.pattern_hypotheses == []
        assert result.clinical_reasoning == {}

    def test_b6_the_provider_method_cannot_emit_candidates(self):
        """Structural: the only ProviderResult it builds hardcodes empty."""
        code = _executable_code(oc.OpenAICompatibleProvider.generate_interview)
        assert "formula_candidates=[]" in code
        assert "pattern_hypotheses=[]" in code

    def test_c_interview_output_cannot_set_purchasability(self):
        code = _executable_code(oc.OpenAICompatibleProvider.generate_interview)
        assert "consumer_purchasable" not in code

    def test_d_interview_output_cannot_set_ranking_eligibility(self):
        for module in (interview_mode, oc.OpenAICompatibleProvider.generate_interview):
            assert "clinical_ranking_eligible" not in _executable_code(module)

    def test_e_interview_output_cannot_create_reviewed_knowledge(self):
        code = _executable_code(oc.OpenAICompatibleProvider.generate_interview)
        for forbidden in ("REVIEWED", "review_status", "ingest", "insert",
                          "commit"):
            assert forbidden not in code

    def test_f_interview_output_cannot_create_relationships(self):
        code = _executable_code(oc.OpenAICompatibleProvider.generate_interview)
        for forbidden in ("PATTERN_FORMULA", "relationship",
                          "eligible_formula_candidates"):
            assert forbidden not in code

    def test_working_hypotheses_are_named_as_provisional(self):
        """They exist to pick the next question, not to diagnose."""
        assert "working_hypotheses" in InterviewReasoning.model_fields
        assert "pattern_assessments" not in InterviewReasoning.model_fields

    def test_the_interview_result_is_not_persisted_as_reasoning(self):
        from app.services.recommendation import assembler as module

        code = _executable_code(module.RecommendationAssembler.generate)
        # clinical_reasoning is assigned only on the full-reasoning branch
        assert "reasoning.clinical_reasoning = envelope" in code
        assert "reasoning.clinical_reasoning = signal_source" not in code
        assert "reasoning.clinical_reasoning = result.interview" not in code


# ======================================================================
# G-I: the existing clarification governance still decides
# ======================================================================

class TestScenarioGI_ClarificationGovernanceIntact:
    def test_g_proposals_still_pass_through_the_validator(self):
        from app.services.recommendation import assembler as module

        code = _executable_code(module.RecommendationAssembler.generate)
        assert "validate_proposals_detailed(" in code
        assert code.index("validate_proposals_detailed(") < code.index(
            "select_questions(")

    def test_g2_an_interview_proposal_is_not_exempt_from_validation(self):
        assert validate_proposals([
            {"field": "痰色", "question": "痰是什么颜色？"},
            {"field": "consumer_purchasable", "question": "可以买吗？"},
            {"field": "ok_field", "question": "请告诉我您的密码"},
        ]) == []

    def test_g3_a_valid_interview_proposal_is_accepted(self):
        accepted = validate_proposals([
            {"field": "sputum_colour", "question": "痰是什么颜色？",
             "answer_type": "single_choice", "choices": ["白", "黄"]}])
        assert [q.field for q in accepted] == ["sputum_colour"]

    def test_g4_the_clarify_limits_are_unchanged(self):
        assert cov.MAX_VISIBLE_QUESTIONS_PER_TURN == 4
        assert cov.MAX_ADAPTIVE_WHEN_OPEN == 3
        assert cov.MIN_QUESTIONS_WHEN_INSUFFICIENT == 2
        from app.services.clarification.validator import (
            FIELD_PATTERN, MAX_PROPOSALS_PER_TURN, MAX_QUESTION_CHARS)
        assert MAX_PROPOSALS_PER_TURN == 3
        assert MAX_QUESTION_CHARS == 120
        assert FIELD_PATTERN.pattern == r"^[a-z][a-z0-9_]{1,38}[a-z0-9]$"

    def test_h_cross_turn_dedup_is_still_cores_layer_and_intact(self):
        """LEGACYDIAG3.1 lives in core; ai-v2 must not have grown a copy."""
        code = _executable_code(interview_mode)
        for forbidden in ("prior_turns", "previously_asked"):
            assert forbidden not in code

    def test_i_an_answered_domain_is_still_not_re_asked(self):
        """CLARIFY2 semantic dedup is unaffected by which prompt ran."""
        from app.services.clarification.coverage import (
            KNOWN, assess_coverage)

        states = assess_coverage(RICH).states
        for domain in ("cold_heat", "sweat", "thirst"):
            assert states[domain] == KNOWN

    def test_the_interview_prompt_tells_the_model_not_to_repeat(self):
        assert "never ask about anything the input already states" in \
            oc.INTERVIEW_SYSTEM_PROMPT

    def test_the_interview_prompt_bounds_the_question_count(self):
        assert "at most 3" in oc.INTERVIEW_SYSTEM_PROMPT
        assert "Two good questions beat four" in oc.INTERVIEW_SYSTEM_PROMPT

    def test_the_interview_prompt_demands_discrimination(self):
        assert "which of your working_hypotheses is leading" in \
            oc.INTERVIEW_SYSTEM_PROMPT


# ======================================================================
# J-N: nothing else moved
# ======================================================================

class TestScenarioJN_BoundariesUnchanged:
    def test_j_k_isolation_remains_cores_responsibility(self):
        """ai-v2 sees one turn and no identity; it cannot leak across cases."""
        assert "user_id" not in RecommendationRequest.model_fields
        assert "case_id" not in RecommendationRequest.model_fields
        assert "correlation_id" not in RecommendationRequest.model_fields

    def test_l_replay_never_reaches_the_provider(self):
        """Idempotent replay is short-circuited before generation."""
        from app.services.integration import base44

        code = _executable_code(base44)
        assert "_execute" in code

    def test_m_the_full_reasoning_path_is_unchanged(self):
        code = _executable_code(oc.OpenAICompatibleProvider.generate_recommendation)
        assert "SYSTEM_PROMPT" in code
        assert "INTERVIEW_SYSTEM_PROMPT" not in code

    def test_m2_the_full_contract_still_asks_for_the_envelope(self):
        for required in ("clinical_reasoning", "formula_hypotheses",
                         "pathogenesis", "treatment_principle"):
            assert required in oc.SYSTEM_PROMPT

    def test_n_the_corpus_gate_is_untouched(self):
        from app.services.recommendation import assembler as module

        code = _executable_code(module.RecommendationAssembler.generate)
        assert "eligible_formula_candidates" in code
        assert "MODEL_FORMULA_CANDIDATES_SUPPRESSED_UNVERIFIED_CORPUS" in code

    def test_n2_retrieval_is_not_gated_on_interview_mode(self):
        """The regression four existing tests caught. Pinned deliberately."""
        from app.services.recommendation import assembler as module

        code = _executable_code(module.RecommendationAssembler.generate)
        assert "not relationship_matches and not interview_mode" not in code

    def test_n3_the_assembler_assigns_no_readiness(self):
        from app.services.recommendation import assembler as module

        tree = ast.parse(textwrap.dedent(inspect.getsource(
            module.RecommendationAssembler.generate)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = ast.unparse(ast.Module(
                    body=[ast.Expr(t) for t in node.targets], type_ignores=[]))
                assert "ready_for_formula_retrieval" not in targets


# ======================================================================
# O-P: telemetry
# ======================================================================

class TestScenarioOP_Telemetry:
    def _payload(self, purpose):
        result = ProviderResult(summary="x", provider="openai-compatible",
                                model="gpt-5.4-mini",
                                inference_purpose=purpose)
        return provider_usage.build_usage_payload(
            generation_id="gen-1", correlation_id="case-1", result=result)

    def test_o_an_interview_call_is_labelled(self):
        assert self._payload(INTERVIEW)["inference_purpose"] == INTERVIEW

    def test_o2_a_full_call_is_labelled(self):
        assert self._payload(FULL_REASONING)["inference_purpose"] == \
            FULL_REASONING

    def test_o3_the_default_is_full_reasoning(self):
        assert ProviderResult(summary="x").inference_purpose == FULL_REASONING

    def test_o4_the_two_are_distinguishable(self):
        assert (self._payload(INTERVIEW)["inference_purpose"]
                != self._payload(FULL_REASONING)["inference_purpose"])

    def test_p_the_purpose_is_not_in_the_clinical_envelope(self):
        from app.schemas.reasoning import (ClinicalReasoningEnvelope,
                                           ReasoningResponse)

        assert "inference_purpose" not in ClinicalReasoningEnvelope.model_fields
        assert "inference_purpose" not in ReasoningResponse.model_fields

    def test_p2_telemetry_carries_no_patient_material(self):
        blob = json.dumps(self._payload(INTERVIEW), ensure_ascii=False)
        for forbidden in ("咳嗽", "发热", "text_input", "symptoms",
                          "system_prompt", "SYSTEM_PROMPT", "messages",
                          "Bearer", "sk-", "password"):
            assert forbidden not in blob


# ======================================================================
# Q-R: failure behaviour
# ======================================================================

class TestScenarioQR_FailsSafely:
    def test_q_a_malformed_interview_block_does_not_raise(self):
        from app.services.clarification.coverage import (
            extract_differential_signals)

        for junk in ({"working_hypotheses": "nope"},
                     {"missing_information": 7},
                     {"working_hypotheses": [{"confidence": 2.0}]}):
            with pytest.raises(Exception):
                InterviewReasoning(**junk)
        # ...and the assembler contains that failure rather than propagating it
        from app.services.recommendation import assembler as module

        code = _executable_code(module.RecommendationAssembler.generate)
        assert "INTERVIEW_REASONING_UNPARSEABLE" in code

    def test_q2_an_absent_interview_block_degrades_to_no_signals(self):
        from app.services.clarification.coverage import (
            DifferentialSignals, extract_differential_signals)

        assert extract_differential_signals(None) == DifferentialSignals()

    def test_q3_an_empty_interview_block_is_valid(self):
        assert InterviewReasoning().is_empty() is True

    def test_q4_signals_still_come_out_of_a_valid_block(self):
        from app.services.clarification.coverage import (
            extract_differential_signals)

        reasoning = InterviewReasoning(working_hypotheses=[
            InterviewHypothesis(name="风寒束表",
                                supporting_findings=["无汗"]),
            InterviewHypothesis(name="风热犯表",
                                contradicting_findings=["明显怕冷"]),
        ])
        signals = extract_differential_signals(reasoning)
        assert "cold_heat" in signals.contradiction_domains
        assert "sweat" in signals.discriminating_domains

    def test_r_a_provider_failure_creates_no_diagnosis(self):
        """ProviderCallError propagates; it is never turned into a result."""
        # X1D-LEGACYDIAG4.1: _call is now a dispatcher over a blocking
        # and a streaming branch. Scoped to the class so the property is
        # checked in BOTH branches -- wider than before, not looser.
        code = _executable_code(oc.OpenAICompatibleProvider)
        assert "ProviderCallError" in code
        for branch in (oc.OpenAICompatibleProvider._call_blocking,
                       oc.OpenAICompatibleProvider._call_streaming):
            body = _executable_code(branch)
            assert "ProviderCallError" in body
            assert "formula_candidates" not in body

    def test_r2_the_failure_class_is_shared_by_both_contracts(self):
        """One transport, one error path, one redaction rule."""
        interview = _executable_code(
            oc.OpenAICompatibleProvider.generate_interview)
        full = _executable_code(
            oc.OpenAICompatibleProvider.generate_recommendation)
        assert "self._call(" in interview
        for branch in (oc.OpenAICompatibleProvider._call_blocking,
                       oc.OpenAICompatibleProvider._call_streaming):
            assert "_redact" in _executable_code(branch)
        assert "httpx" not in interview      # transport is not duplicated

    def test_r3_the_interview_call_redacts_like_the_full_one(self):
        """Every path that can surface a provider body redacts the key first."""
        for branch in (oc.OpenAICompatibleProvider._call_blocking,
                       oc.OpenAICompatibleProvider._call_streaming):
            code = _executable_code(branch)
            assert "self.api_key" in code
            assert "_redact(" in code
            # the raw body must never leave unredacted
            assert "response.text)" not in code


# ======================================================================
# Policy description
# ======================================================================

class TestPolicy:
    def test_the_policy_reports_both_modes(self):
        policy = describe_policy()
        assert set(policy["modes"]) == {INTERVIEW, FULL_REASONING}
        assert policy["max_interview_turns"] == MAX_INTERVIEW_TURNS

    def test_every_route_has_a_reason(self):
        policy = describe_policy()
        for text, expected in ((SPARSE, REASON_MATERIALLY_MISSING),
                               (RICH, REASON_COVERAGE_SUFFICIENT)):
            _mode, reason = decide_mode(accumulated_text=text,
                                        missing_information=[])
            assert reason in policy["reasons"]

    def test_the_policy_carries_no_clinical_content(self):
        blob = json.dumps(describe_policy())
        for token in ("formula", "safety", "purchas", "REVIEWED"):
            assert token not in blob
