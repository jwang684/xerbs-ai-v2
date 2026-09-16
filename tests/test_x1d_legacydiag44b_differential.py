"""X1D-LEGACYDIAG4.4B: carry the differential without letting it harden.

The problem being solved
------------------------
InterviewReasoning has produced working hypotheses since 3.2 and discarded them
at the end of every turn. So turn 2 could not ask the one question that would
separate the two readings turn 1 had just formed.

The problem being avoided
-------------------------
Feeding a model its own previous conclusion builds a machine that agrees with
itself. Three turns of "风热犯卫, still 风热犯卫, definitely 风热犯卫" looks like
conviction and carries no information at all.

Two deterministic rules stand between those, and most of this file tests them:

  * a citation that does not resolve to something the patient actually supplied
    is dropped -- so a model that writes "yellow tongue coating" for a patient
    who never mentioned their tongue does not get that as next turn's premise;

  * a standing may not rise unless the model cites evidence it was not already
    citing -- because without that rule, repetition and evidence are
    indistinguishable.

Both run after the model answers. A prompt instruction is a request; these are
boundaries.
"""

import ast
import asyncio
import inspect
import json
import textwrap

import pytest

from app.schemas.intake import RecommendationRequest
from app.schemas.reasoning import (
    EvidenceRef,
    InterviewCarryState,
    InterviewReasoning,
    ReasoningResponse,
    ResolvableEvidence,
    WorkingDifferentialState,
    WorkingHypothesis,
)
from app.services.clarification.coverage import (
    extract_differential_signals,
    normalize_clinical_domain,
)
from app.services.interview import differential as diff
from app.services.interview.differential import (
    STANDING_ORDER,
    describe_policy,
    domain_for_ref,
    ref_key,
    resolve_ref,
    signals_from_state,
    validate_state,
)
from app.services.llm import openai_compatible as oc


EVIDENCE = ResolvableEvidence(
    has_complaint=True,
    observations=["temperature", "sweating"],
    answers=[{"turn_id": 1, "question_field": "sputum_colour"},
             {"turn_id": 1, "question_field": "aversion_to_cold"}],
)


def hyp(name, standing="PLAUSIBLE", support=None, against=None, disc=None):
    return {
        "pattern_name": name,
        "standing": standing,
        "supporting_evidence": support or [],
        "contradicting_evidence": against or [],
        "unresolved_discriminators": disc or [],
    }


COMPLAINT = {"origin": "COMPLAINT"}
OBS_SWEAT = {"origin": "OBSERVATION", "field": "sweating"}
OBS_TEMP = {"origin": "OBSERVATION", "field": "temperature"}
ANS_SPUTUM = {"origin": "ANSWER", "question_field": "sputum_colour", "turn_id": 1}
ANS_COLD = {"origin": "ANSWER", "question_field": "aversion_to_cold", "turn_id": 1}


def code(obj):
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


# ======================================================================
# 1-2: the state crosses the boundary, patient evidence separately
# ======================================================================

class TestCarryForward:
    def test_1_a_validated_state_survives_the_turn_boundary(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("风寒束表", "PRIMARY_WORKING", [OBS_SWEAT])]},
            EVIDENCE)
        assert state is not None
        carried = InterviewCarryState(prior=state,
                                      resolvable_evidence=EVIDENCE)
        assert carried.prior.hypotheses[0].pattern_name == "风寒束表"

    def test_2_patient_evidence_travels_on_its_own_channel(self):
        """Evidence is not inside the state; the state only points at it."""
        assert "text_input" in RecommendationRequest.model_fields
        assert "interview_state" in RecommendationRequest.model_fields
        state = WorkingDifferentialState(hypotheses=[
            WorkingHypothesis(pattern_name="x",
                              supporting_evidence=[EvidenceRef(**OBS_SWEAT)])])
        blob = state.model_dump_json()
        assert "sweating" in blob          # the reference
        assert "无汗" not in blob           # never the patient's words

    def test_the_intake_field_is_additive_and_optional(self):
        assert RecommendationRequest(text_input="x").interview_state is None


# ======================================================================
# 3-8: a citation is a pointer, never a claim
# ======================================================================

class TestEvidenceReferences:
    def test_3_a_model_assertion_cannot_become_patient_evidence(self):
        """The attack from the design audit, as a test."""
        state, notes = validate_state(
            {"hypotheses": [hyp("风热犯卫", "PRIMARY_WORKING", [
                {"origin": "OBSERVATION", "field": "tongue_coating_yellow"}])]},
            EVIDENCE)
        assert state.hypotheses[0].supporting_evidence == []
        assert "EVIDENCE_REF_UNRESOLVED" in notes
        assert "tongue" not in state.model_dump_json()

    def test_4_an_unsupported_reference_is_dropped_not_the_hypothesis(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("风寒束表", "PLAUSIBLE",
                                [OBS_SWEAT, {"origin": "OBSERVATION",
                                             "field": "never_asked"}])]},
            EVIDENCE)
        assert len(state.hypotheses) == 1
        assert [r.field for r in state.hypotheses[0].supporting_evidence] == \
            ["sweating"]

    def test_5_a_complaint_reference_resolves(self):
        assert resolve_ref(COMPLAINT, EVIDENCE) is not None

    def test_5b_but_not_when_there_is_no_complaint(self):
        assert resolve_ref(COMPLAINT,
                           ResolvableEvidence(has_complaint=False)) is None

    def test_6_a_structured_observation_reference_resolves(self):
        assert resolve_ref(OBS_SWEAT, EVIDENCE) is not None
        assert resolve_ref({"origin": "OBSERVATION", "field": "stool"},
                           EVIDENCE) is None

    def test_7_a_prior_answer_reference_resolves(self):
        assert resolve_ref(ANS_SPUTUM, EVIDENCE) is not None
        assert resolve_ref({"origin": "ANSWER", "question_field": "never_asked"},
                           EVIDENCE) is None

    def test_7b_an_answer_from_a_turn_that_never_asked_it_is_rejected(self):
        assert resolve_ref({"origin": "ANSWER",
                            "question_field": "sputum_colour",
                            "turn_id": 99}, EVIDENCE) is None

    @pytest.mark.parametrize("origin", [
        "MODEL_TEXT", "SUMMARY", "HYPOTHESIS", "IMAGE", "TONGUE_IMAGE",
        "INFERRED_FINDING", "PROVIDER_OUTPUT"])
    def test_8_forbidden_origins_are_rejected(self, origin):
        assert resolve_ref({"origin": origin, "field": "x"}, EVIDENCE) is None
        assert origin not in str(EvidenceRef.model_fields["origin"].annotation)

    def test_8b_the_allowed_origin_set_is_closed(self):
        assert diff.ALLOWED_ORIGINS == ("COMPLAINT", "OBSERVATION", "ANSWER")

    def test_malformed_references_do_not_raise(self):
        for junk in (None, 7, "text", {}, {"origin": None}, []):
            assert resolve_ref(junk, EVIDENCE) is None


# ======================================================================
# 9-14: how a standing may move
# ======================================================================

class TestStandingTransitions:
    PRIOR = WorkingDifferentialState(hypotheses=[
        WorkingHypothesis(pattern_name="风寒束表", standing="PLAUSIBLE",
                          supporting_evidence=[EvidenceRef(**OBS_SWEAT)]),
        WorkingHypothesis(pattern_name="风热犯卫", standing="PRIMARY_WORKING",
                          supporting_evidence=[EvidenceRef(**COMPLAINT)]),
    ])

    def test_9_a_standing_may_stay_the_same(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("风寒束表", "PLAUSIBLE", [OBS_SWEAT])]},
            EVIDENCE, self.PRIOR)
        assert state.hypotheses[0].standing == "PLAUSIBLE"

    def test_10_a_standing_may_weaken_freely(self):
        """Weakening needs no permission. Only strengthening does."""
        state, _ = validate_state(
            {"hypotheses": [hyp("风热犯卫", "WEAKENED", [COMPLAINT])]},
            EVIDENCE, self.PRIOR)
        assert state.hypotheses[0].standing == "WEAKENED"

    def test_11_a_standing_may_rise_with_new_evidence(self):
        state, notes = validate_state(
            {"hypotheses": [hyp("风寒束表", "PRIMARY_WORKING",
                                [OBS_SWEAT, ANS_COLD])]},
            EVIDENCE, self.PRIOR)
        assert state.hypotheses[0].standing == "PRIMARY_WORKING"
        assert "STANDING_ESCALATION_WITHOUT_NEW_EVIDENCE" not in notes

    def test_12_a_standing_may_not_rise_on_repetition_alone(self):
        """The anti-self-confirmation rule, and the reason this phase is safe."""
        state, notes = validate_state(
            {"hypotheses": [hyp("风寒束表", "PRIMARY_WORKING", [OBS_SWEAT])]},
            EVIDENCE, self.PRIOR)
        assert state.hypotheses[0].standing == "PLAUSIBLE"
        assert "STANDING_ESCALATION_WITHOUT_NEW_EVIDENCE" in notes

    def test_12b_nor_by_citing_nothing_at_all(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("风寒束表", "PRIMARY_WORKING", [])]},
            EVIDENCE, self.PRIOR)
        assert state.hypotheses[0].standing == "PLAUSIBLE"

    def test_12c_nor_by_citing_evidence_that_does_not_resolve(self):
        """An invented citation must not buy a promotion either."""
        state, _ = validate_state(
            {"hypotheses": [hyp("风寒束表", "PRIMARY_WORKING",
                                [OBS_SWEAT, {"origin": "OBSERVATION",
                                             "field": "made_up"}])]},
            EVIDENCE, self.PRIOR)
        assert state.hypotheses[0].standing == "PLAUSIBLE"

    def test_13_ruled_out_can_recover_with_new_evidence(self):
        prior = WorkingDifferentialState(hypotheses=[
            WorkingHypothesis(pattern_name="风热犯卫",
                              standing="RULED_OUT_FOR_NOW",
                              supporting_evidence=[])])
        state, _ = validate_state(
            {"hypotheses": [hyp("风热犯卫", "PLAUSIBLE", [ANS_SPUTUM])]},
            EVIDENCE, prior)
        assert state.hypotheses[0].standing == "PLAUSIBLE"

    def test_13b_ruled_out_is_not_a_terminal_state(self):
        assert "RULED_OUT" not in STANDING_ORDER
        assert "RULED_OUT_FOR_NOW" in STANDING_ORDER
        assert describe_policy()["ruled_out_is_reversible"] is True

    def test_14_a_new_alternative_may_appear(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("风寒束表", "PLAUSIBLE", [OBS_SWEAT]),
                            hyp("暑湿犯表", "PLAUSIBLE", [COMPLAINT])]},
            EVIDENCE, self.PRIOR)
        names = [h.pattern_name for h in state.hypotheses]
        assert "暑湿犯表" in names

    def test_14b_a_new_alternative_starts_where_the_model_puts_it(self):
        """No prior standing means no escalation rule to apply."""
        state, _ = validate_state(
            {"hypotheses": [hyp("暑湿犯表", "PRIMARY_WORKING", [COMPLAINT])]},
            EVIDENCE, self.PRIOR)
        assert state.hypotheses[0].standing == "PRIMARY_WORKING"

    def test_a_dropped_hypothesis_simply_does_not_reappear(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("风寒束表", "PLAUSIBLE", [OBS_SWEAT])]},
            EVIDENCE, self.PRIOR)
        assert [h.pattern_name for h in state.hypotheses] == ["风寒束表"]

    def test_an_unknown_standing_cannot_buy_a_promotion(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("风寒束表", "DEFINITELY_THIS", [OBS_SWEAT])]},
            EVIDENCE, self.PRIOR)
        assert state.hypotheses[0].standing in STANDING_ORDER


# ======================================================================
# 15-20: what reaches question selection
# ======================================================================

class TestQuestionSelectionIntegration:
    def test_15_contradicting_evidence_is_retained(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("风热犯卫", "WEAKENED", [COMPLAINT],
                                [OBS_SWEAT, ANS_COLD])]},
            EVIDENCE)
        assert len(state.hypotheses[0].contradicting_evidence) == 2

    def test_16_unresolved_discriminators_survive(self):
        """X1D-LEGACYDIAG4.5 tightened what one has to say to survive.

        A bare {"domain": ...} no longer counts: it has to name the competition
        it resolves, from this state's own live hypotheses. The 4.4B property
        under test -- a valid discriminator reaches the other side intact -- is
        unchanged.
        """
        state, _ = validate_state(
            {"hypotheses": [
                hyp("风寒束表", "PLAUSIBLE", [OBS_SWEAT],
                    disc=[{"domain": "thirst",
                           "separates": ["风寒束表", "风热犯卫"],
                           "if_present_supports": ["风热犯卫"],
                           "if_absent_supports": ["风寒束表"],
                           "rationale": "口渴与否"}]),
                hyp("风热犯卫", "PLAUSIBLE", [COMPLAINT])]},
            EVIDENCE)
        entry = state.hypotheses[0].unresolved_discriminators[0]
        assert entry["domain"] == "thirst"
        assert entry["separates"] == ["风寒束表", "风热犯卫"]
        assert entry["discriminating"] is True

    def test_16b_a_discriminator_naming_no_live_competition_is_dropped(self):
        state, notes = validate_state(
            {"hypotheses": [hyp("风寒束表", "PLAUSIBLE", [OBS_SWEAT],
                                disc=[{"domain": "thirst"}])]},
            EVIDENCE)
        assert state.hypotheses[0].unresolved_discriminators == []
        assert "DISCRIMINATOR_NO_LIVE_COMPETITION" in notes

    def test_17_evidence_gaps_survive_when_they_name_held_patterns(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("A", "PLAUSIBLE", [COMPLAINT]),
                            hyp("B", "PLAUSIBLE", [COMPLAINT])],
             "evidence_gaps": [{"domain": "sweat", "separates": ["A", "B"]}]},
            EVIDENCE)
        assert state.evidence_gaps == [{"domain": "sweat",
                                        "separates": ["A", "B"]}]

    def test_17b_a_gap_naming_patterns_nobody_holds_is_dropped(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("A", "PLAUSIBLE", [COMPLAINT])],
             "evidence_gaps": [{"domain": "sweat", "separates": ["X", "Y"]}]},
            EVIDENCE)
        assert state.evidence_gaps == []

    def test_18_the_validated_state_produces_all_three_signal_kinds(self):
        """The state reaches CLARIFY2 ranking, or this phase changes nothing."""
        state, _ = validate_state(
            {"hypotheses": [
                hyp("风寒束表", "PRIMARY_WORKING", [OBS_SWEAT]),
                hyp("风热犯卫", "WEAKENED", [COMPLAINT], [OBS_TEMP])],
             "evidence_gaps": [{"domain": "thirst",
                                "separates": ["风寒束表", "风热犯卫"]}]},
            EVIDENCE)
        signals = signals_from_state(state)
        assert signals.gap_domains == frozenset({"thirst"})
        assert signals.contradiction_domains == frozenset({"cold_heat"})
        # 汗 is cited by exactly one of the two readings, so it is what
        # separates them -- which is the whole reason to carry the state.
        assert signals.discriminating_domains == frozenset({"sweat"})

    def test_18b_field_identifiers_resolve_by_alias_not_by_prose(self):
        """The bug this replaced: ASCII fields through the text matcher
        matched no Chinese marker and silently yielded no signal at all."""
        assert normalize_clinical_domain("aversion_to_cold") == "cold_heat"
        assert normalize_clinical_domain("sweating") == "sweat"
        assert normalize_clinical_domain("sweat") == "sweat"        # already a key
        assert normalize_clinical_domain("汗") == "sweat"            # Chinese fallback
        assert normalize_clinical_domain("sputum_colour") == "sputum"   # R1
        assert normalize_clinical_domain("cough_timing") is None    # no such domain
        assert normalize_clinical_domain("") is None

    def test_18c_a_complaint_citation_claims_no_domain(self):
        assert domain_for_ref(EvidenceRef(**COMPLAINT)) is None
        assert domain_for_ref(EvidenceRef(**OBS_SWEAT)) == "sweat"

    def test_18d_one_hypothesis_alone_separates_nothing(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("风寒束表", "PLAUSIBLE", [OBS_SWEAT])]},
            EVIDENCE)
        assert signals_from_state(state).discriminating_domains == frozenset()

    def test_18e_the_two_signal_paths_share_one_implementation(self):
        """A gap found from state must count for what a gap from prose counts
        for, so the rule lives in exactly one function."""
        from app.services.clarification import coverage as cov

        assert "signals_from_domains(" in code(cov.extract_differential_signals)
        assert "signals_from_domains(" in code(signals_from_state)

    def test_18f_no_state_yields_no_signals_rather_than_an_error(self):
        from app.services.clarification.coverage import DifferentialSignals

        assert signals_from_state(None) == DifferentialSignals()
        assert signals_from_state(
            WorkingDifferentialState()) == DifferentialSignals()

    def test_19_no_second_question_generator_exists(self):
        source = code(diff)
        for forbidden in ("clarification_proposals", "question", "propose",
                          "generate_"):
            assert forbidden not in source.lower().replace(
                "question_needed", "").replace("question_field", "")

    def test_20_no_new_ranking_pass_exists(self):
        from app.services.clarification import coverage as cov

        assert not hasattr(cov, "WEIGHT_SEPARATES_TOP_TWO")
        assert cov.WEIGHT_ENVELOPE_GAP == 20
        assert cov.WEIGHT_CONTRADICTION == 12
        assert cov.WEIGHT_DISCRIMINATING == 8

    def test_21_the_question_budget_is_unchanged(self):
        from app.services.clarification import coverage as cov
        from app.services.clarification.validator import (
            MAX_PROPOSALS_PER_TURN, REJECTION_REASONS)

        assert cov.MAX_ADAPTIVE_WHEN_OPEN == 3
        assert cov.MAX_VISIBLE_QUESTIONS_PER_TURN == 4
        assert cov.MIN_QUESTIONS_WHEN_INSUFFICIENT == 2
        assert MAX_PROPOSALS_PER_TURN == 3
        assert len(REJECTION_REASONS) == 10


# ======================================================================
# 22-24: nobody outside core may supply state
# ======================================================================

class TestStateOwnership:
    # Points 22-23 -- one case's state never reaching another case or another
    # user -- are properties of the query that loads it, so they are asserted
    # in core's suite, where that query lives.

    def test_22_23_ai_v2_holds_no_case_history_of_its_own(self):
        """It cannot leak across cases because it never keeps any."""
        assert "user_id" not in RecommendationRequest.model_fields
        source = code(diff)
        for forbidden in ("select", "query", "cache", "global ", "_STATE"):
            assert forbidden not in source

    def test_24_the_browser_cannot_inject_working_state(self):
        """ai-v2 accepts it only from the intake core builds."""
        source = code(oc.OpenAICompatibleProvider.generate_interview)
        assert "carry_state" in source
        # it is never read from patient text
        assert "text_input" in source
        assert "json.loads(text_input" not in source

    def test_24b_a_prior_state_is_still_re_validated(self):
        """Carried state does not get a pass just for having been carried."""
        from app.services.recommendation import assembler as module

        source = code(module.RecommendationAssembler.generate)
        assert "validate_state(" in source
        assert source.index("validate_state(") < source.index(
            "extract_differential_signals(")


# ======================================================================
# 32-35: full reasoning and the consumer see nothing
# ======================================================================

class TestIsolation:
    def test_32_full_reasoning_receives_no_working_state(self):
        """The hard anti-anchoring boundary."""
        from app.services.recommendation import assembler as module

        tree = ast.parse(textwrap.dedent(inspect.getsource(
            module.RecommendationAssembler.generate)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = ast.unparse(node.func)
            if target.endswith("generate_recommendation"):
                rendered = ast.unparse(node)
                assert "carry" not in rendered
                assert "interview_state" not in rendered

    def test_32b_the_full_prompt_never_mentions_carried_state(self):
        assert "PREVIOUS_WORKING_DIFFERENTIAL" not in oc.SYSTEM_PROMPT
        assert "working_differential" not in oc.SYSTEM_PROMPT

    def test_33_full_reasoning_may_disagree(self):
        """Nothing constrains the envelope to the interview's primary."""
        source = code(oc.OpenAICompatibleProvider.generate_recommendation)
        assert "carry_state" not in source

    def test_34_the_state_is_not_in_the_consumer_projection(self):
        from app.services.recommendation.consumer_projection import (
            build_consumer_reasoning)
        from app.schemas.reasoning import ClinicalReasoningEnvelope

        projection = build_consumer_reasoning(
            ClinicalReasoningEnvelope(clinical_summary="x"))
        assert "working_differential" not in projection
        assert "standing" not in json.dumps(projection, ensure_ascii=False)

    def test_34b_the_projection_builder_never_reads_it(self):
        from app.services.recommendation import consumer_projection as module

        assert "working_differential" not in code(module)

    def test_35_the_state_is_not_streamed(self):
        from app.services.llm.streaming import STREAMABLE_KEYS

        assert STREAMABLE_KEYS == ("summary", "interview_summary")
        assert "working_differential" not in STREAMABLE_KEYS


# ======================================================================
# 36-42: no authority, and failure goes the safe way
# ======================================================================

class TestAuthorityAndFailure:
    @pytest.mark.parametrize("forbidden", [
        "corpus", "REVIEWED", "VERIFIED", "verification", "PATTERN_FORMULA",
        "FORMULA_HERB", "clinical_ranking_eligible",
        "ready_for_formula_retrieval", "safety", "SafetyEngine",
        "consumer_purchasable", "resolved_product_id", "dosage",
        "administration", "insert", "commit", "session"])
    def test_36_40_the_module_has_no_authority_surface(self, forbidden):
        assert forbidden not in code(diff)

    def test_the_module_assigns_to_no_attribute(self):
        tree = ast.parse(inspect.getsource(diff))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for target in targets:
                assert not isinstance(target, ast.Attribute), \
                    ast.unparse(target)

    @pytest.mark.parametrize("junk", [
        None, 7, "text", [], {}, {"hypotheses": "nope"},
        {"hypotheses": [7, None, "x"]},
        {"hypotheses": [{"pattern_name": ""}]},
        {"hypotheses": [{"supporting_evidence": [1, 2]}]},
    ])
    def test_41_malformed_state_fails_toward_less(self, junk):
        state, _notes = validate_state(junk, EVIDENCE)
        assert state is None or state.hypotheses

    def test_41b_validation_never_raises(self):
        for junk in (object(), {"hypotheses": [{"pattern_name": "x",
                                                "standing": 7}]}):
            validate_state(junk, EVIDENCE)

    def test_42_a_provider_failure_cannot_fabricate_state(self):
        """No state without a model response, and none invented on failure."""
        state, _ = validate_state({}, EVIDENCE)
        assert state is None
        source = code(oc.OpenAICompatibleProvider._call_streaming)
        assert "working_differential" not in source

    def test_42b_an_empty_hypothesis_list_yields_no_state(self):
        state, _ = validate_state({"hypotheses": []}, EVIDENCE)
        assert state is None

    def test_no_numeric_confidence_anywhere_in_the_state(self):
        assert "confidence" not in WorkingHypothesis.model_fields
        assert "confidence" not in WorkingDifferentialState.model_fields
        assert "confidence" not in code(diff)


# ======================================================================
# 43-46: earlier phases untouched
# ======================================================================

class TestNoRegression:
    def test_43_streaming_is_unchanged(self):
        from app.services.llm.streaming import SummaryStreamScanner

        body = json.dumps({"summary": "外感风热。"}, ensure_ascii=False)
        scanner = SummaryStreamScanner()
        out = "".join(scanner.feed(body[:i]) for i in range(1, len(body) + 1))
        assert out == "外感风热。"

    def test_44_the_consumer_projection_is_unchanged(self):
        from app.services.recommendation.consumer_projection import (
            build_consumer_reasoning)
        from app.schemas.reasoning import ClinicalReasoningEnvelope

        projection = build_consumer_reasoning(ClinicalReasoningEnvelope(
            clinical_summary="摘要", pathogenesis="病机"))
        assert set(projection) <= {"summary", "eight_principle",
                                   "pattern_hypotheses", "pathogenesis",
                                   "treatment_principle",
                                   "missing_information", "uncertainty"}

    def test_45_depth_semantics_are_unchanged(self):
        from app.services.interview.mode import (MAX_INTERVIEW_TURNS,
                                                 REASON_DEPTH_REACHED,
                                                 decide_mode)

        assert MAX_INTERVIEW_TURNS == 3
        assert decide_mode(accumulated_text="咳嗽3天。", missing_information=[],
                           interview_depth=3) == ("FULL_REASONING",
                                                  REASON_DEPTH_REACHED)

    def test_45b_carried_state_cannot_extend_the_interview(self):
        from app.services.interview import mode

        assert "interview_state" not in code(mode)
        assert "working_differential" not in code(mode)

    def test_46_one_interview_turn_is_still_one_inference(self):
        from app.services.recommendation import assembler as module

        source = code(module.RecommendationAssembler.generate)
        assert source.count("await self.provider.generate_interview(") == 1
        assert source.count("await self.provider.generate_recommendation(") == 1

    def test_the_response_carries_the_validated_state_only(self):
        assert "working_differential" in ReasoningResponse.model_fields
        assert ReasoningResponse().working_differential == {}
