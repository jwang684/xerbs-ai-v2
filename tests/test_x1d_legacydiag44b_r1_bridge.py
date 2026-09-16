"""X1D-LEGACYDIAG4.4B-R1: the semantic bridge from answer to question.

What R1 repairs
---------------
4.4B carried a working differential correctly and changed it correctly, and
the questions did not diverge at all. The forked staging run measured why:

  * 9 of the 12 discriminator domains the model named were nose, throat or
    sputum -- and none of those existed in the canonical map, so every one was
    discarded before ranking;
  * every citation resolved to COMPLAINT or to turn-1 answers about
    sleep/diet/excretion, because the citable list held the fields that had
    been ASKED, and 寒热/汗/渴 had not been.

So the differential was computing real discriminators and handing them to a
ranker with no name for them, while being unable to cite the answers that
would have settled them.

This file tests the bridge, one property per test:

    patient answer -> canonical domain -> EvidenceRef -> working state
        -> canonical discriminator -> differential signals -> ranking

No ranking weight, question budget, provider call or question generator is
touched by any of it; several tests below exist only to pin that down.
"""

import ast
import inspect
import textwrap

import pytest

from app.schemas.reasoning import (
    EvidenceRef,
    ResolvableEvidence,
    WorkingDifferentialState,
    WorkingHypothesis,
)
from app.services.clarification import coverage as cov
from app.services.clarification.coverage import (
    DOMAINS_BY_KEY,
    KNOWN,
    DifferentialSignals,
    assess_coverage,
    domain_for_field,
    normalize_clinical_domain,
    resolve_domain,
    score_candidate,
)
from app.services.interview.differential import (
    answered_domains,
    domain_for_ref,
    resolve_ref,
    signals_from_state,
    validate_state,
)
from app.services.llm import openai_compatible as oc


# Every domain name the model actually emitted during the failed 4.4B fork,
# with the count it was seen. Kept verbatim -- this is measurement, not design.
FORK_ALIASES = {
    "nasal_symptoms": ("nose", 3),
    "nasal_discharge": ("nose", 1),
    "throat_pain": ("throat", 1),
    "throat": ("throat", 1),
    "sputum_amount": ("sputum", 1),
    "sputum_character": ("sputum", 1),
    "cough_sputum_detail": ("sputum", 1),
    "sweat_detail": ("sweat", 1),
    "sweat": ("sweat", 1),
    "nasal_symptoms ": ("nose", 0),      # trailing space, same domain
}


def code(obj):
    """Executable source only.

    Docstrings are stripped before any source assertion. A comment saying
    "no embeddings, no model call" would otherwise fail a test looking for
    the word "embed" -- the prose describing a boundary is not a breach of it.
    """
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


def answered(field, answer, turn_id=1):
    return {"turn_id": turn_id, "question_field": field, "answer": answer}


# ======================================================================
# 1-5: the alias table, and what it refuses
# ======================================================================

class TestNormalization:
    @pytest.mark.parametrize("alias,expected",
                             [(a, v[0]) for a, v in FORK_ALIASES.items()])
    def test_1_every_alias_the_fork_produced_now_resolves(self, alias,
                                                          expected):
        """Point 1: each observed alias has an explicit expected result."""
        assert normalize_clinical_domain(alias) == expected

    def test_2_an_unknown_domain_returns_none(self):
        for unknown in ("cough_timing", "qi_level", "wind_evil_depth",
                        "completely_made_up", "干支", "42"):
            assert normalize_clinical_domain(unknown) is None

    def test_3_normalization_is_deterministic(self):
        for name in list(FORK_ALIASES) + ["cold_heat", "口渴", "nope"]:
            results = {normalize_clinical_domain(name) for _ in range(20)}
            assert len(results) == 1

    def test_4_no_fuzzy_matching_is_used(self):
        """Point 4: an allowlist, not a similarity score."""
        source = code(cov.normalize_clinical_domain) + code(cov.domain_for_field)
        for forbidden in ("difflib", "ratio", "levenshtein", "embed",
                          "similar", "fuzz", "cosine", "SequenceMatcher"):
            assert forbidden not in source.lower()
        # near-misses do NOT resolve; only exact members do
        assert normalize_clinical_domain("nasal_symptom") == "nose"  # substring alias
        assert normalize_clinical_domain("nsal_symptoms") is None
        assert normalize_clinical_domain("thraot_pain") is None

    def test_5_a_canonical_domain_maps_onto_a_clarify_question_family(self):
        """Point 5: every canonical key is a domain CLARIFY can actually ask."""
        for key in ("nose", "throat", "sputum"):
            domain = DOMAINS_BY_KEY[key]
            assert domain.default_question
            assert domain.aliases
            assert domain.strong

    def test_5b_the_three_new_domains_are_material_to_respiratory(self):
        respiratory = next(p for p in cov.FOCUS_PROFILES
                           if p.key == "respiratory")
        assert {"nose", "throat", "sputum"} <= set(respiratory.material)

    def test_5c_they_were_appended_so_no_existing_weight_moved(self):
        """Position is weight. Inserting would have re-scored the other seven."""
        respiratory = next(p for p in cov.FOCUS_PROFILES
                           if p.key == "respiratory")
        assert respiratory.material[:7] == (
            "cold_heat", "sweat", "thirst", "head_body", "onset_duration",
            "chest_abdomen", "cause")


# ======================================================================
# 6-11: the answers that were missing now reach citable evidence
# ======================================================================

class TestCitableEvidence:
    """Points 6-11. Each domain the fork could not cite, now citable."""

    @pytest.mark.parametrize("field,value,domain", [
        ("chills_or_heat", "怕冷明显，盖被子也冷", "cold_heat"),   # 6
        ("sweating", "一点汗都没有", "sweat"),                    # 7
        ("thirst", "不口渴", "thirst"),                           # 8
        ("sputum_colour", "痰是白的，稀", "sputum"),              # 9
        ("nasal_discharge", "流清鼻涕", "nose"),                  # 10
        ("sore_throat", "咽喉不痛", "throat"),                    # 11
    ])
    def test_6_11_an_answered_field_is_citable_by_its_domain(
            self, field, value, domain):
        evidence = ResolvableEvidence(answers=[answered(field, value)])
        assert resolve_ref({"origin": "ANSWER", "question_field": field},
                           evidence) is not None
        # and by the canonical name the model is likely to use instead
        assert resolve_ref({"origin": "ANSWER", "question_field": domain},
                           evidence) is not None
        assert domain in answered_domains(evidence)

    def test_an_unanswered_question_is_not_citable(self):
        """Point 13's companion: asking is not evidence."""
        assert resolve_ref({"origin": "ANSWER", "question_field": "thirst"},
                           ResolvableEvidence(answers=[])) is None


# ======================================================================
# 12-15: provenance -- the answer is evidence, the question is not
# ======================================================================

class TestProvenance:
    EVIDENCE = ResolvableEvidence(
        answers=[answered("chills_or_heat", "怕冷明显")])

    def test_12_the_record_carries_the_patient_value(self):
        assert self.EVIDENCE.answers[0]["answer"] == "怕冷明显"

    def test_13_the_question_wording_is_not_in_the_record(self):
        """The rule 4.4A named and R1 had to enforce on the wire.

        "您目前是怕冷、发热，还是两者都有？ 怕冷为主" is one string in the
        browser's prose channel. If that string were the evidence, a
        model-written interrogative would be indistinguishable from something
        the patient said.
        """
        blob = str(self.EVIDENCE.model_dump())
        assert "怕冷明显" in blob
        assert "？" not in blob
        assert "您目前是" not in blob

    def test_13b_a_citation_resolves_to_an_answer_not_to_a_question(self):
        source = code(resolve_ref)
        assert "evidence.answers" in source
        assert "question_text" not in source
        assert "clarification_questions" not in source

    def test_14_an_answer_resolves_by_turn_identity(self):
        evidence = ResolvableEvidence(
            answers=[answered("sweating", "无汗", turn_id=2)])
        ok = {"origin": "ANSWER", "question_field": "sweating", "turn_id": 2}
        wrong = {"origin": "ANSWER", "question_field": "sweating",
                 "turn_id": 1}
        assert resolve_ref(ok, evidence) is not None
        assert resolve_ref(wrong, evidence) is None

    def test_15_model_alias_and_question_field_meet_on_one_domain(self):
        """The bridge, in one assertion.

        The patient answered a question filed as `chills_or_heat`. The model
        reasons about `cold_heat`. Before R1 those never met and the citation
        was dropped as a fabrication.
        """
        assert normalize_clinical_domain("chills_or_heat") == \
            normalize_clinical_domain("cold_heat") == "cold_heat"
        assert resolve_ref({"origin": "ANSWER", "question_field": "cold_heat"},
                           self.EVIDENCE) is not None


# ======================================================================
# 16-21: each fork alias, deterministically handled end to end
# ======================================================================

class TestForkAliasesEndToEnd:
    @pytest.mark.parametrize("alias,expected", [
        ("nasal_symptoms", "nose"),        # 16
        ("nasal_discharge", "nose"),       # 17
        ("throat_pain", "throat"),         # 18
        ("sputum_amount", "sputum"),       # 19
        ("sputum_character", "sputum"),    # 20
        ("cough_sputum_detail", "sputum"), # 21
    ])
    def test_16_21_the_alias_survives_into_a_gap_signal(self, alias, expected):
        """Not silently dropped: it reaches the ranker as its domain.

        4.5 requires the discriminator to name its competition; the property
        under test here is still the alias, which must canonicalize whatever
        else changes around it.
        """
        state, _ = validate_state(
            {"hypotheses": [
                {"pattern_name": "风寒束表", "standing": "PLAUSIBLE",
                 "supporting_evidence": [{"origin": "COMPLAINT"}],
                 "unresolved_discriminators": [
                     {"domain": alias,
                      "separates": ["风寒束表", "风热犯表"],
                      "if_present_supports": ["风热犯表"],
                      "if_absent_supports": ["风寒束表"]}]},
                {"pattern_name": "风热犯表", "standing": "PLAUSIBLE",
                 "supporting_evidence": [{"origin": "COMPLAINT"}]},
            ]},
            ResolvableEvidence())
        assert expected in signals_from_state(state).gap_domains

    def test_an_unmappable_discriminator_still_fails_toward_less(self):
        state, _ = validate_state(
            {"hypotheses": [{"pattern_name": "x", "standing": "PLAUSIBLE",
                             "supporting_evidence": [{"origin": "COMPLAINT"}],
                             "unresolved_discriminators": [
                                 {"domain": "qi_transformation_depth",
                                  "separates": ["x", "y"],
                                  "if_present_supports": ["x"],
                                  "if_absent_supports": ["y"]}]},
                            {"pattern_name": "y", "standing": "PLAUSIBLE",
                             "supporting_evidence": [{"origin": "COMPLAINT"}]}]},
            ResolvableEvidence())
        assert signals_from_state(state).gap_domains == frozenset()


# ======================================================================
# 22: an answered domain is not re-asked under another name
# ======================================================================

class TestReAskSuppression:
    TEXT = "咳嗽发热3天。咳嗽有白痰，怕冷明显，无汗。"

    def test_22_an_answered_domain_is_suppressed_whatever_the_alias(self):
        """Point 22, and the concrete症状 the 4.4B fork showed.

        ALPHA answered 无汗 and BETA answered 有汗, and three of the four runs
        were then asked about 汗 again -- because 汗 had been answered under
        one name and re-proposed under another, with nothing comparing them.
        """
        coverage = assess_coverage(self.TEXT)
        assert coverage.states["sweat"] == KNOWN
        assert coverage.states["sputum"] == KNOWN
        for alias in ("sweat_detail", "sweating", "perspiration"):
            domain, certain = resolve_domain(alias, "出汗情况如何？")
            candidate = cov.Candidate(
                field=alias, question="出汗情况如何？", kind="adaptive",
                domain=domain, domain_certain=certain)
            assert score_candidate(candidate, coverage,
                                   DifferentialSignals()) is None

    def test_22b_an_unanswered_domain_is_still_asked(self):
        coverage = assess_coverage(self.TEXT)
        candidate = cov.Candidate(
            field="sore_throat", question="咽喉痛吗？", kind="adaptive",
            domain="throat", domain_certain=True)
        assert score_candidate(candidate, coverage,
                               DifferentialSignals()) is not None

    def test_22c_suppression_compares_canonical_identity_not_wording(self):
        assert resolve_domain("sweat_detail", "")[0] == \
            resolve_domain("sweating", "")[0] == "sweat"

    @pytest.mark.parametrize("answer,domain", [
        # word order: a patient says 痰黄 as readily as 黄痰
        ("痰黄，很黏稠", "sputum"),
        ("痰是白的，很稀", "sputum"),
        ("黄痰", "sputum"),
        # negation: "没有" is an answer, not an absence of one
        ("没有痰", "sputum"),
        ("咽喉不痛也不痒", "throat"),
        ("咽喉痛得厉害", "throat"),
        ("鼻塞，流清鼻涕", "nose"),
        ("无鼻塞流涕", "nose"),
    ])
    def test_22e_an_answer_reads_as_known_in_either_word_order(
            self, answer, domain):
        """The first R1 acceptance run caught this and it is worth a test.

        BETA answered 痰黄，很黏稠 and was asked about their phlegm again: the
        sputum markers held 黄痰 and not 痰黄, so the answer scored PARTIAL and
        stayed askable. The original twelve domains already model both forms --
        寒热 carries 不怕冷 and 渴饮 carries 不渴 -- and the three R1 added had
        shipped without their equivalents.
        """
        states = assess_coverage("发热咳嗽三天。" + answer).states
        assert states[domain] == KNOWN

    def test_22d_no_fuzzy_dedup_was_introduced(self):
        source = code(cov)
        for forbidden in ("difflib", "SequenceMatcher", "embedding",
                          "cosine", "fuzzy"):
            assert forbidden not in source.lower()


# ======================================================================
# 23-24: the signals the ranker actually receives
# ======================================================================

class TestSignalsReachRanking:
    def test_23_a_separable_pair_produces_a_discriminating_domain(self):
        """Point 23. 汗 cited by one reading and not the other separates them."""
        evidence = ResolvableEvidence(
            answers=[answered("sweating", "无汗"),
                     answered("sore_throat", "咽痛明显")])
        state, _ = validate_state({"hypotheses": [
            {"pattern_name": "风寒束表", "standing": "PRIMARY_WORKING",
             "supporting_evidence": [
                 {"origin": "ANSWER", "question_field": "sweat"}]},
            {"pattern_name": "风热犯表", "standing": "PLAUSIBLE",
             "supporting_evidence": [
                 {"origin": "ANSWER", "question_field": "throat"}]},
        ]}, evidence)
        signals = signals_from_state(state)
        assert signals.discriminating_domains == frozenset({"sweat", "throat"})

    def test_24_an_unresolved_discriminator_produces_a_gap_domain(self):
        state, _ = validate_state({"hypotheses": [
            {"pattern_name": "风寒束表", "standing": "PLAUSIBLE",
             "supporting_evidence": [{"origin": "COMPLAINT"}],
             "unresolved_discriminators": [
                 {"domain": "sputum_character",
                  "separates": ["风寒束表", "风热犯表"],
                  "if_present_supports": ["风热犯表"],
                  "if_absent_supports": ["风寒束表"]}]},
            {"pattern_name": "风热犯表", "standing": "PLAUSIBLE",
             "supporting_evidence": [{"origin": "COMPLAINT"}]},
        ], "evidence_gaps": [{"domain": "throat_pain",
                              "separates": ["风寒束表", "风热犯表"]}]},
            ResolvableEvidence())
        signals = signals_from_state(state)
        assert {"sputum", "throat"} <= signals.gap_domains

    def test_24b_a_gap_domain_actually_lifts_that_question(self):
        """The whole point: the signal changes what is asked."""
        coverage = assess_coverage("咳嗽发热3天。")
        candidate = cov.Candidate(
            field="sore_throat", question="咽喉痛吗？", kind="adaptive",
            domain="throat", domain_certain=True)
        without = score_candidate(candidate, coverage, DifferentialSignals())
        with_gap = score_candidate(
            candidate, coverage,
            DifferentialSignals(gap_domains=frozenset({"throat"})))
        assert with_gap.score > without.score
        assert with_gap.score - without.score == cov.WEIGHT_ENVELOPE_GAP


# ======================================================================
# 25-28: nothing else moved
# ======================================================================

class TestNothingElseMoved:
    def test_25_ranking_weights_are_unchanged(self):
        assert cov.WEIGHT_ENVELOPE_GAP == 20
        assert cov.WEIGHT_CONTRADICTION == 12
        assert cov.WEIGHT_DISCRIMINATING == 8
        assert cov.WEIGHT_COMPLAINT_LINK == 10
        assert not hasattr(cov, "WEIGHT_SEPARATES_TOP_TWO")

    def test_25b_focus_weight_constants_are_unchanged(self):
        assert cov.FOCUS_WEIGHT_TOP == 40
        assert cov.FOCUS_WEIGHT_STEP == 4
        assert cov.FOCUS_WEIGHT_FLOOR == 8
        assert cov.FOCUS_WEIGHT_GENERAL == 20
        assert cov.FOCUS_WEIGHT_UNMAPPED == 18

    def test_26_the_question_budget_is_unchanged(self):
        assert cov.MAX_ADAPTIVE_WHEN_OPEN == 3
        assert cov.MAX_ADAPTIVE_WHEN_NARROWING == 1
        assert cov.MAX_VISIBLE_QUESTIONS_PER_TURN == 4
        assert cov.MIN_QUESTIONS_WHEN_INSUFFICIENT == 2

    def test_27_no_second_question_generator_exists(self):
        from app.services.interview import differential as diff

        source = code(diff).lower().replace("question_needed", "").replace(
            "question_field", "")
        for forbidden in ("clarification_proposals", "propose", "generate_"):
            assert forbidden not in source

    def test_28_one_interview_turn_is_still_one_inference(self):
        from app.services.recommendation import assembler as module

        source = code(module.RecommendationAssembler.generate)
        assert source.count("await self.provider.generate_interview(") == 1
        assert source.count("await self.provider.generate_recommendation(") == 1

    def test_28b_normalization_makes_no_model_call(self):
        source = code(cov.normalize_clinical_domain)
        for forbidden in ("await", "provider", "openai", "http", "client"):
            assert forbidden not in source.lower()


# ======================================================================
# 29-31: the 4.4B rules survive R1 intact
# ======================================================================

class TestFourFourBRulesIntact:
    PRIOR = WorkingDifferentialState(hypotheses=[
        WorkingHypothesis(pattern_name="风寒束表", standing="PLAUSIBLE",
                          supporting_evidence=[EvidenceRef(origin="COMPLAINT")])])

    def test_30_anti_escalation_still_holds(self):
        state, notes = validate_state(
            {"hypotheses": [{"pattern_name": "风寒束表",
                             "standing": "PRIMARY_WORKING",
                             "supporting_evidence": [{"origin": "COMPLAINT"}]}]},
            ResolvableEvidence(), self.PRIOR)
        assert state.hypotheses[0].standing == "PLAUSIBLE"
        assert "STANDING_ESCALATION_WITHOUT_NEW_EVIDENCE" in notes

    def test_30b_a_newly_citable_answer_is_legitimate_new_evidence(self):
        """R1 widens what can be cited; it does not widen what may escalate.

        The standing rises here because the patient answered something new,
        which is exactly the condition 4.4B requires -- not because the alias
        table made an old citation resolve.
        """
        evidence = ResolvableEvidence(answers=[answered("sweating", "无汗")])
        state, notes = validate_state(
            {"hypotheses": [{"pattern_name": "风寒束表",
                             "standing": "PRIMARY_WORKING",
                             "supporting_evidence": [
                                 {"origin": "COMPLAINT"},
                                 {"origin": "ANSWER",
                                  "question_field": "sweat"}]}]},
            evidence, self.PRIOR)
        assert state.hypotheses[0].standing == "PRIMARY_WORKING"
        assert "STANDING_ESCALATION_WITHOUT_NEW_EVIDENCE" not in notes

    def test_ruled_out_remains_reversible(self):
        from app.services.interview.differential import describe_policy

        assert describe_policy()["ruled_out_is_reversible"] is True

    def test_29_persistence_shape_is_unchanged(self):
        state, _ = validate_state(
            {"hypotheses": [{"pattern_name": "x", "standing": "PLAUSIBLE",
                             "supporting_evidence": [{"origin": "COMPLAINT"}]}]},
            ResolvableEvidence())
        dumped = state.model_dump(mode="json")
        assert set(dumped) == {"turn_id", "hypotheses", "evidence_gaps"}

    def test_31_an_invalid_ref_is_still_dropped(self):
        state, notes = validate_state(
            {"hypotheses": [{"pattern_name": "风热犯表", "standing": "PLAUSIBLE",
                             "supporting_evidence": [
                                 {"origin": "OBSERVATION",
                                  "field": "tongue_coating_yellow"}]}]},
            ResolvableEvidence())
        assert state.hypotheses[0].supporting_evidence == []
        assert "EVIDENCE_REF_UNRESOLVED" in notes

    def test_31b_the_alias_table_cannot_manufacture_evidence(self):
        """Normalizing a name never asserts the patient answered it."""
        evidence = ResolvableEvidence(answers=[])
        assert normalize_clinical_domain("nasal_symptoms") == "nose"
        assert resolve_ref({"origin": "ANSWER", "question_field": "nose"},
                           evidence) is None
        assert answered_domains(evidence) == set()

    @pytest.mark.parametrize("origin", [
        "MODEL_TEXT", "SUMMARY", "HYPOTHESIS", "IMAGE", "TONGUE_IMAGE",
        "INFERRED_FINDING", "PROVIDER_OUTPUT"])
    def test_forbidden_origins_are_still_refused(self, origin):
        assert resolve_ref({"origin": origin, "field": "sweating"},
                           ResolvableEvidence(
                               answers=[answered("sweating", "无汗")])) is None


# ======================================================================
# 34-41: the earlier phases, untouched
# ======================================================================

class TestEarlierPhases:
    def test_34_full_reasoning_still_receives_no_working_state(self):
        from app.services.recommendation import assembler as module

        tree = ast.parse(textwrap.dedent(inspect.getsource(
            module.RecommendationAssembler.generate)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    ast.unparse(node.func).endswith("generate_recommendation"):
                rendered = ast.unparse(node)
                assert "carry" not in rendered
                assert "interview_state" not in rendered

    def test_34b_the_full_prompt_still_knows_nothing_of_it(self):
        assert "working_differential" not in oc.SYSTEM_PROMPT
        assert "PREVIOUS WORKING DIFFERENTIAL" not in oc.SYSTEM_PROMPT

    def test_35_the_state_is_still_consumer_invisible(self):
        from app.schemas.reasoning import ClinicalReasoningEnvelope
        from app.services.recommendation.consumer_projection import (
            build_consumer_reasoning)

        projection = build_consumer_reasoning(
            ClinicalReasoningEnvelope(clinical_summary="x"))
        assert "working_differential" not in projection
        assert "standing" not in str(projection)

    def test_36_37_38_the_module_still_has_no_authority_surface(self):
        from app.services.interview import differential as diff

        source = code(diff)
        for forbidden in ("corpus", "REVIEWED", "SafetyEngine", "safety",
                          "consumer_purchasable", "dosage",
                          "ready_for_formula_retrieval"):
            assert forbidden not in source

    def test_39_streaming_is_unchanged(self):
        from app.services.llm.streaming import STREAMABLE_KEYS

        assert STREAMABLE_KEYS == ("summary", "interview_summary")

    def test_40_the_projection_whitelist_is_unchanged(self):
        from app.schemas.reasoning import ClinicalReasoningEnvelope
        from app.services.recommendation.consumer_projection import (
            build_consumer_reasoning)

        projection = build_consumer_reasoning(ClinicalReasoningEnvelope(
            clinical_summary="摘要", pathogenesis="病机"))
        assert set(projection) <= {"summary", "eight_principle",
                                   "pattern_hypotheses", "pathogenesis",
                                   "treatment_principle",
                                   "missing_information", "uncertainty"}

    def test_41_depth_semantics_are_unchanged(self):
        from app.services.interview.mode import (MAX_INTERVIEW_TURNS,
                                                 REASON_DEPTH_REACHED,
                                                 decide_mode)

        assert MAX_INTERVIEW_TURNS == 3
        assert decide_mode(accumulated_text="咳嗽3天。",
                           missing_information=[],
                           interview_depth=3) == ("FULL_REASONING",
                                                  REASON_DEPTH_REACHED)


# ======================================================================
# The prompt asks for canonical names -- and is not trusted to deliver them
# ======================================================================

class TestPromptContract:
    def test_the_interview_prompt_names_the_allowed_domains(self):
        prompt = oc.INTERVIEW_SYSTEM_PROMPT
        for key in DOMAINS_BY_KEY:
            assert key in prompt

    def test_the_prompt_is_not_the_enforcement(self):
        """A prompt is a request. The normalizer is the boundary.

        The model emitted seven invented domain names during the 4.4B fork
        while under a contract that told it what shape to use. Asking again,
        more specifically, is worth doing and is not worth trusting.
        """
        state, _ = validate_state(
            {"hypotheses": [{"pattern_name": "x", "standing": "PLAUSIBLE",
                             "supporting_evidence": [{"origin": "COMPLAINT"}],
                             "unresolved_discriminators": [
                                 {"domain": "a_name_the_prompt_forbade"}]}]},
            ResolvableEvidence())
        assert signals_from_state(state).gap_domains == frozenset()
