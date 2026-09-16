"""X1D-LEGACYDIAG4.5: a question has to earn its slot by separating something.

What the three failed acceptances established
---------------------------------------------
4.4B carried the differential. R1 gave it a vocabulary and real provenance.
R2 made the selector honour it, and staging confirmed ``asked ⊆ required``.
And still the two branches asked the same questions. R2 measured why:

    mean WITHIN-fork  required similarity : 0.45
    mean BETWEEN-fork required similarity : 0.70

Two branches holding opposite readings wanted *more similar* discriminators
than one branch wanted across two runs of itself. The selector was faithfully
selecting; the set it was given carried no branch signal.

Phase 0 found the mechanism, and it was not subtle: of 50 discriminators the
model emitted across the R2 runs, **zero** named the hypotheses they separated
-- the schema never asked. Their stated reasons were overwhelmingly
confirmatory ("有没有咽喉痛，以支持风热"). Each hypothesis independently listed
what would CONFIRM it, so the union across three readings came out much the
same whichever one was leading.

So 4.5 changes what a discriminator has to say. Two deterministic rules, and
neither is a weight:

  * it must name at least two DISTINCT LIVE hypotheses of this very state --
    a competition that is not happening buys nothing;
  * it must say what a positive and a negative answer would each favour, and
    those must differ. A finding that supports the same reading either way
    confirms rather than separates, and does not get one of three slots.

The second is the whole idea: the model has to demonstrate that asking would
change its mind.

The danger being avoided
------------------------
A differential allowed to silence questions is a closed tunnel, and the
working state is incomplete by construction. So an entry that names a real
competition without the demonstration is KEPT in the record and merely denied
priority, generic coverage still fills every unused slot, and a single live
hypothesis is never padded with an invented rival.
"""

import ast
import inspect
import textwrap

import pytest

from app.schemas.reasoning import (
    ResolvableEvidence,
    WorkingDifferentialState,
)
from app.services.clarification import coverage as cov
from app.services.clarification.coverage import (
    KNOWN,
    Candidate,
    DifferentialSignals,
    assess_coverage,
    select_questions,
)
from app.services.interview import differential as diff
from app.services.interview.differential import (
    LIVE_STANDINGS,
    describe_policy,
    differential_required_domains,
    signals_from_state,
    validate_state,
)
from app.services.llm import openai_compatible as oc


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


# Neutral synthetic labels throughout. Nothing here encodes the acceptance
# case, and no clinical pattern is special-cased anywhere in this file.
def hypothesis(name, standing="PLAUSIBLE", discriminators=()):
    return {"pattern_name": name, "standing": standing,
            "supporting_evidence": [{"origin": "COMPLAINT"}],
            "unresolved_discriminators": list(discriminators)}


def disc(domain, separates=("A", "B"), present=None, absent=None,
         rationale="because it would move things"):
    entry = {"domain": domain, "separates": list(separates),
             "rationale": rationale}
    if present is not None:
        entry["if_present_supports"] = list(present)
    if absent is not None:
        entry["if_absent_supports"] = list(absent)
    return entry


GOOD = disc("sweat", ("A", "B"), present=["B"], absent=["A"])


def build(*hypotheses, gaps=()):
    raw = {"hypotheses": list(hypotheses)}
    if gaps:
        raw["evidence_gaps"] = list(gaps)
    return validate_state(raw, ResolvableEvidence())


# ======================================================================
# 1-6: a discriminator must name a real competition
# ======================================================================

class TestSeparatesValidation:
    def test_1_it_references_current_live_hypotheses(self):
        state, _ = build(hypothesis("A", "PRIMARY_WORKING", [GOOD]),
                         hypothesis("B"))
        entry = state.hypotheses[0].unresolved_discriminators[0]
        assert entry["separates"] == ["A", "B"]
        assert set(entry["separates"]) <= {h.pattern_name
                                           for h in state.hypotheses}

    def test_2_it_must_separate_at_least_two_distinct_hypotheses(self):
        state, notes = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("sweat", ("A",), present=["A"], absent=["A"])]),
            hypothesis("B"))
        assert state.hypotheses[0].unresolved_discriminators == []
        assert "DISCRIMINATOR_NO_LIVE_COMPETITION" in notes

    def test_3_an_unknown_hypothesis_name_is_rejected(self):
        """Naming a rival it never listed describes a competition that is not
        happening, and must not buy differential priority."""
        state, notes = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("sweat", ("A", "NEVER_PROPOSED"),
                             present=["A"], absent=["NEVER_PROPOSED"])]))
        assert state.hypotheses[0].unresolved_discriminators == []
        assert "DISCRIMINATOR_NO_LIVE_COMPETITION" in notes

    def test_4_duplicate_names_collapse_and_then_fail_the_count(self):
        state, _ = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("sweat", ("A", "A", "A"),
                             present=["A"], absent=["A"])]),
            hypothesis("B"))
        assert state.hypotheses[0].unresolved_discriminators == []

    def test_4b_a_duplicate_alongside_a_real_rival_still_counts_once(self):
        state, _ = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("sweat", ("A", "A", "B"),
                             present=["B"], absent=["A"])]),
            hypothesis("B"))
        assert state.hypotheses[0].unresolved_discriminators[0][
            "separates"] == ["A", "B"]

    def test_5_the_domain_must_be_canonical(self):
        state, _ = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("nasal_symptoms", present=["B"], absent=["A"])]),
            hypothesis("B"))
        assert state.hypotheses[0].unresolved_discriminators[0][
            "domain"] == "nose"

    def test_6_an_unknown_domain_is_rejected(self):
        state, _ = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("qi_depth", present=["B"], absent=["A"])]),
            hypothesis("B"))
        assert state.hypotheses[0].unresolved_discriminators == []

    @pytest.mark.parametrize("junk", [
        None, 7, "text", [], {}, {"domain": None},
        {"domain": "sweat", "separates": "AB"},
        {"domain": "sweat", "separates": [None, 7]}])
    def test_malformed_entries_fail_toward_less(self, junk):
        state, _ = build(hypothesis("A", "PRIMARY_WORKING", [junk]),
                         hypothesis("B"))
        assert state.hypotheses[0].unresolved_discriminators == []


# ======================================================================
# The counterfactual: show that the answer would change your mind
# ======================================================================

class TestCounterfactual:
    def test_a_finding_that_cuts_both_ways_earns_no_slot(self):
        """The core of 4.5, in one test.

        Both answers favouring the same reading is confirmation wearing the
        costume of discrimination -- and Phase 0 showed that is what the model
        was actually producing.
        """
        state, notes = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("sweat", present=["A"], absent=["A"])]),
            hypothesis("B"))
        entry = state.hypotheses[0].unresolved_discriminators[0]
        assert entry["discriminating"] is False
        assert "DISCRIMINATOR_WITHOUT_COUNTERFACTUAL" in notes
        assert differential_required_domains(state) == set()

    def test_a_finding_that_cuts_differently_earns_one(self):
        state, _ = build(hypothesis("A", "PRIMARY_WORKING", [GOOD]),
                         hypothesis("B"))
        assert state.hypotheses[0].unresolved_discriminators[0][
            "discriminating"] is True
        assert differential_required_domains(state) == {"sweat"}

    def test_a_missing_counterfactual_is_kept_but_unprivileged(self):
        """Kept because it is honest reasoning; unprivileged because it has
        not shown that asking would change anything."""
        state, notes = build(
            hypothesis("A", "PRIMARY_WORKING", [disc("sweat")]),
            hypothesis("B"))
        entry = state.hypotheses[0].unresolved_discriminators[0]
        assert entry["domain"] == "sweat"
        assert entry["discriminating"] is False
        assert "DISCRIMINATOR_WITHOUT_COUNTERFACTUAL" in notes
        assert differential_required_domains(state) == set()

    def test_the_counterfactual_names_are_themselves_checked(self):
        state, _ = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("sweat", present=["GHOST"], absent=["A"])]),
            hypothesis("B"))
        entry = state.hypotheses[0].unresolved_discriminators[0]
        assert entry["if_present_supports"] == []
        assert entry["discriminating"] is False


# ======================================================================
# 7-8: reasoning is never evidence
# ======================================================================

class TestRationaleIsNotEvidence:
    RATIONALE = "the patient obviously has a yellow tongue coating"

    def test_7_a_rationale_cannot_become_patient_evidence(self):
        state, _ = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("sweat", present=["B"], absent=["A"],
                             rationale=self.RATIONALE)]),
            hypothesis("B"))
        entry = state.hypotheses[0].unresolved_discriminators[0]
        assert entry["rationale"] == self.RATIONALE
        # It is prose in the record and it is not a citation. The hypothesis
        # cites the complaint and nothing else -- the tongue the rationale
        # asserts was never reported by anyone, and stays uncited.
        assert [r.origin for r in state.hypotheses[0].supporting_evidence] ==             ["COMPLAINT"]
        assert all(r.field is None
                   for r in state.hypotheses[0].supporting_evidence)
        from app.services.interview.differential import resolve_ref

        assert resolve_ref({"origin": "OBSERVATION",
                            "field": "tongue_coating"},
                           ResolvableEvidence()) is None

    def test_8_counterfactual_fields_cannot_become_patient_evidence(self):
        state, _ = build(hypothesis("A", "PRIMARY_WORKING", [GOOD]),
                         hypothesis("B"))
        signals = signals_from_state(state)
        # they produce a GAP (something still unknown), never a citation
        assert "sweat" in signals.gap_domains
        assert all(not h.supporting_evidence or
                   all(r.origin == "COMPLAINT" for r in h.supporting_evidence)
                   for h in state.hypotheses)

    def test_the_rationale_is_bounded(self):
        state, _ = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("sweat", present=["B"], absent=["A"],
                             rationale="x" * 5000)]),
            hypothesis("B"))
        assert len(state.hypotheses[0].unresolved_discriminators[0][
            "rationale"]) <= diff.MAX_RATIONALE_CHARS


# ======================================================================
# 9-14: standings, and what is already answered
# ======================================================================

class TestStandingsAndKnown:
    def test_9_an_already_known_domain_is_not_proposed(self):
        """Server-side suppression is the authority; the prompt also says it."""
        assert "ALREADY_KNOWN" in oc.INTERVIEW_SYSTEM_PROMPT
        source = code(oc.OpenAICompatibleProvider.generate_interview)
        assert "known_domains" in source
        coverage = assess_coverage("发热咳嗽三天。咳嗽有白痰，无汗。")
        assert coverage.states["sputum"] == KNOWN
        candidate = Candidate(field="sputum_q", question="痰？",
                              domain="sputum", kind="adaptive",
                              domain_certain=True, payload={})
        assert cov.score_candidate(
            candidate, coverage,
            DifferentialSignals(gap_domains=frozenset({"sputum"}))) is None

    def test_10_an_unresolved_domain_survives(self):
        state, _ = build(hypothesis("A", "PRIMARY_WORKING", [GOOD]),
                         hypothesis("B"))
        assert "sweat" in signals_from_state(state).gap_domains

    def test_11_primary_and_plausible_may_be_separated(self):
        state, _ = build(hypothesis("A", "PRIMARY_WORKING", [GOOD]),
                         hypothesis("B", "PLAUSIBLE"))
        assert differential_required_domains(state) == {"sweat"}

    def test_12_plausible_and_plausible_may_be_separated(self):
        state, _ = build(hypothesis("A", "PLAUSIBLE", [GOOD]),
                         hypothesis("B", "PLAUSIBLE"))
        assert differential_required_domains(state) == {"sweat"}

    def test_13_ruled_out_for_now_cannot_drive_a_discriminator(self):
        state, _ = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("sweat", ("A", "B"), present=["B"],
                             absent=["A"])]),
            hypothesis("B", "RULED_OUT_FOR_NOW"))
        assert state.hypotheses[0].unresolved_discriminators == []
        assert differential_required_domains(state) == set()

    def test_14_weakened_matches_the_r2_semantics(self):
        assert LIVE_STANDINGS == ("PRIMARY_WORKING", "PLAUSIBLE")
        state, _ = build(hypothesis("A", "WEAKENED", [GOOD]),
                         hypothesis("B", "PLAUSIBLE"))
        # A is not live, so its discriminator earns nothing
        assert differential_required_domains(state) == set()
        assert describe_policy()["discriminator_requires_live_pair"] is True
        assert describe_policy()["discriminator_requires_counterfactual"] is True


# ======================================================================
# 15-19: the tunnel that must not form
# ======================================================================

class TestNoTunnel:
    def test_15_one_live_hypothesis_fabricates_no_competitor(self):
        state, _ = build(hypothesis("A", "PRIMARY_WORKING", [GOOD]))
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0].unresolved_discriminators == []
        assert differential_required_domains(state) == set()

    def test_15b_the_prompt_says_so_too(self):
        assert "do not invent a second one" in oc.INTERVIEW_SYSTEM_PROMPT

    def test_16_generic_coverage_still_fills_capacity(self):
        text = "发热咳嗽三天。"
        candidates = [
            Candidate(field="sweat_q", question="出汗？", domain="sweat",
                      kind="adaptive", domain_certain=True,
                      payload={"field": "sweat_q"}),
            Candidate(field="head_q", question="头痛？", domain="head_body",
                      kind="adaptive", domain_certain=True,
                      payload={"field": "head_q"}),
            Candidate(field="chest_q", question="胸闷？",
                      domain="chest_abdomen", kind="adaptive",
                      domain_certain=True, payload={"field": "chest_q"}),
        ]
        selection = select_questions(
            deterministic=[], adaptive=candidates,
            coverage=assess_coverage(text, {"sweat"}),
            signals=DifferentialSignals(gap_domains=frozenset({"sweat"})),
            differential_domains={"sweat"})
        fields = [q["field"] for q in selection.adaptive]
        assert fields[0] == "sweat_q"
        assert "head_q" in fields and "chest_q" in fields

    def test_17_a_new_valid_alternative_can_enter_discrimination(self):
        """The differential is not frozen to turn 1."""
        prior = WorkingDifferentialState(hypotheses=[
            {"pattern_name": "A", "standing": "PRIMARY_WORKING",
             "supporting_evidence": [{"origin": "COMPLAINT"}]}])
        state, _ = validate_state(
            {"hypotheses": [
                hypothesis("A", "PLAUSIBLE"),
                hypothesis("NEW", "PLAUSIBLE",
                           [disc("thirst", ("NEW", "A"), present=["NEW"],
                                 absent=["A"])])]},
            ResolvableEvidence(), prior)
        assert "NEW" in {h.pattern_name for h in state.hypotheses}
        assert differential_required_domains(state) == {"thirst"}

    def test_18_a_prior_hypothesis_is_not_evidence(self):
        prior = WorkingDifferentialState(hypotheses=[
            {"pattern_name": "A", "standing": "PRIMARY_WORKING",
             "supporting_evidence": [{"origin": "COMPLAINT"}]}])
        state, notes = validate_state(
            {"hypotheses": [hypothesis("A", "PRIMARY_WORKING")]},
            ResolvableEvidence(), prior)
        assert "STANDING_ESCALATION_WITHOUT_NEW_EVIDENCE" not in notes
        assert state.hypotheses[0].standing == "PRIMARY_WORKING"
        # the prior never becomes a citation
        assert [r.origin for r in state.hypotheses[0].supporting_evidence] == \
            ["COMPLAINT"]

    def test_19_current_evidence_can_replace_the_prior_primary(self):
        prior = WorkingDifferentialState(hypotheses=[
            {"pattern_name": "A", "standing": "PRIMARY_WORKING",
             "supporting_evidence": [{"origin": "COMPLAINT"}]}])
        state, _ = validate_state(
            {"hypotheses": [hypothesis("A", "WEAKENED"),
                            hypothesis("B", "PRIMARY_WORKING")]},
            ResolvableEvidence(), prior)
        top = {h.pattern_name: h.standing for h in state.hypotheses}
        assert top["A"] == "WEAKENED"
        assert top["B"] == "PRIMARY_WORKING"

    def test_19b_the_prompt_forbids_agreeing_with_itself(self):
        prompt = oc.INTERVIEW_SYSTEM_PROMPT
        assert "agreeing with yourself is not evidence" in prompt
        assert "has learned nothing" in prompt


# ======================================================================
# 20-27: nothing gained authority
# ======================================================================

class TestAuthorityUnchanged:
    def test_20_no_numeric_confidence_was_added(self):
        state, _ = build(hypothesis("A", "PRIMARY_WORKING", [GOOD]),
                         hypothesis("B"))
        blob = state.model_dump_json()
        assert "confidence" not in blob
        assert "score" not in blob
        assert "confidence" not in code(diff)

    @pytest.mark.parametrize("forbidden", [
        "corpus", "REVIEWED", "VERIFIED", "verification_state",
        "PATTERN_FORMULA", "FORMULA_HERB", "clinical_ranking_eligible",
        "ready_for_formula_retrieval", "SafetyEngine", "safety",
        "consumer_purchasable", "resolved_product_id", "dosage",
        "administration", "insert", "commit", "session"])
    def test_21_26_no_authority_surface_exists(self, forbidden):
        assert forbidden not in code(diff)

    def test_the_module_still_assigns_to_no_attribute(self):
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

    def test_26_the_structure_is_never_consumer_visible(self):
        from app.schemas.reasoning import ClinicalReasoningEnvelope
        from app.services.recommendation.consumer_projection import (
            build_consumer_reasoning)

        projection = build_consumer_reasoning(
            ClinicalReasoningEnvelope(clinical_summary="x"))
        for forbidden in ("if_present_supports", "if_absent_supports",
                          "separates", "rationale", "discriminating",
                          "working_differential"):
            assert forbidden not in str(projection)

    def test_27_full_reasoning_receives_no_working_state(self):
        from app.services.recommendation import assembler as module

        tree = ast.parse(textwrap.dedent(inspect.getsource(
            module.RecommendationAssembler.generate)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    ast.unparse(node.func).endswith("generate_recommendation"):
                rendered = ast.unparse(node)
                for forbidden in ("carry", "interview_state", "known_domains",
                                  "required_domains"):
                    assert forbidden not in rendered

    def test_27b_the_full_prompt_knows_nothing_of_any_of_it(self):
        for forbidden in ("if_present_supports", "separates",
                          "working_differential", "ALREADY_KNOWN"):
            assert forbidden not in oc.SYSTEM_PROMPT


# ======================================================================
# 28-37: budgets, calls, and the earlier phases
# ======================================================================

class TestFrozen:
    def test_28_29_budgets_are_unchanged(self):
        from app.services.clarification.validator import MAX_PROPOSALS_PER_TURN

        assert MAX_PROPOSALS_PER_TURN == 3
        assert cov.MAX_ADAPTIVE_WHEN_OPEN == 3
        assert cov.MAX_ADAPTIVE_WHEN_NARROWING == 1
        assert cov.MAX_VISIBLE_QUESTIONS_PER_TURN == 4
        assert cov.MIN_QUESTIONS_WHEN_INSUFFICIENT == 2

    def test_30_interview_depth_is_unchanged(self):
        from app.services.interview.mode import (MAX_INTERVIEW_TURNS,
                                                 REASON_DEPTH_REACHED,
                                                 decide_mode)

        assert MAX_INTERVIEW_TURNS == 3
        assert decide_mode(accumulated_text="咳嗽3天。", missing_information=[],
                           interview_depth=3) == ("FULL_REASONING",
                                                  REASON_DEPTH_REACHED)

    def test_31_one_provider_call_per_turn(self):
        from app.services.recommendation import assembler as module

        source = code(module.RecommendationAssembler.generate)
        assert source.count("await self.provider.generate_interview(") == 1
        assert source.count("await self.provider.generate_recommendation(") == 1

    def test_31b_the_known_domain_list_costs_no_call(self):
        from app.services.recommendation import assembler as module

        source = code(module.RecommendationAssembler.generate)
        head = source.split("generate_interview(")[0]
        assert "assess_coverage(" in head          # a pure function, in-process
        assert head.count("await ") <= 1

    def test_32_no_second_generator_exists(self):
        source = code(diff).lower().replace("question_needed", "").replace(
            "question_field", "")
        for forbidden in ("clarification_proposals", "propose", "generate_"):
            assert forbidden not in source

    def test_33_34_r1_provenance_and_canonicalization_intact(self):
        from app.services.interview.differential import (
            normalize_clinical_domain, resolve_ref)

        evidence = ResolvableEvidence(answers=[
            {"turn_id": 1, "question_field": "chills_or_heat",
             "answer": "怕冷"}])
        assert resolve_ref({"origin": "ANSWER", "question_field": "cold_heat"},
                           evidence) is not None
        assert normalize_clinical_domain("nasal_symptoms") == "nose"

    def test_35_r1_re_ask_suppression_intact(self):
        coverage = assess_coverage("发热咳嗽三天。咳嗽有白痰，无汗。")
        assert coverage.states["sputum"] == KNOWN
        assert coverage.states["sweat"] == KNOWN

    def test_36_r2_focus_veto_repair_intact(self):
        text = "恶寒发热两天，头痛，全身酸痛，咳嗽"
        assert assess_coverage(text).states["throat"] == "NOT_RELEVANT"
        assert assess_coverage(text, {"throat"}).states["throat"] == "UNKNOWN"

    def test_37_r2_eligibility_intact(self):
        assert "differential_domains" in code(select_questions)
        assert cov.WEIGHT_ENVELOPE_GAP == 20
        assert cov.WEIGHT_CONTRADICTION == 12
        assert cov.WEIGHT_DISCRIMINATING == 8

    def test_43_streaming_unchanged(self):
        from app.services.llm.streaming import STREAMABLE_KEYS

        assert STREAMABLE_KEYS == ("summary", "interview_summary")

    def test_44_consumer_projection_unchanged(self):
        from app.schemas.reasoning import ClinicalReasoningEnvelope
        from app.services.recommendation.consumer_projection import (
            build_consumer_reasoning)

        projection = build_consumer_reasoning(ClinicalReasoningEnvelope(
            clinical_summary="摘要", pathogenesis="病机"))
        assert set(projection) <= {"summary", "eight_principle",
                                   "pattern_hypotheses", "pathogenesis",
                                   "treatment_principle",
                                   "missing_information", "uncertainty"}


# ======================================================================
# The prompt asks generally, and encodes no case
# ======================================================================

class TestPromptIsGeneral:
    def test_it_reasons_pair_first(self):
        prompt = oc.INTERVIEW_SYSTEM_PROMPT
        for step in ("STEP 1", "STEP 2", "STEP 3", "STEP 4", "STEP 5",
                     "STEP 6"):
            assert step in prompt
        assert prompt.index("STEP 5") < prompt.index("STEP 6")

    def test_it_encodes_no_clinical_case(self):
        """U: general discrimination logic, never this acceptance case."""
        prompt = oc.INTERVIEW_SYSTEM_PROMPT
        for forbidden in ("风寒", "风热", "发热咳嗽", "外感风", "麻黄", "桂枝",
                          "银翘", "痰热"):
            assert forbidden not in prompt

    def test_the_domain_list_is_the_canonical_one(self):
        for key in cov.DOMAINS_BY_KEY:
            assert key in oc.INTERVIEW_SYSTEM_PROMPT

    def test_the_validator_is_the_authority_not_the_prompt(self):
        """A prompt is a request; the boundary is deterministic code."""
        state, notes = build(
            hypothesis("A", "PRIMARY_WORKING",
                       [disc("sweat", ("A", "UNLISTED"), present=["A"],
                             absent=["UNLISTED"])]),
            hypothesis("B"))
        assert state.hypotheses[0].unresolved_discriminators == []
        assert "DISCRIMINATOR_NO_LIVE_COMPETITION" in notes
