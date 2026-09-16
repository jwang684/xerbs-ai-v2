"""X1D-LEGACYDIAG4.4B-R2: the differential decides what is ELIGIBLE, not what scores.

Where R1 stopped
----------------
R1 repaired the vocabulary and the provenance, and the forked run then showed
two branches holding opposite hypotheses -- 风寒 twice, 风热 twice -- asking the
same questions. The candidate table said why, and neither cause was a weight:

  * a domain the validated differential explicitly needed could be struck out
    as NOT_RELEVANT by the coarse focus profile before its gap weight applied.
    On the pain-focus complaint that removed sweat, thirst, nose, throat AND
    sputum, and 病因/旧病 were asked instead;

  * the three adaptive slots were filled from one generic pool by one rule, so
    a domain nothing was waiting on could take the slot of a domain the
    differential depended on. Two opposite differentials therefore drew the
    same questions.

R2 adds exactly two ideas, both eligibility and neither numeric: a required
domain is not vetoed by coarse focus, and required candidates are offered the
scarce slots first. Every weight and every budget is asserted unchanged below.

The danger being avoided
------------------------
A differential that could *silence* questions would be a closed tunnel: the
model's working state is incomplete by construction, so a domain it forgot must
stay askable. Generic coverage therefore keeps every slot the differential does
not use, and several tests here exist only to hold that door open.
"""

import ast
import inspect
import textwrap

import pytest

from app.schemas.reasoning import (
    MissingInformation,
    ResolvableEvidence,
    WorkingDifferentialState,
)
from app.services.clarification import coverage as cov
from app.services.clarification.coverage import (
    KNOWN,
    NOT_RELEVANT,
    UNKNOWN,
    Candidate,
    DifferentialSignals,
    assess_coverage,
    score_candidate,
    select_questions,
)
from app.services.interview.differential import (
    LIVE_STANDINGS,
    describe_policy,
    differential_required_domains,
    live_hypotheses,
    validate_state,
)


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


# X1D-LEGACYDIAG4.5: a discriminator now has to name the competition it
# resolves and show that the two answers would move it differently. These
# fixtures build already-validated state, so they state it directly; the
# validation of that shape is tested in the 4.5 file.
def disc(domain, separates=("A", "B")):
    return {"domain": domain, "separates": list(separates),
            "if_present_supports": [separates[0]],
            "if_absent_supports": [separates[1]],
            "discriminating": True}


def state(*hypotheses, gaps=()):
    return WorkingDifferentialState(
        hypotheses=[{"pattern_name": n, "standing": s,
                     "unresolved_discriminators": [disc(d) for d in ds]}
                    for n, s, ds in hypotheses],
        evidence_gaps=list(gaps))


def adaptive(field, question, domain, priority="medium"):
    return Candidate(field=field, question=question, domain=domain,
                     kind="adaptive", model_priority=priority,
                     domain_certain=True, payload={"field": field,
                                                   "question": question})


# ======================================================================
# 1-7: what counts as live, and what it may require
# ======================================================================

class TestRequiredDomains:
    def test_1_it_derives_only_from_validated_state(self):
        """Raw model output never reaches this; validate_state runs first."""
        raw = {"hypotheses": [
            {"pattern_name": "A", "standing": "PLAUSIBLE",
             "supporting_evidence": [{"origin": "COMPLAINT"}],
             "unresolved_discriminators": [
                 {"domain": "throat_pain", "separates": ["A", "B"],
                  "if_present_supports": ["A"],
                  "if_absent_supports": ["B"]}]},
            {"pattern_name": "B", "standing": "PLAUSIBLE",
             "supporting_evidence": [{"origin": "COMPLAINT"}]}]}
        validated, _ = validate_state(raw, ResolvableEvidence())
        assert differential_required_domains(validated) == {"throat"}
        source = code(differential_required_domains)
        assert "validate" not in source          # it consumes, never validates
        assert "raw" not in source

    def test_2_aliases_are_canonicalized_before_membership(self):
        required = differential_required_domains(
            state(("A", "PRIMARY_WORKING",
                   ["nasal_symptoms", "sputum_character", "throat_pain"])))
        assert required == {"nose", "sputum", "throat"}

    def test_3_an_unknown_domain_cannot_become_required(self):
        assert differential_required_domains(
            state(("A", "PRIMARY_WORKING",
                   ["qi_depth", "made_up", ""]))) == set()

    def test_4_a_primary_working_discriminator_contributes(self):
        assert differential_required_domains(
            state(("A", "PRIMARY_WORKING", ["sweat"]))) == {"sweat"}

    def test_5_a_plausible_discriminator_contributes(self):
        assert differential_required_domains(
            state(("A", "PLAUSIBLE", ["thirst"]))) == {"thirst"}

    def test_6_weakened_is_deliberately_not_live(self):
        """Stated as a test because it is a judgement, not an accident.

        WEAKENED is the reading the evidence currently argues against. Letting
        it claim one of three scarce slots is how an interview stays open on a
        fading alternative -- and it is not silenced: it keeps its place in the
        carried state and is re-evaluated every turn.
        """
        assert LIVE_STANDINGS == ("PRIMARY_WORKING", "PLAUSIBLE")
        assert differential_required_domains(
            state(("A", "WEAKENED", ["sweat"]))) == set()
        assert describe_policy()["live_standings"] == list(LIVE_STANDINGS)

    def test_7_ruled_out_for_now_does_not_drive_questions(self):
        assert differential_required_domains(
            state(("A", "RULED_OUT_FOR_NOW", ["sweat"]))) == set()

    def test_7b_but_it_returns_the_moment_its_standing_is_raised(self):
        """Reversible, exactly as 4.4B requires -- through the standing, not
        through a back door here."""
        assert differential_required_domains(
            state(("A", "PLAUSIBLE", ["sweat"]))) == {"sweat"}

    def test_a_gap_counts_only_between_live_hypotheses(self):
        required = differential_required_domains(state(
            ("A", "PRIMARY_WORKING", []),
            ("B", "RULED_OUT_FOR_NOW", []),
            gaps=[{"domain": "sweat", "separates": ["A", "B"]},
                  {"domain": "thirst", "separates": ["B", "C"]}]))
        assert required == {"sweat"}          # thirst separates nothing live

    def test_no_state_requires_nothing(self):
        assert differential_required_domains(None) == set()
        assert differential_required_domains(WorkingDifferentialState()) == set()
        assert live_hypotheses(None) == []

    def test_no_numeric_confidence_is_introduced(self):
        assert "confidence" not in code(differential_required_domains)


# ======================================================================
# 8-11: the focus veto, overridden narrowly
# ======================================================================

class TestFocusVetoOverride:
    # A pain-focus complaint: the exact shape that failed in R1.
    TEXT = "恶寒发热两天，头痛，全身酸痛，咳嗽"

    def test_8_a_required_unknown_domain_survives_the_focus_veto(self):
        assert assess_coverage(self.TEXT).states["throat"] == NOT_RELEVANT
        assert assess_coverage(self.TEXT, {"throat"}).states["throat"] == UNKNOWN
        candidate = adaptive("sore_throat", "咽喉痛吗？", "throat")
        assert score_candidate(candidate, assess_coverage(self.TEXT),
                               DifferentialSignals()) is None
        assert score_candidate(
            candidate, assess_coverage(self.TEXT, {"throat"}),
            DifferentialSignals(gap_domains=frozenset({"throat"}))) is not None

    def test_9_the_same_domain_without_the_requirement_is_still_vetoed(self):
        candidate = adaptive("sore_throat", "咽喉痛吗？", "throat")
        assert score_candidate(candidate, assess_coverage(self.TEXT),
                               DifferentialSignals()) is None

    def test_10_a_contradicted_domain_survives_the_veto_too(self):
        assert assess_coverage(self.TEXT, {"sweat"}).states["sweat"] == UNKNOWN
        candidate = adaptive("sweating", "出汗吗？", "sweat")
        scored = score_candidate(
            candidate, assess_coverage(self.TEXT, {"sweat"}),
            DifferentialSignals(contradiction_domains=frozenset({"sweat"})))
        assert scored is not None
        assert scored.components["differential"] == cov.WEIGHT_CONTRADICTION

    def test_11_surviving_grants_no_new_numeric_bonus(self):
        """Eligibility only. The score is built from the existing parts."""
        candidate = adaptive("sore_throat", "咽喉痛吗？", "throat")
        scored = score_candidate(
            candidate, assess_coverage(self.TEXT, {"throat"}),
            DifferentialSignals(gap_domains=frozenset({"throat"})))
        assert set(scored.components) == {
            "focus", "coverage", "differential", "complaint_link",
            "model_priority"}
        assert scored.score == sum(scored.components.values())
        assert "eligible" not in scored.components
        assert "required" not in scored.components
        assert "differential_required" not in scored.components

    def test_11b_it_competes_rather_than_wins(self):
        """A required domain still has to beat the others on existing weights."""
        coverage = assess_coverage(self.TEXT, {"throat"})
        required = score_candidate(
            adaptive("sore_throat", "咽喉痛吗？", "throat"), coverage,
            DifferentialSignals(gap_domains=frozenset({"throat"})))
        material = score_candidate(
            adaptive("cause_of_onset", "发病前受凉了吗？", "cause"), coverage,
            DifferentialSignals())
        assert required.score > 0 and material.score > 0

    def test_the_override_never_invents_a_domain(self):
        states = assess_coverage(self.TEXT, {"not_a_domain", "", None}).states
        assert "not_a_domain" not in states
        assert len(states) == len(cov.DOMAINS)


# ======================================================================
# 12-14: the weights, frozen
# ======================================================================

class TestWeightsFrozen:
    def test_12_the_differential_weights_are_unchanged(self):
        assert cov.WEIGHT_ENVELOPE_GAP == 20
        assert cov.WEIGHT_DISCRIMINATING == 8
        assert cov.MAX_DIFFERENTIAL_WEIGHT == 24

    def test_13_the_contradiction_weight_is_unchanged(self):
        assert cov.WEIGHT_CONTRADICTION == 12

    def test_14_every_other_weight_is_unchanged(self):
        assert cov.WEIGHT_COMPLAINT_LINK == 10
        assert cov.FOCUS_WEIGHT_TOP == 40
        assert cov.FOCUS_WEIGHT_STEP == 4
        assert cov.FOCUS_WEIGHT_FLOOR == 8
        assert cov.FOCUS_WEIGHT_GENERAL == 20
        assert cov.FOCUS_WEIGHT_UNMAPPED == 18
        assert cov.FOCUS_WEIGHT_UNMAPPED_OFF_TOPIC == 4
        assert cov.COVERAGE_WEIGHT == {UNKNOWN: 25, "PARTIAL": 12}
        assert cov.MODEL_PRIORITY_WEIGHT == {"high": 6, "medium": 3, "low": 0}
        assert cov.MIN_ADAPTIVE_SCORE == 40
        assert cov.PRIORITY_FLOOR_SCORE == 1000

    def test_14b_no_new_weight_was_added(self):
        weights = {n for n in dir(cov)
                   if n.startswith("WEIGHT_") or n.startswith("FOCUS_WEIGHT")}
        assert weights == {
            "WEIGHT_ENVELOPE_GAP", "WEIGHT_CONTRADICTION",
            "WEIGHT_DISCRIMINATING", "WEIGHT_COMPLAINT_LINK",
            "FOCUS_WEIGHT_TOP", "FOCUS_WEIGHT_STEP", "FOCUS_WEIGHT_FLOOR",
            "FOCUS_WEIGHT_GENERAL", "FOCUS_WEIGHT_UNMAPPED",
            "FOCUS_WEIGHT_UNMAPPED_OFF_TOPIC"}

    def test_14c_the_scoring_function_itself_is_unchanged(self):
        source = code(score_candidate)
        assert "differential_domains" not in source
        assert "required" not in source


# ======================================================================
# 15-16: patient evidence still wins
# ======================================================================

class TestAnsweredDomainsWin:
    TEXT = "发热咳嗽三天。咳嗽有白痰，怕冷明显，无汗。"

    def test_15_a_known_domain_stays_excluded_even_if_required(self):
        coverage = assess_coverage(self.TEXT, {"sputum", "sweat", "cold_heat"})
        for domain in ("sputum", "sweat", "cold_heat"):
            assert coverage.states[domain] == KNOWN
            assert score_candidate(
                adaptive("q_" + domain, "?", domain), coverage,
                DifferentialSignals(
                    gap_domains=frozenset({domain}))) is None

    def test_16_an_answered_alias_cannot_resurrect_the_domain(self):
        coverage = assess_coverage(self.TEXT, {"sputum"})
        for alias in ("sputum_character", "sputum_amount",
                      "cough_sputum_detail"):
            domain, certain = cov.resolve_domain(alias, "痰是什么颜色？")
            assert domain == "sputum" and certain
            assert score_candidate(
                adaptive(alias, "痰是什么颜色？", domain), coverage,
                DifferentialSignals(
                    gap_domains=frozenset({"sputum"}))) is None

    def test_the_override_changes_state_only_never_materiality(self):
        """Deliberately narrow.

        Materiality also drives unknown_material() and therefore
        assess_sufficiency, so adding a required domain to material would
        silently turn a one-question narrowing turn into a three-question open
        one. Lifting the state alone leaves the budget decision untouched.
        """
        plain = assess_coverage("恶寒发热两天，头痛，全身酸痛，咳嗽")
        lifted = assess_coverage("恶寒发热两天，头痛，全身酸痛，咳嗽",
                                 {"throat", "sputum"})
        assert plain.states["throat"] == NOT_RELEVANT
        assert lifted.states["throat"] == UNKNOWN
        assert plain.material == lifted.material
        assert plain.unknown_material() == lifted.unknown_material()
        assert cov.assess_sufficiency(plain, DifferentialSignals(), False) ==             cov.assess_sufficiency(lifted, DifferentialSignals(), False)


# ======================================================================
# 17-19: generic coverage is guided, never closed off
# ======================================================================

class TestGenericCoverageSurvives:
    TEXT = "发热咳嗽三天。"

    def build(self, required):
        coverage = assess_coverage(self.TEXT, required)
        signals = DifferentialSignals(gap_domains=frozenset(required))
        candidates = [
            adaptive("sweating", "出汗吗？", "sweat"),
            adaptive("nasal_discharge", "鼻塞流涕吗？", "nose"),
            adaptive("head_body_ache", "头身痛吗？", "head_body"),
        ]
        return select_questions(
            deterministic=[], adaptive=candidates, coverage=coverage,
            signals=signals, differential_domains=required)

    def test_17_generic_material_domains_remain_eligible(self):
        selection = self.build({"sweat"})
        fields = [q["field"] for q in selection.adaptive]
        assert "head_body_ache" in fields          # required nothing, still asked

    def test_18_generic_domains_fill_the_unused_capacity(self):
        """One required candidate, three slots: generic takes the other two."""
        selection = self.build({"sweat"})
        assert len(selection.adaptive) == cov.MAX_ADAPTIVE_WHEN_OPEN
        assert selection.adaptive[0]["field"] == "sweating"

    def test_19_the_differential_cannot_suppress_all_generic_assessment(self):
        """Absence from the working state is not evidence of irrelevance.

        The state is model output and is incomplete by construction. If a
        domain it forgot became unaskable, the differential would be a closed
        tunnel rather than a guide.
        """
        selection = self.build(set())
        assert len(selection.adaptive) == 3
        selection = self.build({"sweat", "nose", "head_body"})
        assert len(selection.adaptive) == 3

    def test_19b_a_required_candidate_takes_a_slot_from_a_higher_scoring_one(self):
        """The tier, demonstrated: rank alone would not have chosen this."""
        coverage = assess_coverage(self.TEXT, {"nose"})
        signals = DifferentialSignals()      # no gap weight at all
        candidates = [
            adaptive("cold_heat_q", "怕冷还是怕热？", "cold_heat", "high"),
            adaptive("thirst_q", "口渴吗？", "thirst", "high"),
            adaptive("head_q", "头身痛吗？", "head_body", "high"),
            adaptive("nose_q", "鼻塞流涕吗？", "nose", "low"),
        ]
        without = select_questions(
            deterministic=[], adaptive=candidates, coverage=coverage,
            signals=signals, differential_domains=set())
        with_req = select_questions(
            deterministic=[], adaptive=candidates, coverage=coverage,
            signals=signals, differential_domains={"nose"})
        assert "nose_q" not in [q["field"] for q in without.adaptive]
        assert "nose_q" in [q["field"] for q in with_req.adaptive]
        # and it did not get there by scoring higher
        scored = {s.candidate.field: s.score for s in with_req.scores}
        assert scored["nose_q"] < max(scored.values())


# ======================================================================
# 20-27: budget, architecture, and divergence without weights
# ======================================================================

class TestArchitecture:
    def test_20_the_budget_is_unchanged(self):
        assert cov.MAX_ADAPTIVE_WHEN_OPEN == 3
        assert cov.MAX_ADAPTIVE_WHEN_NARROWING == 1
        assert cov.MAX_VISIBLE_QUESTIONS_PER_TURN == 4
        assert cov.MIN_QUESTIONS_WHEN_INSUFFICIENT == 2
        from app.services.clarification.validator import MAX_PROPOSALS_PER_TURN

        assert MAX_PROPOSALS_PER_TURN == 3

    def test_21_there_is_still_one_ranking_pass(self):
        """One scorer, one ordering key. The tier is a sort key, not a pass."""
        source = code(select_questions)
        assert source.count("score_candidate(") == 1
        assert "tier(pair[1])" in source
        # no second scoring function anywhere in the module
        assert len([n for n in dir(cov)
                    if n.startswith("score_") or n.startswith("rank_")]) == 1

    def test_22_no_second_provider_call(self):
        from app.services.recommendation import assembler as module

        source = code(module.RecommendationAssembler.generate)
        assert source.count("await self.provider.generate_interview(") == 1
        assert source.count("await self.provider.generate_recommendation(") == 1

    def test_23_no_prompt_changed(self):
        from app.services.llm import openai_compatible as oc

        assert "differential_required" not in oc.INTERVIEW_SYSTEM_PROMPT
        assert "eligib" not in oc.INTERVIEW_SYSTEM_PROMPT.lower()
        assert "working_differential" not in oc.SYSTEM_PROMPT

    def test_24_no_model_or_provider_setting_changed(self):
        from app.services.llm import openai_compatible as oc

        source = code(oc.OpenAICompatibleProvider)
        assert "temperature" not in source
        assert "max_tokens" not in source
        assert "'response_format':" in source and "json_object" in source

    def test_25_eligibility_differs_when_the_required_sets_differ(self):
        """The R2 claim, stated at the level it actually operates."""
        text = "发热咳嗽三天。怕冷明显，痰是白的，咽喉不痛。"
        alpha_required = {"nose"}
        beta_required = {"thirst"}
        candidates = [adaptive("nose_q", "鼻塞吗？", "nose"),
                      adaptive("thirst_q", "口渴吗？", "thirst"),
                      adaptive("head_q", "头身痛吗？", "head_body"),
                      adaptive("chest_q", "胸闷吗？", "chest_abdomen")]

        def run(required):
            return [q["field"] for q in select_questions(
                deterministic=[], adaptive=candidates,
                coverage=assess_coverage(text, required),
                signals=DifferentialSignals(gap_domains=frozenset(required)),
                differential_domains=required).adaptive]

        assert run(alpha_required) != run(beta_required)
        assert "nose_q" in run(alpha_required)
        assert "thirst_q" in run(beta_required)

    def test_26_identical_coverage_no_longer_forces_identical_selection(self):
        """Exactly the R1 bottleneck, as a regression test.

        Both branches answered the same questions, so their coverage is
        identical. Before R2 that alone determined the candidate pool and the
        two branches could not diverge.
        """
        text = "发热咳嗽三天。怕冷明显，痰是白的，咽喉不痛。"
        coverage_a = assess_coverage(text, {"nose"})
        coverage_b = assess_coverage(text, {"thirst"})
        assert coverage_a.states["head_body"] == coverage_b.states["head_body"]
        # Five candidates for three slots, so the tier has to decide. The two
        # branch-specific ones score lowest and would both be dropped on rank.
        candidates = [adaptive("head_q", "头身痛吗？", "head_body", "high"),
                      adaptive("chest_q", "胸闷吗？", "chest_abdomen", "high"),
                      adaptive("cause_q", "受凉了吗？", "cause", "high"),
                      adaptive("nose_q", "鼻塞吗？", "nose", "low"),
                      adaptive("thirst_q", "口渴吗？", "thirst", "low")]
        a = select_questions(deterministic=[], adaptive=candidates,
                             coverage=coverage_a,
                             signals=DifferentialSignals(),
                             differential_domains={"nose"})
        b = select_questions(deterministic=[], adaptive=candidates,
                             coverage=coverage_b,
                             signals=DifferentialSignals(),
                             differential_domains={"thirst"})
        assert [q["field"] for q in a.adaptive] != [q["field"] for q in b.adaptive]

    def test_27_divergence_needed_no_weight_change(self):
        """The same candidates, the same weights, a different selection."""
        assert cov.WEIGHT_ENVELOPE_GAP == 20
        assert cov.WEIGHT_CONTRADICTION == 12
        assert cov.WEIGHT_DISCRIMINATING == 8
        source = code(cov.signals_from_domains) + code(score_candidate)
        assert "differential_domains" not in source


# ======================================================================
# 28-30: the deterministic question keeps its identity
# ======================================================================

class TestDeterministicProvenance:
    def test_28_a_deterministic_question_carries_its_canonical_field(self):
        """R1 found these arriving at the browser with field=None.

        core's adapter has always keyed its lookup on
        missing_information[].question, and this schema never carried it, so
        the lookup could not match. The field was known here all along.
        """
        from app.services.reasoning.engine import CORE_FIELDS

        for field, _reason, question in CORE_FIELDS:
            item = MissingInformation(field=field, reason="x",
                                      question=question)
            assert item.question == question
            domain, certain = cov.resolve_domain(item.field, item.question)
            assert domain is not None and certain

    def test_28b_the_engine_populates_it(self):
        from app.services.reasoning import engine

        assert "question=q" in code(engine.DiagnosticReasoningEngine)

    def test_29_it_stays_optional_where_no_identity_exists(self):
        """No invented identity: absent stays absent rather than guessed."""
        assert MissingInformation(field="x", reason="y").question is None
        assert MissingInformation.model_fields["question"].default is None

    def test_30_the_field_maps_onto_the_canonical_domain_system(self):
        assert cov.domain_for_field("sleep") == "sleep"
        assert cov.domain_for_field("stool") == "excretion"
        assert cov.domain_for_field("temperature") == "cold_heat"
        assert cov.domain_for_field("appetite") == "diet"
        assert cov.domain_for_field("duration") == "onset_duration"


# ======================================================================
# 31-42: everything the earlier phases established
# ======================================================================

class TestNoRegression:
    def test_31_re_ask_suppression_is_intact(self):
        coverage = assess_coverage("发热咳嗽三天。咳嗽有白痰，无汗。")
        assert coverage.states["sputum"] == KNOWN
        assert coverage.states["sweat"] == KNOWN
        assert score_candidate(adaptive("sputum_q", "痰？", "sputum"),
                               coverage, DifferentialSignals()) is None

    def test_36_depth_is_unchanged(self):
        from app.services.interview.mode import (MAX_INTERVIEW_TURNS,
                                                 REASON_DEPTH_REACHED,
                                                 decide_mode)

        assert MAX_INTERVIEW_TURNS == 3
        assert decide_mode(accumulated_text="咳嗽3天。", missing_information=[],
                           interview_depth=3) == ("FULL_REASONING",
                                                  REASON_DEPTH_REACHED)

    def test_37_full_reasoning_independence_is_unchanged(self):
        from app.services.recommendation import assembler as module

        tree = ast.parse(textwrap.dedent(inspect.getsource(
            module.RecommendationAssembler.generate)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    ast.unparse(node.func).endswith("generate_recommendation"):
                rendered = ast.unparse(node)
                assert "carry" not in rendered
                assert "required_domains" not in rendered
                assert "interview_state" not in rendered

    def test_38_the_state_is_still_consumer_invisible(self):
        from app.schemas.reasoning import ClinicalReasoningEnvelope
        from app.services.recommendation.consumer_projection import (
            build_consumer_reasoning)

        projection = build_consumer_reasoning(
            ClinicalReasoningEnvelope(clinical_summary="x"))
        assert "working_differential" not in projection
        assert "standing" not in str(projection)
        assert "differential_required" not in str(projection)

    @pytest.mark.parametrize("forbidden", [
        "REVIEWED", "VERIFIED", "verification_state", "PATTERN_FORMULA",
        "FORMULA_HERB", "clinical_ranking_eligible",
        "ready_for_formula_retrieval", "SafetyEngine", "consumer_purchasable",
        "resolved_product_id", "dosage", "administration"])
    def test_39_authority_isolation_is_unchanged(self, forbidden):
        from app.services.interview import differential as diff

        assert forbidden not in code(diff)

    def test_40_streaming_is_unchanged(self):
        from app.services.llm.streaming import STREAMABLE_KEYS

        assert STREAMABLE_KEYS == ("summary", "interview_summary")

    def test_41_the_consumer_projection_is_unchanged(self):
        from app.schemas.reasoning import ClinicalReasoningEnvelope
        from app.services.recommendation.consumer_projection import (
            build_consumer_reasoning)

        projection = build_consumer_reasoning(ClinicalReasoningEnvelope(
            clinical_summary="摘要", pathogenesis="病机"))
        assert set(projection) <= {"summary", "eight_principle",
                                   "pattern_hypotheses", "pathogenesis",
                                   "treatment_principle",
                                   "missing_information", "uncertainty"}

    def test_42_the_4_4b_rules_still_hold(self):
        prior = WorkingDifferentialState(hypotheses=[
            {"pattern_name": "A", "standing": "PLAUSIBLE",
             "supporting_evidence": [{"origin": "COMPLAINT"}]}])
        validated, notes = validate_state(
            {"hypotheses": [{"pattern_name": "A", "standing": "PRIMARY_WORKING",
                             "supporting_evidence": [{"origin": "COMPLAINT"}]}]},
            ResolvableEvidence(), prior)
        assert validated.hypotheses[0].standing == "PLAUSIBLE"
        assert "STANDING_ESCALATION_WITHOUT_NEW_EVIDENCE" in notes
