"""X1D-LEGACYDIAG4.6: the model picks the competition, the system picks the question.

What 4.5 measured
-----------------
    competition shape : within 0.650   between 0.000
    canonical domains : within 0.333   between 0.340

Which competition each branch was in separated perfectly and reproducibly.
Which domain it chose to settle that competition did not reproduce at all --
several findings separate the same two readings about equally well, and
nothing preferred one consistently. With a mean required-set size of 1.8,
two runs resolving the SAME competition by different valid means scored 0.00.

Phase 0 then found that the system had nothing to choose between: in all ten
runs the domains the model named were exactly the domains it asked about. A
preference layer over "the domains the model identified" would have been
inert, the way R2's tier was inert while proposals equalled budget.

So 4.6 splits the decision in two:

    the model   : which competition am I in, and what could settle it
    this code   : which of those candidates is put to the patient first

The ordering below mentions no pattern, no complaint and no domain by name. It
is three properties of an interview -- untouched beats half-touched, bounded
beats open-ended, and canonical registry order breaks the rest -- and it would
behave identically for two competing readings of anything at all.

The danger being avoided
------------------------
Deterministic code must not diagnose. It never adds a domain, never invents a
clinical score, and never resurrects something the patient already answered;
it sorts candidates the model itself named. And when it cannot help -- no
competition, no valid discriminator, spare capacity -- generic coverage still
fills the turn.
"""

import ast
import inspect
import textwrap

import pytest

from app.schemas.reasoning import ResolvableEvidence, WorkingDifferentialState
from app.services.clarification import coverage as cov
from app.services.clarification.coverage import (
    KNOWN,
    PARTIAL,
    UNKNOWN,
    Candidate,
    DifferentialSignals,
    assess_coverage,
    governed_question_candidates,
    order_discriminators,
    preference_rank,
    select_questions,
)
from app.services.interview import differential as diff
from app.services.interview.differential import (
    competition_key,
    competitions,
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


# Synthetic labels only. Nothing here names a pattern or a complaint.
def disc(domain, separates=("A", "B"), present=("B",), absent=("A",),
         also=()):
    return {"domain": domain, "separates": list(separates),
            "if_present_supports": list(present),
            "if_absent_supports": list(absent),
            "also_resolved_by": list(also), "rationale": "r"}


def hyp(name, standing="PLAUSIBLE", discriminators=()):
    return {"pattern_name": name, "standing": standing,
            "supporting_evidence": [{"origin": "COMPLAINT"}],
            "unresolved_discriminators": list(discriminators)}


def build(*hypotheses):
    return validate_state({"hypotheses": list(hypotheses)},
                          ResolvableEvidence())


# ======================================================================
# 1-6: competition identity, and who may be in one
# ======================================================================

class TestCompetitionKey:
    def test_1_it_is_order_independent(self):
        assert competition_key(["A", "B"]) == competition_key(["B", "A"])
        assert competition_key(["B", "A", "B"]) == ("A", "B")

    def test_1b_it_is_total(self):
        for junk in (None, 7, "AB", {}, [], [None, "  "]):
            assert isinstance(competition_key(junk), tuple)

    def test_2_only_live_hypotheses_enter_a_competition(self):
        state, _ = build(hyp("A", "PRIMARY_WORKING", [disc("sweat")]),
                         hyp("B", "PLAUSIBLE"),
                         hyp("C", "WEAKENED", [disc("thirst", ("A", "C"))]))
        found = competitions(state)
        assert list(found) == [("A", "B")]
        assert all(set(key) <= {"A", "B"} for key in found)

    def test_3_4_an_unknown_rival_is_rejected(self):
        state, notes = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("sweat", ("A", "GHOST"), present=("GHOST",),
                      absent=("A",))]),
            hyp("B"))
        assert state.hypotheses[0].unresolved_discriminators == []
        assert "DISCRIMINATOR_NO_LIVE_COMPETITION" in notes
        assert competitions(state) == {}

    def test_5_a_new_hypothesis_cannot_arrive_through_separates(self):
        """The only door into the differential is the hypotheses list."""
        state, _ = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("sweat", ("A", "SMUGGLED"), present=("SMUGGLED",))]),
            hyp("B"))
        assert {h.pattern_name for h in state.hypotheses} == {"A", "B"}
        assert "SMUGGLED" not in state.model_dump_json()

    def test_6_it_may_take_part_once_it_is_in_the_validated_state(self):
        state, _ = build(hyp("A", "PRIMARY_WORKING",
                             [disc("sweat", ("A", "NEW"), present=("NEW",),
                                   absent=("A",))]),
                         hyp("NEW", "PLAUSIBLE"))
        assert competitions(state) == {("A", "NEW"): {"sweat"}}

    def test_no_fuzzy_matching_is_used_on_names(self):
        """A near-miss is not a match. B1 is not B."""
        state, _ = build(hyp("A", "PRIMARY_WORKING",
                             [disc("sweat", ("A", "B1"), present=("B1",))]),
                         hyp("B"))
        assert state.hypotheses[0].unresolved_discriminators == []
        source = code(diff)
        # "ratio" would match "rationale", so name the real offenders
        for forbidden in ("difflib", "SequenceMatcher", "similarity_ratio",
                          "embed", "fuzz", "startswith", "endswith",
                          "levenshtein"):
            assert forbidden not in source


# ======================================================================
# 7-12: grouping the valid candidates
# ======================================================================

class TestValidDiscriminatorSet:
    def test_7_domains_group_by_competition(self):
        state, _ = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("sweat", ("A", "B"), also=["thirst"]),
                 disc("nose", ("A", "C"), present=("C",), absent=("A",))]),
            hyp("B"), hyp("C"))
        found = competitions(state)
        assert found[("A", "B")] == {"sweat", "thirst"}
        assert found[("A", "C")] == {"nose"}

    def test_8_an_invalid_discriminator_contributes_nothing(self):
        state, _ = build(hyp("A", "PRIMARY_WORKING", [disc("sweat", ("A",))]),
                         hyp("B"))
        assert competitions(state) == {}

    def test_9_a_known_domain_is_excluded_from_candidates(self):
        coverage = assess_coverage("咳嗽三天。无汗，不渴。")
        assert coverage.states["sweat"] == KNOWN
        assert governed_question_candidates(["sweat"], coverage) == []

    def test_10_an_unknown_domain_is_excluded(self):
        state, _ = build(hyp("A", "PRIMARY_WORKING",
                             [disc("sweat", also=["not_a_domain"])]),
                         hyp("B"))
        assert competitions(state)[("A", "B")] == {"sweat"}

    def test_11_a_domain_with_no_governed_question_is_excluded(self):
        coverage = assess_coverage("咳嗽三天。")
        built = {c.field for c in governed_question_candidates(
            ["sweat", "not_a_domain"], coverage)}
        assert built == {"sweat"}

    def test_12_a_valid_counterfactual_is_still_required(self):
        state, _ = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("sweat", present=("A",), absent=("A",), also=["nose"])]),
            hyp("B"))
        assert competitions(state) == {}
        assert differential_required_domains(state) == set()

    def test_the_alternatives_join_the_required_set(self):
        state, _ = build(hyp("A", "PRIMARY_WORKING",
                             [disc("sweat", also=["nose", "thirst"])]),
                         hyp("B"))
        assert differential_required_domains(state) == {"sweat", "nose",
                                                        "thirst"}

    def test_deterministic_code_never_adds_a_domain(self):
        source = code(diff.competitions) + code(order_discriminators)
        assert "DOMAINS_BY_KEY.get" not in source or "for key in DOMAINS" \
            not in source
        state, _ = build(hyp("A", "PRIMARY_WORKING", [disc("sweat")]),
                         hyp("B"))
        assert competitions(state)[("A", "B")] == {"sweat"}


# ======================================================================
# 13-20: the preference, and its tie-break
# ======================================================================

class TestPreference:
    TEXT = "咳嗽三天。"

    def test_13_it_is_stable_across_repeated_calls(self):
        coverage = assess_coverage(self.TEXT)
        first = order_discriminators(["throat", "sweat", "nose", "thirst"],
                                     coverage)
        for _ in range(50):
            assert order_discriminators(["nose", "thirst", "throat", "sweat"],
                                        coverage) == first
            assert order_discriminators(["sweat", "nose", "throat", "thirst"],
                                        coverage) == first

    def test_14_15_it_does_not_look_at_hypothesis_names(self):
        """Same domains, unrelated readings, identical order."""
        coverage = assess_coverage(self.TEXT)
        order = order_discriminators(["throat", "sweat"], coverage)
        for names in (("A", "B"), ("Pattern 1", "Pattern 2"),
                      ("肝气郁结", "心血不足"), ("x", "y")):
            state, _ = build(
                hyp(names[0], "PRIMARY_WORKING",
                    [disc("sweat", names, present=(names[1],),
                          absent=(names[0],), also=["throat"])]),
                hyp(names[1]))
            domains = competitions(state)[competition_key(names)]
            assert order_discriminators(domains, coverage) == order
        source = code(preference_rank) + code(order_discriminators)
        assert "pattern" not in source.lower()
        assert "hypothes" not in source.lower()

    def test_16_17_no_clinical_special_casing_exists(self):
        source = code(preference_rank) + code(order_discriminators)
        for forbidden in ("风寒", "风热", "cold_heat", "sweat", "throat",
                          "nose", "sputum", "thirst", "respiratory"):
            assert forbidden not in source
        # the order table is built from the registry, not written by hand
        assert cov.DOMAIN_ORDER == {d.key: i
                                    for i, d in enumerate(cov.DOMAINS)}

    def test_18_untouched_outranks_half_touched(self):
        """PARTIAL means the patient said something adjacent; the question is
        then harder to phrase and the answer harder to read."""
        coverage = assess_coverage("咳嗽三天。有点怕冷。")
        assert coverage.states["cold_heat"] == PARTIAL
        assert coverage.states["sweat"] == UNKNOWN
        assert order_discriminators(["cold_heat", "sweat"],
                                    coverage)[0] == "sweat"

    def test_18b_bounded_outranks_open_ended(self):
        coverage = assess_coverage(self.TEXT)
        assert not cov.DOMAINS_BY_KEY["onset_duration"].default_choices
        assert cov.DOMAINS_BY_KEY["sweat"].default_choices
        assert order_discriminators(["onset_duration", "sweat"],
                                    coverage)[0] == "sweat"

    def test_19_20_the_tie_break_is_canonical_registry_order(self):
        """Deliberately arbitrary and documented. Inventing a number to
        separate two genuinely equivalent domains would be false precision."""
        coverage = assess_coverage(self.TEXT)
        equivalent = ["throat", "sweat", "nose", "thirst"]
        for domain in equivalent:
            assert preference_rank(domain, coverage)[:2] == (0, 0)
        ordered = order_discriminators(equivalent, coverage)
        assert ordered == sorted(equivalent,
                                 key=lambda d: cov.DOMAIN_ORDER[d])
        assert cov.DOMAIN_ORDER["sweat"] < cov.DOMAIN_ORDER["throat"]

    def test_the_rank_is_a_total_order(self):
        coverage = assess_coverage(self.TEXT)
        ranks = [preference_rank(d, coverage) for d in cov.DOMAINS_BY_KEY]
        assert len(set(ranks)) == len(ranks)

    def test_no_numeric_clinical_score_was_invented(self):
        source = code(preference_rank)
        assert "float" not in source
        assert "0." not in source
        for forbidden in ("utility", "value", "severity", "confidence",
                          "weight"):
            assert forbidden not in source.lower()


# ======================================================================
# 21-24: the tunnel that must not form
# ======================================================================

class TestNoTunnel:
    TEXT = "咳嗽三天。"

    def adaptive(self, field, domain, priority="medium"):
        return Candidate(field=field, question="q", domain=domain,
                         kind="adaptive", model_priority=priority,
                         domain_certain=True,
                         payload={"field": field, "question": "q"})

    def test_21_22_generic_coverage_still_fills_spare_capacity(self):
        candidates = [self.adaptive("sweat_q", "sweat"),
                      self.adaptive("head_q", "head_body"),
                      self.adaptive("chest_q", "chest_abdomen")]
        selection = select_questions(
            deterministic=[], adaptive=candidates,
            coverage=assess_coverage(self.TEXT, {"sweat"}),
            signals=DifferentialSignals(gap_domains=frozenset({"sweat"})),
            differential_domains={"sweat"})
        fields = [q["field"] for q in selection.adaptive]
        assert fields[0] == "sweat_q"
        assert {"head_q", "chest_q"} <= set(fields)

    def test_22b_with_no_competition_everything_is_generic(self):
        candidates = [self.adaptive("head_q", "head_body"),
                      self.adaptive("chest_q", "chest_abdomen")]
        selection = select_questions(
            deterministic=[], adaptive=candidates,
            coverage=assess_coverage(self.TEXT),
            signals=DifferentialSignals(), differential_domains=set())
        assert len(selection.adaptive) == 2

    def test_23_one_live_hypothesis_fabricates_no_competition(self):
        state, _ = build(hyp("A", "PRIMARY_WORKING", [disc("sweat")]))
        assert competitions(state) == {}
        assert differential_required_domains(state) == set()

    def test_24_preference_cannot_resurrect_an_answered_domain(self):
        coverage = assess_coverage("咳嗽三天。无汗。")
        assert coverage.states["sweat"] == KNOWN
        assert governed_question_candidates(["sweat", "nose"], coverage) and \
            {c.field for c in governed_question_candidates(
                ["sweat", "nose"], coverage)} == {"nose"}
        candidate = Candidate(field="sweat_q", question="q", domain="sweat",
                              kind="adaptive", domain_certain=True, payload={})
        assert cov.score_candidate(
            candidate, assess_coverage("咳嗽三天。无汗。", {"sweat"}),
            DifferentialSignals(gap_domains=frozenset({"sweat"}))) is None


# ======================================================================
# 25-37: everything the earlier phases established
# ======================================================================

class TestEarlierPhasesIntact:
    def test_25_26_27_r1_provenance_canonicalization_and_dedup(self):
        from app.services.interview.differential import (
            normalize_clinical_domain, resolve_ref)

        evidence = ResolvableEvidence(answers=[
            {"turn_id": 1, "question_field": "chills_or_heat",
             "answer": "怕冷"}])
        assert resolve_ref({"origin": "ANSWER", "question_field": "cold_heat"},
                           evidence) is not None
        assert normalize_clinical_domain("nasal_symptoms") == "nose"
        assert assess_coverage("咳嗽三天。无汗。").states["sweat"] == KNOWN

    def test_28_29_r2_focus_veto_and_eligibility(self):
        text = "恶寒发热两天，头痛，全身酸痛，咳嗽"
        assert assess_coverage(text).states["throat"] == "NOT_RELEVANT"
        assert assess_coverage(text, {"throat"}).states["throat"] == "UNKNOWN"
        assert "differential_domains" in code(select_questions)

    def test_30_31_the_45_rules_are_unchanged(self):
        state, notes = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("sweat", present=("A",), absent=("A",))]),
            hyp("B"))
        assert state.hypotheses[0].unresolved_discriminators[0][
            "discriminating"] is False
        assert "DISCRIMINATOR_WITHOUT_COUNTERFACTUAL" in notes
        prompt = oc.INTERVIEW_SYSTEM_PROMPT
        assert "STEP 1" in prompt and "STEP 6" in prompt
        assert "confirms rather than separates" in prompt

    def test_32_33_no_confidence_and_no_new_clinical_score(self):
        state, _ = build(hyp("A", "PRIMARY_WORKING", [disc("sweat")]),
                         hyp("B"))
        assert "confidence" not in state.model_dump_json()
        assert "confidence" not in code(diff)

    def test_34_clarify_weights_are_unchanged(self):
        assert cov.WEIGHT_ENVELOPE_GAP == 20
        assert cov.WEIGHT_CONTRADICTION == 12
        assert cov.WEIGHT_DISCRIMINATING == 8
        assert cov.WEIGHT_COMPLAINT_LINK == 10
        assert cov.FOCUS_WEIGHT_TOP == 40
        assert cov.COVERAGE_WEIGHT == {UNKNOWN: 25, PARTIAL: 12}
        assert cov.MIN_ADAPTIVE_SCORE == 40
        assert not hasattr(cov, "WEIGHT_SEPARATES_TOP_TWO")

    def test_34b_the_ordering_changed_no_score(self):
        source = code(select_questions)
        assert "preference_rank" in source
        assert source.count("score_candidate(") == 1
        assert "score +" not in source and "score *" not in source

    def test_35_the_budget_is_unchanged(self):
        from app.services.clarification.validator import MAX_PROPOSALS_PER_TURN
        from app.services.interview.mode import MAX_INTERVIEW_TURNS

        assert MAX_PROPOSALS_PER_TURN == 3
        assert cov.MAX_ADAPTIVE_WHEN_OPEN == 3
        assert cov.MAX_ADAPTIVE_WHEN_NARROWING == 1
        assert cov.MAX_VISIBLE_QUESTIONS_PER_TURN == 4
        assert MAX_INTERVIEW_TURNS == 3

    def test_36_37_one_provider_call_and_no_second_generator(self):
        from app.services.recommendation import assembler as module

        source = code(module.RecommendationAssembler.generate)
        assert source.count("await self.provider.generate_interview(") == 1
        assert source.count("await self.provider.generate_recommendation(") == 1
        differential = code(diff).lower()
        for forbidden in ("clarification_proposals", "propose", "await"):
            assert forbidden not in differential

    def test_38_full_reasoning_isolation(self):
        from app.services.recommendation import assembler as module

        tree = ast.parse(textwrap.dedent(inspect.getsource(
            module.RecommendationAssembler.generate)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    ast.unparse(node.func).endswith("generate_recommendation"):
                rendered = ast.unparse(node)
                for forbidden in ("carry", "interview_state", "known_domains",
                                  "required_domains", "competition"):
                    assert forbidden not in rendered

    @pytest.mark.parametrize("forbidden", [
        "corpus", "REVIEWED", "VERIFIED", "verification_state",
        "PATTERN_FORMULA", "FORMULA_HERB", "clinical_ranking_eligible",
        "ready_for_formula_retrieval", "SafetyEngine", "safety",
        "consumer_purchasable", "resolved_product_id", "dosage"])
    def test_44_authority_isolation(self, forbidden):
        assert forbidden not in code(diff)

    def test_45_no_consumer_leak(self):
        from app.schemas.reasoning import ClinicalReasoningEnvelope
        from app.services.recommendation.consumer_projection import (
            build_consumer_reasoning)

        projection = build_consumer_reasoning(
            ClinicalReasoningEnvelope(clinical_summary="x"))
        for forbidden in ("also_resolved_by", "separates", "competition",
                          "discriminating", "working_differential",
                          "if_present_supports"):
            assert forbidden not in str(projection)

    def test_46_streaming_unchanged(self):
        from app.services.llm.streaming import STREAMABLE_KEYS

        assert STREAMABLE_KEYS == ("summary", "interview_summary")

    def test_47_projection_unchanged(self):
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
# The emission contract
# ======================================================================

class TestEmissionTightening:
    def test_the_prompt_demands_exact_names(self):
        prompt = oc.INTERVIEW_SYSTEM_PROMPT
        assert "copied EXACTLY from your own pattern_name values" in prompt
        assert "also_resolved_by" in prompt

    def test_the_prompt_sends_the_live_names(self):
        source = code(oc.OpenAICompatibleProvider.generate_interview)
        assert "carry_state" in source

    def test_the_validator_remains_the_authority(self):
        state, notes = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("sweat", ("A", "not listed anywhere"))]),
            hyp("B"))
        assert state.hypotheses[0].unresolved_discriminators == []
        assert "DISCRIMINATOR_NO_LIVE_COMPETITION" in notes

    def test_it_encodes_no_clinical_case(self):
        prompt = oc.INTERVIEW_SYSTEM_PROMPT
        for forbidden in ("风寒", "风热", "发热咳嗽", "外感风", "麻黄", "桂枝"):
            assert forbidden not in prompt


# ======================================================================
# X1D-LEGACYDIAG4.6 Phase 1: a domain already answered is never asked again
# ======================================================================
#
# Phase 0 of the 4.6 resumption measured that preference_rank ranks a KNOWN
# domain as though untouched -- its first element separates PARTIAL from
# everything else and nothing more -- so ordering ALONE would put an
# already-answered domain first. That is not a defect in the sort key.
# Askability is not the sort key's job. It belongs to
# governed_question_candidates, which admits only UNKNOWN and PARTIAL, and it
# is the only thing standing between a validated differential and asking the
# patient something they have already told us.
#
# So the invariant is pinned where production actually relies on it, rather
# than duplicated into the sort key where it would be a second copy of the
# same decision.

class TestKnownDomainIsNeverReAsked:

    # 无汗 answers the sweat domain outright. Nothing here speaks to 鼻.
    TEXT = "发热无汗身痛"

    def _state(self):
        return build(
            hyp("A", "PRIMARY_WORKING",
                [disc("sweat", ("A", "B")), disc("nose", ("A", "B"))]),
            hyp("B"))

    def _pipeline(self):
        """Exactly the production order, with nothing in between."""
        state, _ = self._state()
        required = differential_required_domains(state)
        signals = signals_from_state(state)
        coverage = assess_coverage(
            self.TEXT, required | set(signals.contradiction_domains))
        candidates = governed_question_candidates(required, coverage, [])
        selection = select_questions(
            deterministic=[], adaptive=candidates, coverage=coverage,
            signals=signals, differential_domains=required)
        return required, coverage, candidates, selection

    def test_1_the_answered_domain_is_still_eligible(self):
        """Eligibility is about the differential, not about the record."""
        required, coverage, _, _ = self._pipeline()
        assert {"sweat", "nose"} <= required
        assert coverage.states["sweat"] == KNOWN
        assert coverage.states["nose"] in (UNKNOWN, PARTIAL)

    def test_2_governed_candidates_exclude_the_known_domain(self):
        _, _, candidates, _ = self._pipeline()
        domains = {c.domain for c in candidates}
        assert "sweat" not in domains
        assert "nose" in domains

    def test_3_it_never_reaches_selection_as_an_adaptive_candidate(self):
        _, _, _, selection = self._pipeline()
        for scored in selection.scores:
            if scored.candidate.kind == "adaptive":
                assert scored.candidate.domain != "sweat"

    def test_4_the_patient_is_not_asked_it_again(self):
        _, _, _, selection = self._pipeline()
        asked = " ".join(
            str(q.get("question", "")) for q in
            (list(selection.adaptive) + list(selection.deterministic)
             + list(selection.fallback)))
        assert cov.DOMAINS_BY_KEY["sweat"].default_question not in asked
        for marker in ("出汗", "汗"):
            assert marker not in asked

    def test_5_the_remaining_discriminator_can_still_be_selected(self):
        _, _, _, selection = self._pipeline()
        chosen = {c.get("field") for c in selection.adaptive}
        assert "nose" in chosen

    def test_6_nothing_here_grants_authority(self):
        _, _, candidates, selection = self._pipeline()
        blob = str([c.payload for c in candidates]) + str(selection.adaptive)
        for forbidden in ("clinical_ranking_eligible", "VERIFIED", "REVIEWED",
                          "verification_state", "PATTERN_FORMULA",
                          "FORMULA_HERB", "ready_for_formula_retrieval",
                          "consumer_purchasable", "resolved_product_id",
                          "purchase", "dosage", "administration"):
            assert forbidden not in blob
        # Every question is a governed default, never generated here.
        for candidate in candidates:
            assert (candidate.question
                    == cov.DOMAINS_BY_KEY[candidate.domain].default_question)
            assert candidate.payload["source"] == "xerbs-ai-v2-coverage"


class TestOrderDiscriminatorsIsNotAProductionFilter:
    """The helper limitation, pinned so nobody wires it in by mistake."""

    TEXT = "发热无汗身痛"

    def test_it_would_rank_an_answered_domain_first(self):
        """Documented, not fixed: it assumes a pre-filtered input."""
        coverage = assess_coverage(self.TEXT, {"sweat", "nose"})
        assert coverage.states["sweat"] == KNOWN
        assert order_discriminators({"sweat", "nose"}, coverage)[0] == "sweat"
        # ... while the production filter drops it on the same input.
        produced = governed_question_candidates(
            {"sweat", "nose"}, coverage, [])
        assert [c.domain for c in produced] == ["nose"]

    def test_preference_rank_does_not_encode_askability(self):
        coverage = assess_coverage(self.TEXT, {"sweat", "nose"})
        # KNOWN scores the same first element as an untouched domain. This is
        # asserted so that changing it is a deliberate act, not a silent one.
        assert preference_rank("sweat", coverage)[0] == 0

    def test_neither_helper_has_a_production_caller(self):
        """AST over app/, so a new call site fails this test immediately."""
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(cov.__file__)))
        called = set()
        for dirpath, _, filenames in os.walk(root):
            if "__pycache__" in dirpath:
                continue
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read())
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    name = (func.id if isinstance(func, ast.Name)
                            else getattr(func, "attr", None))
                    if name in ("order_discriminators", "competitions"):
                        # competition_key is reached only from competitions.
                        called.add((name, os.path.basename(path)))
        assert called == set(), called
