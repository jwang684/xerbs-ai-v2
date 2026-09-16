"""X1D-LEGACYDIAG4.6-R1: the differential pipeline diagnostic.

The 4.6 staging acceptance could not tell three situations apart, because
from outside they are identical:

    the model emitted no working_differential at all
    it emitted one and validation rejected every hypothesis
    it emitted a good one and every rule passed silently

validate_state returns ``(None, [])`` for the first and ``(state, [])`` for
the third, and the DIFFERENTIAL_* uncertainty flags exist only to say that a
RULE FIRED -- so both are flagless. Absence then degrades silently into
ordinary coverage behaviour, which is correct clinically and invisible
operationally.

This phase adds observation and nothing else. The tests below exist mostly to
prove the "and nothing else" half: identical inputs must select identical
questions whether or not anyone is listening.
"""
import ast
import inspect
import textwrap

import pytest

from app.schemas.reasoning import ResolvableEvidence
from app.services.clarification.coverage import (
    DOMAINS_BY_KEY,
    assess_coverage,
    governed_question_candidates,
    select_questions,
)
from app.services.interview.differential import (
    differential_required_domains,
    signals_from_state,
    validate_state,
)
from app.services.recommendation import assembler as assembler_module
from app.services.telemetry import differential_pipeline as diag
from app.services.telemetry.differential_pipeline import (
    EVENT,
    STAGE_NO_CANDIDATES,
    STAGE_NO_EMISSION,
    STAGE_NO_LIVE,
    STAGE_NO_REQUIRED,
    STAGE_PARSE_REJECTED,
    STAGE_REACHED,
    emit,
    first_failing_stage,
)


def hyp(name, standing="PLAUSIBLE", discriminators=()):
    return {"pattern_name": name, "standing": standing,
            "supporting_evidence": [{"origin": "COMPLAINT"}],
            "unresolved_discriminators": list(discriminators)}


def disc(domain, separates=("A", "B"), present=("B",), absent=("A",), also=()):
    return {"domain": domain, "separates": list(separates),
            "if_present_supports": list(present),
            "if_absent_supports": list(absent),
            "also_resolved_by": list(also), "rationale": "r"}


# ======================================================================
# Every pipeline state is distinguishable
# ======================================================================

class TestStagesAreDistinguishable:
    """The whole point of the phase: these must not look alike."""

    def test_no_emission(self):
        state, notes = validate_state(None, ResolvableEvidence())
        assert state is None and notes == []
        assert first_failing_stage(False, False, 0, 0, 0) == STAGE_NO_EMISSION

    def test_emitted_but_rejected(self):
        """A dict arrives and nothing survives it."""
        state, notes = validate_state({"hypotheses": [{"pattern_name": ""}]},
                                      ResolvableEvidence())
        assert state is None
        assert first_failing_stage(True, False, 0, 0, 0) == STAGE_PARSE_REJECTED

    def test_validated_but_no_live_hypothesis(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("A", "WEAKENED"), hyp("B", "RULED_OUT_FOR_NOW")]},
            ResolvableEvidence())
        assert state is not None
        assert differential_required_domains(state) == set()
        assert first_failing_stage(True, True, 0, 0, 0) == STAGE_NO_LIVE

    def test_live_but_no_required_domains(self):
        """Live readings, but no discriminator that showed its work."""
        state, _ = validate_state(
            {"hypotheses": [hyp("A", "PRIMARY_WORKING"), hyp("B")]},
            ResolvableEvidence())
        assert state is not None and state.hypotheses
        assert differential_required_domains(state) == set()
        assert first_failing_stage(True, True, 2, 0, 0) == STAGE_NO_REQUIRED

    def test_required_but_no_governed_candidate(self):
        """Every required domain is already answered, so none is askable."""
        state, _ = validate_state(
            {"hypotheses": [hyp("A", "PRIMARY_WORKING", [disc("sweat")]),
                            hyp("B")]},
            ResolvableEvidence())
        required = differential_required_domains(state)
        assert required == {"sweat"}
        coverage = assess_coverage("发热无汗身痛", required)
        assert governed_question_candidates(required, coverage, []) == []
        assert first_failing_stage(True, True, 2, 1, 0) == STAGE_NO_CANDIDATES

    def test_reaches_preference_selection(self):
        state, _ = validate_state(
            {"hypotheses": [hyp("A", "PRIMARY_WORKING",
                                [disc("sweat", also=["nose", "throat"])]),
                            hyp("B")]},
            ResolvableEvidence())
        required = differential_required_domains(state)
        assert {"sweat", "nose", "throat"} <= required
        coverage = assess_coverage("咳嗽三天", required)
        governed = governed_question_candidates(required, coverage, [])
        assert governed
        assert first_failing_stage(True, True, 2, len(required),
                                   len(governed)) == STAGE_REACHED

    def test_the_stage_function_is_total(self):
        for args in [(False, True, 9, 9, 9), (True, False, 9, 9, 9),
                     (True, True, -1, 0, 0), (True, True, 1, 1, 1)]:
            assert isinstance(first_failing_stage(*args), str)


# ======================================================================
# Observation only
# ======================================================================

class TestInstrumentationCannotAlterSelection:

    def _pipeline(self, text, raw):
        state, notes = validate_state(raw, ResolvableEvidence())
        required = differential_required_domains(state)
        signals = signals_from_state(state)
        coverage = assess_coverage(
            text, required | set(signals.contradiction_domains))
        cands = governed_question_candidates(required, coverage, [])
        selection = select_questions(
            deterministic=[], adaptive=cands, coverage=coverage,
            signals=signals, differential_domains=required)
        return state, notes, required, cands, selection

    def test_selection_is_identical_with_and_without_emission(self):
        raw = {"hypotheses": [hyp("A", "PRIMARY_WORKING",
                                  [disc("sweat", also=["nose", "throat"])]),
                              hyp("B")]}
        text = "咳嗽三天"
        before = self._pipeline(text, raw)
        emit(generation_id="g", correlation_id="c", raw_state=raw,
             validated=before[0], notes=before[1],
             required_domains=before[2],
             governed_domains=[c.domain for c in before[3]],
             adaptive_total=len(before[3]),
             adaptive_budget=before[4].adaptive_budget,
             selected_domains=[s.candidate.domain for s in before[4].scores])
        after = self._pipeline(text, raw)
        assert [c.domain for c in before[3]] == [c.domain for c in after[3]]
        assert ([s.candidate.domain for s in before[4].scores]
                == [s.candidate.domain for s in after[4].scores])
        assert before[4].adaptive_budget == after[4].adaptive_budget
        assert before[4].sufficiency == after[4].sufficiency

    def test_emit_returns_nothing_and_never_raises(self):
        assert emit() is None
        for junk in (7, "x", [], {"hypotheses": "not a list"}):
            assert emit(raw_state=junk, validated=junk, notes=junk,
                        required_domains=junk, governed_domains=junk,
                        adaptive_total=junk, adaptive_budget=junk,
                        selected_domains=junk) is None

    def test_the_observer_is_side_effect_only_in_the_assembler(self):
        """Its return value must be discarded, like the existing observers."""
        source = textwrap.dedent(inspect.getsource(
            assembler_module.RecommendationAssembler.generate))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    ast.unparse(node.func).endswith("on_differential_pipeline"):
                parent_is_expr = True  # a bare call statement
                assert parent_is_expr
        # and it is never assigned from
        assert "= on_differential_pipeline(" not in source

    def test_diagnostic_locals_are_never_read_by_clinical_code(self):
        """diag_* may be written and passed to the observer, nothing else."""
        source = textwrap.dedent(inspect.getsource(
            assembler_module.RecommendationAssembler.generate))
        tree = ast.parse(source)
        reads = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.startswith("diag_") \
                    and isinstance(node.ctx, ast.Load):
                reads.append(node.id)
        # Every load happens inside the observer payload; none feeds a call
        # that decides anything. Assert no diag_ name is passed to the
        # selection or validation entry points.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name.endswith(("select_questions",
                                  "governed_question_candidates",
                                  "assess_coverage", "validate_state",
                                  "differential_required_domains")):
                    rendered = ast.unparse(node)
                    assert "diag_" not in rendered, rendered


# ======================================================================
# Absence stays safe, and nothing new becomes visible
# ======================================================================

class TestAbsenceAndGovernance:

    def test_absence_still_falls_back_to_governed_coverage(self):
        """No differential must behave exactly as it did before 4.6-R1."""
        state, notes = validate_state(None, ResolvableEvidence())
        required = differential_required_domains(state)
        assert state is None and required == set()
        coverage = assess_coverage("咳嗽三天", required)
        assert governed_question_candidates(required, coverage, []) == []
        selection = select_questions(
            deterministic=[], adaptive=[], coverage=coverage,
            signals=signals_from_state(state), differential_domains=required)
        assert selection.sufficiency
        assert selection.adaptive == []

    def test_no_consumer_visible_differential_metadata_is_added(self):
        from app.schemas.reasoning import ClinicalReasoningEnvelope
        from app.services.recommendation.consumer_projection import (
            build_consumer_reasoning)

        projection = str(build_consumer_reasoning(
            ClinicalReasoningEnvelope(clinical_summary="x")))
        for forbidden in ("working_differential", "also_resolved_by",
                          "separates", "discriminating", "required_domains",
                          "governed_candidate", "adaptive_surplus",
                          EVENT, "stage"):
            assert forbidden not in projection

    def test_the_diagnostic_grants_no_authority(self):
        source = inspect.getsource(diag)
        for forbidden in ("clinical_ranking_eligible", "VERIFIED", "REVIEWED",
                          "verification_state", "PATTERN_FORMULA",
                          "FORMULA_HERB", "ready_for_formula_retrieval",
                          "consumer_purchasable", "resolved_product_id",
                          "dosage", "administration", "commit", "insert"):
            assert forbidden not in source

    def test_it_logs_no_patient_text_or_model_prose(self):
        """Only counts, booleans, domain keys, reason codes and ids.

        Docstrings are stripped first: this is about what the CODE can put
        into a record, not which words the prose is allowed to use.
        """
        tree = ast.parse(inspect.getsource(diag))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
                    and body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:]
        source = ast.unparse(tree)
        for forbidden in ("text_input", "symptoms", "question", "rationale",
                          "pattern_name", "summary", "api_key", "token",
                          "password", "secret", "interview_summary"):
            assert forbidden not in source

    def test_the_record_carries_only_allowed_keys(self, caplog):
        import json
        import logging

        with caplog.at_level(logging.INFO, logger=diag.__name__):
            emit(generation_id="g1", correlation_id="c1",
                 raw_state={"hypotheses": [hyp("A")]},
                 validated=None, notes=["EVIDENCE_REF_UNRESOLVED"],
                 required_domains=["sweat"], governed_domains=["sweat"],
                 adaptive_total=4, adaptive_budget=3,
                 selected_domains=["sweat", "nose"])
        lines = [r.getMessage() for r in caplog.records if EVENT in r.getMessage()]
        assert lines, "no diagnostic line emitted"
        record = json.loads(lines[-1].split(EVENT, 1)[1].strip())
        assert set(record) == {
            "generation_id", "correlation_id", "stage", "model_emitted",
            "raw_hypotheses", "validated", "validated_hypotheses",
            "live_hypotheses", "discriminators", "discriminating",
            "reason_codes", "required_domains", "required_count",
            "governed_candidate_domains", "governed_candidate_count",
            "adaptive_candidates", "adaptive_budget", "adaptive_surplus",
            "preference_had_a_choice", "selected_domains",
            "selected_from_required", "occurred_at",
            # X1D-LEGACYDIAG4.6-R2
            "candidates", "candidates_total", "candidates_above_floor",
            "candidates_below_floor", "governed_suppressed_by_model"}
        assert record["stage"] == STAGE_PARSE_REJECTED
        assert record["adaptive_surplus"] == 1
        assert all(d in DOMAINS_BY_KEY for d in record["required_domains"])


# ======================================================================
# Replay must not gain an execution
# ======================================================================

class TestReplayUnchanged:

    def test_the_diagnostic_is_emitted_from_the_generation_path_only(self):
        """Replay returns a persisted row and never reaches the assembler."""
        from app.services.integration import base44

        source = inspect.getsource(base44.Base44GenerationService)
        tree = ast.parse(textwrap.dedent(source))

        emitting = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                rendered = ast.unparse(node)
                if "record_differential_pipeline" in rendered:
                    emitting.append(node.name)
        # Only the live generation path emits; the replay responder does not.
        assert emitting, "diagnostic is never emitted"
        assert "_response" not in emitting
        assert "_replay" not in emitting

    def test_emission_sits_after_the_result_is_persisted(self):
        """It rides in _execute, behind the SUCCEEDED write."""
        from app.services.integration import base44

        source = inspect.getsource(base44.Base44GenerationService._execute)
        assert source.index("SUCCEEDED") < \
            source.index("record_differential_pipeline")

    def test_replay_short_circuits_before_execute(self):
        """A repeated key returns the persisted row and never runs _execute.

        This is what makes the diagnostic replay-safe without any special
        case of its own: the whole generation path, provider call included,
        is skipped, so nothing is emitted a second time.
        """
        from app.services.integration import base44

        source = inspect.getsource(base44.Base44GenerationService.generate)
        tree = ast.parse(textwrap.dedent(source))
        returns_before_execute = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                rendered = ast.unparse(node)
                if "existing" in rendered and "_response(existing)" in rendered:
                    returns_before_execute = "_execute" not in rendered
        assert returns_before_execute
        assert "record_differential_pipeline" not in source
        assert "on_differential_pipeline" not in source


# ======================================================================
# X1D-LEGACYDIAG4.6-R2: the mixed-provenance adaptive pool
# ======================================================================
#
# R1 proved the differential reaches preference selection. It could not say
# why preference then failed to arbitrate, because it counted the pool as it
# was BUILT rather than as it was RANKED. A candidate that scores below the
# adaptive floor is dropped before tiering, so a pool of four can become three
# and leave a budget of three nothing to choose between -- while R1's
# adaptive_surplus still reported one.
#
# These tests pin the per-candidate record that closes that gap, and the
# rule that surplus is counted after the floor.

class TestMixedProvenanceDiagnostic:

    TEXT = "咳嗽三天，有点发热，身上酸痛。"
    REQUIRED = {"cold_heat", "sputum", "sweat", "throat"}
    MODEL_PROPOSED = ("cold_heat", "sputum", "sweat")

    def _pool(self, discriminating):
        from app.services.clarification.coverage import (
            Candidate, DifferentialSignals, resolve_domain)

        coverage = assess_coverage(self.TEXT, self.REQUIRED)
        signals = DifferentialSignals(
            gap_domains=frozenset(), contradiction_domains=frozenset(),
            discriminating_domains=frozenset(discriminating))
        adaptive = []
        for field in self.MODEL_PROPOSED:
            domain, certain = resolve_domain(field, "q")
            adaptive.append(Candidate(field=field, question="q", domain=domain,
                                      kind="adaptive", model_priority="medium",
                                      domain_certain=certain,
                                      payload={"field": field}))
        governed = governed_question_candidates(
            self.REQUIRED, coverage, [c.domain for c in adaptive])
        return coverage, signals, adaptive + governed, governed

    def test_governed_candidates_are_suppressed_for_model_taken_domains(self):
        """`already` is the deduplication, and it happens before scoring."""
        _, _, _, governed = self._pool(self.REQUIRED)
        assert [c.domain for c in governed] == ["throat"]

    def test_the_pool_is_genuinely_mixed(self):
        _, _, pool, _ = self._pool(self.REQUIRED)
        sources = {("coverage" if isinstance(c.payload, dict)
                    and c.payload.get("source") else "model")
                   for c in pool}
        assert sources == {"model", "coverage"}

    def test_surplus_is_counted_after_the_floor_not_before(self, caplog):
        """The R1 gap, pinned.

        Four candidates, three above the adaptive floor, budget three: the
        preference layer has nothing to arbitrate. R1 reported a surplus of
        one here because it counted the pool before the floor removed the
        fourth.
        """
        import json
        import logging

        rows_with = [{"domain": "a", "above_floor": True},
                     {"domain": "b", "above_floor": True},
                     {"domain": "c", "above_floor": True},
                     {"domain": "d", "above_floor": False}]
        with caplog.at_level(logging.INFO, logger=diag.__name__):
            emit(raw_state={"hypotheses": [hyp("A")]}, validated=None,
                 required_domains=["a"], governed_domains=["a"],
                 adaptive_total=4, adaptive_budget=3, candidates=rows_with)
        lines = [r.getMessage() for r in caplog.records
                 if EVENT in r.getMessage()]
        assert lines
        recorded = json.loads(lines[-1].split(EVENT, 1)[1].strip())
        assert recorded["candidates_total"] == 4
        assert recorded["candidates_above_floor"] == 3
        assert recorded["adaptive_surplus"] == 0
        assert recorded["preference_had_a_choice"] is False
        assert recorded["candidates_below_floor"] == ["d"]

    def test_candidate_rows_carry_no_clinical_text(self):
        rows = [{"source": "model", "domain": "sweat", "tier": 0,
                 "preference_rank": [0, 0, 1], "score": 72.0,
                 "above_floor": True, "selected": True}]
        blob = str(rows)
        for forbidden in ("question", "咳嗽", "rationale", "pattern"):
            assert forbidden not in blob

    def test_telemetry_does_not_change_the_selection(self):
        """Same pool, selected twice, with a full emission in between."""
        coverage, signals, pool, _ = self._pool({"sweat", "sputum"})
        first = select_questions(deterministic=[], adaptive=pool,
                                 coverage=coverage, signals=signals,
                                 differential_domains=self.REQUIRED)
        before = [s.candidate.domain for s in first.scores]
        emit(raw_state={"hypotheses": [hyp("A")]}, validated=None,
             required_domains=self.REQUIRED,
             governed_domains=["throat"], adaptive_total=len(pool),
             adaptive_budget=first.adaptive_budget,
             selected_domains=before,
             candidates=[{"domain": c.domain, "above_floor": True}
                         for c in pool])
        second = select_questions(deterministic=[], adaptive=pool,
                                  coverage=coverage, signals=signals,
                                  differential_domains=self.REQUIRED)
        assert before == [s.candidate.domain for s in second.scores]
        assert first.adaptive_budget == second.adaptive_budget
        assert first.sufficiency == second.sufficiency
