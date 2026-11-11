"""X1D-P20: compound primary-syndrome matching, audited with a synthetic matrix.

The matcher (knowledge/pattern_match.py) splits a model pattern name only at
punctuation and whitespace and requires a component to EQUAL a governed name or
alias (a trailing 证 ignored). Only the selected primary's match drives
retrieval (PRIMARY_ANCHORED_FAIL_CLOSED_V1); explanatory prose and secondary
names never do.

One confirmed defect is fixed here, fail-closed and without any new clinical
equivalence: a single name whose components equal two DIFFERENT governed
patterns ("风热犯肺，风寒束表") used to anchor on whichever pattern had been
ingested first. It now anchors on neither.

Two behaviours are deliberately UNCHANGED and pinned as-is, because changing
them needs a clinical decision, not an engineering one (see the P20 report):
a governed pattern combined with a different, non-governed pattern
("风热犯肺，痰热壅肺"), and a qualifier on the governed name ("风热犯肺（已排除）").

All objects are SYNTHETIC in-memory fixtures; the hypotheses are scripted.
"""

import pytest

from app.services.coverage import harness as H
from app.services.knowledge.pattern_match import COMPOUND_PIECE, NORMALIZED_CANONICAL
from app.services.reasoning import primary as P
from tests.test_x1d_p19r_approved_cough_contract import (APPROVED, FLU_TEXT, SYN_SRC, YES_NONE, assemble,
                                                         cough_slice, hyp)
from tests.test_x1d_p7_source_verification import env, store  # noqa: F401


def outcome(store, *hyps, context=YES_NONE, text=None):
    kw = {"text": text} if text else {}
    resp = assemble(store, *hyps, context=context, **kw)
    primary = next((a for a in resp.reasoning.pattern_assessments if a.is_primary), None)
    return resp, primary


# ----------------------------------------------------------------------
# Clearly safe under the existing contract
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name, mechanism", [("风热犯肺", NORMALIZED_CANONICAL), ("风热犯肺证", "EXACT_CANONICAL")])
def test_exact_and_canonical_primary_reach_formula_8_only_with_satisfied_applicability(store, name, mechanism):
    cough_slice(store)
    resp, primary = outcome(store, hyp(name, 0.8))
    assert primary.governed_pattern_name == "风热犯肺证" and primary.match_mechanism == mechanism
    assert [c.name for c in resp.formula_candidates] == ["桑菊饮加减"]
    resp, _ = outcome(store, hyp(name, 0.8), context=None)
    assert resp.formula_candidates == [] and resp.reasoning.condition_scope_required == ["cough_primary"]


def test_pattern_plus_pathomechanism_matches_as_documented(store):
    """The P6 design case: '<pattern>，<pathomechanism>' (e.g. 风寒束表，肺气失宣)."""
    cough_slice(store)
    resp, primary = outcome(store, hyp("风寒束表，肺气失宣", 0.8), context=None, text=FLU_TEXT)
    assert primary.match_mechanism == COMPOUND_PIECE
    assert [c.name for c in resp.formula_candidates] == ["麻黄汤加味"]


@pytest.mark.parametrize("hyps", [
    [hyp("痰热壅肺", 0.7), hyp("风热犯肺", 0.3)],
    [hyp("风寒束表", 0.7), hyp("风热犯肺", 0.3)],
])
def test_a_secondary_never_selects_formula_8(store, hyps):
    cough_slice(store)
    resp, _ = outcome(store, *hyps)
    assert "桑菊饮加减" not in [c.name for c in resp.formula_candidates]
    assert resp.reasoning.condition_scope_required == []


def test_the_wind_cold_primary_still_resolves_its_own_formula_with_a_wind_heat_secondary(store):
    cough_slice(store)
    resp, _ = outcome(store, hyp("风寒束表", 0.7), hyp("风热犯肺", 0.3), text=FLU_TEXT)
    assert [c.name for c in resp.formula_candidates] == ["麻黄汤加味"]
    assert P.SECONDARY_GOVERNED_MATCH_NOT_USED in resp.uncertainty_flags


def test_explanatory_prose_never_matches(store):
    """Only the hypothesis NAME is matched; its reasoning text is evidence, not a name."""
    cough_slice(store)
    h = {"name": "肝郁气滞", "confidence": 0.8, "reasoning": "需与风热犯肺鉴别，但不支持风热犯肺。"}
    resp, primary = outcome(store, h)
    assert primary.corpus_match is False and resp.formula_candidates == []
    assert P.PRIMARY_PATTERN_NOT_IN_GOVERNED_CORPUS in resp.uncertainty_flags


@pytest.mark.parametrize("hyps", [
    [],                                                    # missing primary
    [hyp("风热犯肺", 0.5), hyp("风寒束表", 0.5)],             # ambiguous (tie)
])
def test_missing_or_ambiguous_primary_fails_closed(store, hyps):
    cough_slice(store)
    resp, _ = outcome(store, *hyps)
    assert resp.formula_candidates == [] and resp.reasoning.condition_scope_required == []


MALFORMED = [
    [{"name": "风热犯肺", "confidence": "high", "reasoning": "r"}],   # non-numeric confidence
    [{"name": "", "confidence": 0.9, "reasoning": "r"}],             # empty name
    [{"name": "风热犯肺", "confidence": 1.7, "reasoning": "r"}],      # out-of-range confidence
]


@pytest.mark.parametrize("hyps", MALFORMED)
def test_a_malformed_primary_is_never_selected_by_the_engine(store, hyps):
    from app.schemas.intake import RecommendationRequest
    from app.services.knowledge.resolver import KnowledgeResolver
    from app.services.reasoning.engine import DiagnosticReasoningEngine
    cough_slice(store)
    r = DiagnosticReasoningEngine(store, KnowledgeResolver(store, record_gaps=False)).analyze(
        RecommendationRequest(text_input="合成：咳嗽三天"), hyps)
    assert r.ready_for_formula_retrieval is False
    assert not any(a.is_primary for a in r.pattern_assessments)


def test_a_malformed_hypothesis_shape_fails_the_turn_rather_than_producing_a_formula(store):
    """Observed (unchanged here): the response schema rejects it, so the whole turn errors
    and core reports a system failure. Fail-closed, but as an error -- see the P20 report."""
    from pydantic import ValidationError
    cough_slice(store)
    with pytest.raises(ValidationError):
        assemble(store, {"name": "风热犯肺", "confidence": "high"}, context=YES_NONE)


def test_a_low_confidence_but_unique_primary_is_still_gated_by_applicability(store):
    """Confidence is not a gate (unchanged); the applicability questions still are."""
    cough_slice(store)
    resp, _ = outcome(store, hyp("风热犯肺", 0.12), context=None)
    assert resp.formula_candidates == [] and resp.reasoning.condition_scope_required == ["cough_primary"]


@pytest.mark.parametrize("ctx", [{"cough_primary": "NO"}, {"cough_primary": "UNKNOWN"},
                                 {"cough_primary": "YES", "cough_exclusion_features": "PRESENT"}])
def test_correct_primary_with_failed_applicability_fails_closed(store, ctx):
    cough_slice(store)
    resp, _ = outcome(store, hyp("风热犯肺", 0.8), context=ctx)
    assert resp.formula_candidates == []


# ----------------------------------------------------------------------
# Confirmed defect, fixed fail-closed
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["风热犯肺，风寒束表", "风寒束表，风热犯肺", "风寒束表、风热犯肺证"])
@pytest.mark.parametrize("ctx", [None, YES_NONE])
def test_a_primary_naming_two_governed_patterns_anchors_on_neither(store, name, ctx):
    cough_slice(store)
    resp, primary = outcome(store, hyp(name, 0.8), context=ctx, text=FLU_TEXT)
    assert resp.formula_candidates == []                              # was 麻黄汤加味, by ingestion order
    assert resp.reasoning.condition_scope_required == []
    assert primary.corpus_match is False and primary.pattern_id is None
    assert P.MULTIPLE_GOVERNED_PATTERNS_MATCHED in resp.uncertainty_flags
    assert P.PRIMARY_PATTERN_NOT_IN_GOVERNED_CORPUS not in resp.uncertainty_flags
    assert resp.reasoning.ready_for_formula_retrieval is False


def test_a_secondary_naming_two_patterns_does_not_flag_the_primary(store):
    cough_slice(store)
    resp, _ = outcome(store, hyp("风寒束表", 0.8), hyp("风热犯肺，风寒束表", 0.3), text=FLU_TEXT)
    assert [c.name for c in resp.formula_candidates] == ["麻黄汤加味"]
    assert P.MULTIPLE_GOVERNED_PATTERNS_MATCHED not in resp.uncertainty_flags


def test_the_harness_mirrors_the_fail_closed_rule(store):
    pid_snap = H.CorpusSnapshot(
        environment="staging",
        sources=(H.SourceSnap(SYN_SRC, "SOURCE_VERIFIED", 3, live_verification=True),),
        entities=(H.EntitySnap("p-heat", "pattern", "风热犯肺证", "SOURCE_VERIFIED", 3, source_ids=(SYN_SRC,),
                               live_verification=True, applicability=APPROVED),
                  H.EntitySnap("p-cold", "pattern", "风寒束表", "SOURCE_VERIFIED", 3, source_ids=(SYN_SRC,),
                               live_verification=True)),
        relationships=(), core_formula_names=())
    r = H.classify_turn(pid_snap, H.TurnInput(1, (H.Hypothesis("风热犯肺，风寒束表", 0.8),)))
    assert r.primary_match is False
    assert r.hypothesis_explanations[0][1] == H.MULTIPLE_GOVERNED_PATTERNS


def test_single_governed_match_helper():
    assert P.single_governed_match([]) == []
    one = [{"pattern_id": "a"}, {"pattern_id": "a"}]
    assert P.single_governed_match(one) == one
    assert P.single_governed_match([{"pattern_id": "a"}, {"pattern_id": "b"}]) == []


# ----------------------------------------------------------------------
# Unchanged, pending an explicit clinical decision (pinned so a change is deliberate)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["风热犯肺，痰热壅肺", "风热犯肺，热壅于肺", "风热犯肺（已排除）"])
def test_policy_pending_compound_labels_still_match_and_remain_gated(store, name):
    cough_slice(store)
    resp, primary = outcome(store, hyp(name, 0.8), context=None)
    assert primary.match_mechanism == COMPOUND_PIECE and primary.governed_pattern_name == "风热犯肺证"
    assert resp.formula_candidates == [] and resp.reasoning.condition_scope_required == ["cough_primary"]
