"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P15: primary-anchored, fail-closed formula retrieval.

P14 replayed 71 Staging trace turns and found 18 whose top diagnosis was
wind-heat but whose lower-ranked 风寒束表 hypothesis would still fetch
麻黄汤加味, because retrieval pooled every matched hypothesis. Only the
unambiguous primary may drive readiness and retrieval now, and every
pattern-derived candidate says which hypothesis produced it.

Runs the real engine and assembler over the real governed store (the P7
ingest -> submit -> source-verify flow); the provider is scripted, so no model
is called.
"""

import asyncio
import hashlib
from pathlib import Path

import pytest

from app.schemas.intake import RecommendationRequest
from app.services.coverage import harness as H
from app.services.llm.provider import LLMProvider, ProviderResult
from app.services.reasoning import primary as P
from app.services.reasoning.engine import DiagnosticReasoningEngine
from app.services.knowledge.resolver import KnowledgeResolver
from app.services.recommendation.assembler import RecommendationAssembler
from tests.test_x1d_p7_source_verification import env, full_slice, store  # noqa: F401

# Every basic field answered, so readiness depends on the pattern alone.
COMPLETE = "恶寒发热两天，体温38度，怕冷，无汗，身痛头痛，鼻流清涕。食欲一般，大便正常，睡眠尚可。"


def hyp(name, confidence, reasoning="r"):
    return {"name": name, "confidence": confidence, "reasoning": reasoning}


def engine_for(store):
    return DiagnosticReasoningEngine(store, KnowledgeResolver(store, record_gaps=False))


def analyze(store, *hyps, text=COMPLETE):
    return engine_for(store).analyze(RecommendationRequest(text_input=text), list(hyps))


class Scripted(LLMProvider):
    def __init__(self, hyps):
        self.hyps = hyps

    async def generate_recommendation(self, **kwargs):
        return ProviderResult(summary="scripted", pattern_hypotheses=list(self.hyps),
                              provider="scripted", model="scripted-v1")


def assemble(store, *hyps, text=COMPLETE):
    asm = RecommendationAssembler(Scripted(hyps))
    asm.corpus = store
    asm.reasoning = engine_for(store)
    return asyncio.run(asm.generate(RecommendationRequest(text_input=text, interview_depth=9)))


# ----------------------------------------------------------------------
# 6-8. one deterministic definition of "primary"
# ----------------------------------------------------------------------
def test_the_highest_confidence_hypothesis_is_primary_even_off_index_zero():
    s = P.select_primary([hyp("风热犯肺", 0.3), hyp("风寒束表", 0.8), hyp("燥邪犯肺", 0.1)])
    assert (s.outcome, s.index, s.rank, s.name, s.confidence) == (P.PRIMARY_SELECTED, 1, 2, "风寒束表", 0.8)


@pytest.mark.parametrize("hyps", [[hyp("风寒束表", 0.5), hyp("风热犯肺", 0.5)],
                                  [hyp("风寒束表", 0.4), hyp("风热犯肺", 0.4), hyp("x", 0.1)]])
def test_a_tie_for_highest_confidence_fails_closed(hyps):
    assert P.select_primary(hyps).outcome == P.AMBIGUOUS_PRIMARY_HYPOTHESIS


@pytest.mark.parametrize("bad", [None, "0.8", True, float("nan"), float("inf"), -0.1, 1.5])
def test_a_missing_or_invalid_confidence_fails_closed(bad):
    assert P.select_primary([hyp("风寒束表", bad), hyp("风热犯肺", 0.2)]).outcome == P.INVALID_PRIMARY_HYPOTHESIS
    assert P.select_primary([{"name": "风寒束表"}]).outcome == P.INVALID_PRIMARY_HYPOTHESIS


def test_malformed_or_empty_lists_fail_closed_and_position_is_never_a_fallback():
    assert P.select_primary([]).outcome == P.NO_PATTERN_HYPOTHESES
    assert P.select_primary(None).outcome == P.NO_PATTERN_HYPOTHESES
    assert P.select_primary(["风寒束表"]).outcome == P.INVALID_PRIMARY_HYPOTHESIS
    assert P.select_primary([hyp("", 0.9), hyp("风寒束表", 0.2)]).outcome == P.INVALID_PRIMARY_HYPOTHESIS


def test_an_ambiguous_primary_blocks_retrieval_even_when_both_are_governed(store):
    full_slice(store)
    r = analyze(store, hyp("风寒束表", 0.5), hyp("风寒束表，肺气失宣", 0.5))
    assert r.primary_selection == P.AMBIGUOUS_PRIMARY_HYPOTHESIS
    assert r.ready_for_formula_retrieval is False
    assert all(a.corpus_match for a in r.pattern_assessments)       # still matched, still reported
    assert P.AMBIGUOUS_PRIMARY_HYPOTHESIS in r.uncertainty_flags


def test_an_invalid_confidence_in_model_output_fails_closed_without_crashing(store):
    full_slice(store)
    r = analyze(store, hyp("风寒束表", None), hyp("风热犯肺", 0.2))
    assert r.primary_selection == P.INVALID_PRIMARY_HYPOTHESIS and r.ready_for_formula_retrieval is False
    assert r.pattern_assessments[0].model_confidence == 0.0
    # A numeric string is coerced by the response schema but is not a number to
    # the selector, so the whole turn still fails closed.
    resp = assemble(store, hyp("风寒束表", "0.8"), hyp("风热犯肺", 0.2))
    assert resp.formula_candidates == []
    assert resp.reasoning.primary_selection == P.INVALID_PRIMARY_HYPOTHESIS


# ----------------------------------------------------------------------
# 1, 2, 9, 10. the legitimate path, with provenance
# ----------------------------------------------------------------------
def test_a_single_fenghan_hypothesis_retrieves_mahuangtang_jiawei_with_provenance(store):
    pid, fid, _ = full_slice(store)
    r = analyze(store, hyp("风寒束表", 0.8))
    assert r.primary_selection == P.PRIMARY_SELECTED and r.ready_for_formula_retrieval is True
    resp = assemble(store, hyp("风寒束表", 0.8))
    assert [c.name for c in resp.formula_candidates] == ["麻黄汤加味"]
    c = resp.formula_candidates[0]
    assert c.derived_from == {"pattern_id": pid, "pattern_name": "风寒束表", "hypothesis_rank": 1,
                              "hypothesis_name": "风寒束表", "model_confidence": 0.8,
                              "match_mechanism": "EXACT_CANONICAL"}
    assert c.retrieval_policy == "PRIMARY_ANCHORED_FAIL_CLOSED_V1" == P.RETRIEVAL_POLICY
    assert c.governance["basis"] == "SOURCE_VERIFIED"        # source-verification semantics unchanged


def test_primary_fenghan_with_an_unrelated_secondary_retrieves_from_the_primary(store):
    pid, _, _ = full_slice(store)
    resp = assemble(store, hyp("风寒束表，肺气失宣", 0.7), hyp("痰热壅肺", 0.2))
    assert [c.name for c in resp.formula_candidates] == ["麻黄汤加味"]
    d = resp.formula_candidates[0].derived_from
    assert (d["pattern_id"], d["hypothesis_rank"], d["hypothesis_name"], d["match_mechanism"]) == (
        pid, 1, "风寒束表，肺气失宣", "COMPOUND_PIECE")
    assert P.SECONDARY_GOVERNED_MATCH_NOT_USED not in resp.uncertainty_flags


def test_the_primary_off_index_zero_is_the_one_whose_provenance_is_recorded(store):
    full_slice(store)
    resp = assemble(store, hyp("痰热壅肺", 0.2), hyp("风寒束表证", 0.7))
    d = resp.formula_candidates[0].derived_from
    assert (d["hypothesis_rank"], d["hypothesis_name"], d["match_mechanism"]) == (2, "风寒束表证", "NORMALIZED_CANONICAL")


# ----------------------------------------------------------------------
# 3, 4, 5. the P14 defect
# ----------------------------------------------------------------------
P14_CASE = (hyp("风热犯肺", 0.46), hyp("风寒束表", 0.32), hyp("表证夹轻度肺气失宣", 0.22))


def test_wind_heat_primary_never_fetches_a_formula_through_the_wind_cold_secondary(store):
    full_slice(store)
    resp = assemble(store, *P14_CASE)
    assert [c.name for c in resp.formula_candidates] == []
    assert not any(c.derived_from for c in resp.formula_candidates)


def test_the_p14_case_says_why(store):
    full_slice(store)
    r = analyze(store, *P14_CASE)
    assert P.PRIMARY_PATTERN_NOT_IN_GOVERNED_CORPUS in r.uncertainty_flags
    assert P.SECONDARY_GOVERNED_MATCH_NOT_USED in r.uncertainty_flags
    # the secondary match is kept for diagnostics, just not used
    sec = r.pattern_assessments[1]
    assert sec.corpus_match and not sec.is_primary and sec.match_mechanism == "EXACT_CANONICAL"
    assert r.pattern_assessments[0].is_primary and not r.pattern_assessments[0].corpus_match


def test_a_secondary_match_never_turns_readiness_on(store):
    full_slice(store)
    assert analyze(store, *P14_CASE).ready_for_formula_retrieval is False
    assert analyze(store, hyp("风寒束表", 0.46)).ready_for_formula_retrieval is True
    # HIGH-field readiness logic is unchanged
    assert analyze(store, hyp("风寒束表", 0.8), text="身痛头痛").ready_for_formula_retrieval is False


def test_the_interview_engine_readiness_is_primary_anchored_too():
    from app.schemas.reasoning import PatternAssessment
    sec_only = [PatternAssessment(name="风热犯肺", model_confidence=0.5, is_primary=True, corpus_match=False),
                PatternAssessment(pattern_id="pat-1", name="风寒束表", model_confidence=0.3, corpus_match=True)]
    assert P.primary_governed_assessment(sec_only) is None
    src = Path(__file__).resolve().parents[1].joinpath("app/services/interview/engine.py").read_text(encoding="utf-8")
    assert "any(x.corpus_match" not in src and "primary_governed_assessment(" in src


# ----------------------------------------------------------------------
# 18. indication fallback is independent of pattern hypotheses, and unchanged
# ----------------------------------------------------------------------
def test_indication_fallback_does_not_consume_pattern_hypotheses():
    import inspect
    from app.services.knowledge.persistent_clinical import PersistentClinicalStore
    assert list(inspect.signature(PersistentClinicalStore.eligible_formula_candidates).parameters) == [
        "self", "symptoms", "text_input"]
    src = inspect.getsource(RecommendationAssembler.generate)
    assert "self.corpus.eligible_formula_candidates(\n                request.symptoms,\n                request.text_input," in src


def test_indication_candidates_carry_no_pattern_provenance(store, monkeypatch):
    full_slice(store)
    fake = [{"formula_id": "frm-q", "name": "清肺排毒汤", "confidence": 0.5, "rationale": "indication",
             "ingredients": [], "safety_flags": []}]
    monkeypatch.setattr(store, "eligible_formula_candidates", lambda symptoms, text_input="": fake)
    resp = assemble(store, *P14_CASE)
    assert [c.name for c in resp.formula_candidates] == ["清肺排毒汤"]
    assert resp.formula_candidates[0].derived_from is None and resp.formula_candidates[0].retrieval_policy is None


# ----------------------------------------------------------------------
# 16. harness parity with the new production rule
# ----------------------------------------------------------------------
def _snapshot(pid, fid, rid):
    src = "nhc-natcm-flu-dx-tx-2025"
    return H.CorpusSnapshot(
        environment="staging", sources=(H.SourceSnap(src, "SOURCE_VERIFIED", 3, live_verification=True),),
        entities=(H.EntitySnap(pid, "pattern", "风寒束表", "SOURCE_VERIFIED", 3, source_ids=(src,), live_verification=True),
                  H.EntitySnap(fid, "formula", "麻黄汤加味", "SOURCE_VERIFIED", 3, source_ids=(src,), live_verification=True)),
        relationships=(H.RelationshipSnap(rid, pid, fid, "SOURCE_VERIFIED", 3, evidence_source_ids=(src,),
                                          live_verification=True),),
        core_formula_names=("麻黄汤加味",))


PARITY_CASES = [
    (hyp("风寒束表", 0.8),),
    (hyp("风寒束表，肺气失宣", 0.7), hyp("痰热壅肺", 0.2)),
    P14_CASE,
    (hyp("痰热壅肺", 0.2), hyp("风寒束表证", 0.7)),
    (hyp("风寒束表", 0.5), hyp("风热犯肺", 0.5)),
    (hyp("风寒束表", "0.8"), hyp("风热犯肺", 0.2)),
    (hyp("肝阳上亢", 0.9),),
]


@pytest.mark.parametrize("case", PARITY_CASES)
def test_the_harness_current_reality_agrees_with_the_assembler(store, case):
    pid, fid, rid = full_slice(store)
    resp = assemble(store, *case)
    turn = H.TurnInput(ordinal=1, hypotheses=tuple(H.Hypothesis(h["name"], h["confidence"]) for h in case))
    r = H.classify_turn(_snapshot(pid, fid, rid), turn)
    assert [c.name for c in resp.formula_candidates] == [c.name for c in r.candidates]
    if r.detail == "SECONDARY_GOVERNED_MATCH_NOT_USED":
        assert P.SECONDARY_GOVERNED_MATCH_NOT_USED in resp.uncertainty_flags


def test_the_harness_keeps_the_historical_pooled_rule_as_a_simulation(store):
    pid, fid, rid = full_slice(store)
    turn = H.TurnInput(ordinal=1, hypotheses=tuple(H.Hypothesis(h["name"], h["confidence"]) for h in P14_CASE))
    snap = _snapshot(pid, fid, rid)
    assert H.classify_turn(snap, turn, policy=H.POOLED_ALL_HYPOTHESES_HISTORICAL).outcome == H.FULL_PATH
    assert H.classify_turn(snap, turn).outcome == H.NO_PATTERN_MATCH


# ----------------------------------------------------------------------
# 19, 20. no write path; protected file untouched
# ----------------------------------------------------------------------
def test_the_primary_selector_and_harness_have_no_database_access():
    root = Path(__file__).resolve().parents[1]
    for rel in ("app/services/reasoning/primary.py", "app/services/coverage/harness.py",
                "app/services/coverage/measure.py", "app/services/coverage/whatif.py"):
        text = (root / rel).read_text(encoding="utf-8")
        for banned in ("app.db", "sqlalchemy", "get_session_factory"):
            assert banned not in text, (rel, banned)


def test_the_protected_audit_script_is_untouched():
    path = Path(__file__).resolve().parents[1] / "scripts/corpus_coverage_audit.py"
    if not path.exists():
        pytest.skip("protected local file not present in this checkout")
    data = path.read_bytes()
    assert len(data) == 8894
    assert hashlib.sha256(data).hexdigest().startswith("7edb88f74f7f6258")
