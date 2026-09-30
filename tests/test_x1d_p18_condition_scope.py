"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P18: condition-scoped governed retrieval.

A relationship whose pattern or formula content carries
applicability={"condition": "COUGH_PRIMARY"} may produce a formula candidate
only when the patient's explicit answer says cough is the main/only complaint.
Everything else fails closed; a never-answered context asks one fixed question.

The scoped relationship here is a SYNTHETIC in-memory fixture built through
the real ingest -> submit -> source-verify flow. Nothing real is created.
"""

import asyncio

import pytest

from app.schemas.clinical_workflow import ClinicalEntityType
from app.schemas.intake import RecommendationRequest
from app.services.coverage import harness as H
from app.services.governance import applicability as A
from app.services.governance import canonical
from app.services.llm.provider import LLMProvider, ProviderResult
from app.services.reasoning import primary as P
from app.services.reasoning.engine import DiagnosticReasoningEngine
from app.services.knowledge.resolver import KnowledgeResolver
from app.services.recommendation.assembler import RecommendationAssembler
from tests.test_x1d_p7_source_verification import (  # noqa: F401
    SRC, env, full_slice, ingest, make_relationship, store, submit_source, verify)

COMPLETE = "恶寒发热两天，体温38度，怕冷，无汗，身痛头痛，鼻流清涕。食欲一般，大便正常，睡眠尚可。"
COUGH_TEXT = "咳嗽三天，咳嗽频剧，咽喉燥痛，痰黄，发热，体温38度。食欲一般，大便正常，睡眠尚可。"
SCOPED = {"condition": "COUGH_PRIMARY"}


def hyp(name, confidence):
    return {"name": name, "confidence": confidence, "reasoning": "r"}


class Scripted(LLMProvider):
    def __init__(self, hyps):
        self.hyps = hyps

    async def generate_recommendation(self, **kwargs):
        return ProviderResult(summary="scripted", pattern_hypotheses=list(self.hyps),
                              provider="scripted", model="scripted-v1")


def assemble(store, *hyps, context=None, text=COUGH_TEXT, fallback=None):
    asm = RecommendationAssembler(Scripted(hyps))
    asm.corpus = store
    asm.reasoning = DiagnosticReasoningEngine(store, KnowledgeResolver(store, record_gaps=False))
    if fallback is not None:
        store.eligible_formula_candidates = lambda symptoms, text_input="": list(fallback)
    return asyncio.run(asm.generate(RecommendationRequest(
        text_input=text, interview_depth=9, condition_context=context)))


def scoped_slice(store, *, pattern_applicability=SCOPED, formula_applicability=None, with_flu=True):
    """Flu slice (unrestricted, as on Staging) plus a synthetic cough-scoped link."""
    if with_flu:
        full_slice(store)
    extra_p = {} if pattern_applicability is None else {"applicability": pattern_applicability}
    extra_f = {} if formula_applicability is None else {"applicability": formula_applicability}
    pid = ingest(store, ClinicalEntityType.PATTERN, "风热犯肺证", "xerbs:pattern:synthetic-cough:风热犯肺证", **extra_p)
    fid = ingest(store, ClinicalEntityType.FORMULA, "桑菊饮加减", "xerbs:formula:synthetic-cough:桑菊饮加减",
                 ingredients=["桑叶", "菊花", "苦杏仁", "连翘", "薄荷", "桔梗", "芦根", "甘草"], **extra_f)
    if not with_flu:
        submit_source(store)
        verify(store, "SOURCE", SRC)
    verify(store, "CLINICAL_ENTITY", pid)
    verify(store, "CLINICAL_ENTITY", fid)
    rid = make_relationship(store, pid, fid)
    verify(store, "CLINICAL_RELATIONSHIP", rid)
    return pid, fid, rid


# ----------------------------------------------------------------------
# The evaluator itself
# ----------------------------------------------------------------------
@pytest.mark.parametrize("ctx,expected", [
    ({"cough_primary": "YES"}, A.SATISFIED),
    ({"cough_primary": "NO"}, A.REJECTED),
    ({"cough_primary": "UNKNOWN"}, A.REJECTED),
    ({"cough_primary": "yes"}, A.REJECTED),        # malformed answer never widens
    ({"cough_primary": "是"}, A.REJECTED),
    ({}, A.MISSING),
    (None, A.MISSING),
    ({"cough_primary": None}, A.MISSING),
])
def test_only_an_explicit_yes_satisfies_cough_primary(ctx, expected):
    outcome, conditions, missing = A.evaluate([{"applicability": SCOPED}, {}], ctx)
    assert outcome == expected
    assert missing == (("cough_primary",) if expected == A.MISSING else ())


@pytest.mark.parametrize("bad", ["COUGH_PRIMARY", None, [], {"condition": "PREGNANCY"},
                                 {"condition": "COUGH_PRIMARY", "severity": "mild"}, {"cond": "COUGH_PRIMARY"},
                                 {"condition": ["COUGH_PRIMARY"]}])
def test_unsupported_or_malformed_applicability_fails_closed(bad):
    assert A.evaluate([{"applicability": bad}, {}], {"cough_primary": "YES"})[0] == A.UNSUPPORTED
    assert not A.eligible(A.UNSUPPORTED)


def test_no_applicability_key_is_unrestricted():
    assert A.evaluate([{}, {"name": "x"}], None)[0] == A.NOT_SCOPED


def test_applicability_is_bound_into_the_entity_content_hash():
    """Broadening scope later is a content change -> a new version the verification no longer covers."""
    assert "applicability" not in canonical.ENTITY_EXCLUDED_FIELDS
    snap = {"name": "风热犯肺证", "aliases": [], "indications": []}
    a = canonical.digest(canonical.entity_subject(external_id="x", entity_type="pattern",
                                                  snapshot={**snap, "applicability": SCOPED}))
    b = canonical.digest(canonical.entity_subject(external_id="x", entity_type="pattern", snapshot=snap))
    assert a != b


# ----------------------------------------------------------------------
# CASE 1, 17. unrestricted content is unchanged
# ----------------------------------------------------------------------
@pytest.mark.parametrize("ctx", [None, {"cough_primary": "YES"}, {"cough_primary": "NO"}])
def test_the_unrestricted_fenghan_path_is_unchanged_whatever_the_context(store, ctx):
    scoped_slice(store)
    resp = assemble(store, hyp("风寒束表", 0.8), context=ctx, text=COMPLETE)
    assert [c.name for c in resp.formula_candidates] == ["麻黄汤加味"]
    assert resp.formula_candidates[0].retrieval_policy == P.RETRIEVAL_POLICY
    assert resp.formula_candidates[0].governance["applicability"] == []
    assert resp.reasoning.condition_scope_required == []
    assert not {A.FLAG_CONTEXT_REQUIRED, A.FLAG_NOT_SATISFIED, A.FLAG_UNSUPPORTED} & set(resp.uncertainty_flags)


# ----------------------------------------------------------------------
# CASE 2-5, 13. the four answers
# ----------------------------------------------------------------------
def test_yes_makes_the_scoped_relationship_eligible(store):
    pid, _, _ = scoped_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context={"cough_primary": "YES"})
    assert [c.name for c in resp.formula_candidates] == ["桑菊饮加减"]
    c = resp.formula_candidates[0]
    assert c.governance["applicability"] == ["COUGH_PRIMARY"]
    assert c.derived_from["pattern_id"] == pid and c.derived_from["hypothesis_rank"] == 1
    assert A.FLAG_SATISFIED in resp.uncertainty_flags
    assert resp.reasoning.condition_scope_required == []


@pytest.mark.parametrize("answer", ["NO", "UNKNOWN"])
def test_no_or_unknown_blocks_and_never_asks_again(store, answer):
    scoped_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context={"cough_primary": answer})
    assert resp.formula_candidates == []
    assert resp.reasoning.condition_scope_required == []          # answered: not re-asked
    assert A.FLAG_NOT_SATISFIED in resp.uncertainty_flags


def test_missing_context_asks_the_fixed_question_and_creates_no_candidate(store):
    scoped_slice(store)
    fallback = [{"formula_id": "frm-q", "name": "清肺排毒汤", "confidence": 0.5, "rationale": "indication",
                 "ingredients": [], "safety_flags": []}]
    resp = assemble(store, hyp("风热犯肺", 0.8), context=None, fallback=fallback)
    assert resp.formula_candidates == []                           # not even the indication fallback
    assert resp.reasoning.condition_scope_required == ["cough_primary"]
    assert A.FLAG_CONTEXT_REQUIRED in resp.uncertainty_flags
    assert resp.reasoning.ready_for_formula_retrieval is True     # reasoning is complete; scope is not


# ----------------------------------------------------------------------
# CASE 6-8. the same case continues after the answer
# ----------------------------------------------------------------------
@pytest.mark.parametrize("answer,expected", [("YES", ["桑菊饮加减"]), ("NO", []), ("UNKNOWN", [])])
def test_after_the_answer_the_case_continues_without_asking_again(store, answer, expected):
    scoped_slice(store)
    first = assemble(store, hyp("风热犯肺", 0.8), context=None)
    assert first.reasoning.condition_scope_required == ["cough_primary"]
    second = assemble(store, hyp("风热犯肺", 0.8), context={"cough_primary": answer})
    assert [c.name for c in second.formula_candidates] == expected
    assert second.reasoning.condition_scope_required == []


# ----------------------------------------------------------------------
# CASE 9, 10. unsupported / malformed applicability
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", [{"condition": "PREGNANCY"}, "COUGH_PRIMARY", {"condition": "COUGH_PRIMARY", "x": 1}])
def test_unsupported_scope_fails_closed_end_to_end(store, bad):
    scoped_slice(store, pattern_applicability=bad)
    resp = assemble(store, hyp("风热犯肺", 0.8), context={"cough_primary": "YES"})
    assert resp.formula_candidates == []
    assert resp.reasoning.condition_scope_required == []           # nothing to ask can fix it
    assert A.FLAG_UNSUPPORTED in resp.uncertainty_flags


def test_scope_on_the_formula_side_is_enforced_too(store):
    scoped_slice(store, pattern_applicability=None, formula_applicability=SCOPED)
    assert assemble(store, hyp("风热犯肺", 0.8), context=None).reasoning.condition_scope_required == ["cough_primary"]
    assert [c.name for c in assemble(store, hyp("风热犯肺", 0.8),
                                     context={"cough_primary": "YES"}).formula_candidates] == ["桑菊饮加减"]


# ----------------------------------------------------------------------
# CASE 11, 12. the P15 primary rule decides first
# ----------------------------------------------------------------------
def test_a_secondary_scoped_match_never_triggers_the_scope_question(store):
    scoped_slice(store)
    resp = assemble(store, hyp("痰热壅肺", 0.6), hyp("风热犯肺", 0.3), context=None)
    assert resp.formula_candidates == []
    assert resp.reasoning.condition_scope_required == []
    assert P.SECONDARY_GOVERNED_MATCH_NOT_USED in resp.uncertainty_flags
    assert A.FLAG_CONTEXT_REQUIRED not in resp.uncertainty_flags


def test_an_ambiguous_primary_retrieves_nothing_and_asks_nothing(store):
    scoped_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.5), hyp("风寒束表", 0.5), context=None)
    assert resp.formula_candidates == [] and resp.reasoning.condition_scope_required == []
    assert P.AMBIGUOUS_PRIMARY_HYPOTHESIS in resp.uncertainty_flags


# ----------------------------------------------------------------------
# CASE 14, 15. nothing but the explicit answer counts
# ----------------------------------------------------------------------
def test_a_cough_keyword_in_free_text_is_not_an_answer(store):
    scoped_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context=None,
                    text="主要是咳嗽，咳嗽为主要症状，只有咳嗽。体温38度，发热两天。食欲一般，大便正常，睡眠尚可。")
    assert resp.formula_candidates == []
    assert resp.reasoning.condition_scope_required == ["cough_primary"]


def test_an_unrelated_context_field_is_not_an_answer(store):
    scoped_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context={"sputum": "YES", "cough": "YES"})
    assert resp.formula_candidates == []
    assert resp.reasoning.condition_scope_required == ["cough_primary"]


# ----------------------------------------------------------------------
# CASE 18. the gate runs before any candidate exists
# ----------------------------------------------------------------------
def test_the_store_creates_no_candidate_before_scope_eligibility(store):
    pid, _, rid = scoped_slice(store, with_flu=False)
    report = []
    assert store.eligible_formula_candidates_for_patterns([pid], condition_context=None, scope_report=report) == []
    assert report == [{"relationship_id": rid, "outcome": A.MISSING, "conditions": ["COUGH_PRIMARY"],
                       "missing_context": ["cough_primary"]}]
    # the pre-P18 call signature still works and still fails closed
    assert store.eligible_formula_candidates_for_patterns([pid]) == []


# ----------------------------------------------------------------------
# Harness: the same gate, the same outcomes
# ----------------------------------------------------------------------
def _snapshot(pid, fid, rid, applicability=SCOPED):
    return H.CorpusSnapshot(
        environment="staging", sources=(H.SourceSnap(SRC, "SOURCE_VERIFIED", 3, live_verification=True),),
        entities=(H.EntitySnap(pid, "pattern", "风热犯肺证", "SOURCE_VERIFIED", 3, source_ids=(SRC,),
                               live_verification=True, applicability=applicability),
                  H.EntitySnap(fid, "formula", "桑菊饮加减", "SOURCE_VERIFIED", 3, source_ids=(SRC,),
                               live_verification=True)),
        relationships=(H.RelationshipSnap(rid, pid, fid, "SOURCE_VERIFIED", 3, evidence_source_ids=(SRC,),
                                          live_verification=True),),
        core_formula_names=("桑菊饮加减",))


@pytest.mark.parametrize("ctx,outcome", [
    ({"cough_primary": "YES"}, H.FULL_PATH),
    ({"cough_primary": "NO"}, H.MATCHED_PRIMARY_SCOPE_REJECTED),
    ({"cough_primary": "UNKNOWN"}, H.MATCHED_PRIMARY_SCOPE_REJECTED),
    (None, H.MATCHED_PRIMARY_SCOPE_MISSING),
])
def test_the_harness_reports_scope_states_and_agrees_with_the_store(store, ctx, outcome):
    pid, fid, rid = scoped_slice(store, with_flu=False)
    turn = H.TurnInput(ordinal=1, hypotheses=(H.Hypothesis("风热犯肺", 0.8),), condition_context=ctx)
    r = H.classify_turn(_snapshot(pid, fid, rid), turn)
    assert r.outcome == outcome
    if outcome == H.FULL_PATH:
        assert r.detail == H.MATCHED_PRIMARY_SCOPE_SATISFIED
    if outcome == H.MATCHED_PRIMARY_SCOPE_MISSING:
        assert r.detail == "cough_primary"
    prod = store.eligible_formula_candidates_for_patterns([pid], condition_context=ctx)
    assert [c["name"] for c in prod] == [c.name for c in r.candidates]


def test_the_harness_fails_closed_on_unsupported_scope():
    turn = H.TurnInput(ordinal=1, hypotheses=(H.Hypothesis("风热犯肺", 0.8),), condition_context={"cough_primary": "YES"})
    for bad in ({"condition": "PREGNANCY"}, None, "COUGH_PRIMARY"):
        assert H.classify_turn(_snapshot("p", "f", "r", bad), turn).outcome == H.UNSUPPORTED_SCOPE_FAIL_CLOSED


def test_historical_turns_carry_no_scope_answer():
    """No fabricated answers: a trace turn has no condition context, so a scoped path is MISSING."""
    from app.services.coverage import measure as M
    t = M.sanitize_trace_row({"ai_response_summary": {"pattern_hypotheses": [{"name": "风热犯肺", "confidence": 0.8}]},
                              "condition_context": {"cough_primary": "YES"}}, 1)
    assert t.condition_context is None
    assert H.classify_turn(_snapshot("p", "f", "r"), t).outcome == H.MATCHED_PRIMARY_SCOPE_MISSING
