"""X1D-P19E: the provisional two-condition cough scope, and its isolation.

风热犯肺证 -> 桑菊饮加减 (《咳嗽中医诊疗专家共识意见（2021）》§5.3.2) needs two
patient-answered conditions: cough is the main/only complaint (P18), and none of
the P19C draft exclusion features (§1.1 / §4.2.3). The second is a
PROVISIONAL ENGINEERING RULE awaiting clinical review.

Isolation: in every deployment provisional rules are disabled by a code
constant, so a link carrying a provisional condition fails closed -- no
candidate, no question -- whatever the request says. Only in-process test code
enables them. All clinical objects here are SYNTHETIC in-memory fixtures built
through the real ingest -> submit -> source-verify flow.
"""

import asyncio
from pathlib import Path

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
    SRC, env, full_slice, ingest, make_relationship, store, verify)

COUGH_TEXT = "咳嗽三天，咳嗽频剧，咽喉燥痛，痰黄，发热，体温38度。食欲一般，大便正常，睡眠尚可。"
FLU_TEXT = "恶寒发热两天，体温38度，怕冷，无汗，身痛头痛，鼻流清涕。食欲一般，大便正常，睡眠尚可。"
TWO = {"conditions": ["COUGH_PRIMARY", "COUGH_EXCLUSION_FEATURES_NONE"]}
COMPOSITION = [{"herb": "桑叶", "dose": "9g"}, {"herb": "菊花", "dose": "9g"}, {"herb": "苦杏仁", "dose": "9g"},
               {"herb": "连翘", "dose": "9g"}, {"herb": "薄荷", "dose": "6g（后下）"}, {"herb": "桔梗", "dose": "9g"},
               {"herb": "芦根", "dose": "15g"}, {"herb": "甘草", "dose": "6g"}]
YES_NONE = {"cough_primary": "YES", "cough_exclusion_features": "NONE"}


@pytest.fixture
def provisional_on(monkeypatch):
    monkeypatch.setattr(A, "_PROVISIONAL_ENGINEERING_RULES_ENABLED", True)


class Scripted(LLMProvider):
    def __init__(self, hyps):
        self.hyps = hyps

    async def generate_recommendation(self, **kwargs):
        return ProviderResult(summary="scripted", pattern_hypotheses=list(self.hyps),
                              provider="scripted", model="scripted-v1")


def hyp(name, confidence):
    return {"name": name, "confidence": confidence, "reasoning": "r"}


def assemble(store, *hyps, context=None, text=COUGH_TEXT):
    asm = RecommendationAssembler(Scripted(hyps))
    asm.corpus = store
    asm.reasoning = DiagnosticReasoningEngine(store, KnowledgeResolver(store, record_gaps=False))
    return asyncio.run(asm.generate(RecommendationRequest(text_input=text, interview_depth=9,
                                                          condition_context=context)))


def cough_slice(store, applicability=TWO):
    full_slice(store)   # the real-shaped flu slice, unrestricted
    pid = ingest(store, ClinicalEntityType.PATTERN, "风热犯肺证", "xerbs:pattern:synthetic-cough-2021:风热犯肺证",
                 applicability=applicability)
    fid = ingest(store, ClinicalEntityType.FORMULA, "桑菊饮加减", "xerbs:formula:synthetic-cough-2021:桑菊饮加减",
                 ingredients=[c["herb"] for c in COMPOSITION], composition=COMPOSITION)
    verify(store, "CLINICAL_ENTITY", pid)
    verify(store, "CLINICAL_ENTITY", fid)
    rid = make_relationship(store, pid, fid)
    verify(store, "CLINICAL_RELATIONSHIP", rid)
    return pid, fid, rid


# ----------------------------------------------------------------------
# Isolation: what every deployment ships
# ----------------------------------------------------------------------
def test_provisional_rules_are_off_by_a_code_constant_not_bound_to_any_input():
    assert A.provisional_rules_enabled() is False
    src = Path(A.__file__).read_text(encoding="utf-8")
    assert "_PROVISIONAL_ENGINEERING_RULES_ENABLED = False" in src
    # Nothing outside this module's own constant can turn it on.
    for binding in ("import os", "os.environ", "getenv", "get_settings", "from app.core"):
        assert binding not in src, binding


@pytest.mark.parametrize("ctx", [None, {"cough_primary": "YES"}, YES_NONE,
                                 {"cough_primary": "YES", "cough_exclusion_features": "NONE", "provisional": "on"}])
def test_with_provisional_rules_off_the_link_fails_closed_and_asks_nothing(store, ctx):
    """X1D-P19R made this stricter: the provisional pattern is not even matchable, so
    ordinary traffic behaves exactly as if it had never been ingested."""
    cough_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context=ctx)
    assert resp.formula_candidates == []
    assert resp.reasoning.condition_scope_required == []          # no patient-visible question at all
    assert resp.reasoning.pattern_assessments[0].corpus_match is False
    assert resp.reasoning.ready_for_formula_retrieval is False
    assert P.PRIMARY_PATTERN_NOT_IN_GOVERNED_CORPUS in resp.uncertainty_flags
    assert not [f for f in resp.uncertainty_flags if f.startswith("CONDITION_SCOPE")]


def test_with_provisional_rules_off_the_fenghan_path_is_unchanged(store):
    cough_slice(store)
    resp = assemble(store, hyp("风寒束表", 0.8), context=None, text=FLU_TEXT)
    assert [c.name for c in resp.formula_candidates] == ["麻黄汤加味"]
    assert "rule_status" not in resp.formula_candidates[0].governance


# ----------------------------------------------------------------------
# The provisional path, in-process only
# ----------------------------------------------------------------------
def test_the_questions_come_in_order_and_only_one_at_a_time(store, provisional_on):
    cough_slice(store)
    assert assemble(store, hyp("风热犯肺", 0.8), context=None).reasoning.condition_scope_required == ["cough_primary"]
    second = assemble(store, hyp("风热犯肺", 0.8), context={"cough_primary": "YES"})
    assert second.formula_candidates == []
    assert second.reasoning.condition_scope_required == ["cough_exclusion_features"]
    # an exclusion answer without cough-primary never skips the first question
    assert assemble(store, hyp("风热犯肺", 0.8), context={"cough_exclusion_features": "NONE"}
                    ).reasoning.condition_scope_required == ["cough_primary"]


def test_both_conditions_satisfied_produce_the_labelled_provisional_candidate(store, provisional_on):
    pid, fid, _ = cough_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context=YES_NONE)
    assert [c.name for c in resp.formula_candidates] == ["桑菊饮加减"]
    c = resp.formula_candidates[0]
    g = c.governance
    assert g["basis"] == "SOURCE_VERIFIED"
    assert g["rule_status"] == A.PROVISIONAL_ENGINEERING_RULE
    assert g["clinical_review"] == A.PENDING_CLINICAL_REVIEW            # never "approved"
    assert g["applicability"] == ["COUGH_EXCLUSION_FEATURES_NONE", "COUGH_PRIMARY"]
    assert g["composition"] == COMPOSITION
    assert c.derived_from["pattern_id"] == pid and c.derived_from["hypothesis_rank"] == 1
    assert c.retrieval_policy == P.RETRIEVAL_POLICY
    assert A.FLAG_PROVISIONAL in resp.uncertainty_flags


@pytest.mark.parametrize("ctx", [
    {"cough_primary": "NO"}, {"cough_primary": "UNKNOWN"},
    {"cough_primary": "YES", "cough_exclusion_features": "PRESENT"},
    {"cough_primary": "YES", "cough_exclusion_features": "UNKNOWN"},
    {"cough_primary": "YES", "cough_exclusion_features": "none"},
    {"cough_primary": "NO", "cough_exclusion_features": "NONE"},
])
def test_any_other_answer_blocks_and_asks_nothing_more(store, provisional_on, ctx):
    cough_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context=ctx)
    assert resp.formula_candidates == []
    assert resp.reasoning.condition_scope_required == []
    assert A.FLAG_NOT_SATISFIED in resp.uncertainty_flags


def test_secondary_only_and_ambiguous_primary_never_reach_the_provisional_link(store, provisional_on):
    cough_slice(store)
    sec = assemble(store, hyp("痰热壅肺", 0.6), hyp("风热犯肺", 0.3), context=YES_NONE)
    assert sec.formula_candidates == [] and sec.reasoning.condition_scope_required == []
    assert P.SECONDARY_GOVERNED_MATCH_NOT_USED in sec.uncertainty_flags
    amb = assemble(store, hyp("风热犯肺", 0.5), hyp("风寒束表", 0.5), context=YES_NONE)
    assert amb.formula_candidates == [] and amb.reasoning.condition_scope_required == []


def test_the_fenghan_path_is_unchanged_with_provisional_rules_on(store, provisional_on):
    cough_slice(store)
    resp = assemble(store, hyp("风寒束表", 0.8), context=YES_NONE, text=FLU_TEXT)
    assert [c.name for c in resp.formula_candidates] == ["麻黄汤加味"]
    assert "rule_status" not in resp.formula_candidates[0].governance


# ----------------------------------------------------------------------
# The ordered-list contract
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", [{"conditions": []}, {"conditions": ["COUGH_PRIMARY", "COUGH_PRIMARY"]},
                                 {"conditions": ["COUGH_PRIMARY", "PREGNANCY"]}, {"conditions": "COUGH_PRIMARY"},
                                 {"conditions": ["COUGH_PRIMARY"], "condition": "COUGH_PRIMARY"},
                                 {"conditions": [["COUGH_PRIMARY"]]}])
def test_malformed_condition_lists_fail_closed(provisional_on, bad):
    assert A.evaluate([{"applicability": bad}, {}], YES_NONE)[0] == A.UNSUPPORTED


def test_the_single_condition_form_keeps_its_p18_meaning():
    assert A.evaluate([{"applicability": {"condition": "COUGH_PRIMARY"}}, {}], {"cough_primary": "YES"})[0] == A.SATISFIED
    assert A.evaluate([{"applicability": {"conditions": ["COUGH_PRIMARY"]}}, {}], {"cough_primary": "YES"})[0] == A.SATISFIED


def test_the_condition_list_is_bound_into_the_content_hash():
    snap = {"name": "风热犯肺证", "aliases": []}
    digest = lambda a: canonical.digest(canonical.entity_subject(external_id="x", entity_type="pattern",
                                                                 snapshot={**snap, "applicability": a}))
    assert digest(TWO) != digest({"condition": "COUGH_PRIMARY"})
    assert digest(TWO) != digest({"conditions": ["COUGH_EXCLUSION_FEATURES_NONE", "COUGH_PRIMARY"]})


def test_the_harness_mirrors_the_provisional_gate(store, provisional_on):
    pid, fid, rid = cough_slice(store)
    snap = H.CorpusSnapshot(
        environment="staging", sources=(H.SourceSnap(SRC, "SOURCE_VERIFIED", 3, live_verification=True),),
        entities=(H.EntitySnap(pid, "pattern", "风热犯肺证", "SOURCE_VERIFIED", 3, source_ids=(SRC,),
                               live_verification=True, applicability=TWO),
                  H.EntitySnap(fid, "formula", "桑菊饮加减", "SOURCE_VERIFIED", 3, source_ids=(SRC,),
                               live_verification=True)),
        relationships=(H.RelationshipSnap(rid, pid, fid, "SOURCE_VERIFIED", 3, evidence_source_ids=(SRC,),
                                          live_verification=True),),
        core_formula_names=("桑菊饮加减",))
    for ctx, outcome in [(YES_NONE, H.FULL_PATH), ({"cough_primary": "YES"}, H.MATCHED_PRIMARY_SCOPE_MISSING),
                         ({"cough_primary": "YES", "cough_exclusion_features": "PRESENT"},
                          H.MATCHED_PRIMARY_SCOPE_REJECTED)]:
        r = H.classify_turn(snap, H.TurnInput(1, (H.Hypothesis("风热犯肺", 0.8),), condition_context=ctx))
        assert r.outcome == outcome, ctx
        prod = store.eligible_formula_candidates_for_patterns([pid], condition_context=ctx)
        assert [c["name"] for c in prod] == [c.name for c in r.candidates]
