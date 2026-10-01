"""X1D-P19R: the approved, versioned cough applicability contract X1D-P19C-V1.

风热犯肺证 -> 桑菊饮加减 (《咳嗽中医诊疗专家共识意见（2021）》§5.3.2) may produce a
formula only when, in order, the patient answers 是 to cough-primary and 都没有 to
the P19C §3 exclusion question. The contract pins those exact conditions and the
reviewed packet's SHA-256.

The project owner approved the packet by instruction; its §6 approval record is
not filled in. The contract therefore ships INACTIVE: content that references it
is invisible to matching. Tests activate it in-process only. All clinical objects
here are SYNTHETIC in-memory fixtures built through the real flows.
"""

import asyncio

import pytest

from app.schemas.clinical_workflow import ClinicalEntityType
from app.schemas.intake import RecommendationRequest
from app.schemas.safety import RelationshipCreateRequest
from app.services.coverage import harness as H
from app.services.governance import applicability as A
from app.services.governance import canonical, lifecycle
from app.services.llm.provider import LLMProvider, ProviderResult
from app.services.reasoning import primary as P
from app.services.reasoning.engine import DiagnosticReasoningEngine
from app.services.knowledge.resolver import KnowledgeResolver
from app.services.recommendation.assembler import RecommendationAssembler
from app.services.safety.engine import SafetyEngine
from tests.test_x1d_p7_source_verification import (  # noqa: F401
    SRC, env, full_slice, ingest, make_relationship, store, submit_source, verify)

CONTRACT = "X1D-P19C-V1"
APPROVED = {"conditions": ["COUGH_PRIMARY", "COUGH_EXCLUSION_FEATURES_NONE_P19C_V1"], "contract_version": CONTRACT}
COUGH_TEXT = "咳嗽三天，咳嗽频剧，咽喉燥痛，痰黄，发热，体温38度。食欲一般，大便正常，睡眠尚可。"
FLU_TEXT = "恶寒发热两天，体温38度，怕冷，无汗，身痛头痛，鼻流清涕。食欲一般，大便正常，睡眠尚可。"
YES_NONE = {"cough_primary": "YES", "cough_exclusion_features": "NONE"}
SYN_SRC = "synthetic-cough-2021"


@pytest.fixture
def contract_on(monkeypatch):
    entry = dict(A.APPLICABILITY_CONTRACTS[CONTRACT], active=True, approval_record=A.APPROVAL_RECORD_COMPLETE)
    monkeypatch.setitem(A.APPLICABILITY_CONTRACTS, CONTRACT, entry)


class Scripted(LLMProvider):
    def __init__(self, hyps):
        self.hyps = hyps

    async def generate_recommendation(self, **kwargs):
        return ProviderResult(summary="scripted", pattern_hypotheses=list(self.hyps), provider="stub", model="p19r")


def hyp(name, confidence):
    return {"name": name, "confidence": confidence, "reasoning": "r"}


def assemble(store, *hyps, context=None, text=COUGH_TEXT):
    asm = RecommendationAssembler(Scripted(hyps))
    asm.corpus = store
    asm.reasoning = DiagnosticReasoningEngine(store, KnowledgeResolver(store, record_gaps=False))
    return asyncio.run(asm.generate(RecommendationRequest(text_input=text, interview_depth=9,
                                                          condition_context=context)))


def cough_slice(store, applicability=APPROVED):
    full_slice(store)
    pid = ingest(store, ClinicalEntityType.PATTERN, "风热犯肺证", "xerbs:pattern:synthetic-cough-2021:风热犯肺证",
                 sources=(SYN_SRC,), applicability=applicability)
    fid = ingest(store, ClinicalEntityType.FORMULA, "桑菊饮加减", "xerbs:formula:synthetic-cough-2021:桑菊饮加减",
                 sources=(SYN_SRC,), ingredients=["桑叶", "菊花", "苦杏仁", "连翘", "薄荷", "桔梗", "芦根", "甘草"])
    submit_source(store, SYN_SRC)
    verify(store, "SOURCE", SYN_SRC)
    verify(store, "CLINICAL_ENTITY", pid)
    verify(store, "CLINICAL_ENTITY", fid)
    rid = make_relationship(store, pid, fid, source_id=SYN_SRC)
    verify(store, "CLINICAL_RELATIONSHIP", rid)
    return pid, fid, rid


# ----------------------------------------------------------------------
# The contract itself
# ----------------------------------------------------------------------
def test_the_contract_pins_conditions_packet_and_approval_basis():
    c = A.APPLICABILITY_CONTRACTS[CONTRACT]
    assert c["conditions"] == ("COUGH_PRIMARY", "COUGH_EXCLUSION_FEATURES_NONE_P19C_V1")
    assert c["review_packet_sha256"] == "82db5e2a57a47cf2aac0e42d5d2dffe623afd846859cbd820049a2ccd27a1b5e"
    assert c["approval_basis"] == "OWNER_INSTRUCTION"
    # §6 of the packet is not filled in: the contract ships inactive
    assert c["approval_record"] == A.APPROVAL_RECORD_INCOMPLETE and c["active"] is False
    assert A.contract_active(CONTRACT) is False


# 13 (unapproved stays inaccessible) -- the shipped default
@pytest.mark.parametrize("ctx", [None, {"cough_primary": "YES"}, YES_NONE])
def test_an_inactive_contract_leaves_ordinary_traffic_exactly_as_before(store, ctx):
    cough_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context=ctx)
    assert resp.formula_candidates == []
    assert resp.reasoning.condition_scope_required == []
    assert resp.reasoning.pattern_assessments[0].corpus_match is False   # not even matchable
    assert resp.reasoning.ready_for_formula_retrieval is False
    assert not [f for f in resp.uncertainty_flags if f.startswith("CONDITION_SCOPE")]


# ----------------------------------------------------------------------
# 1-8: the approved decision table (contract active in-process)
# ----------------------------------------------------------------------
def test_1_yes_and_none_satisfy_and_produce_the_labelled_candidate(store, contract_on):
    pid, _, _ = cough_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context=YES_NONE)
    assert [c.name for c in resp.formula_candidates] == ["桑菊饮加减"]
    g = resp.formula_candidates[0].governance
    assert g["basis"] == "SOURCE_VERIFIED" and g["clinical_review"] == "NOT_PERFORMED"
    assert g["rule_status"] == A.APPROVED_APPLICABILITY_CONTRACT
    assert g["applicability_contract"] == CONTRACT and g["applicability_approval_basis"] == "OWNER_INSTRUCTION"
    assert resp.formula_candidates[0].derived_from["pattern_id"] == pid


@pytest.mark.parametrize("second", ["PRESENT", "UNKNOWN"])
def test_2_3_yes_then_有_or_不清楚_reject_and_never_ask_again(store, contract_on, second):
    cough_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context={"cough_primary": "YES", "cough_exclusion_features": second})
    assert resp.formula_candidates == [] and resp.reasoning.condition_scope_required == []
    assert A.FLAG_NOT_SATISFIED in resp.uncertainty_flags


def test_4_yes_then_missing_asks_the_second_question_once(store, contract_on):
    cough_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context={"cough_primary": "YES"})
    assert resp.formula_candidates == []
    assert resp.reasoning.condition_scope_required == ["cough_exclusion_features"]


@pytest.mark.parametrize("first", ["NO", "UNKNOWN"])
def test_5_6_no_or_unknown_reject_without_the_second_question(store, contract_on, first):
    cough_slice(store)
    resp = assemble(store, hyp("风热犯肺", 0.8), context={"cough_primary": first})
    assert resp.formula_candidates == [] and resp.reasoning.condition_scope_required == []


def test_7_missing_first_answer_asks_the_first_question_once(store, contract_on):
    cough_slice(store)
    assert assemble(store, hyp("风热犯肺", 0.8), context=None).reasoning.condition_scope_required == ["cough_primary"]
    # a second answer alone never skips the first question
    assert assemble(store, hyp("风热犯肺", 0.8), context={"cough_exclusion_features": "NONE"}
                    ).reasoning.condition_scope_required == ["cough_primary"]


@pytest.mark.parametrize("ctx", [{"cough_primary": "是", "cough_exclusion_features": "NONE"},
                                 {"cough_primary": "YES", "cough_exclusion_features": "都没有"},
                                 {"cough_primary": "YES", "cough_exclusion_features": "none"}])
def test_8_malformed_answers_fail_closed(store, contract_on, ctx):
    cough_slice(store)
    assert assemble(store, hyp("风热犯肺", 0.8), context=ctx).formula_candidates == []


# ----------------------------------------------------------------------
# 9-11: the primary rule and model inference
# ----------------------------------------------------------------------
def test_9_10_secondary_only_and_ambiguous_primary_never_reach_the_link(store, contract_on):
    cough_slice(store)
    for hyps in ([hyp("痰热壅肺", 0.6), hyp("风热犯肺", 0.3)], [hyp("风热犯肺", 0.5), hyp("风寒束表", 0.5)]):
        resp = assemble(store, *hyps, context=YES_NONE)
        assert resp.formula_candidates == [] and resp.reasoning.condition_scope_required == []


def test_11_model_wording_and_free_text_never_establish_scope(store, contract_on):
    cough_slice(store)
    resp = assemble(store, hyp("风热犯肺（以咳嗽为主，无哮鸣无咯血）", 0.9), context=None,
                    text="只有咳嗽，没有喘息、没有咯血、没有脓痰，也没有慢性肺病。体温38度，发热两天。食欲一般，大便正常，睡眠尚可。")
    assert resp.formula_candidates == []
    assert resp.reasoning.condition_scope_required == ["cough_primary"]


# ----------------------------------------------------------------------
# 14: contract version mismatch fails closed
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"conditions": ["COUGH_PRIMARY", "COUGH_EXCLUSION_FEATURES_NONE_P19C_V1"], "contract_version": "X1D-P19C-V2"},
    {"conditions": ["COUGH_EXCLUSION_FEATURES_NONE_P19C_V1", "COUGH_PRIMARY"], "contract_version": CONTRACT},
    {"conditions": ["COUGH_PRIMARY"], "contract_version": CONTRACT},
    {"conditions": ["COUGH_PRIMARY", "COUGH_EXCLUSION_FEATURES_NONE_P19C_V1"]},          # contract-only cond, no contract
    {"conditions": ["COUGH_PRIMARY", "COUGH_EXCLUSION_FEATURES_NONE_P19C_V1"], "contract_version": CONTRACT, "x": 1},
])
def test_14_a_mismatched_contract_fails_closed(store, contract_on, bad):
    cough_slice(store, applicability=bad)
    resp = assemble(store, hyp("风热犯肺", 0.8), context=YES_NONE)
    assert resp.formula_candidates == []


def test_14_the_contract_reference_is_bound_into_the_content_hash():
    snap = {"name": "风热犯肺证", "aliases": []}
    d = lambda a: canonical.digest(canonical.entity_subject(external_id="x", entity_type="pattern",  # noqa: E731
                                                            snapshot={**snap, "applicability": a}))
    assert d(APPROVED) != d({**APPROVED, "contract_version": "X1D-P19C-V2"})
    assert d(APPROVED) != d({"conditions": APPROVED["conditions"]})


def test_the_relationship_locator_is_part_of_the_evidence_hash(store):
    full_slice(store)
    pid = store.match_reviewed_patterns("风寒束表")[0]["pattern_id"]
    eng = SafetyEngine(); eng.Session = store.Session
    other = ingest(store, ClinicalEntityType.FORMULA, "测试方", "xerbs:formula:synthetic:测试方", ingredients=["甘草"])
    verify(store, "CLINICAL_ENTITY", other)
    rel = eng.create_relationship(RelationshipCreateRequest(
        source_entity_id=pid, target_entity_id=other, relationship_type="PATTERN_FORMULA", source_id=SRC,
        locator="p.1468 §5.3.2; applicability contract X1D-P19C-V1", actor_id="corpus-author"))
    assert rel["evidence_hash"] == lifecycle.evidence_hash([{"source_id": SRC, "source_version": 1,
                                                             "locator": "p.1468 §5.3.2; applicability contract X1D-P19C-V1"}])
    assert rel["evidence_hash"] != lifecycle.evidence_hash([{"source_id": SRC, "source_version": 1}])


# ----------------------------------------------------------------------
# 15: the 风寒 path is unchanged
# ----------------------------------------------------------------------
@pytest.mark.parametrize("activate", [False, True])
def test_15_the_fenghan_path_is_unchanged(store, monkeypatch, activate):
    if activate:
        monkeypatch.setitem(A.APPLICABILITY_CONTRACTS, CONTRACT,
                            dict(A.APPLICABILITY_CONTRACTS[CONTRACT], active=True,
                                 approval_record=A.APPROVAL_RECORD_COMPLETE))
    cough_slice(store)
    resp = assemble(store, hyp("风寒束表", 0.8), context=YES_NONE, text=FLU_TEXT)
    assert [c.name for c in resp.formula_candidates] == ["麻黄汤加味"]
    assert "rule_status" not in resp.formula_candidates[0].governance


def test_active_flag_alone_is_not_enough_without_a_complete_record(store, monkeypatch):
    monkeypatch.setitem(A.APPLICABILITY_CONTRACTS, CONTRACT, dict(A.APPLICABILITY_CONTRACTS[CONTRACT], active=True))
    cough_slice(store)
    assert assemble(store, hyp("风热犯肺", 0.8), context=YES_NONE).formula_candidates == []


def test_the_harness_mirrors_the_contract_gate(store, contract_on):
    pid, fid, rid = cough_slice(store)
    snap = H.CorpusSnapshot(
        environment="staging", sources=(H.SourceSnap(SYN_SRC, "SOURCE_VERIFIED", 3, live_verification=True),),
        entities=(H.EntitySnap(pid, "pattern", "风热犯肺证", "SOURCE_VERIFIED", 3, source_ids=(SYN_SRC,),
                               live_verification=True, applicability=APPROVED),
                  H.EntitySnap(fid, "formula", "桑菊饮加减", "SOURCE_VERIFIED", 3, source_ids=(SYN_SRC,),
                               live_verification=True)),
        relationships=(H.RelationshipSnap(rid, pid, fid, "SOURCE_VERIFIED", 3, evidence_source_ids=(SYN_SRC,),
                                          live_verification=True),),
        core_formula_names=("桑菊饮加减",))
    for ctx, outcome in [(YES_NONE, H.FULL_PATH), ({"cough_primary": "YES"}, H.MATCHED_PRIMARY_SCOPE_MISSING),
                         ({"cough_primary": "YES", "cough_exclusion_features": "PRESENT"},
                          H.MATCHED_PRIMARY_SCOPE_REJECTED)]:
        r = H.classify_turn(snap, H.TurnInput(1, (H.Hypothesis("风热犯肺", 0.8),), condition_context=ctx))
        assert r.outcome == outcome
        prod = store.eligible_formula_candidates_for_patterns([pid], condition_context=ctx)
        assert [c["name"] for c in prod] == [c.name for c in r.candidates]
