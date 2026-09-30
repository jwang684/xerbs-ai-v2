"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P13: the deterministic governed coverage harness.

The harness is only worth anything if (a) its baseline IS production -- same
matcher, same eligibility, checked against the real store -- and (b) it cannot
change anything: no session, no write, no model, no alias without evidence.
"""

import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from app.services.coverage import harness as H
from app.services.coverage import measure as M
from app.services.coverage.whatif import CONSUMER_PAIRS, PAIRS, SEVERE_PAIRS, build_whatif, verify_pairs_against_source
from app.services.knowledge.pattern_match import matches_reviewed_name
# Real governed state, built through the real ingest -> submit -> source-verify flow.
from tests.test_x1d_p7_source_verification import env, full_slice, store  # noqa: F401

SRC = "nhc-natcm-flu-dx-tx-2025"
MILD_SCOPE = "流行性感冒 轻症辨证治疗方案（2025年版）"
CORE_ROWS = ("银翘散", "麻黄汤", "藿香正气散", "清肺排毒汤", "麻黄汤加味")


def staging_like(**changes):
    """The accepted Staging corpus shape (P11/P12), as a snapshot."""
    snap = H.CorpusSnapshot(
        environment="staging",
        sources=(H.SourceSnap(SRC, "SOURCE_VERIFIED", 3, live_verification=True),
                 H.SourceSnap("nhsa-2024-YPSN202400007", "REVIEWED", 3)),
        entities=(
            H.EntitySnap("pat-fh", "pattern", "风寒束表", "SOURCE_VERIFIED", 3, source_ids=(SRC,),
                         live_verification=True, source_scope=MILD_SCOPE),
            H.EntitySnap("frm-mh", "formula", "麻黄汤加味", "SOURCE_VERIFIED", 3, source_ids=(SRC,),
                         live_verification=True, source_scope=MILD_SCOPE),
            H.EntitySnap("frm-qf", "formula", "清肺排毒汤", "REVIEWED", 3, source_ids=("nhsa-2024-YPSN202400007",),
                         clinical_ranking_eligible=True),
            H.EntitySnap("corpus-legacy-1", "formula", "银翘散", "DRAFT", 1),
        ),
        relationships=(H.RelationshipSnap("rel-fh", "pat-fh", "frm-mh", "SOURCE_VERIFIED", 3,
                                          evidence_source_ids=(SRC,), live_verification=True),),
        core_formula_names=CORE_ROWS)
    return H.CorpusSnapshot(**{**snap.__dict__, **changes})


def turn(*names, origin="TEST"):
    return H.TurnInput(ordinal=1, hypotheses=tuple(H.Hypothesis(n, 0.7) for n in names), origin=origin)


def classify(snap, *names, matcher=H.PRODUCTION):
    return H.classify_turn(snap, turn(*names), matcher)


# Verbatim excerpts of 《流行性感冒诊疗方案（2025年版）》 section (五), whitespace as the PDF text has it.
SOURCE_EXCERPT = (
    "1.轻症辨证治疗方案 （1）风热犯卫 症见：…… 基本方药：银翘散加减。 （2）风寒束表 症见：…… "
    "基本方药：麻黄汤加味。 （3）表寒里热 …… 基本方药：大青龙汤加减。 （4）热毒袭肺 …… "
    "基本方药：麻杏石甘汤加减。 2.重症辨证治疗方案 （1）毒热壅盛 …… 基本方药：宣白承气汤加减。 "
    "（2）毒热内陷，内闭外脱 …… 基本方药：参附汤加减。 3.恢复期辨证治疗方案 气阴两虚，正气未复 …… "
    "基本方药：沙参麦门冬汤加减。")


# ----------------------------------------------------------------------
# 1. 风寒束表 reaches the 麻黄汤加味 path
# ----------------------------------------------------------------------
def test_fenghan_shubiao_reaches_mahuangtang_jiawei_and_core():
    r = classify(staging_like(), "风寒束表")
    assert r.outcome == H.FULL_PATH
    assert r.primary_match and r.any_match
    assert r.hypothesis_explanations[0][1] == H.EXACT_CANONICAL
    assert [c.name for c in r.candidates] == ["麻黄汤加味"]
    c = r.candidates[0]
    assert (c.basis, c.source_ids, c.source_scope) == ("SOURCE_VERIFIED", (SRC,), MILD_SCOPE)
    assert r.core_formula == "麻黄汤加味"

    compound = classify(staging_like(), "风寒束表，郁而化热")
    assert compound.outcome == H.FULL_PATH
    assert compound.hypothesis_explanations[0][1] == H.COMPOUND_PIECE


def test_a_secondary_hypothesis_carried_the_turn_only_under_the_historical_pooled_rule():
    """X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P15: pooling is now a historical simulation only."""
    t = H.TurnInput(ordinal=1, hypotheses=(H.Hypothesis("痰热壅肺", 0.6), H.Hypothesis("风寒束表，肺气失宣", 0.3)))
    old = H.classify_turn(staging_like(), t, policy=H.POOLED_ALL_HYPOTHESES_HISTORICAL)
    assert not old.primary_match and old.any_match and old.outcome == H.FULL_PATH
    now = H.classify_turn(staging_like(), t)
    assert now.any_match and not now.primary_match
    assert (now.outcome, now.detail) == (H.NO_PATTERN_MATCH, "SECONDARY_GOVERNED_MATCH_NOT_USED")


# ----------------------------------------------------------------------
# 2. normalization never breaks an exact (or any production) match
# ----------------------------------------------------------------------
def test_neutral_normalization_is_a_strict_superset_of_production():
    neutral = H.Matcher(name="n", neutral_normalization=True)
    s1, _ = build_whatif(staging_like(), CONSUMER_PAIRS, label="S1")
    for name, _why in M.CURATED:
        prod = H.match_patterns(s1, name)
        wide = H.match_patterns(s1, name, neutral)
        assert [m.pattern_id for m in prod] == [m.pattern_id for m in wide][:len(prod)]
        if prod:
            assert prod[0].explanation == wide[0].explanation
    assert H.match_patterns(s1, "风寒束表", neutral)[0].explanation == H.EXACT_CANONICAL
    # typography-only: these now match ...
    assert H.match_patterns(s1, "风寒束表—肺气失宣", neutral)[0].explanation == H.NEUTRAL_NORMALIZATION
    assert H.match_patterns(s1, "风寒 束表", neutral)[0].explanation == H.NEUTRAL_NORMALIZATION
    # ... and no clinical fragment does.
    for bad in ("风寒", "风寒束肺", "外感风寒束表", "风热犯肺", "气阴两虚"):
        assert H.match_patterns(s1, bad, neutral) == [], bad


def test_the_production_explanation_agrees_with_the_production_matcher():
    snap = staging_like()
    pat = snap.entity("pat-fh")
    for name, _why in M.CURATED:
        assert (H.PRODUCTION.explain(name, pat) != H.NO_MATCH) == matches_reviewed_name(name, [pat.name]), name


# ----------------------------------------------------------------------
# 3. unknown pattern -> NO_PATTERN_MATCH (and a gated one -> PATTERN_INELIGIBLE)
# ----------------------------------------------------------------------
def test_unknown_pattern_is_no_pattern_match():
    r = classify(staging_like(), "肝阳上亢")
    assert r.outcome == H.NO_PATTERN_MATCH and not r.any_match and r.candidates == ()


def test_a_named_but_ungoverned_pattern_is_pattern_ineligible():
    snap = staging_like()
    unverified = snap.with_additions(entities=(
        H.EntitySnap("pat-draft", "pattern", "风热犯卫", "DRAFT", 1, source_ids=(SRC,)),))
    r = classify(unverified, "风热犯卫")
    assert (r.outcome, r.detail) == (H.PATTERN_INELIGIBLE, "PATTERN_STATUS_DRAFT")
    # a withdrawn verification is not a verification
    revoked = H.CorpusSnapshot(**{**snap.__dict__, "entities": tuple(
        H.EntitySnap(**{**e.__dict__, "live_verification": False}) if e.entity_id == "pat-fh" else e
        for e in snap.entities)})
    assert classify(revoked, "风寒束表").outcome == H.PATTERN_INELIGIBLE
    # and outside Staging source-bounded retrieval does not exist
    prod_env = staging_like(environment="production")
    r = classify(prod_env, "风寒束表")
    assert r.outcome == H.PATTERN_INELIGIBLE and "DISABLED" in r.detail


# ----------------------------------------------------------------------
# 4. matched pattern, no relationship -> NO_RELATIONSHIP
# ----------------------------------------------------------------------
def test_matched_pattern_without_relationship_is_no_relationship():
    snap = staging_like(relationships=())
    r = classify(snap, "风寒束表")
    assert r.any_match and r.outcome == H.NO_RELATIONSHIP


# ----------------------------------------------------------------------
# 5. relationship to an unavailable formula -> the right failure state
# ----------------------------------------------------------------------
def test_relationship_to_an_unavailable_formula_is_formula_ineligible():
    snap = staging_like().with_additions(relationships=(
        H.RelationshipSnap("rel-legacy", "pat-fh", "corpus-legacy-1", "SOURCE_VERIFIED", 3,
                           evidence_source_ids=(SRC,), live_verification=True),))
    only_legacy = H.CorpusSnapshot(**{**snap.__dict__, "relationships": snap.relationships[1:]})
    r = classify(only_legacy, "风寒束表")
    assert r.outcome == H.FORMULA_INELIGIBLE and "DRAFT" in r.detail
    # the relationship itself unverified -> RELATIONSHIP_INELIGIBLE
    unverified_rel = staging_like(relationships=(H.RelationshipSnap(
        "rel-fh", "pat-fh", "frm-mh", "IN_REVIEW", 2, evidence_source_ids=(SRC,)),))
    assert classify(unverified_rel, "风寒束表").outcome == H.RELATIONSHIP_INELIGIBLE
    # a governed formula Core has no canonical row for -> CORE_CANONICAL_MISSING
    no_core = staging_like(core_formula_names=("清肺排毒汤",))
    assert classify(no_core, "风寒束表").outcome == H.CORE_CANONICAL_MISSING
    # a duplicated Core name is refused, exactly like formula_resolver_service
    dup = staging_like(core_formula_names=CORE_ROWS + ("麻黄汤加味",))
    assert classify(dup, "风寒束表").outcome == H.CORE_CANONICAL_MISSING


# ----------------------------------------------------------------------
# 6. simulation never writes to any database
# ----------------------------------------------------------------------
def test_simulation_opens_no_session_and_leaves_the_snapshot_unchanged(monkeypatch):
    import app.db.session as session_mod

    def refuse(*a, **k):
        raise AssertionError("the coverage harness must never open a database session")
    monkeypatch.setattr(session_mod, "get_session_factory", refuse)
    monkeypatch.setattr("sqlalchemy.create_engine", refuse)

    base = staging_like()
    before = copy.deepcopy(base)
    report = M.build_report(base, [turn("风寒束表"), turn("风热犯卫"), turn("毒热壅盛")],
                            source_text=SOURCE_EXCERPT)
    assert base == before and base.simulated_object_count == 0
    assert report["simulated_objects"]["S1"] > 0
    with pytest.raises(Exception):
        base.entities[0].name = "x"   # frozen


def test_the_coverage_package_has_no_database_or_model_imports():
    root = Path(__file__).resolve().parents[1]
    for f in list((root / "app/services/coverage").glob("*.py")) + [root / "scripts/governed_coverage_harness.py"]:
        text = f.read_text(encoding="utf-8")
        for banned in ("app.db", "sqlalchemy", "get_session_factory", "services.llm", "anthropic", "openai"):
            assert banned not in text, (f.name, banned)


# ----------------------------------------------------------------------
# 7. an alias is invalid without source evidence
# ----------------------------------------------------------------------
def test_an_alias_without_source_evidence_is_refused_and_never_applied():
    s1, _ = build_whatif(staging_like(), CONSUMER_PAIRS, label="S1")
    texts = {SRC: SOURCE_EXCERPT}
    claim = H.AliasProposal(pattern_name="风热犯卫", alias="风热犯肺")
    matcher, verdicts = H.alias_matcher(s1, [claim], texts)
    assert verdicts == [{"pattern_name": "风热犯卫", "alias": "风热犯肺",
                         "verdict": H.CLINICAL_EQUIVALENCE_EVIDENCE_REQUIRED, "reason": "NO_SOURCE_EVIDENCE"}]
    assert classify(s1, "风热犯肺", matcher=matcher).outcome == H.NO_PATTERN_MATCH

    invented = H.AliasProposal("风热犯卫", "风热犯肺", SRC, "风热犯卫又称风热犯肺")
    assert H.evaluate_alias(invented, texts, s1) == (H.CLINICAL_EQUIVALENCE_EVIDENCE_REQUIRED,
                                                     "QUOTE_NOT_FOUND_IN_SOURCE_TEXT")
    half = H.AliasProposal("风热犯卫", "风热犯肺", SRC, "（1）风热犯卫")
    assert H.evaluate_alias(half, texts, s1)[1] == "QUOTE_DOES_NOT_NAME_BOTH"

    # Only a source that itself names both, verbatim, makes an alias -- and then only in simulation.
    fixture_text = {SRC: SOURCE_EXCERPT + " 风热犯卫（亦称风热犯肺）"}
    backed = H.AliasProposal("风热犯卫", "风热犯肺", SRC, "风热犯卫（亦称风热犯肺）")
    matcher, verdicts = H.alias_matcher(s1, [backed], fixture_text)
    assert verdicts[0]["verdict"] == H.SOURCE_BACKED
    r = classify(s1, "风热犯肺", matcher=matcher)
    assert r.hypothesis_explanations[0][1] == H.SOURCE_BACKED_ALIAS
    assert classify(s1, "风热犯肺").outcome == H.NO_PATTERN_MATCH   # production untouched


def test_derived_alias_proposals_are_all_claims_in_the_report():
    report = M.build_report(staging_like(), [turn("风热犯肺"), turn("风寒束表")], source_text=SOURCE_EXCERPT)
    verdicts = report["alias_verdicts"]["on_S1"]
    assert {"pattern_name": "风热犯卫", "alias": "风热犯肺", "verdict": H.CLINICAL_EQUIVALENCE_EVIDENCE_REQUIRED,
            "reason": "NO_SOURCE_EVIDENCE"} in verdicts
    assert all(v["verdict"] != H.SOURCE_BACKED for v in verdicts)
    s3 = report["strategies"]["S3_SOURCE_BACKED_ALIASES_ON_S1"]["metrics"]
    assert s3 == report["strategies"]["S1_CANONICAL_EXPANSION"]["metrics"]


# ----------------------------------------------------------------------
# 8. no PII in the output
# ----------------------------------------------------------------------
def test_the_report_carries_no_pii():
    rows = [
        {"id": 987654, "user_id": "8d3c2f0e-user-uuid", "consumer_input": {"symptoms": ["张三 咳嗽 电话13800138000"]},
         "turn_count": 2, "recommendation_state": "NEEDS_MORE_INFORMATION",
         "ai_response_summary": {
             "summary": "患者张三，alice@example.com", "consumer_reasoning": {"x": "张三"},
             "pattern_hypotheses": [
                 {"name": "风寒束表", "confidence": 0.8, "reasoning": "张三自述恶寒，电话13800138000"},
                 {"name": "alice@example.com", "confidence": 0.1, "reasoning": "x"},
                 {"name": "风热犯肺 13800138000", "confidence": 0.1}],
             "uncertainty_flags": ["MISSING_CLINICAL_INFORMATION", "张三"],
             "formula_candidates": [{"formula_id": "frm-mh", "rationale": "张三"}],
             "answered_fields": [{"field": "sleep", "answer": "张三不清楚"}]}},
        {"id": 306, "user_id": "u2", "ai_response_summary": {"pattern_hypotheses": []}},
    ]
    turns, redacted = M.sanitize_trace_rows(rows)
    assert len(turns) == 1 and redacted == 2
    t = turns[0]
    assert [h.name for h in t.hypotheses] == ["风寒束表", M.REDACTED_NAME, M.REDACTED_NAME]
    assert t.uncertainty_flags == ("MISSING_CLINICAL_INFORMATION",)
    assert t.recorded_formula_ids == ("frm-mh",) and t.turn_count == 2
    assert t.ordinal == 1   # a position, not the row id
    report = json.dumps(M.build_report(staging_like(), turns, redacted_names=redacted), ensure_ascii=False)
    for leak in ("张三", "13800138000", "alice@example.com", "8d3c2f0e", "user_id", "consumer_input",
                 "reasoning\"", "987654", "不清楚"):
        assert leak not in report, leak
    assert M.COVERAGE_LABEL in report


# ----------------------------------------------------------------------
# 9. the harness baseline agrees with the production store
# ----------------------------------------------------------------------
PARITY_NAMES = ("风寒束表", "风寒束表证", "风寒束表，郁而化热", "风寒束表（肺气失宣）", "风寒", "风寒束肺",
                "外感风寒束表", "风热犯肺", "肝阳上亢")


def _snapshot_of_full_slice(pid, fid, rid, *, relationship_verified=True):
    return H.CorpusSnapshot(
        environment="staging",
        sources=(H.SourceSnap(SRC, "SOURCE_VERIFIED", 3, live_verification=True),),
        entities=(H.EntitySnap(pid, "pattern", "风寒束表", "SOURCE_VERIFIED", 3, source_ids=(SRC,),
                               live_verification=True),
                  H.EntitySnap(fid, "formula", "麻黄汤加味", "SOURCE_VERIFIED", 3, source_ids=(SRC,),
                               live_verification=True)),
        relationships=(H.RelationshipSnap(rid, pid, fid, "SOURCE_VERIFIED" if relationship_verified else "IN_REVIEW",
                                          3 if relationship_verified else 2, evidence_source_ids=(SRC,),
                                          live_verification=relationship_verified),),
        core_formula_names=CORE_ROWS)


def test_harness_agrees_with_the_production_store_for_fenghan_shubiao(store):
    pid, fid, rid = full_slice(store)
    snap = _snapshot_of_full_slice(pid, fid, rid)
    for name in PARITY_NAMES:
        prod = store.match_reviewed_patterns(name)
        mine = H.match_patterns(snap, name)
        assert [m["pattern_id"] for m in prod] == [m.pattern_id for m in mine], name
        assert [m["governance_basis"] for m in prod] == [m.governance_basis for m in mine], name
        if prod:
            prod_c = store.eligible_formula_candidates_for_patterns([prod[0]["pattern_id"]])
            mine_c = H.formula_candidates(snap, [mine[0].pattern_id])
            assert [(c["formula_id"], c["name"], c["governance"]["basis"], tuple(c["governance"]["source_ids"]))
                    for c in prod_c] == [(c.formula_id, c.name, c.basis, c.source_ids) for c in mine_c], name
    assert H.classify_turn(snap, turn("风寒束表")).outcome == H.FULL_PATH


def test_harness_agrees_with_the_store_when_the_relationship_is_unverified(store):
    pid, fid, rid = full_slice(store, verify_relationship=False)
    snap = _snapshot_of_full_slice(pid, fid, rid, relationship_verified=False)
    assert store.eligible_formula_candidates_for_patterns([pid]) == []
    assert H.formula_candidates(snap, [pid]) == []
    assert H.classify_turn(snap, turn("风寒束表")).outcome == H.RELATIONSHIP_INELIGIBLE


# ----------------------------------------------------------------------
# 10. severe pairs are reported apart, never as consumer coverage
# ----------------------------------------------------------------------
def test_severe_pairs_are_separated_from_consumer_coverage():
    assert {p.pattern_name for p in SEVERE_PAIRS} == {"毒热壅盛", "毒热内陷，内闭外脱"}
    assert not {p.pattern_name for p in SEVERE_PAIRS} & {p.pattern_name for p in CONSUMER_PAIRS}
    turns = [turn("毒热壅盛"), turn("风热犯卫"), turn("肝阳上亢")]
    report = M.build_report(staging_like(), turns, source_text=SOURCE_EXCERPT)
    s1 = report["strategies"]["S1_CANONICAL_EXPANSION"]["metrics"]
    assert s1["FULL_GOVERNED_PATH"]["count"] == 1        # 风热犯卫 only
    severe = {s["pair"]: s for s in report["severe_pairs"]}
    assert severe["S1"]["trace_turns_reaching_it"] == 1
    assert severe["S1"]["reported_as"] == "SAFETY SIGNAL, NOT CONSUMER COVERAGE"

    s1_sev, _ = build_whatif(staging_like(), PAIRS, label="S1+SEVERE")
    results = M.run(s1_sev, turns)
    assert results[0].governed and results[0].severe_only
    m = M.metrics(results)
    assert m["FULL_GOVERNED_PATH"]["count"] == 1
    assert m["SEVERE_ONLY_GOVERNED_PATH_NOT_CONSUMER_COVERAGE"] == 1


# ----------------------------------------------------------------------
# What-if corpus and the read-only loader
# ----------------------------------------------------------------------
def test_every_whatif_pair_is_located_verbatim_in_the_source():
    assert all(verify_pairs_against_source(PAIRS, SOURCE_EXCERPT).values())
    assert not verify_pairs_against_source(PAIRS, SOURCE_EXCERPT.replace("风热犯卫", "风热犯肺"))["P2"]


def test_whatif_reuses_live_objects_and_labels_every_invention():
    s1, prov = build_whatif(staging_like(), CONSUMER_PAIRS, label="S1")
    p1 = next(r for r in prov if r["pair"] == "P1")
    assert (p1["pattern_object"], p1["formula_object"], p1["relationship_object"]) == (
        "EXISTING:pat-fh", "EXISTING:frm-mh", "EXISTING:rel-fh")
    assert all(r["pattern_object"] == "SIMULATED" for r in prov if r["pair"] != "P1")
    r = classify(s1, "风热犯卫")
    assert r.outcome == H.CORE_CANONICAL_MISSING and r.candidates[0].simulated
    ceiling, _ = build_whatif(staging_like(), CONSUMER_PAIRS, label="C", add_core_rows=True)
    assert classify(ceiling, "风热犯卫").outcome == H.FULL_PATH


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/governed_coverage_harness.py"
    spec = importlib.util.spec_from_file_location("governed_coverage_harness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_staging(revoked=False):
    sv = [{"governance_object_type": t, "semantic_object_id": s, "object_version": 2, "decision": "SOURCE_VERIFIED",
           "target_system": "xerbs-ai-v2", "target_environment": "staging",
           "revoked_at": "2026-09-30T00:00:00Z" if revoked and t == "CLINICAL_ENTITY" else None}
          for t, s in (("SOURCE", SRC), ("CLINICAL_ENTITY", "xerbs:pattern:风寒束表"),
                       ("CLINICAL_ENTITY", "xerbs:formula:麻黄汤加味"), ("CLINICAL_RELATIONSHIP", "rel:x"))]
    routes = {
        "/api/v1/knowledge/clinical/sources": {"results": [{"source_id": SRC, "review_status": "SOURCE_VERIFIED",
                                                            "version": 3}]},
        "/api/v1/knowledge/clinical/sources/%s/entities" % SRC: {"results": [
            {"entity_id": "pat-fh", "entity_type": "pattern"}, {"entity_id": "frm-mh", "entity_type": "formula"}]},
        "/api/v1/governance/pending-review": {"items": []},
        "/api/v1/knowledge/clinical/entities/pattern/pat-fh": {
            "entity_type": "pattern", "name": "风寒束表", "version": 3, "review_status": "SOURCE_VERIFIED",
            "clinical_ranking_eligible": False, "sources": [{"source_id": SRC}],
            "content": {"aliases": [], "source_scope": MILD_SCOPE}, "retired_at": None},
        "/api/v1/knowledge/clinical/entities/formula/frm-mh": {
            "entity_type": "formula", "name": "麻黄汤加味", "version": 3, "review_status": "SOURCE_VERIFIED",
            "clinical_ranking_eligible": False, "sources": [{"source_id": SRC}], "content": {}, "retired_at": None},
        "/api/v1/governance/pending-review/CLINICAL_ENTITY/pat-fh": {"semantic_object_id": "xerbs:pattern:风寒束表"},
        "/api/v1/governance/pending-review/CLINICAL_ENTITY/frm-mh": {"semantic_object_id": "xerbs:formula:麻黄汤加味"},
        "/api/v1/safety/relationships": {"results": [{
            "id": "rel-fh", "source_entity_id": "pat-fh", "target_entity_id": "frm-mh",
            "relationship_type": "PATTERN_FORMULA", "review_status": "SOURCE_VERIFIED", "source_id": SRC}]},
        "/api/v1/governance/pending-review/CLINICAL_RELATIONSHIP/rel-fh": {
            "object_version": 3, "semantic_object_id": "rel:x", "evidence": [{"source_id": SRC}]},
        "/rest/v1/source_verification_attestations": sv,
        "/rest/v1/herbal_formulas": [{"id": i, "name": n} for i, n in enumerate(CORE_ROWS, 1)],
        "/rest/v1/product_formula_mappings": [{"formula_id": 4}],
        "/rest/v1/integration_recommendation_trace": [
            {"turn_count": 1, "recommendation_state": "NEEDS_MORE_INFORMATION",
             "pattern_hypotheses": [{"name": "风寒束表，肺气失宣", "confidence": 0.8, "reasoning": "患者自述"}],
             "uncertainty_flags": [], "formula_candidates": []}],
    }
    methods = []

    def handler(request):
        methods.append(request.method)
        assert request.method == "GET"
        return httpx.Response(200, json=routes[request.url.path])
    return httpx.MockTransport(handler), methods


@pytest.mark.parametrize("revoked", [False, True])
def test_the_loader_reads_by_get_only_and_rebuilds_live_verification(revoked):
    mod = _load_script()
    transport, methods = _fake_staging(revoked)
    ai = mod.GetOnlyClient("https://xerbs-ai-v2-staging.example", {}, host_must_contain="staging",
                           transport=transport)
    core = mod.GetOnlyClient("https://project.supabase.co", {}, host_must_contain="supabase.co",
                             transport=transport)
    snap = mod.load_snapshot(ai, core, "staging")
    turns, redacted = mod.load_trace_turns(core)
    assert set(methods) == {"GET"} and redacted == 0
    r = H.classify_turn(snap, turns[0])
    assert r.outcome == (H.PATTERN_INELIGIBLE if revoked else H.FULL_PATH)
    assert snap.core_mapped_formula_names == ("清肺排毒汤",)


def test_the_loader_refuses_writes_and_non_staging_hosts():
    mod = _load_script()
    with pytest.raises(mod.ReadOnlyViolation):
        mod.GetOnlyClient("https://xerbs-ai-v2-production.example", {}, host_must_contain="staging")
    client = mod.GetOnlyClient("https://x-staging.example", {}, host_must_contain="staging")
    for verb in ("post", "put", "patch", "delete", "request", "stream", "send"):
        with pytest.raises(mod.ReadOnlyViolation):
            getattr(client, verb)


def test_the_influenza_scope_signal_survives_a_formula_without_its_own_scope():
    """Staging shape: the scope sits on the pattern, not the formula; the source is the flu guideline."""
    snap = staging_like()
    snap = H.CorpusSnapshot(**{**snap.__dict__, "entities": tuple(
        H.EntitySnap(**{**e.__dict__, "source_scope": None}) if e.entity_id == "frm-mh" else e
        for e in snap.entities)})
    r = classify(snap, "风寒束表")
    assert r.candidates[0].source_scope == MILD_SCOPE
    sig = M.safety_signals(snap, [r])
    assert sig["governed_turns_resting_on_influenza_scoped_source"] == 1
    assert sig["scope_gate_in_matcher_or_retrieval"] is False
    assert sig["governed_formulas"][0]["clinical_review"] == "NOT_PERFORMED"
