"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P7: the human SOURCE-VERIFICATION lane.

Source verification says one thing: "this object faithfully represents its
cited authoritative source". It is not a clinical review, so it must never be
readable as one -- not by the clinical approval path, not by reviewed-only
search, not by clinical ranking -- and it may only feed retrieval where
source-bounded retrieval is explicitly enabled (Staging).
"""

import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.db.base import Base
from app.db.models import ClinicalEntity, GovernedObjectReviewEvent
from app.schemas.clinical_knowledge import SourceRef, SourceSubmitReviewRequest
from app.schemas.clinical_workflow import ClinicalEntityType, IngestionBatchRequest, IngestionItem
from app.schemas.intake import RecommendationRequest
from app.schemas.safety import RelationshipCreateRequest
from app.services.governance import lifecycle
from app.services.governance.attested_review import AttestedReviewError, AttestedReviewService
from app.services.governance.source_verification import (
    SOURCE_VERIFICATION_STATEMENT, SOURCE_VERIFICATION_STATEMENT_VERSION,
    SOURCE_VERIFIED_BY_ATTESTATION, SourceVerificationService)
from app.services.knowledge.persistent_clinical import PersistentClinicalStore
from app.services.knowledge.resolver import KnowledgeResolver
from app.services.reasoning.engine import DiagnosticReasoningEngine
from app.services.safety.engine import SafetyEngine
from tests.governed_fixtures import StubCoreAttestationClient, core_attestation_for, local_binding_for

SRC = "nhc-natcm-flu-dx-tx-2025"
VERIFIER = "xerbs-core:admin:4242"


@pytest.fixture
def env(monkeypatch):
    """Run as Staging unless a test says otherwise."""
    monkeypatch.setattr(get_settings(), "environment", "staging")
    return get_settings()


@pytest.fixture
def store(env):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return PersistentClinicalStore(sessionmaker(bind=engine, expire_on_commit=False))


def sv_record(store, object_type, object_id, **overrides):
    """What core would hold for a well-formed source verification of this object."""
    b = local_binding_for(object_type, object_id, store.Session)
    rec = {
        "attestation_id": "sv-SYNTHETIC-" + uuid.uuid4().hex[:12],
        "verification_kind": "SOURCE_VERIFICATION",
        "decision": "SOURCE_VERIFIED",
        "statement": SOURCE_VERIFICATION_STATEMENT,
        "statement_version": SOURCE_VERIFICATION_STATEMENT_VERSION,
        "target_system": "xerbs-ai-v2",
        "target_environment": get_settings().environment,
        "governance_object_type": object_type,
        "semantic_object_id": b.semantic_object_id,
        "object_version": b.object_version,
        "content_hash": b.content_hash,
        "evidence_hash": b.evidence_hash,
        "verifier_subject": VERIFIER,
        "author_subject": b.author_subject,
        "submitter_subject": b.submitter_subject,
        "last_material_editor_subject": b.last_material_editor_subject,
        "is_active": True,
        "is_revoked": False,
    }
    rec.update(overrides)
    return rec


def verify(store, object_type, object_id, **overrides):
    rec = sv_record(store, object_type, object_id, **overrides)
    svc = SourceVerificationService(client=StubCoreAttestationClient(rec), session_factory=store.Session)
    return svc.verify(object_type=object_type, object_id=object_id, verification_id=rec["attestation_id"])


def ingest(store, typ, name, external_id, sources=(SRC,), **payload):
    req = IngestionBatchRequest(submitted_by="corpus-author", source_label="p7", items=[IngestionItem(
        entity_type=typ, external_id=external_id, payload={"name": name, **payload},
        sources=[SourceRef(source_id=s, title="流行性感冒诊疗方案（2025年版）", source_type="OFFICIAL_CLINICAL_GUIDELINE") for s in sources])])
    eid = store.ingest(req).created_entity_ids[0]
    store.submit_for_review(typ, eid, "corpus-submitter")
    return eid


def submit_source(store, source_id=SRC):
    cur = store.get_source(source_id)
    store.submit_source_for_review(source_id, SourceSubmitReviewRequest(submitted_by="corpus-submitter", expected_version=cur.version))


def make_relationship(store, pid, fid, source_id=SRC):
    eng = SafetyEngine(); eng.Session = store.Session
    rel = eng.create_relationship(RelationshipCreateRequest(
        source_entity_id=pid, target_entity_id=fid, relationship_type="PATTERN_FORMULA",
        source_id=source_id, actor_id="corpus-author"))
    AttestedReviewService(session_factory=store.Session).submit(
        object_type="CLINICAL_RELATIONSHIP", object_id=rel["id"], submitted_by="corpus-submitter")
    return rel["id"]


def full_slice(store, *, verify_relationship=True):
    pid = ingest(store, ClinicalEntityType.PATTERN, "风寒束表", "xerbs:pattern:nhc-flu-2025:风寒束表",
                 indications=["恶寒", "无汗"])
    fid = ingest(store, ClinicalEntityType.FORMULA, "麻黄汤加味", "xerbs:formula:nhc-flu-2025:麻黄汤加味",
                 ingredients=["炙麻黄", "炒杏仁", "桂枝", "葛根", "羌活", "苏叶", "炙甘草"])
    submit_source(store)
    verify(store, "SOURCE", SRC)
    verify(store, "CLINICAL_ENTITY", pid)
    verify(store, "CLINICAL_ENTITY", fid)
    rid = make_relationship(store, pid, fid)
    if verify_relationship:
        verify(store, "CLINICAL_RELATIONSHIP", rid)
    return pid, fid, rid


# ----------------------------------------------------------------------
# 1, 7. Distinct from clinical APPROVED
# ----------------------------------------------------------------------

def test_source_verified_is_its_own_state_not_reviewed(store):
    pid, fid, rid = full_slice(store)
    with store.Session() as s:
        for eid in (pid, fid):
            e = s.get(ClinicalEntity, eid)
            assert e.review_status == "SOURCE_VERIFIED"
            assert e.governance_provenance == "SOURCE_VERIFIED"
            assert e.review_attestation_id is None
        approvals = s.scalars(select(GovernedObjectReviewEvent).where(
            GovernedObjectReviewEvent.action == "APPROVED_BY_ATTESTATION")).all()
        assert approvals == []
        assert s.scalars(select(GovernedObjectReviewEvent).where(
            GovernedObjectReviewEvent.action == SOURCE_VERIFIED_BY_ATTESTATION)).all()
    assert store.get_source(SRC).review_status == "SOURCE_VERIFIED"
    # clinical gates keep refusing it
    assert store.search("风寒束表", ["pattern"], reviewed_only=True) == []
    assert store.stats().ranking_eligible_patterns == 0
    assert store.stats().ranking_eligible_formulas == 0
    assert store.eligible_formula_candidates([], "发热恶寒") == []


def test_a_clinical_attestation_cannot_be_used_as_a_source_verification(store):
    pid = ingest(store, ClinicalEntityType.PATTERN, "风寒束表", "xerbs:pattern:x:风寒束表")
    clinical = core_attestation_for("CLINICAL_ENTITY", pid, session_factory=store.Session)
    svc = SourceVerificationService(client=StubCoreAttestationClient(clinical), session_factory=store.Session)
    with pytest.raises(AttestedReviewError) as e:
        svc.verify(object_type="CLINICAL_ENTITY", object_id=pid, verification_id=clinical["attestation_id"])
    assert e.value.code == "NOT_A_SOURCE_VERIFICATION"


def test_a_source_verification_cannot_masquerade_as_clinical_approval(store):
    pid = ingest(store, ClinicalEntityType.PATTERN, "风寒束表", "xerbs:pattern:x:风寒束表")
    rec = sv_record(store, "CLINICAL_ENTITY", pid, reviewer_subject=VERIFIER)
    clinical = AttestedReviewService(client=StubCoreAttestationClient(rec), session_factory=store.Session)
    with pytest.raises(AttestedReviewError) as e:
        clinical.approve(object_type="CLINICAL_ENTITY", object_id=pid, attestation_id=rec["attestation_id"])
    assert e.value.code == "DECISION_NOT_APPROVED"


def test_a_source_verified_object_cannot_then_be_clinically_approved_in_place(store):
    pid = ingest(store, ClinicalEntityType.PATTERN, "风寒束表", "xerbs:pattern:x:风寒束表")
    verify(store, "CLINICAL_ENTITY", pid)
    rec = core_attestation_for("CLINICAL_ENTITY", pid, session_factory=store.Session)
    clinical = AttestedReviewService(client=StubCoreAttestationClient(rec), session_factory=store.Session)
    with pytest.raises(AttestedReviewError) as e:
        clinical.approve(object_type="CLINICAL_ENTITY", object_id=pid, attestation_id=rec["attestation_id"])
    assert e.value.code == "OBJECT_NOT_IN_REVIEW"


@pytest.mark.parametrize("field,value,code", [
    ("decision", "APPROVED", "DECISION_NOT_VERIFIED"),
    ("verification_kind", "CLINICAL_REVIEW", "NOT_A_SOURCE_VERIFICATION"),
    ("statement", "I approve this treatment.", "STATEMENT_MISMATCH"),
    ("statement_version", "source-verification/0", "STATEMENT_MISMATCH"),
    ("is_revoked", True, "ATTESTATION_REVOKED"),
    ("is_active", False, "ATTESTATION_NOT_ACTIVE"),
    ("target_environment", "production", "TARGET_ENVIRONMENT_MISMATCH"),
    ("governance_object_type", "SOURCE", "OBJECT_TYPE_MISMATCH"),
])
def test_the_record_must_say_exactly_what_a_source_verification_says(store, field, value, code):
    pid = ingest(store, ClinicalEntityType.PATTERN, "风寒束表", "xerbs:pattern:x:风寒束表")
    with pytest.raises(AttestedReviewError) as e:
        verify(store, "CLINICAL_ENTITY", pid, **{field: value})
    assert e.value.code == code


# ----------------------------------------------------------------------
# 2, 3. Human only
# ----------------------------------------------------------------------

@pytest.mark.parametrize("subject", [
    "xerbs-ai-v2:principal:corpus-bot", "xerbs-core:service:xerbs-core",
    "xerbs-ai-v2:admin:1", "admin:4242", ""])
def test_only_an_authenticated_core_admin_can_verify(store, subject):
    pid = ingest(store, ClinicalEntityType.PATTERN, "风寒束表", "xerbs:pattern:x:风寒束表")
    with pytest.raises(AttestedReviewError) as e:
        verify(store, "CLINICAL_ENTITY", pid, verifier_subject=subject)
    assert e.value.code == "VERIFIER_SUBJECT_INVALID"


def test_safety_rules_cannot_be_source_verified(store):
    svc = SourceVerificationService(client=StubCoreAttestationClient(None), session_factory=store.Session)
    with pytest.raises(AttestedReviewError) as e:
        svc.verify(object_type="SAFETY_RULE", object_id="x", verification_id="sv-1")
    assert e.value.code == "UNKNOWN_OBJECT_TYPE"


# ----------------------------------------------------------------------
# 4. Separation of duties
# ----------------------------------------------------------------------

def test_the_submitter_cannot_verify_their_own_submission(store):
    req = IngestionBatchRequest(submitted_by="corpus-author", source_label="p7", items=[IngestionItem(
        entity_type=ClinicalEntityType.PATTERN, external_id="xerbs:pattern:x:风寒束表",
        payload={"name": "风寒束表"}, sources=[SourceRef(source_id=SRC, title="t", source_type="T")])])
    pid = store.ingest(req).created_entity_ids[0]
    store.submit_for_review(ClinicalEntityType.PATTERN, pid, VERIFIER)
    with pytest.raises(AttestedReviewError) as e:
        verify(store, "CLINICAL_ENTITY", pid)
    assert e.value.code == "SEPARATION_OF_DUTIES"


# ----------------------------------------------------------------------
# 5, 6. Stale version / hash / evidence
# ----------------------------------------------------------------------

@pytest.mark.parametrize("field,value,code", [
    ("object_version", 99, "VERSION_MISMATCH"),
    ("content_hash", "0" * 64, "CONTENT_HASH_MISMATCH"),
    ("evidence_hash", "f" * 64, "EVIDENCE_HASH_MISMATCH"),
    ("semantic_object_id", "xerbs:pattern:other", "SUBJECT_MISMATCH"),
])
def test_a_stale_or_mismatched_binding_is_refused(store, field, value, code):
    pid = ingest(store, ClinicalEntityType.PATTERN, "风寒束表", "xerbs:pattern:x:风寒束表")
    with pytest.raises(AttestedReviewError) as e:
        verify(store, "CLINICAL_ENTITY", pid, **{field: value})
    assert e.value.code == code
    with store.Session() as s:
        assert s.get(ClinicalEntity, pid).review_status == "IN_REVIEW"


def test_replaying_the_same_verification_is_idempotent(store):
    pid = ingest(store, ClinicalEntityType.PATTERN, "风寒束表", "xerbs:pattern:x:风寒束表")
    rec = sv_record(store, "CLINICAL_ENTITY", pid)
    svc = SourceVerificationService(client=StubCoreAttestationClient(rec), session_factory=store.Session)
    first = svc.verify(object_type="CLINICAL_ENTITY", object_id=pid, verification_id=rec["attestation_id"])
    again = svc.verify(object_type="CLINICAL_ENTITY", object_id=pid, verification_id=rec["attestation_id"])
    assert first["idempotent_replay"] is False and again["idempotent_replay"] is True


# ----------------------------------------------------------------------
# 9-12. Source-bounded retrieval
# ----------------------------------------------------------------------

def test_the_full_source_verified_slice_is_retrievable_on_staging(store):
    pid, fid, _ = full_slice(store)
    found = store.match_reviewed_patterns("风寒束表，肺气失宣")
    assert [x["pattern_id"] for x in found] == [pid]
    assert found[0]["governance_basis"] == "SOURCE_VERIFIED"
    cands = store.eligible_formula_candidates_for_patterns([pid])
    assert [c["name"] for c in cands] == ["麻黄汤加味"]
    gov = cands[0]["governance"]
    assert gov["basis"] == "SOURCE_VERIFIED" and gov["clinical_review"] == "NOT_PERFORMED"
    assert gov["source_ids"] == [SRC]
    assert gov["formula_external_id"] == "xerbs:formula:nhc-flu-2025:麻黄汤加味"
    assert "SOURCE_VERIFIED_NOT_CLINICALLY_REVIEWED" in cands[0]["safety_flags"]


def test_engine_says_the_pattern_is_source_verified_not_clinically_reviewed(store):
    full_slice(store)
    engine = DiagnosticReasoningEngine(store, KnowledgeResolver(store, record_gaps=False))
    r = engine.analyze(RecommendationRequest(text_input="恶寒发热两天，无汗，身痛头痛，鼻流清涕。"),
                       [{"name": "风寒束表，肺气失宣", "confidence": 0.8, "reasoning": "r"}])
    assert r.pattern_assessments[0].corpus_match is True
    assert "PATTERN_SOURCE_VERIFIED_NOT_CLINICALLY_REVIEWED" in r.uncertainty_flags


@pytest.mark.parametrize("environment", ["production", "development", "", "Staging-2"])
def test_source_verified_content_is_invisible_outside_staging(store, monkeypatch, environment):
    pid, _, _ = full_slice(store)
    monkeypatch.setattr(get_settings(), "environment", environment)
    assert store.match_reviewed_patterns("风寒束表") == []
    assert store.eligible_formula_candidates_for_patterns([pid]) == []


def test_the_production_allowlist_is_a_code_constant():
    assert lifecycle.SOURCE_BOUNDED_RETRIEVAL_ENVIRONMENTS == frozenset({"staging"})
    assert not lifecycle.source_bounded_retrieval_enabled("production")


def test_unverified_content_is_not_retrievable(store):
    pid = ingest(store, ClinicalEntityType.PATTERN, "风寒束表", "xerbs:pattern:x:风寒束表")
    assert store.match_reviewed_patterns("风寒束表") == []          # IN_REVIEW pattern
    submit_source(store)
    verify(store, "CLINICAL_ENTITY", pid)
    assert store.match_reviewed_patterns("风寒束表") == []          # source not yet verified
    verify(store, "SOURCE", SRC)
    assert store.match_reviewed_patterns("风寒束表")                 # now both verified


def test_unsupported_aliases_do_not_match(store):
    full_slice(store)
    for name in ("风寒表证", "太阳伤寒表实证", "风寒"):
        assert store.match_reviewed_patterns(name) == [], name


def test_the_relationship_must_itself_be_verified(store):
    pid, _, _ = full_slice(store, verify_relationship=False)
    assert store.match_reviewed_patterns("风寒束表")       # pattern is fine
    assert store.eligible_formula_candidates_for_patterns([pid]) == []


def test_the_relationship_source_must_be_the_endpoints_source(store):
    other = "some-other-verified-source"
    pid = ingest(store, ClinicalEntityType.PATTERN, "风寒束表", "xerbs:pattern:x:风寒束表")
    fid = ingest(store, ClinicalEntityType.FORMULA, "麻黄汤加味", "xerbs:formula:x:麻黄汤加味",
                 ingredients=["炙麻黄"])
    # a second, unrelated source verified too
    ingest(store, ClinicalEntityType.HERB, "炙麻黄", "xerbs:herb:x:炙麻黄", sources=(other,))
    for src in (SRC, other):
        submit_source(store, src); verify(store, "SOURCE", src)
    verify(store, "CLINICAL_ENTITY", pid); verify(store, "CLINICAL_ENTITY", fid)
    rid = make_relationship(store, pid, fid, source_id=other)
    verify(store, "CLINICAL_RELATIONSHIP", rid)
    assert store.eligible_formula_candidates_for_patterns([pid]) == []
