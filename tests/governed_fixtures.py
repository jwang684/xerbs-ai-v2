"""Test-only helpers for governed objects (X1D-AIV2-GOV2-C1).

Why this exists
---------------
Before GOV2-C1, creating a relationship or a safety rule made it REVIEWED and
immediately effective, so a test that wanted to exercise *downstream* behaviour
-- pattern->formula retrieval, safety screening, blocking -- could just create
one. That shortcut was the defect: authority came from a literal in the source.

Creation now yields DRAFT, and the only route to REVIEWED is a verified human
clinical review attestation from xerbs-core, which this service cannot produce
and AI-GOV2 has not yet wired up. Tests that are really about retrieval or
screening still need an object in the reviewed state, so they use the helper
below.

What this is NOT
----------------
It is not a bypass, and it adds no capability to the application. It writes
rows directly with a test session, exactly as a test fixture may; no
application code path can reach it, and importing it from ``app/`` would be a
mistake worth failing a review over. The attestation ids it writes are
unmistakably synthetic.

Think of it as standing in for the future AI-GOV2 integration: "assume core
has already attested this object, now assert what retrieval does".
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db.models import ClinicalRelationship, GovernedObjectSource, SafetyRule
from app.db.session import get_session_factory
from app.services.governance import lifecycle

#: Prefix that makes every fixture-written attestation identifiable at a glance
#: and greppable in any database it ever reaches.
SYNTHETIC_ATTESTATION_PREFIX = "att-SYNTHETIC-TEST-"


def _synthetic_attestation_id() -> str:
    return SYNTHETIC_ATTESTATION_PREFIX + uuid.uuid4().hex[:12]


def simulate_core_attestation_for_test(object_type: str, object_id: str,
                                       *, ensure_reviewed_evidence: bool = True):
    """Put one governed object into the state a verified core attestation
    would leave it in.

    ``ensure_reviewed_evidence`` also promotes the object's attached Sources to
    REVIEWED, because eligibility requires at least one -- the same bar
    clinical entities have had since Phase 12C-2D3.
    """
    Session = get_session_factory()
    with Session.begin() as s:
        model = {"CLINICAL_RELATIONSHIP": ClinicalRelationship,
                 "SAFETY_RULE": SafetyRule}[object_type]
        row = s.get(model, object_id)
        assert row is not None, "%s %s does not exist" % (object_type, object_id)

        if ensure_reviewed_evidence:
            from app.db.models import SourceRegistry
            source_ids = list(s.scalars(select(GovernedObjectSource.source_id).where(
                GovernedObjectSource.object_type == object_type,
                GovernedObjectSource.object_id == object_id)).all())
            for sid in source_ids:
                src = s.get(SourceRegistry, sid)
                if src is not None:
                    src.review_status = "REVIEWED"

    # X1D-AIV2-ATTEST1: stamping the column is no longer enough, and that is
    # the point -- eligibility now requires a durable APPROVED_BY_ATTESTATION
    # event that only the verified transition writes. So this helper runs the
    # real flow: submit, then approve against a stub core whose attestation
    # matches the object's recomputed binding.
    from app.services.governance.attested_review import attested_review_service
    attested_review_service.submit(object_type=object_type, object_id=object_id,
                                   submitted_by="synth-submitter")
    result, _record, _svc = attested_approve(object_type, object_id)
    return result["review_attestation_id"]


# ----------------------------------------------------------------------
# X1D-AIV2-ATTEST1: attested approval, through the real verification path
# ----------------------------------------------------------------------
# The helper above stamps rows. This one does not: it builds an attestation
# that a correctly-behaving xerbs-core would have produced, hands it to a stub
# lookup, and runs the actual AttestedReviewService. So a test that uses it
# exercises every check -- authoritative state, subject binding, version,
# hashes, provenance subjects, IN_REVIEW gate, TOCTOU -- rather than skipping
# them. Overriding any field is how the negative tests force a mismatch.


class StubCoreAttestationClient:
    """Stands in for core's internal lookup. Returns what it is given."""

    def __init__(self, record=None, *, raises=None):
        self.record = record
        self.raises = raises
        self.calls = []

    def configured(self) -> bool:
        return True

    def fetch(self, attestation_id, *, correlation_id=None):
        self.calls.append((attestation_id, correlation_id))
        if self.raises is not None:
            raise self.raises
        if self.record is None or self.record.get("attestation_id") != attestation_id:
            from app.services.governance.attestation_client import AttestationNotFound
            raise AttestationNotFound(attestation_id)
        return self.record


def local_binding_for(object_type: str, object_id: str, session_factory=None):
    """What ai-v2 currently believes about the object."""
    from app.services.governance.attested_review import ADAPTERS
    Session = session_factory or get_session_factory()
    with Session() as s:
        _row, binding = ADAPTERS[object_type](s, object_id)
        return binding


def core_attestation_for(object_type: str, object_id: str,
                         session_factory=None, **overrides):
    """A well-formed attestation matching the object's current binding.

    Reviewer is a synthetic core admin subject. This function never writes
    anything; it only describes what core would have recorded.
    """
    from app.core.config import get_settings
    binding = local_binding_for(object_type, object_id, session_factory)
    record = {
        "attestation_id": SYNTHETIC_ATTESTATION_PREFIX + uuid.uuid4().hex[:12],
        "decision": "APPROVED",
        "target_system": "xerbs-ai-v2",
        "target_environment": get_settings().environment,
        "governance_object_type": object_type,
        "semantic_object_id": binding.semantic_object_id,
        "object_version": binding.object_version,
        "content_hash": binding.content_hash,
        "evidence_hash": binding.evidence_hash,
        "reviewer_subject": "xerbs-core:admin:4242",
        "reviewer_authority": {"permission": "clinical:review"},
        "author_subject": binding.author_subject,
        "submitter_subject": binding.submitter_subject,
        "last_material_editor_subject": binding.last_material_editor_subject,
        "reviewed_at": "2026-01-01T00:00:00+00:00",
        "policy_version": "clinical-review/1",
        "supersedes_attestation_id": None,
        "superseded_by_attestation_id": None,
        "revoked_at": None,
        "revocation_reason": None,
        "is_revoked": False,
        "is_superseded": False,
        "is_active": True,
    }
    record.update(overrides)
    if "is_active" not in overrides:
        record["is_active"] = (record["decision"] == "APPROVED"
                               and not record["is_revoked"]
                               and not record["is_superseded"])
    return record


def attested_approve(object_type: str, object_id: str, *, record=None,
                     raises=None, expected_version=None, session_factory=None,
                     **overrides):
    """Run the real attested-review flow with a stubbed core lookup."""
    from app.services.governance.attested_review import AttestedReviewService
    if record is None and raises is None:
        record = core_attestation_for(object_type, object_id,
                                      session_factory=session_factory, **overrides)
    client = StubCoreAttestationClient(record, raises=raises)
    service = AttestedReviewService(client=client, session_factory=session_factory)
    result = service.approve(object_type=object_type, object_id=object_id,
                             attestation_id=(record or {}).get(
                                 "attestation_id", "att-SYNTHETIC-TEST-missing"),
                             expected_version=expected_version,
                             correlation_id="corr-SYNTHETIC",
                             idempotency_key="idem-SYNTHETIC")
    return result, record, service


def submit_and_attest_entity(entity_type: str, entity_id: str,
                             submitted_by: str = "synth-author"):
    """DRAFT -> IN_REVIEW -> REVIEWED, the latter only via attestation."""
    from app.schemas.clinical_workflow import ClinicalEntityType
    from app.services.knowledge.persistent_clinical import PersistentClinicalStore
    store = PersistentClinicalStore()
    store.submit_for_review(ClinicalEntityType(entity_type), entity_id, submitted_by)
    return attested_approve("CLINICAL_ENTITY", entity_id)


class _FakeResponse:
    """Shape-compatible stand-in so call sites keep reading ``.status_code``."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


def approve_entity_for_test(entity_type: str, entity_id: str,
                            session_factory=None):
    """Attested approval for tests that used the now-closed local path.

    X1D-AIV2-ATTEST1 closed ``POST /entities/{t}/{id}/review`` with
    decision=APPROVE, because a reviewer_role in a request body is a
    self-assertion. Tests whose subject is something else -- retrieval,
    eligibility, audit ordering -- still need a REVIEWED entity, so they go
    through the real verification path instead.

    Entities ingested without an ``external_id`` cannot be attestation
    subjects, which is correct and is a real deployment prerequisite. Here we
    assign a synthetic one so the fixture can proceed; production objects need
    authored identities, not generated ones.
    """
    from app.db.models import ClinicalEntity
    from app.schemas.clinical_workflow import ClinicalEntityType
    from app.services.knowledge.persistent_clinical import PersistentClinicalStore

    Session = session_factory or get_session_factory()
    with Session.begin() as s:
        row = s.get(ClinicalEntity, entity_id)
        assert row is not None, "entity %s does not exist" % entity_id
        if not row.external_id:
            row.external_id = "xerbs-ai-v2-test:%s:%s" % (row.entity_type, entity_id)

    from app.api.governance import _STATUS
    from app.services.governance.attested_review import AttestedReviewError
    try:
        result, _record, _svc = attested_approve(
            "CLINICAL_ENTITY", entity_id, session_factory=session_factory)
    except AttestedReviewError as exc:
        # Mirror the endpoint's mapping so call sites that assert on a status
        # code keep describing the same contract.
        return _FakeResponse({"code": exc.code, "message": exc.message},
                             status_code=_STATUS.get(exc.code, 422))
    detail = PersistentClinicalStore(session_factory).get_entity_detail(
        ClinicalEntityType(entity_type), entity_id)
    return _FakeResponse({
        "entity_id": entity_id,
        "review_status": detail.review_status.value,
        "version": detail.version,
        "clinical_ranking_eligible": detail.clinical_ranking_eligible,
        "attested": result,
    })


def drive_source_to_reviewed(client, source_id: str):
    """DRAFT -> IN_REVIEW -> REVIEWED for one Source.

    Source governance is a separate, still-open lifecycle: a Source is
    bibliographic metadata, not a clinical claim, and X1D-AIV2-ATTEST1 did not
    close it. ``expected_version`` is required by both request schemas, and
    omitting it fails with a quiet 422 -- which is how an earlier version of
    this helper left every test source at DRAFT.
    """
    got = client.get("/api/v1/knowledge/clinical/sources/%s" % source_id)
    assert got.status_code == 200, got.text
    state = got.json()

    if state["review_status"] == "REVIEWED":
        return state
    if state["review_status"] == "DRAFT":
        r = client.post(
            "/api/v1/knowledge/clinical/sources/%s/submit-review" % source_id,
            json={"submitted_by": "synth-author",
                  "expected_version": state["version"]})
        assert r.status_code == 200, r.text
        state = r.json()
    assert state["review_status"] == "IN_REVIEW", state["review_status"]

    # X1D-AIV2-GOVCLOSURE1: source APPROVE is closed to request-body roles.
    # Approval runs the real attested path against a stub core, so the fixture
    # exercises verification rather than skipping it.
    attested_approve("SOURCE", source_id)
    got = client.get("/api/v1/knowledge/clinical/sources/%s" % source_id)
    assert got.json()["review_status"] == "REVIEWED", got.text
    return got.json()


def approve_source_for_test(source_id: str, session_factory=None):
    """Attested source approval for store-level tests.

    The request-body approval path is closed; a Source reaching REVIEWED is
    what makes everything downstream ranking-eligible, so it goes through the
    same verification as any other governed object.
    """
    result, _record, _svc = attested_approve("SOURCE", source_id,
                                             session_factory=session_factory)
    return result
