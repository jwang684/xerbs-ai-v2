"""Human SOURCE VERIFICATION: verify, then transition (X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P7).

What it means, and what it does not
-----------------------------------
A source verification is a human decision, made in xerbs-core by an admin who
holds the explicit ``source:verify`` authority, that says exactly this::

    I verified that this object accurately represents the cited authoritative
    source and does not add unsupported clinical claims. This verification is
    source verification, not independent clinical endorsement.

It is **not** a clinical review. It never sets REVIEWED, never sets the
ATTESTED provenance, never fills ``review_attestation_id``, and never writes
APPROVED_BY_ATTESTATION -- so every existing clinical gate (ranking, reviewed-
only search, safety screening, attested approval) goes on refusing it. The
object moves to its own state, SOURCE_VERIFIED, which only the explicit
source-bounded retrieval path reads, and only in the environments listed in
``lifecycle.SOURCE_BOUNDED_RETRIEVAL_ENVIRONMENTS``.

How it is verified
------------------
Exactly as the clinical lane is, and through the same adapters: the record is
fetched from core over an authenticated internal call (a different table and a
different path from clinical attestations), and every bound field -- semantic
id, version, content hash, evidence hash, author/submitter/editor -- is
recomputed here from current rows and must match, inside the transaction that
writes the transition. Beyond that, the record must state it IS a source
verification, carry the exact statement above at the exact statement version,
and name a human core admin who is none of the author, submitter or editor.

Durable proof
-------------
Success appends ``SOURCE_VERIFIED_BY_ATTESTATION`` at the new version. That
event -- not the status string -- is what source-bounded retrieval requires,
and it does not carry forward to a later version.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import (ClinicalEntityVersion, GovernedObjectReviewEvent)
from app.db.session import get_session_factory
from app.services.governance import identity, lifecycle
from app.services.governance.attestation_client import (
    AttestationLookupError, CoreSourceVerificationClient)
from app.services.governance.attested_review import (
    ADAPTERS, CLINICAL_ENTITY, CLINICAL_RELATIONSHIP, SOURCE,
    AttestedReviewError, AttestedReviewService)

#: The one sentence a verifier stands behind. Byte-identical to xerbs-core's
#: SOURCE_VERIFICATION_STATEMENT (migration 038); a record carrying any other
#: text is refused, so the meaning cannot drift on one side only.
SOURCE_VERIFICATION_STATEMENT = (
    "I verified that this object accurately represents the cited authoritative "
    "source and does not add unsupported clinical claims. This verification is "
    "source verification, not independent clinical endorsement.")
SOURCE_VERIFICATION_STATEMENT_VERSION = "source-verification/1"
VERIFICATION_KIND = "SOURCE_VERIFICATION"
VERIFIED_DECISION = "SOURCE_VERIFIED"

SOURCE_VERIFIED_BY_ATTESTATION = "SOURCE_VERIFIED_BY_ATTESTATION"
#: Reserved for the revocation lane. Nothing writes it yet; eligibility already
#: honours it, so a revocation added later cannot be ignored by retrieval.
SOURCE_VERIFICATION_REVOKED = "SOURCE_VERIFICATION_REVOKED"

#: Safety rules are deliberately absent: a safety rule is a clinical judgement
#: about harm, and "it matches the source" is not a safety review.
VERIFIABLE_OBJECT_TYPES = (SOURCE, CLINICAL_ENTITY, CLINICAL_RELATIONSHIP)

#: Only an authenticated human core admin can be the verifier.
HUMAN_VERIFIER_PREFIX = "xerbs-core:admin:"


def has_verified_source_verification(s, object_type: str, object_id: str,
                                     version: int) -> bool:
    """Durable proof that THIS version was source-verified, and not withdrawn."""
    row = s.scalars(select(GovernedObjectReviewEvent).where(
        GovernedObjectReviewEvent.object_type == object_type,
        GovernedObjectReviewEvent.object_id == object_id,
        GovernedObjectReviewEvent.action == SOURCE_VERIFIED_BY_ATTESTATION,
        GovernedObjectReviewEvent.version == version)).first()
    if row is None or not row.attestation_id:
        return False
    revoked = s.scalars(select(GovernedObjectReviewEvent).where(
        GovernedObjectReviewEvent.object_type == object_type,
        GovernedObjectReviewEvent.object_id == object_id,
        GovernedObjectReviewEvent.action == SOURCE_VERIFICATION_REVOKED,
        GovernedObjectReviewEvent.attestation_id == row.attestation_id)).first()
    return revoked is None


def _apply_source_verification(s, row, object_type, verification_id,
                               verifier_subject, verified_version: int) -> int:
    """Write SOURCE_VERIFIED and return the new lifecycle position.

    Mirrors the clinical approval's version handling so a later content change
    moves the object past the verified position. Never touches REVIEWED,
    ATTESTED, ``reviewed_by`` or ``review_attestation_id``.
    """
    now = datetime.now(timezone.utc)
    new_version = verified_version + 1
    row.review_status = lifecycle.SOURCE_VERIFIED
    if object_type == SOURCE:
        row.updated_at = now
        row.version = new_version
        from app.services.knowledge.persistent_clinical import PersistentClinicalStore
        PersistentClinicalStore._add_source_event(
            s, row, "SOURCE_VERIFIED", verifier_subject, "SOURCE_VERIFIER",
            lifecycle.IN_REVIEW, lifecycle.SOURCE_VERIFIED,
            "source-verified against xerbs-core verification %s; "
            "not a clinical review" % verification_id)
        return new_version

    row.governance_provenance = lifecycle.SOURCE_VERIFIED_PROVENANCE
    if object_type == CLINICAL_RELATIONSHIP:
        row.updated_at = now
        row.version = new_version
        return new_version

    # Clinical entity: record the position in the snapshot table, content
    # unchanged. Not clinically ranking eligible -- that stays a REVIEWED-only
    # fact.
    row.current_version = new_version
    latest = s.scalars(select(ClinicalEntityVersion).where(
        ClinicalEntityVersion.entity_id == row.id,
        ClinicalEntityVersion.version == verified_version)).first()
    snapshot = dict(latest.snapshot) if latest is not None else {}
    snapshot["review_status"] = lifecycle.SOURCE_VERIFIED
    snapshot["version"] = new_version
    snapshot["clinical_ranking_eligible"] = False
    s.add(ClinicalEntityVersion(
        id="ver-%s" % uuid4().hex[:16], entity_id=row.id,
        version=new_version, snapshot=snapshot, created_by=verifier_subject))
    return new_version


class SourceVerificationService:
    """Verify a core SOURCE-VERIFICATION record, then transition one object."""

    def __init__(self, client: CoreSourceVerificationClient | None = None,
                 session_factory=None):
        self._client = client or CoreSourceVerificationClient()
        self.Session = session_factory or get_session_factory()

    @staticmethod
    def _assert_record_is_a_source_verification(rec: Mapping[str, Any],
                                                object_type: str) -> None:
        def refuse(message, code):
            raise AttestedReviewError(message, code=code)

        if rec.get("verification_kind") != VERIFICATION_KIND:
            refuse("record is %r, not a source verification"
                   % rec.get("verification_kind"), "NOT_A_SOURCE_VERIFICATION")
        if rec.get("decision") != VERIFIED_DECISION:
            refuse("verification decision is %r, not %s"
                   % (rec.get("decision"), VERIFIED_DECISION), "DECISION_NOT_VERIFIED")
        if rec.get("statement") != SOURCE_VERIFICATION_STATEMENT:
            refuse("verification does not carry the source-verification statement",
                   "STATEMENT_MISMATCH")
        if rec.get("statement_version") != SOURCE_VERIFICATION_STATEMENT_VERSION:
            refuse("statement version %r is not %s"
                   % (rec.get("statement_version"),
                      SOURCE_VERIFICATION_STATEMENT_VERSION), "STATEMENT_MISMATCH")
        if rec.get("is_revoked"):
            refuse("verification has been revoked", "ATTESTATION_REVOKED")
        if not rec.get("is_active"):
            refuse("verification is not active", "ATTESTATION_NOT_ACTIVE")
        if rec.get("target_system") != "xerbs-ai-v2":
            refuse("verification targets %r, not this service"
                   % rec.get("target_system"), "TARGET_SYSTEM_MISMATCH")
        expected_env = (get_settings().environment or "").strip().lower()
        if (rec.get("target_environment") or "").strip().lower() != expected_env:
            refuse("verification targets environment %r; this deployment is %r"
                   % (rec.get("target_environment"), expected_env),
                   "TARGET_ENVIRONMENT_MISMATCH")
        if rec.get("governance_object_type") != object_type:
            refuse("verification is for %r, not %r"
                   % (rec.get("governance_object_type"), object_type),
                   "OBJECT_TYPE_MISMATCH")
        verifier = rec.get("verifier_subject") or ""
        if not (identity.is_valid_subject(verifier)
                and verifier.startswith(HUMAN_VERIFIER_PREFIX)):
            refuse("verifier must be an authenticated xerbs-core admin subject",
                   "VERIFIER_SUBJECT_INVALID")

    def verify(self, *, object_type: str, object_id: str, verification_id: str,
               expected_version: int | None = None,
               correlation_id: str | None = None,
               idempotency_key: str | None = None) -> dict:
        if object_type not in VERIFIABLE_OBJECT_TYPES:
            raise AttestedReviewError(
                "object_type %r cannot be source-verified" % object_type,
                code="UNKNOWN_OBJECT_TYPE")
        try:
            rec = self._client.fetch(verification_id, correlation_id=correlation_id)
        except AttestationLookupError as exc:
            raise AttestedReviewError(exc.message, code=exc.code) from exc

        self._assert_record_is_a_source_verification(rec, object_type)
        verifier = rec["verifier_subject"]

        with self.Session.begin() as s:
            row, local = ADAPTERS[object_type](s, object_id)

            existing = s.scalars(select(GovernedObjectReviewEvent).where(
                GovernedObjectReviewEvent.object_type == object_type,
                GovernedObjectReviewEvent.object_id == object_id,
                GovernedObjectReviewEvent.action == SOURCE_VERIFIED_BY_ATTESTATION)).all()
            for event in existing:
                if event.version == local.object_version:
                    if event.attestation_id == verification_id:
                        return {"object_type": object_type, "object_id": object_id,
                                "review_status": row.review_status,
                                "object_version": local.object_version,
                                "verification_id": verification_id,
                                "idempotent_replay": True}
                    raise AttestedReviewError(
                        "version %d was already source-verified under %s"
                        % (event.version, event.attestation_id),
                        code="ALREADY_VERIFIED_BY_DIFFERENT_RECORD")

            if expected_version is not None and int(expected_version) != int(local.object_version):
                raise AttestedReviewError(
                    "expected version %s, object is at version %s"
                    % (expected_version, local.object_version), code="VERSION_CONFLICT")
            if local.review_status != lifecycle.IN_REVIEW:
                raise AttestedReviewError(
                    "only IN_REVIEW objects can be source-verified; this one is %s"
                    % local.review_status, code="OBJECT_NOT_IN_REVIEW")

            # Same binding rule as the clinical lane, same TOCTOU position.
            AttestedReviewService._assert_binding_matches(rec, local)

            # Separation of duties, re-checked here against this service's own
            # record of who wrote and submitted the object.
            conflicts = [f for f, v in (("author", local.author_subject),
                                        ("submitter", local.submitter_subject),
                                        ("editor", local.last_material_editor_subject))
                         if v and v == verifier]
            if conflicts:
                raise AttestedReviewError(
                    "verifier is the object's %s" % "/".join(conflicts),
                    code="SEPARATION_OF_DUTIES")

            before = local.review_status
            new_version = _apply_source_verification(
                s, row, object_type, verification_id, verifier, local.object_version)
            s.add(GovernedObjectReviewEvent(
                event_id="govevt-%s" % uuid4().hex[:16],
                object_type=object_type, object_id=object_id,
                action=SOURCE_VERIFIED_BY_ATTESTATION,
                actor_subject=verifier,
                from_status=before, to_status=lifecycle.SOURCE_VERIFIED,
                version=new_version, attestation_id=verification_id,
                notes="source verification (%s) verified against xerbs-core; NOT a "
                      "clinical review; correlation=%s; idempotency=%s"
                      % (SOURCE_VERIFICATION_STATEMENT_VERSION,
                         correlation_id or "-", idempotency_key or "-")))
            return {"object_type": object_type, "object_id": object_id,
                    "review_status": lifecycle.SOURCE_VERIFIED,
                    "object_version": new_version,
                    "verified_version": local.object_version,
                    "verification_id": verification_id,
                    "verification_kind": VERIFICATION_KIND,
                    "idempotent_replay": False}


source_verification_service = SourceVerificationService()
