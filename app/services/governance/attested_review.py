"""Attested human review: verify, then transition (X1D-AIV2-ATTEST1).

The whole point in one sentence: **ai-v2 recomputes the subject from its own
rows and refuses unless core's attestation names exactly that.**

Core holds the human decision. It does not hold the object, so it cannot know
whether the thing it approved is still the thing ai-v2 stores. That half is
this file's job, and it is why the pushed payload is never trusted on its own:

  * the attestation is fetched from core over an authenticated internal call,
    so a fabricated id does not resolve;
  * every bound field is recomputed locally from current rows, so a stale or
    mismatched attestation is refused;
  * the comparison happens immediately before the transition, inside the same
    transaction, so a change between review and approval loses the race rather
    than slipping through.

Durable proof of approval
-------------------------
Success appends a ``governed_object_review_event`` with action
``APPROVED_BY_ATTESTATION`` carrying the attestation id and the exact version.
That row -- not the ``review_attestation_id`` column -- is what ranking
eligibility requires. Populating the column by hand, through a direct UPDATE or
some future careless code path, therefore confers nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import (ClinicalEntity, ClinicalEntityVersion,
                           ClinicalRelationship, EntitySource,
                           GovernedObjectReviewEvent, GovernedObjectSource,
                           ReviewEvent, SafetyRule, SourceRegistry,
                           SourceReviewEvent)
from app.db.session import get_session_factory
from app.services.governance import canonical, identity, lifecycle
from app.services.governance.attestation_client import (AttestationLookupError,
                                                        CoreAttestationClient)

APPROVED_BY_ATTESTATION = "APPROVED_BY_ATTESTATION"

SOURCE = "SOURCE"
CLINICAL_ENTITY = "CLINICAL_ENTITY"
CLINICAL_RELATIONSHIP = "CLINICAL_RELATIONSHIP"
SAFETY_RULE = "SAFETY_RULE"

OBJECT_TYPES = (SOURCE, CLINICAL_ENTITY, CLINICAL_RELATIONSHIP, SAFETY_RULE)


class AttestedReviewError(Exception):
    """Approval refused. Carries a stable code so the API can map it."""

    def __init__(self, message: str, *, code: str):
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class LocalBinding:
    """What ai-v2 believes about the object, computed from its own rows."""

    semantic_object_id: str | None
    object_version: int
    content_hash: str | None
    evidence_hash: str | None
    author_subject: str | None
    submitter_subject: str | None
    last_material_editor_subject: str | None
    review_status: str


# ----------------------------------------------------------------------
# Adapters. One per object type, because the four storage models genuinely
# differ -- forcing them through one generic SQL shape would be the kind of
# cleverness that quietly approves the wrong row.
# ----------------------------------------------------------------------

def _evidence_rows(s, object_type, object_id) -> list[dict]:
    rows = s.scalars(select(GovernedObjectSource).where(
        GovernedObjectSource.object_type == object_type,
        GovernedObjectSource.object_id == object_id)).all()
    return [{"source_id": r.source_id, "source_version": r.source_version,
             "locator": r.locator} for r in rows]


def _latest_actor(s, model, id_column, object_id, action) -> str | None:
    row = s.scalars(select(model).where(
        id_column == object_id, model.action == action)
        .order_by(model.version.desc()).limit(1)).first()
    return getattr(row, "actor_id", None) if row is not None else None


def _bind_source(s, object_id) -> tuple[Any, LocalBinding]:
    row = s.get(SourceRegistry, object_id)
    if row is None:
        raise AttestedReviewError("source %s not found" % object_id,
                                  code="OBJECT_NOT_FOUND")
    # SourceRegistry carries a real integer `version` that _transition bumps on
    # every lifecycle move, so the attestation binds a truthful version rather
    # than a hardcoded 1.
    subject = canonical.source_subject(
        source_id=row.source_id, title=row.title, citation=row.citation,
        url=row.url, source_type=row.source_type)
    submitter = _latest_actor(s, SourceReviewEvent, SourceReviewEvent.source_id,
                              object_id, "SUBMITTED_FOR_REVIEW")
    return row, LocalBinding(
        semantic_object_id=row.source_id,
        object_version=row.version,
        content_hash=canonical.digest(subject),
        # A Source IS its own evidence; there is no separate evidence set to
        # bind, and inventing an empty-set hash would assert something false.
        evidence_hash=None,
        author_subject=identity.as_subject(row.created_by),
        submitter_subject=identity.as_subject(submitter),
        last_material_editor_subject=None,
        review_status=row.review_status)


def _bind_entity(s, object_id) -> tuple[Any, LocalBinding]:
    row = s.get(ClinicalEntity, object_id)
    if row is None:
        raise AttestedReviewError("clinical entity %s not found" % object_id,
                                  code="OBJECT_NOT_FOUND")
    if not row.external_id:
        raise AttestedReviewError(
            "clinical entity %s has no semantic identity; it cannot be the "
            "subject of a cross-service attestation" % object_id,
            code="SEMANTIC_IDENTITY_MISSING")
    version_row = s.scalars(select(ClinicalEntityVersion).where(
        ClinicalEntityVersion.entity_id == object_id,
        ClinicalEntityVersion.version == row.current_version)).first()
    if version_row is None:
        raise AttestedReviewError(
            "clinical entity %s has no snapshot at version %d"
            % (object_id, row.current_version), code="OBJECT_STATE_INVALID")

    evidence = [{"source_id": es.source_id, "source_version": es.source_version,
                 "locator": es.locator}
                for es in s.scalars(select(EntitySource).where(
                    EntitySource.entity_id == object_id)).all()]
    if not evidence:
        # The local approval path enforced this before ATTEST1 closed it.
        # Carrying it forward matters: an attestation can bind an empty
        # evidence set perfectly well, so without this an entity could be
        # approved with no source at all.
        raise AttestedReviewError(
            "clinical entity %s has no attached source; approval requires at "
            "least one explicit source" % object_id,
            code="EVIDENCE_REQUIRED")
    subject = canonical.entity_subject(
        external_id=row.external_id, entity_type=row.entity_type,
        snapshot=version_row.snapshot)
    submitter = _latest_actor(s, ReviewEvent, ReviewEvent.entity_id,
                              object_id, "SUBMITTED_FOR_REVIEW")
    ingested = _latest_actor(s, ReviewEvent, ReviewEvent.entity_id,
                             object_id, "INGESTED_AS_DRAFT")
    return row, LocalBinding(
        semantic_object_id=row.external_id,
        object_version=row.current_version,
        content_hash=canonical.digest(subject),
        evidence_hash=canonical.evidence_hash(evidence),
        author_subject=identity.as_subject(ingested),
        submitter_subject=identity.as_subject(submitter),
        # The last identity to write a content-bearing version.
        last_material_editor_subject=identity.as_subject(version_row.created_by),
        review_status=row.review_status)


def _bind_relationship(s, object_id) -> tuple[Any, LocalBinding]:
    row = s.get(ClinicalRelationship, object_id)
    if row is None:
        raise AttestedReviewError("relationship %s not found" % object_id,
                                  code="OBJECT_NOT_FOUND")
    src = s.get(ClinicalEntity, row.source_entity_id)
    tgt = s.get(ClinicalEntity, row.target_entity_id)
    if src is None or tgt is None or not src.external_id or not tgt.external_id:
        raise AttestedReviewError(
            "relationship %s cannot be bound: both endpoints need a semantic "
            "identity" % object_id, code="SEMANTIC_IDENTITY_MISSING")

    external_id = identity.relationship_external_id(
        src.external_id, row.relationship_type, tgt.external_id)
    content_hash = lifecycle.relationship_content_hash(
        src.external_id, row.relationship_type, tgt.external_id)
    evidence = _evidence_rows(s, CLINICAL_RELATIONSHIP, object_id)
    return row, LocalBinding(
        semantic_object_id=external_id,
        object_version=row.version,
        content_hash=content_hash,
        evidence_hash=canonical.evidence_hash(evidence),
        author_subject=identity.as_subject(row.created_by),
        submitter_subject=row.submitter_subject,
        last_material_editor_subject=row.last_material_editor_subject,
        review_status=row.review_status)


def _bind_safety_rule(s, object_id) -> tuple[Any, LocalBinding]:
    row = s.get(SafetyRule, object_id)
    if row is None:
        raise AttestedReviewError("safety rule %s not found" % object_id,
                                  code="OBJECT_NOT_FOUND")
    target = s.get(ClinicalEntity, row.target_entity_id)
    if target is None or not target.external_id:
        raise AttestedReviewError(
            "safety rule %s cannot be bound: its target needs a semantic "
            "identity" % object_id, code="SEMANTIC_IDENTITY_MISSING")

    external_id = identity.safety_rule_external_id(
        target.external_id, row.rule_type, row.trigger_term)
    content_hash = lifecycle.safety_rule_content_hash(
        target_external_id=target.external_id, rule_type=row.rule_type,
        trigger_term=row.trigger_term, severity=row.severity,
        action=row.action, message=row.message)
    evidence = _evidence_rows(s, SAFETY_RULE, object_id)
    return row, LocalBinding(
        semantic_object_id=external_id,
        object_version=row.version,
        content_hash=content_hash,
        evidence_hash=canonical.evidence_hash(evidence),
        author_subject=identity.as_subject(row.created_by),
        submitter_subject=row.submitter_subject,
        last_material_editor_subject=row.last_material_editor_subject,
        review_status=row.review_status)


ADAPTERS = {
    SOURCE: _bind_source,
    CLINICAL_ENTITY: _bind_entity,
    CLINICAL_RELATIONSHIP: _bind_relationship,
    SAFETY_RULE: _bind_safety_rule,
}


def _apply_approval(s, row, object_type, attestation_id, reviewer_subject,
                    attested_version: int) -> int:
    """Write the REVIEWED state, and return the new lifecycle position.

    Approval advances the version, as every other governed transition does.
    The attestation bound ``attested_version`` and the binding comparison has
    already happened against it, so nothing is invalidated retroactively; what
    the bump buys is that a clinical entity's snapshot stays consistent with
    its row, and that a later content change moves the object past the
    approved position and drops eligibility.
    """
    now = datetime.now(timezone.utc)
    new_version = attested_version + 1
    row.review_status = lifecycle.REVIEWED
    if object_type == SOURCE:
        # SourceRegistry predates GOV2 and has reviewed_by, not the GOV2 column
        # set. Marked as an externally asserted subject, never as a local
        # identity -- ai-v2 has no human users and must not appear to.
        row.reviewed_by = reviewer_subject
        row.updated_at = now
        row.version = new_version
    else:
        # Explicit per type. Setting an attribute that is not a mapped column
        # is silently accepted by SQLAlchemy and lost on commit -- which is
        # exactly how the entity attestation reference went missing before
        # migration 0009 added the column. Never assume the field is there.
        row.governance_provenance = lifecycle.ATTESTED
        row.review_attestation_id = attestation_id
        row.reviewed_at = now
        if object_type in (CLINICAL_RELATIONSHIP, SAFETY_RULE):
            row.updated_at = now
            row.version = new_version
        else:
            # Clinical entity: the snapshot table is the record of what the
            # object looked like at each position, so approval writes one too.
            # Its content is unchanged -- only review_status moves -- and
            # review_status is excluded from the content hash by rule 8, so
            # this cannot alter what was approved.
            row.current_version = new_version
            latest = s.scalars(select(ClinicalEntityVersion).where(
                ClinicalEntityVersion.entity_id == row.id,
                ClinicalEntityVersion.version == attested_version)).first()
            snapshot = dict(latest.snapshot) if latest is not None else {}
            snapshot["review_status"] = lifecycle.REVIEWED
            snapshot["version"] = new_version
            # Same canonical rule the store reports elsewhere: REVIEWED plus at
            # least one REVIEWED Source. Leaving the stale value would make the
            # snapshot disagree with the entity detail.
            from app.db.models import EntitySource as _ES
            reviewed_sources = s.query(_ES).join(
                SourceRegistry, SourceRegistry.source_id == _ES.source_id).filter(
                _ES.entity_id == row.id,
                SourceRegistry.review_status == "REVIEWED").count()
            snapshot["clinical_ranking_eligible"] = reviewed_sources > 0
            s.add(ClinicalEntityVersion(
                id="ver-%s" % uuid4().hex[:16], entity_id=row.id,
                version=new_version, snapshot=snapshot,
                created_by=reviewer_subject))
    return new_version


class AttestedReviewService:
    """Verify a core attestation, then transition one governed object."""

    def __init__(self, client: CoreAttestationClient | None = None,
                 session_factory=None):
        self._client = client or CoreAttestationClient()
        self.Session = session_factory or get_session_factory()

    # -- verification ---------------------------------------------------
    @staticmethod
    def _assert_attestation_is_authoritative(att: Mapping[str, Any],
                                             object_type: str) -> None:
        """Everything that is true of the attestation itself, before we look
        at the object. Approval is never inferred from the row existing."""
        if att.get("decision") != "APPROVED":
            raise AttestedReviewError(
                "attestation decision is %r, not APPROVED" % att.get("decision"),
                code="DECISION_NOT_APPROVED")
        if att.get("is_revoked"):
            raise AttestedReviewError("attestation has been revoked",
                                      code="ATTESTATION_REVOKED")
        if att.get("is_superseded"):
            raise AttestedReviewError(
                "attestation has been superseded by %s"
                % att.get("superseded_by_attestation_id"),
                code="ATTESTATION_SUPERSEDED")
        if not att.get("is_active"):
            raise AttestedReviewError("attestation is not active",
                                      code="ATTESTATION_NOT_ACTIVE")

        if att.get("target_system") != "xerbs-ai-v2":
            raise AttestedReviewError(
                "attestation targets %r, not this service"
                % att.get("target_system"), code="TARGET_SYSTEM_MISMATCH")

        expected_env = (get_settings().environment or "").strip().lower()
        if (att.get("target_environment") or "").strip().lower() != expected_env:
            # A staging-signed attestation must be structurally incapable of
            # approving production, and vice versa.
            raise AttestedReviewError(
                "attestation targets environment %r; this deployment is %r"
                % (att.get("target_environment"), expected_env),
                code="TARGET_ENVIRONMENT_MISMATCH")

        if att.get("governance_object_type") != object_type:
            raise AttestedReviewError(
                "attestation is for %r, not %r"
                % (att.get("governance_object_type"), object_type),
                code="OBJECT_TYPE_MISMATCH")

        if not identity.is_valid_subject(att.get("reviewer_subject") or ""):
            raise AttestedReviewError(
                "attestation reviewer_subject is not a valid namespaced subject",
                code="REVIEWER_SUBJECT_MALFORMED")

    @staticmethod
    def _assert_binding_matches(att: Mapping[str, Any], local: LocalBinding) -> None:
        """The half core cannot do: is the approved thing the stored thing?"""
        def mismatch(field, expected, got, code):
            raise AttestedReviewError(
                "%s mismatch: attestation says %r, this service computed %r"
                % (field, expected, got), code=code)

        if att.get("semantic_object_id") != local.semantic_object_id:
            mismatch("semantic_object_id", att.get("semantic_object_id"),
                     local.semantic_object_id, "SUBJECT_MISMATCH")
        if int(att.get("object_version") or -1) != int(local.object_version):
            mismatch("object_version", att.get("object_version"),
                     local.object_version, "VERSION_MISMATCH")
        if att.get("content_hash") != local.content_hash:
            mismatch("content_hash", att.get("content_hash"),
                     local.content_hash, "CONTENT_HASH_MISMATCH")
        if (att.get("evidence_hash") or None) != (local.evidence_hash or None):
            mismatch("evidence_hash", att.get("evidence_hash"),
                     local.evidence_hash, "EVIDENCE_HASH_MISMATCH")

        # Provenance subjects. Core checked independence against these values;
        # if they no longer describe the object, the independence core checked
        # was about something else.
        for field, expected, got, code in (
            ("author_subject", att.get("author_subject"), local.author_subject,
             "AUTHOR_SUBJECT_MISMATCH"),
            ("submitter_subject", att.get("submitter_subject"),
             local.submitter_subject, "SUBMITTER_SUBJECT_MISMATCH"),
            ("last_material_editor_subject",
             att.get("last_material_editor_subject"),
             local.last_material_editor_subject, "EDITOR_SUBJECT_MISMATCH"),
        ):
            if (expected or None) != (got or None):
                mismatch(field, expected, got, code)

    # -- locally-authorisable transition ---------------------------------
    def submit(self, *, object_type: str, object_id: str,
               submitted_by: str) -> dict:
        """DRAFT/REJECTED -> IN_REVIEW. Confers no authority whatsoever.

        Uses lifecycle.next_state, so it shares the table that refuses APPROVE
        from every state. A submitter is recorded as a machine principal, and
        core later checks that the approver is not that same subject.
        """
        if object_type not in ADAPTERS:
            raise AttestedReviewError("unknown object_type %r" % object_type,
                                      code="UNKNOWN_OBJECT_TYPE")
        subject = identity.as_subject(submitted_by)
        with self.Session.begin() as s:
            row, local = ADAPTERS[object_type](s, object_id)
            try:
                target = lifecycle.next_state(local.review_status, "SUBMIT")
            except lifecycle.GovernanceLifecycleError as exc:
                raise AttestedReviewError(str(exc), code="INVALID_TRANSITION") from exc

            before = row.review_status
            row.review_status = target
            new_version = local.object_version
            if object_type in (CLINICAL_RELATIONSHIP, SAFETY_RULE):
                # Submission advances the lifecycle position, as it does for
                # clinical entities. Approval deliberately does not: the
                # attestation binds this exact version.
                new_version = local.object_version + 1
                row.version = new_version
                row.submitter_subject = subject
                row.updated_at = datetime.now(timezone.utc)

            s.add(GovernedObjectReviewEvent(
                event_id="govevt-%s" % uuid4().hex[:16],
                object_type=object_type, object_id=object_id,
                action="SUBMITTED_FOR_REVIEW", actor_subject=subject,
                from_status=before, to_status=target,
                version=new_version, attestation_id=None))
            return {"object_type": object_type, "object_id": object_id,
                    "review_status": target, "object_version": new_version,
                    "submitter_subject": subject}

    # -- the operation --------------------------------------------------
    def approve(self, *, object_type: str, object_id: str, attestation_id: str,
                expected_version: int | None = None,
                correlation_id: str | None = None,
                idempotency_key: str | None = None) -> dict:
        if object_type not in ADAPTERS:
            raise AttestedReviewError("unknown object_type %r" % object_type,
                                      code="UNKNOWN_OBJECT_TYPE")

        try:
            att = self._client.fetch(attestation_id, correlation_id=correlation_id)
        except AttestationLookupError as exc:
            # Every lookup failure -- unconfigured, unreachable, malformed,
            # unknown -- fails closed. None of them is ever read as approval.
            raise AttestedReviewError(exc.message, code=exc.code) from exc

        self._assert_attestation_is_authoritative(att, object_type)

        with self.Session.begin() as s:
            row, local = ADAPTERS[object_type](s, object_id)

            # Idempotent replay: the same attestation already applied to this
            # exact version returns the existing result rather than doing it
            # again. Checked before the IN_REVIEW gate, because after a
            # successful approval the object is no longer IN_REVIEW.
            existing = s.scalars(select(GovernedObjectReviewEvent).where(
                GovernedObjectReviewEvent.object_type == object_type,
                GovernedObjectReviewEvent.object_id == object_id,
                GovernedObjectReviewEvent.action == APPROVED_BY_ATTESTATION)
                .order_by(GovernedObjectReviewEvent.version.desc())).all()
            for event in existing:
                if event.version == local.object_version:
                    if event.attestation_id == attestation_id:
                        return {"object_type": object_type, "object_id": object_id,
                                "review_status": row.review_status,
                                "object_version": local.object_version,
                                "review_attestation_id": event.attestation_id,
                                "idempotent_replay": True}
                    raise AttestedReviewError(
                        "version %d of this object was already approved under "
                        "attestation %s; a different attestation cannot be "
                        "applied to the same version"
                        % (event.version, event.attestation_id),
                        code="ALREADY_APPROVED_BY_DIFFERENT_ATTESTATION")

            if expected_version is not None and int(expected_version) != int(local.object_version):
                raise AttestedReviewError(
                    "expected version %s, object is at version %s"
                    % (expected_version, local.object_version),
                    code="VERSION_CONFLICT")

            if local.review_status != lifecycle.IN_REVIEW:
                raise AttestedReviewError(
                    "only IN_REVIEW objects can receive an attested review "
                    "decision; this one is %s" % local.review_status,
                    code="OBJECT_NOT_IN_REVIEW")

            # TOCTOU: recomputed from rows loaded inside this transaction,
            # compared immediately before the write.
            self._assert_binding_matches(att, local)

            before = local.review_status
            new_version = _apply_approval(
                s, row, object_type, attestation_id,
                att.get("reviewer_subject"), local.object_version)

            s.add(GovernedObjectReviewEvent(
                event_id="govevt-%s" % uuid4().hex[:16],
                object_type=object_type, object_id=object_id,
                action=APPROVED_BY_ATTESTATION,
                # Recorded as the EXTERNALLY VERIFIED subject. ai-v2 did not
                # authenticate this person and must never appear to have.
                actor_subject=att.get("reviewer_subject"),
                from_status=before, to_status=lifecycle.REVIEWED,
                version=new_version, attestation_id=attestation_id,
                notes="verified against xerbs-core; correlation=%s; "
                      "idempotency=%s; reviewer identity asserted by core, not "
                      "authenticated here"
                      % (correlation_id or "-", idempotency_key or "-")))

            return {"object_type": object_type, "object_id": object_id,
                    "review_status": lifecycle.REVIEWED,
                    "object_version": new_version,
                    "attested_version": local.object_version,
                    "review_attestation_id": attestation_id,
                    "idempotent_replay": False}


def has_verified_attested_approval(s, object_type: str, object_id: str,
                                   version: int, attestation_id: str | None) -> bool:
    """Durable proof that THIS version was approved under THIS attestation.

    This is what ranking eligibility consults -- not the
    ``review_attestation_id`` column. A direct UPDATE that sets the column
    creates no event, so it confers nothing; and an event from an earlier
    version does not carry forward, so changing the object drops eligibility
    until it is reviewed again.
    """
    if not attestation_id:
        return False
    row = s.scalars(select(GovernedObjectReviewEvent).where(
        GovernedObjectReviewEvent.object_type == object_type,
        GovernedObjectReviewEvent.object_id == object_id,
        GovernedObjectReviewEvent.action == APPROVED_BY_ATTESTATION,
        GovernedObjectReviewEvent.version == version,
        GovernedObjectReviewEvent.attestation_id == attestation_id)).first()
    return row is not None


attested_review_service = AttestedReviewService()
