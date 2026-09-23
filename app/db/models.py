from datetime import datetime, timezone
from sqlalchemy import String, Integer, DateTime, Text, ForeignKey, UniqueConstraint, Index
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


def utcnow():
    return datetime.now(timezone.utc)


class ClinicalEntity(Base):
    __tablename__ = "clinical_entity"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # X1D-AIV2-GOV2-C1: stable semantic identity, for cross-service attestation
    # and for a future governed export/import. Supplied by authoring tooling,
    # never derived from `name` -- a display name is mutable, and deriving
    # identity from it would fork the identity on rename. NULL for rows
    # authored before GOV2; the migration does not invent one.
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True)
    entity_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    review_status: Mapped[str] = mapped_column(String(20), nullable=False, default="DRAFT", index=True)
    current_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # X1D-AIV2-GOV2-C1: how this object came to hold its review_status. A
    # legacy REVIEWED row is not evidence that a human reviewed anything.
    governance_provenance: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    # X1D-AIV2-ATTEST1: the attestation seam for entities. 0008 gave it to
    # relationships and safety rules only, because entity approval was still
    # local then. Closing that path proved the column is needed here too --
    # without it the approval wrote to a phantom Python attribute and the
    # reference was silently lost.
    review_attestation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    migration_origin: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_by_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("clinical_entity.id"), nullable=True)
    __table_args__ = (Index("ix_clinical_entity_type_status", "entity_type", "review_status"),)


class ClinicalEntityVersion(Base):
    __tablename__ = "clinical_entity_version"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_id: Mapped[str] = mapped_column(String(64), ForeignKey("clinical_entity.id", ondelete="RESTRICT"), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (UniqueConstraint("entity_id", "version", name="uq_clinical_entity_version"),)


class SourceRegistry(Base):
    __tablename__ = "source_registry"
    source_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    citation: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(String(80), nullable=False, default="UNSPECIFIED")
    review_status: Mapped[str] = mapped_column(String(20), nullable=False, default="DRAFT", index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=True)


class EntitySource(Base):
    __tablename__ = "entity_source"
    entity_id: Mapped[str] = mapped_column(String(64), ForeignKey("clinical_entity.id", ondelete="CASCADE"), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(128), ForeignKey("source_registry.source_id", ondelete="RESTRICT"), primary_key=True)
    source_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    locator: Mapped[str | None] = mapped_column(Text, nullable=True)


class IngestionBatch(Base):
    __tablename__ = "ingestion_batch"
    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    submitted_by: Mapped[str] = mapped_column(String(255), nullable=False)
    source_label: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class IngestionItemRow(Base):
    __tablename__ = "ingestion_item"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(64), ForeignKey("ingestion_batch.batch_id", ondelete="CASCADE"), nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String(64), ForeignKey("clinical_entity.id", ondelete="RESTRICT"), nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class ReviewEvent(Base):
    __tablename__ = "review_event"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_id: Mapped[str] = mapped_column(String(64), ForeignKey("clinical_entity.id", ondelete="RESTRICT"), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(20), nullable=False)
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(255), nullable=False)
    actor_role: Mapped[str | None] = mapped_column(String(80), nullable=True)
    from_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    # X1D-AUDITORDER1. version is the causal order of this entity's lifecycle:
    # _transition bumps it exactly once per governed change and stamps the event
    # with it. Ordering read it that way long before anything enforced it, and
    # two concurrent writers could both commit the same number -- the optimistic
    # check reads without a lock. This makes the claim the database's to keep.
    __table_args__ = (UniqueConstraint("entity_id", "version",
                                       name="uq_review_event_entity_version"),)


class SourceReviewEvent(Base):
    """Append-only governance trail for canonical Sources.

    Mirrors ReviewEvent, but keyed to source_registry so the existing
    review_event -> clinical_entity foreign key stays intact.
    """

    __tablename__ = "source_review_event"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(128), ForeignKey("source_registry.source_id", ondelete="RESTRICT"), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(255), nullable=False)
    actor_role: Mapped[str | None] = mapped_column(String(80), nullable=True)
    from_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (UniqueConstraint("source_id", "version",
                                       name="uq_source_review_event_source_version"),)


class AuditEvent(Base):
    __tablename__ = "audit_event"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # Canonical Source identity is kept in its own column rather than sharing the
    # clinical entity_id namespace, which is unvalidated client-supplied text.
    source_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    actor_id: Mapped[str] = mapped_column(String(255), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    # X1D-AUDITORDER1. The aggregate's version, copied here so this table can be
    # ordered causally instead of by (created_at, event_id) -- a random uuid
    # tiebreaker that returned SOURCE_APPROVED before SOURCE_CREATED whenever the
    # clock tied, which on a 1-2ms tick it regularly did.
    #
    # Nullable on purpose: telemetry and gap events belong to no aggregate, have
    # no causal position, and are never returned by an ordered query. NULLs are
    # distinct under UNIQUE in both SQLite and PostgreSQL, so they coexist freely.
    #
    # Internal. Not exposed by SourceAuditEventRecord and not part of any API
    # response -- the ordering it produces is the visible effect, not the number.
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    __table_args__ = (
        UniqueConstraint("source_id", "version",
                         name="uq_audit_event_source_version"),
        UniqueConstraint("entity_id", "version",
                         name="uq_audit_event_entity_version"),
    )

class ClinicalRelationship(Base):
    __tablename__ = "clinical_relationship"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_entity_id: Mapped[str] = mapped_column(String(64), ForeignKey("clinical_entity.id", ondelete="CASCADE"), nullable=False, index=True)
    target_entity_id: Mapped[str] = mapped_column(String(64), ForeignKey("clinical_entity.id", ondelete="CASCADE"), nullable=False, index=True)
    relationship_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    review_status: Mapped[str] = mapped_column(String(20), nullable=False, default="DRAFT", index=True)
    source_id: Mapped[str | None] = mapped_column(String(128), ForeignKey("source_registry.source_id", ondelete="RESTRICT"), nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    # ---- X1D-AIV2-GOV2-C1 ------------------------------------------------
    # Semantic identity, derived from the endpoints: rel:<src>|<TYPE>|<tgt>.
    # Stable across evidence and version changes; changes only when the triple
    # does -- and a changed triple is a different relationship, not an edit.
    external_id: Mapped[str | None] = mapped_column(String(600), nullable=True, unique=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    governance_provenance: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    # The attestation seam. Only a verified xerbs-core human decision may ever
    # fill this; nothing in this service can. Its absence is what keeps a
    # GOV2-era object out of ranking.
    review_attestation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    submitter_subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_material_editor_subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    __table_args__ = (UniqueConstraint("source_entity_id", "target_entity_id", "relationship_type", name="uq_clinical_relationship"),)


class SafetyRule(Base):
    __tablename__ = "safety_rule"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    target_entity_id: Mapped[str] = mapped_column(String(64), ForeignKey("clinical_entity.id", ondelete="CASCADE"), nullable=False, index=True)
    rule_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    trigger_term: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="WARN")
    action: Mapped[str] = mapped_column(String(20), nullable=False, default="WARN")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    review_status: Mapped[str] = mapped_column(String(20), nullable=False, default="DRAFT", index=True)
    source_id: Mapped[str | None] = mapped_column(String(128), ForeignKey("source_registry.source_id", ondelete="RESTRICT"), nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    # ---- X1D-AIV2-GOV2-C1: same treatment, same reasons --------------------
    # A safety rule drives BLOCK, so the "REVIEWED at creation" defect mattered
    # here at least as much as it did for relationships.
    external_id: Mapped[str | None] = mapped_column(String(600), nullable=True, unique=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    governance_provenance: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    review_attestation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    submitter_subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_material_editor_subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ----------------------------------------------------------------------
# X1D-AIV2-GOV2-C1: durable provenance for the two object types that never
# had any.
# ----------------------------------------------------------------------
# Clinical entities already have clinical_entity_version, entity_source and
# review_event, and those are reused unchanged -- duplicating them would be
# scope creep. Relationships and safety rules had none of the three. Rather
# than build two near-identical sets, they share one, discriminated by
# object_type. The shape mirrors the entity tables so the vocabulary is the
# same one a reviewer already knows.


class GovernedObjectVersion(Base):
    """Append-only snapshot of one governed object at one version."""

    __tablename__ = "governed_object_version"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    object_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    object_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(600), nullable=True)
    snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (
        UniqueConstraint("object_type", "object_id", "version",
                         name="uq_governed_object_version"),
    )


class GovernedObjectSource(Base):
    """One evidence reference attached to one governed object.

    Mirrors entity_source, including `locator` -- "which section of the
    document" is part of what a reviewer read, and Production's relationship
    could not express it at all.
    """

    __tablename__ = "governed_object_source"
    object_type: Mapped[str] = mapped_column(String(40), primary_key=True)
    object_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(128), ForeignKey("source_registry.source_id", ondelete="RESTRICT"), primary_key=True)
    source_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    locator: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class GovernedObjectReviewEvent(Base):
    """One lifecycle transition of a governed object.

    `attestation_id` is the reference to the xerbs-core human decision. It
    stays NULL for every event this service can write on its own, which is
    every event it can write today.
    """

    __tablename__ = "governed_object_review_event"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    object_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    object_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(40), nullable=False)
    actor_subject: Mapped[str] = mapped_column(String(128), nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    to_status: Mapped[str] = mapped_column(String(20), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    attestation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (
        # X1D-AIV2-ATTEST1: per ACTION, not per version. Several transitions
        # can legitimately share a version -- CREATED and SUBMITTED_FOR_REVIEW
        # do -- and the version is bound by the attestation, so bumping it on
        # approval would invalidate the attestation that authorised it.
        UniqueConstraint("object_type", "object_id", "version", "action",
                         name="uq_governed_object_review_event_action"),
    )


class DiagnosticInterview(Base):
    __tablename__ = "diagnostic_interview"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="OPEN", index=True)
    intake_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    reasoning_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    current_turn: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    convergence_score: Mapped[float] = mapped_column(nullable=False, default=0.0)
    convergence_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class DiagnosticInterviewTurn(Base):
    __tablename__ = "diagnostic_interview_turn"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    interview_id: Mapped[str] = mapped_column(String(64), ForeignKey("diagnostic_interview.id", ondelete="CASCADE"), nullable=False, index=True)
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    answers: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    asked_questions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    reasoning_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (UniqueConstraint("interview_id", "turn_number", name="uq_diagnostic_interview_turn"),)


class GenerationRequest(Base):
    __tablename__ = "generation_request"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    correlation_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    external_request_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    organization_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING", index=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    response_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
