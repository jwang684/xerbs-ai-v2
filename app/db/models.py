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
    entity_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    review_status: Mapped[str] = mapped_column(String(20), nullable=False, default="DRAFT", index=True)
    current_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class EntitySource(Base):
    __tablename__ = "entity_source"
    entity_id: Mapped[str] = mapped_column(String(64), ForeignKey("clinical_entity.id", ondelete="CASCADE"), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(128), ForeignKey("source_registry.source_id", ondelete="RESTRICT"), primary_key=True)


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


class AuditEvent(Base):
    __tablename__ = "audit_event"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    actor_id: Mapped[str] = mapped_column(String(255), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

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
