from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.clinical_knowledge import SourceRef


class ClinicalEntityType(str, Enum):
    PATTERN = "pattern"
    FORMULA = "formula"
    HERB = "herb"


class ReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_CHANGES = "REQUEST_CHANGES"


class IngestionItem(BaseModel):
    entity_type: ClinicalEntityType
    external_id: str | None = None
    payload: dict[str, Any]
    sources: list[SourceRef] = Field(default_factory=list)


class IngestionBatchRequest(BaseModel):
    submitted_by: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    items: list[IngestionItem] = Field(min_length=1, max_length=500)


class ReviewActionRequest(BaseModel):
    reviewer_id: str = Field(min_length=1)
    reviewer_role: str = "CLINICAL_REVIEWER"
    decision: ReviewDecision
    notes: str | None = None
    expected_version: int | None = Field(default=None, ge=1)


class SubmitForReviewRequest(BaseModel):
    submitted_by: str = Field(min_length=1)
    notes: str | None = None
    expected_version: int | None = Field(default=None, ge=1)


class RetireRequest(BaseModel):
    actor_id: str = Field(min_length=1)
    actor_role: str = "CLINICAL_REVIEWER"
    notes: str | None = None
    expected_version: int | None = Field(default=None, ge=1)


class SupersedeRequest(BaseModel):
    actor_id: str = Field(min_length=1)
    actor_role: str = "CLINICAL_ADMIN"
    superseded_by_id: str = Field(min_length=1)
    expected_version: int | None = Field(default=None, ge=1)


class WorkflowEvent(BaseModel):
    event_id: str
    entity_type: ClinicalEntityType
    entity_id: str
    action: str
    actor_id: str
    from_status: str | None = None
    to_status: str | None = None
    version: int
    notes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IngestionBatchResult(BaseModel):
    batch_id: str
    submitted_by: str
    source_label: str
    created_entity_ids: list[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
