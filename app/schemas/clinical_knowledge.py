from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ReviewStatus(str, Enum):
    DRAFT = "DRAFT"
    IN_REVIEW = "IN_REVIEW"
    REVIEWED = "REVIEWED"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"


class SourceRef(BaseModel):
    source_id: str
    title: str
    citation: str | None = None
    url: str | None = None
    source_type: str = "UNSPECIFIED"


class SourceFieldConflict(BaseModel):
    """One canonical Source field whose submitted value differs from the persisted one."""

    field: Literal['title', 'citation', 'url', 'source_type']
    persisted: str | None = None
    submitted: str | None = None


class SourceConflictDetail(BaseModel):
    """Machine-readable body for a 409 canonical-source conflict.

    A source_id must never refer to conflicting source metadata. When ingestion
    submits an existing source_id with different canonical fields the batch is
    rejected whole; the persisted Source is never mutated or overwritten.
    """

    error: Literal['SOURCE_METADATA_CONFLICT'] = 'SOURCE_METADATA_CONFLICT'
    source_id: str
    message: str
    conflicting_fields: list[SourceFieldConflict] = Field(default_factory=list)


class SourceRecord(BaseModel):
    """One canonical source_registry row, returned verbatim.

    Exposes only fields the registry actually persists; nothing bibliographic
    is synthesised. Governance state is included so a review UI can render a
    Source without a second call.
    """

    source_id: str
    title: str
    citation: str | None = None
    url: str | None = None
    source_type: str
    review_status: ReviewStatus
    version: int
    created_by: str | None = None
    reviewed_by: str | None = None
    created_at: datetime
    updated_at: datetime | None = None


class SourceCreateRequest(BaseModel):
    """Create one canonical Source in DRAFT."""

    source_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=500)
    citation: str | None = None
    url: str | None = None
    source_type: str = "UNSPECIFIED"
    actor_id: str = Field(min_length=1)


class SourceAlreadyExistsDetail(BaseModel):
    """Machine-readable body for a 409 from the Source create endpoint."""

    error: Literal['SOURCE_ALREADY_EXISTS'] = 'SOURCE_ALREADY_EXISTS'
    source_id: str
    message: str


class SourceSubmitReviewRequest(BaseModel):
    submitted_by: str = Field(min_length=1)
    notes: str | None = None
    expected_version: int = Field(ge=1)


class SourceReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_CHANGES = "REQUEST_CHANGES"


class SourceReviewActionRequest(BaseModel):
    reviewer_id: str = Field(min_length=1)
    reviewer_role: str = "CLINICAL_REVIEWER"
    decision: SourceReviewDecision
    notes: str | None = None
    expected_version: int = Field(ge=1)


class SourceReviewEventRecord(BaseModel):
    """One persisted source_review_event row, returned verbatim."""

    event_id: str
    source_id: str
    action: str
    actor_id: str
    actor_role: str | None = None
    from_status: str | None = None
    to_status: str | None = None
    version: int
    notes: str | None = None
    created_at: datetime


class SourceReviewHistoryResponse(BaseModel):
    source_id: str
    count: int = 0
    results: list[SourceReviewEventRecord] = Field(default_factory=list)


class SourceAuditEventRecord(BaseModel):
    """One audit_event row carrying this Source's structured source_id."""

    event_id: str
    event_type: str
    source_id: str
    actor_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class SourceAuditHistoryResponse(BaseModel):
    source_id: str
    count: int = 0
    results: list[SourceAuditEventRecord] = Field(default_factory=list)


class SourceListResponse(BaseModel):
    """A page of canonical Sources.

    `count` is the total number of Sources matching the filters, before
    limit/offset are applied, so a caller can page through it.
    """

    count: int = 0
    limit: int = 0
    offset: int = 0
    results: list[SourceRecord] = Field(default_factory=list)


class SourceEntityRef(BaseModel):
    """A clinical entity that currently cites a canonical Source.

    Identity fields only, read from the live clinical_entity row.
    """

    entity_id: str
    entity_type: str
    name: str
    current_version: int
    review_status: ReviewStatus
    clinical_ranking_eligible: bool


class SourceEntitiesResponse(BaseModel):
    """Entities citing one Source through entity_source.

    Reverse lookup over entity_source only. clinical_relationship.source_id and
    safety_rule.source_id are deliberately NOT mixed in here.
    """

    source_id: str
    count: int = 0
    results: list[SourceEntityRef] = Field(default_factory=list)


class PatternRecord(BaseModel):
    pattern_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    indications: list[str] = Field(default_factory=list)
    exclusion_flags: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.DRAFT
    version: int = 1
    sources: list[SourceRef] = Field(default_factory=list)

    @property
    def clinical_ranking_eligible(self) -> bool:
        return self.review_status == ReviewStatus.REVIEWED and bool(self.sources)


class HerbRecord(BaseModel):
    herb_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    interaction_flags: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.DRAFT
    version: int = 1
    sources: list[SourceRef] = Field(default_factory=list)

    @property
    def clinical_ranking_eligible(self) -> bool:
        return self.review_status == ReviewStatus.REVIEWED and bool(self.sources)


class ClinicalFormulaRecord(BaseModel):
    formula_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    pattern_ids: list[str] = Field(default_factory=list)
    indications: list[str] = Field(default_factory=list)
    ingredients: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    interaction_flags: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.DRAFT
    version: int = 1
    sources: list[SourceRef] = Field(default_factory=list)
    migration_origin: str | None = None

    @property
    def clinical_ranking_eligible(self) -> bool:
        return self.review_status == ReviewStatus.REVIEWED and bool(self.sources)


class CorpusStats(BaseModel):
    patterns: int
    formulas: int
    herbs: int
    ranking_eligible_patterns: int
    ranking_eligible_formulas: int
    ranking_eligible_herbs: int


class CorpusSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    entity_types: list[str] = Field(default_factory=lambda: ["pattern", "formula", "herb"])
    reviewed_only: bool = False
    limit: int = Field(default=20, ge=1, le=100)
