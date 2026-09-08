from enum import Enum
from typing import Literal

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
