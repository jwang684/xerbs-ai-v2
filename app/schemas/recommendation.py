from typing import Literal
from pydantic import BaseModel, Field
from app.schemas.safety import SafetyAssessment
from app.schemas.reasoning import ReasoningResponse


class PatternHypothesis(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class FormulaCandidate(BaseModel):
    formula_id: str | None = None
    name: str
    confidence: float = Field(ge=0, le=1)
    rationale: str
    ingredients: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    catalog_matches: list[dict] = Field(default_factory=list)
    safety_assessment: SafetyAssessment | None = None


class ModelProvenance(BaseModel):
    provider: str
    model: str
    prompt_version: str


class RecommendationResponse(BaseModel):
    request_id: str | None = None
    status: Literal["DRAFT_AI_RECOMMENDATION"] = "DRAFT_AI_RECOMMENDATION"
    summary: str
    pattern_hypotheses: list[PatternHypothesis] = Field(default_factory=list)
    formula_candidates: list[FormulaCandidate] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    model_confidence: float = Field(ge=0, le=1)
    provenance: ModelProvenance
    requires_practitioner_review: bool = True
    reasoning: ReasoningResponse | None = None
