from pydantic import BaseModel, Field

class StructuredSymptom(BaseModel):
    name: str
    source: str = "explicit"

class EvidenceItem(BaseModel):
    text: str
    source: str = "patient_report"

class PatternAssessment(BaseModel):
    pattern_id: str | None = None
    name: str
    model_confidence: float = Field(ge=0, le=1)
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    contradictions: list[EvidenceItem] = Field(default_factory=list)
    corpus_match: bool = False

class MissingInformation(BaseModel):
    field: str
    reason: str
    priority: str = "MEDIUM"

class ConvergenceMetrics(BaseModel):
    score: float = Field(ge=0, le=1)
    evidence_sufficiency: float = Field(ge=0, le=1)
    pattern_stability: float = Field(ge=0, le=1)
    verified_pattern_strength: float = Field(ge=0, le=1)
    contradiction_penalty: float = Field(ge=0, le=1)
    stable_pattern_names: list[str] = Field(default_factory=list)
    changed_pattern_names: list[str] = Field(default_factory=list)
    contradiction_count: int = Field(ge=0, default=0)
    rationale: list[str] = Field(default_factory=list)

class ReasoningResponse(BaseModel):
    request_id: str | None = None
    structured_symptoms: list[StructuredSymptom] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    followup_questions: list[str] = Field(default_factory=list)
    pattern_assessments: list[PatternAssessment] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    ready_for_formula_retrieval: bool = False
    convergence: ConvergenceMetrics | None = None
