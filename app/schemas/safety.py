from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

class PatientContext(BaseModel):
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    diagnoses: list[str] = Field(default_factory=list)
    pregnancy: bool | None = None
    age: int | None = Field(default=None, ge=0, le=130)

class SafetyFinding(BaseModel):
    rule_id: str | None = None
    target_entity_id: str | None = None
    rule_type: str
    severity: Literal['INFO','WARN','HIGH','CRITICAL'] = 'WARN'
    action: Literal['INFO','WARN','BLOCK'] = 'WARN'
    trigger_term: str
    message: str

class SafetyAssessment(BaseModel):
    risk_level: Literal['NONE','LOW','MODERATE','HIGH','CRITICAL']
    eligible_for_selection: bool
    findings: list[SafetyFinding] = Field(default_factory=list)

class SafetyScreenRequest(BaseModel):
    formula_id: str | None = None
    formula_name: str
    ingredients: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    patient_context: PatientContext = Field(default_factory=PatientContext)

class RelationshipCreateRequest(BaseModel):
    source_entity_id: str
    target_entity_id: str
    relationship_type: Literal['PATTERN_FORMULA','FORMULA_HERB']
    source_id: str | None = None
    actor_id: str
    actor_role: str = 'CLINICAL_REVIEWER'

class SafetyRuleCreateRequest(BaseModel):
    target_entity_id: str
    rule_type: Literal['CONTRAINDICATION','DRUG_INTERACTION','ALLERGY','PREGNANCY','AGE','DIAGNOSIS']
    trigger_term: str
    severity: Literal['INFO','WARN','HIGH','CRITICAL'] = 'WARN'
    action: Literal['INFO','WARN','BLOCK'] = 'WARN'
    message: str
    source_id: str | None = None
    actor_id: str
    actor_role: str = 'CLINICAL_REVIEWER'


class RelationshipType(str, Enum):
    PATTERN_FORMULA = 'PATTERN_FORMULA'
    FORMULA_HERB = 'FORMULA_HERB'


class RelationshipDirection(str, Enum):
    """Direction of a relationship relative to the queried entity."""

    OUTGOING = 'outgoing'
    INCOMING = 'incoming'
    BOTH = 'both'


class ClinicalRelationshipRecord(BaseModel):
    """One persisted clinical_relationship row, returned verbatim.

    Entity type/name are read from the referenced clinical_entity rows so a
    detail view can render the counterpart without a second call. Nothing is
    inferred from names, indications, TSE data or model output.
    """

    id: str
    source_entity_id: str
    source_entity_type: str | None = None
    source_entity_name: str | None = None
    target_entity_id: str
    target_entity_type: str | None = None
    target_entity_name: str | None = None
    relationship_type: RelationshipType
    review_status: str
    source_id: str | None = None
    created_by: str
    created_at: datetime
    direction: Literal['outgoing', 'incoming']


class RelationshipQueryResponse(BaseModel):
    entity_id: str
    direction: RelationshipDirection
    relationship_type: RelationshipType | None = None
    count: int = 0
    results: list[ClinicalRelationshipRecord] = Field(default_factory=list)
