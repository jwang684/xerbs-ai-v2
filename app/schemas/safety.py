from pydantic import BaseModel, Field
from typing import Literal

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
