from typing import Literal

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

class ClarificationQuestion(BaseModel):
    """A model-proposed question that passed deterministic validation.

    Carried separately from followup_questions, which are the deterministic
    engine's own and take precedence. No rationale prose is carried: the field
    identifier is the whole justification, and reasoning text would be
    chain-of-thought by another name.
    """

    field: str
    question: str
    answer_type: str = "short_text"
    priority: str = "medium"
    choices: list[str] = Field(default_factory=list)
    source: str = "xerbs-ai-v2-adaptive"


class ReasoningResponse(BaseModel):
    request_id: str | None = None
    structured_symptoms: list[StructuredSymptom] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    followup_questions: list[str] = Field(default_factory=list)
    pattern_assessments: list[PatternAssessment] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    ready_for_formula_retrieval: bool = False
    convergence: ConvergenceMetrics | None = None
    # X1D-CLARIFY1: validated adaptive questions. Optional and defaulted, so
    # every existing caller and stored snapshot stays valid.
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    # X1D-LEGACYDIAG2: model reasoning, retained with no authority.
    clinical_reasoning: "ClinicalReasoningEnvelope | None" = None
    # X1D-CLARIFY2: whether anything material is still worth asking.
    #
    # This governs question emission only. It is NOT a clinical readiness
    # signal and must never be read as one -- ready_for_formula_retrieval above
    # is the only field that gates corpus retrieval, and it is computed from
    # missing_information and corpus matching exactly as before.
    clarification_sufficiency: str | None = None
    # X1D-CLARIFY2: 十问歌-derived coverage of the accumulated patient facts,
    # as domain -> KNOWN / PARTIAL / UNKNOWN / NOT_RELEVANT. Operational
    # observation: it records which domains the patient has spoken to, and
    # never asserts a value for one they have not.
    clinical_coverage: dict = Field(default_factory=dict)


# ======================================================================
# X1D-LEGACYDIAG2: the clinical reasoning envelope
# ======================================================================
#
# The legacy system's perceived quality came mostly from what its prompt
# demanded, not from a better model. Its output contract required a primary
# and a secondary pattern each with cited evidence, eight-principle
# differentiation, pathogenesis, treatment principle and formula rationale,
# down to dosage ranges and decoction method. The current contract asks for
# five fields, so most of that reasoning was never requested -- and the
# formula hypotheses that were produced got discarded rather than kept.
#
# This envelope asks for the reasoning again and keeps it. What it does not do
# is give any of it authority. The legacy system's failure was not that it
# reasoned broadly; it was that free-form prose became, via regex, a
# purchasable formula with no verification anywhere in between.
#
# So every clinical object here carries status=MODEL_HYPOTHESIS and nothing
# else. These are evidence targets for later verification, not facts. They
# never enter formula_candidates, never reach SafetyEngine, and never touch
# purchasability -- those paths are unchanged and untouched by this phase.

MODEL_HYPOTHESIS = "MODEL_HYPOTHESIS"


class HypothesisBase(BaseModel):
    """Marks provenance on every clinical object the model produces.

    Literal rather than a plain string: nothing downstream can widen this to a
    verified state without changing the type, which makes the escalation
    visible in review instead of silent.
    """

    status: Literal["MODEL_HYPOTHESIS"] = MODEL_HYPOTHESIS


class PatternHypothesisDetail(HypothesisBase):
    name: str
    role: Literal["primary", "secondary"] = "secondary"
    confidence: float = Field(default=0.0, ge=0, le=1)
    supporting_findings: list[str] = Field(default_factory=list)
    contradicting_findings: list[str] = Field(default_factory=list)


class IngredientHypothesis(HypothesisBase):
    name: str
    # Free text on purpose: the legacy system emitted ranges such as "6-9g",
    # and parsing that into a number would manufacture a precision the model
    # never expressed. Nothing computes with this field.
    dosage: str | None = None
    role: str | None = None


class FormulaHypothesis(HypothesisBase):
    name: str
    confidence: float = Field(default=0.0, ge=0, le=1)
    rationale: str | None = None
    ingredients: list[IngredientHypothesis] = Field(default_factory=list)
    administration: str | None = None
    contraindications: list[str] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list)


class ClinicalReasoningEnvelope(BaseModel):
    """Richer TCM reasoning, retained internally, carrying no authority.

    Every field is optional and defaulted. A model that omits a section
    produces a valid envelope rather than a parse failure, which is what keeps
    a richer contract from making the clinical path less reliable -- and what
    lets the model leave a field out instead of inventing it.
    """

    clinical_summary: str | None = None
    tcm_diagnosis_hypotheses: list[str] = Field(default_factory=list)
    eight_principle_differentiation: dict[str, str] = Field(default_factory=dict)
    pattern_hypotheses: list[PatternHypothesisDetail] = Field(default_factory=list)
    pathogenesis: str | None = None
    treatment_principle: str | None = None
    formula_hypotheses: list[FormulaHypothesis] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    model_confidence: float = Field(default=0.0, ge=0, le=1)

    def is_empty(self) -> bool:
        return not any([
            self.clinical_summary, self.tcm_diagnosis_hypotheses,
            self.eight_principle_differentiation, self.pattern_hypotheses,
            self.pathogenesis, self.treatment_principle,
            self.formula_hypotheses, self.missing_information,
        ])


# ======================================================================
# X1D-LEGACYDIAG3.2: the interview contract
# ======================================================================
#
# A patient mid-interview needs the next one to three questions. The full
# contract asks the model for a summary, pattern hypotheses, formula
# candidates, and the whole ClinicalReasoningEnvelope -- pathogenesis,
# treatment principle, formula hypotheses with ingredients, dosages,
# administration and contraindications -- and then shows the patient two short
# questions. Measured on staging that is ~2000-2600 completion tokens to
# deliver perhaps a hundred, and completion tokens are ~5.8ms each.
#
# So this is a deliberately small contract for the interview turns. It is NOT
# ClinicalReasoningEnvelope and must never become one: no formula candidates,
# no dosages, no herb composition, no treatment plan, no final syndrome
# diagnosis, no safety clearance, no eligibility of any kind. Those fields are
# absent by construction rather than filtered later, which is what makes the
# absence testable.
#
# working_hypotheses are reasoning aids for choosing the next question. They
# are not diagnoses, never become REVIEWED knowledge, never create
# relationships, and are not persisted as governed knowledge.


class InterviewHypothesis(BaseModel):
    """A provisional reading the next question is meant to separate.

    Carries findings for and against so CLARIFY2 ranking can tell which
    domains actually discriminate -- the same signal the envelope supplies on
    a full turn.
    """

    name: str
    confidence: float = Field(default=0.0, ge=0, le=1)
    supporting_findings: list[str] = Field(default_factory=list)
    contradicting_findings: list[str] = Field(default_factory=list)


class InterviewReasoning(BaseModel):
    """What the model returns on an interview turn, and nothing more."""

    interview_summary: str | None = None
    working_hypotheses: list[InterviewHypothesis] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    # Advisory only. Deterministic governance decides whether a case may
    # proceed; the model never gets to declare a case finished.
    information_sufficient: bool = False

    @property
    def pattern_hypotheses(self) -> list[InterviewHypothesis]:
        """Duck-typed alias so CLARIFY2 signal extraction reads this
        unchanged. extract_differential_signals uses getattr for exactly this
        reason; giving it the shape it already knows avoids a second code path
        through the ranking logic."""
        return self.working_hypotheses

    def is_empty(self) -> bool:
        return not any([self.interview_summary, self.working_hypotheses,
                        self.missing_information])
