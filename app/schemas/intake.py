from pydantic import BaseModel, Field
from app.schemas.safety import PatientContext


class RecommendationRequest(BaseModel):
    request_id: str | None = None
    text_input: str = Field(min_length=1, max_length=5000)
    symptoms: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    image_data: str | None = None
    image_filename: str | None = None
    language: str = "zh"
    patient_context: PatientContext = Field(default_factory=PatientContext)
    # X1D-LEGACYDIAG3.2.1: how many governed turns this case has already
    # completed, NOT counting this submission. Turn 1 arrives as 0.
    #
    # Derived by core from its own trace rows, never from the browser and
    # never from model output. It exists so the deterministic router can stop
    # interviewing after a bounded number of turns; it carries no clinical
    # meaning and grants nothing.
    interview_depth: int = Field(default=0, ge=0, le=50)
