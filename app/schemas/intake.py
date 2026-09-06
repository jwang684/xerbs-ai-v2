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
