from typing import Literal
from pydantic import BaseModel, Field
from app.schemas.intake import RecommendationRequest
from app.schemas.recommendation import RecommendationResponse

GenerationStatus = Literal["PENDING","SUCCEEDED","FAILED"]

class Base44GenerateRequest(BaseModel):
    organization_id: str | None = None
    request_id: str
    intake: RecommendationRequest

class Base44GenerationResponse(BaseModel):
    generation_id: str
    correlation_id: str
    request_id: str
    organization_id: str | None = None
    status: GenerationStatus
    retry_count: int = 0
    recommendation: RecommendationResponse | None = None
    error_code: str | None = None
    error_message: str | None = None
    trust_score_owned_by_ai_service: bool = False
    contract_version: str = "base44-xerbs-ai-v1"
