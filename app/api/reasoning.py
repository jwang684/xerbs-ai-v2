from fastapi import APIRouter
from app.schemas.intake import RecommendationRequest
from app.schemas.reasoning import ReasoningResponse
from app.services.reasoning.engine import DiagnosticReasoningEngine

router=APIRouter(prefix="/api/v1/reasoning",tags=["reasoning"])

@router.post("/analyze",response_model=ReasoningResponse)
async def analyze_reasoning(request: RecommendationRequest):
    return DiagnosticReasoningEngine().analyze(request)
