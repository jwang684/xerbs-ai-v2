from fastapi import APIRouter, HTTPException
from app.schemas.intake import RecommendationRequest
from app.schemas.recommendation import RecommendationResponse
from app.services.llm.factory import get_provider
from app.services.recommendation.assembler import RecommendationAssembler

router = APIRouter(prefix="/api/v1/recommendations", tags=["recommendations"])


@router.post("/generate", response_model=RecommendationResponse)
async def generate_recommendation(request: RecommendationRequest):
    try:
        service = RecommendationAssembler(get_provider())
        return await service.generate(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
