from fastapi import APIRouter
from app.schemas.followup import FollowUpRequest, FollowUpResponse
from app.services.followup.defaults import default_tcm_questions

router = APIRouter(prefix="/api/v1/followups", tags=["followups"])


@router.post("/generate", response_model=FollowUpResponse)
async def generate_followups(request: FollowUpRequest):
    # Phase 2 extraction: preserve the legacy Ten Questions fallback as a
    # deterministic, testable baseline. Personalized LLM questions come next.
    return FollowUpResponse(
        request_id=request.request_id,
        questions=default_tcm_questions(request.language),
        source="legacy-ten-questions-v1",
    )
