from fastapi import APIRouter, HTTPException
from app.schemas.interview import InterviewStartRequest, InterviewAnswerRequest, InterviewState
from app.schemas.recommendation import RecommendationResponse
from app.services.interview.engine import AdaptiveInterviewEngine, InterviewNotFoundError
from app.services.llm.factory import ProviderConfigurationError, get_provider
from app.services.recommendation.assembler import RecommendationAssembler

router=APIRouter(prefix='/api/v1/interviews',tags=['adaptive-diagnostic-interview'])

# Resolved on first use, not at import: a provider misconfiguration must surface
# as an explicit 503 on the affected endpoint rather than preventing the service
# from booting at all. The engine keeps interview state, so it stays a singleton.
_engine: AdaptiveInterviewEngine | None = None


def _get_engine() -> AdaptiveInterviewEngine:
    global _engine
    if _engine is None:
        try:
            _engine = AdaptiveInterviewEngine(get_provider())
        except ProviderConfigurationError as e:
            raise HTTPException(503, detail=str(e)) from e
    return _engine

@router.post('/start',response_model=InterviewState)
async def start(req:InterviewStartRequest):
    return await _get_engine().start(req.intake,req.max_questions_per_round)

@router.get('/{interview_id}',response_model=InterviewState)
def get_interview(interview_id:str):
    try: return _get_engine().get(interview_id)
    except InterviewNotFoundError as e: raise HTTPException(404,detail=str(e)) from e

@router.post('/{interview_id}/answers',response_model=InterviewState)
async def answers(interview_id:str,req:InterviewAnswerRequest):
    try: return await _get_engine().answer(interview_id,req)
    except InterviewNotFoundError as e: raise HTTPException(404,detail=str(e)) from e
    except ValueError as e: raise HTTPException(422,detail=str(e)) from e

@router.post('/{interview_id}/complete',response_model=InterviewState)
def complete(interview_id:str):
    try: return _get_engine().complete(interview_id)
    except InterviewNotFoundError as e: raise HTTPException(404,detail=str(e)) from e

@router.post('/{interview_id}/recommendation',response_model=RecommendationResponse)
async def recommendation(interview_id:str):
    try:
        state=_get_engine().get(interview_id)
    except InterviewNotFoundError as e:
        raise HTTPException(404,detail=str(e)) from e
    try:
        result=await RecommendationAssembler(get_provider()).generate(state.intake)
        _get_engine().complete(interview_id)
        return result
    except RuntimeError as e:
        raise HTTPException(503,detail=str(e)) from e
