from fastapi import APIRouter, HTTPException
from app.schemas.interview import InterviewStartRequest, InterviewAnswerRequest, InterviewState
from app.schemas.recommendation import RecommendationResponse
from app.services.interview.engine import AdaptiveInterviewEngine, InterviewNotFoundError
from app.services.llm.factory import get_provider
from app.services.recommendation.assembler import RecommendationAssembler

router=APIRouter(prefix='/api/v1/interviews',tags=['adaptive-diagnostic-interview'])
engine=AdaptiveInterviewEngine(get_provider())

@router.post('/start',response_model=InterviewState)
async def start(req:InterviewStartRequest):
    return await engine.start(req.intake,req.max_questions_per_round)

@router.get('/{interview_id}',response_model=InterviewState)
def get_interview(interview_id:str):
    try: return engine.get(interview_id)
    except InterviewNotFoundError as e: raise HTTPException(404,detail=str(e)) from e

@router.post('/{interview_id}/answers',response_model=InterviewState)
async def answers(interview_id:str,req:InterviewAnswerRequest):
    try: return await engine.answer(interview_id,req)
    except InterviewNotFoundError as e: raise HTTPException(404,detail=str(e)) from e
    except ValueError as e: raise HTTPException(422,detail=str(e)) from e

@router.post('/{interview_id}/complete',response_model=InterviewState)
def complete(interview_id:str):
    try: return engine.complete(interview_id)
    except InterviewNotFoundError as e: raise HTTPException(404,detail=str(e)) from e

@router.post('/{interview_id}/recommendation',response_model=RecommendationResponse)
async def recommendation(interview_id:str):
    try:
        state=engine.get(interview_id)
    except InterviewNotFoundError as e:
        raise HTTPException(404,detail=str(e)) from e
    try:
        result=await RecommendationAssembler(get_provider()).generate(state.intake)
        engine.complete(interview_id)
        return result
    except RuntimeError as e:
        raise HTTPException(503,detail=str(e)) from e
