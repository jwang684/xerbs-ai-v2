from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field
from app.schemas.intake import RecommendationRequest
from app.schemas.reasoning import ReasoningResponse, ConvergenceMetrics

InterviewStatus = Literal["OPEN", "READY_FOR_RECOMMENDATION", "COMPLETED"]

class InterviewStartRequest(BaseModel):
    intake: RecommendationRequest
    max_questions_per_round: int = Field(default=3, ge=1, le=5)

class InterviewAnswer(BaseModel):
    question_id: str
    answer: str = Field(min_length=1, max_length=2000)

class InterviewAnswerRequest(BaseModel):
    answers: list[InterviewAnswer] = Field(min_length=1, max_length=5)
    max_questions_per_round: int = Field(default=3, ge=1, le=5)

class InterviewQuestion(BaseModel):
    question_id: str
    field: str
    question: str
    priority: str
    information_gain_score: float = Field(ge=0, le=1)

class InterviewTurn(BaseModel):
    turn_number: int
    answers: list[InterviewAnswer] = Field(default_factory=list)
    asked_questions: list[InterviewQuestion] = Field(default_factory=list)
    reasoning: ReasoningResponse
    created_at: datetime

class InterviewState(BaseModel):
    interview_id: str
    status: InterviewStatus
    intake: RecommendationRequest
    reasoning: ReasoningResponse
    next_questions: list[InterviewQuestion] = Field(default_factory=list)
    turn_count: int
    convergence_score: float = Field(ge=0, le=1)
    convergence: ConvergenceMetrics | None = None
    created_at: datetime
    updated_at: datetime
