from pydantic import BaseModel, Field


class FollowUpQuestion(BaseModel):
    id: int
    question: str
    question_type: str = "choice"
    options: list[str] | None = None
    category: str


class FollowUpRequest(BaseModel):
    request_id: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    initial_summary: str = ""
    language: str = "zh"


class FollowUpResponse(BaseModel):
    request_id: str | None = None
    questions: list[FollowUpQuestion]
    source: str
