from pydantic import BaseModel, Field


class TSESearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=20, ge=1, le=100)


class TSESymptomRetrievalRequest(BaseModel):
    symptoms: list[str] = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
