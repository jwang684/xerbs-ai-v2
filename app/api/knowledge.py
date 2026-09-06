from fastapi import APIRouter

from app.schemas.knowledge import TSESearchRequest, TSESymptomRetrievalRequest
from app.services.knowledge.tse_repository import TSEKnowledgeRepository

router = APIRouter(prefix="/api/v1/knowledge/tse", tags=["knowledge"])
repo = TSEKnowledgeRepository()


@router.get("/stats")
def stats():
    return repo.stats()


@router.post("/search")
def search(request: TSESearchRequest):
    return {"query": request.query, "results": repo.search_catalog(request.query, limit=request.limit)}


@router.post("/symptom-metadata")
def symptom_metadata(request: TSESymptomRetrievalRequest):
    return {
        "symptoms": request.symptoms,
        "policy": "RETRIEVAL_ONLY_NOT_CLINICAL_RECOMMENDATION",
        "results": repo.retrieve_symptom_metadata(request.symptoms, limit=request.limit),
    }
