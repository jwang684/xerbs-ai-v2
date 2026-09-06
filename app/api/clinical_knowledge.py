from fastapi import APIRouter, HTTPException
from app.schemas.clinical_knowledge import CorpusSearchRequest
from app.schemas.clinical_workflow import ClinicalEntityType, IngestionBatchRequest, ReviewActionRequest, SubmitForReviewRequest, RetireRequest, SupersedeRequest
from app.services.knowledge.persistent_clinical import PersistentClinicalStore, PersistentWorkflowError

router = APIRouter(prefix="/api/v1/knowledge/clinical", tags=["clinical-knowledge"])
store = PersistentClinicalStore()

@router.get("/stats")
def stats(): return store.stats()

@router.post("/search")
def search(request: CorpusSearchRequest):
    return {"query":request.query,"reviewed_only":request.reviewed_only,"results":store.search(request.query,request.entity_types,request.reviewed_only,request.limit)}

@router.get("/governance")
def governance():
    return {"policy":"REVIEW_GATED_PERSISTENT_CLINICAL_CORPUS","clinical_ranking_eligibility":"review_status=REVIEWED AND at_least_one_source","draft_records_may_rank":False,"legacy_tse_symptom_metadata_may_rank":False,"legacy_formula_fixtures_may_rank":False,"trust_score_owned_by_ai_service":False,"ingestion_can_set_review_status":False,"approval_requires_explicit_source":True,"version_history_append_only":True,"persistence":"SQLALCHEMY_DATABASE","optimistic_concurrency":True,"reviewer_role_enforced":True,"retire_and_supersede_supported":True}

@router.post("/ingest")
def ingest(request: IngestionBatchRequest):
    try: return store.ingest(request)
    except PersistentWorkflowError as e: raise HTTPException(422,detail=str(e)) from e

@router.get("/batches/{batch_id}")
def batch(batch_id:str):
    try: return store.get_batch(batch_id)
    except PersistentWorkflowError as e: raise HTTPException(404,detail=str(e)) from e

@router.post("/entities/{entity_type}/{entity_id}/submit-review")
def submit_review(entity_type:ClinicalEntityType,entity_id:str,request:SubmitForReviewRequest):
    try: return store.submit_for_review(entity_type,entity_id,request.submitted_by,request.notes,request.expected_version)
    except PersistentWorkflowError as e: raise HTTPException(409,detail=str(e)) from e

@router.post("/entities/{entity_type}/{entity_id}/review")
def review(entity_type:ClinicalEntityType,entity_id:str,request:ReviewActionRequest):
    try: return store.review(entity_type,entity_id,request)
    except PersistentWorkflowError as e: raise HTTPException(409,detail=str(e)) from e

@router.post("/entities/{entity_type}/{entity_id}/retire")
def retire(entity_type:ClinicalEntityType,entity_id:str,request:RetireRequest):
    try: return store.retire(entity_type,entity_id,request.actor_id,request.actor_role,request.notes,request.expected_version)
    except PersistentWorkflowError as e: raise HTTPException(409,detail=str(e)) from e

@router.post("/entities/{entity_type}/{entity_id}/supersede")
def supersede(entity_type:ClinicalEntityType,entity_id:str,request:SupersedeRequest):
    try: return store.supersede(entity_type,entity_id,request.superseded_by_id,request.actor_id,request.actor_role,request.expected_version)
    except PersistentWorkflowError as e: raise HTTPException(409,detail=str(e)) from e

@router.get("/entities/{entity_type}/{entity_id}/history")
def history(entity_type:ClinicalEntityType,entity_id:str):
    try: return {"results":store.get_history(entity_type,entity_id)}
    except PersistentWorkflowError as e: raise HTTPException(404,detail=str(e)) from e

@router.get("/audit")
def audit(entity_type:ClinicalEntityType|None=None,entity_id:str|None=None): return {"results":store.audit(entity_type,entity_id)}
