from fastapi import APIRouter, HTTPException, Query
from app.schemas.clinical_knowledge import CorpusSearchRequest, SourceConflictDetail, SourceEntitiesResponse, SourceListResponse, SourceRecord
from app.schemas.clinical_workflow import ClinicalEntityDetail, ClinicalEntityType, IngestionBatchRequest, ReviewActionRequest, SubmitForReviewRequest, RetireRequest, SupersedeRequest
from app.services.knowledge.persistent_clinical import PersistentClinicalStore, PersistentWorkflowError, SourceConflictError

router = APIRouter(prefix="/api/v1/knowledge/clinical", tags=["clinical-knowledge"])
store = PersistentClinicalStore()

@router.get("/stats")
def stats(): return store.stats()

@router.post("/search")
def search(request: CorpusSearchRequest):
    return {"query":request.query,"reviewed_only":request.reviewed_only,"results":store.search(request.query,request.entity_types,request.reviewed_only,request.limit)}

@router.get("/sources", response_model=SourceListResponse)
def list_sources(
    source_id:str|None=Query(default=None,min_length=1,description="Exact canonical source_id"),
    source_type:str|None=Query(default=None,min_length=1,description="Exact source_type"),
    query:str|None=Query(default=None,min_length=1,description="Case-insensitive substring over title, citation and url"),
    limit:int=Query(default=20,ge=1,le=100),
    offset:int=Query(default=0,ge=0),
):
    total,results=store.list_sources(source_id,source_type,query,limit,offset)
    return SourceListResponse(count=total,limit=limit,offset=offset,results=results)

@router.get("/sources/{source_id}", response_model=SourceRecord)
def get_source(source_id:str):
    try: return store.get_source(source_id)
    except PersistentWorkflowError as e: raise HTTPException(404,detail=str(e)) from e

@router.get("/sources/{source_id}/entities", response_model=SourceEntitiesResponse)
def source_entities(source_id:str):
    try: results=store.get_source_entities(source_id)
    except PersistentWorkflowError as e: raise HTTPException(404,detail=str(e)) from e
    return SourceEntitiesResponse(source_id=source_id,count=len(results),results=results)

@router.get("/governance")
def governance():
    return {"policy":"REVIEW_GATED_PERSISTENT_CLINICAL_CORPUS","clinical_ranking_eligibility":"review_status=REVIEWED AND at_least_one_source","draft_records_may_rank":False,"legacy_tse_symptom_metadata_may_rank":False,"legacy_formula_fixtures_may_rank":False,"trust_score_owned_by_ai_service":False,"ingestion_can_set_review_status":False,"approval_requires_explicit_source":True,"version_history_append_only":True,"persistence":"SQLALCHEMY_DATABASE","optimistic_concurrency":True,"reviewer_role_enforced":True,"retire_and_supersede_supported":True}

@router.post("/ingest", responses={409: {"model": SourceConflictDetail, "description": "Canonical source metadata conflict"}})
def ingest(request: IngestionBatchRequest):
    try: return store.ingest(request)
    except SourceConflictError as e: raise HTTPException(409,detail=e.to_detail().model_dump(mode="json")) from e
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

@router.get("/entities/{entity_type}/{entity_id}", response_model=ClinicalEntityDetail)
def entity_detail(entity_type:ClinicalEntityType,entity_id:str):
    try: return store.get_entity_detail(entity_type,entity_id)
    except PersistentWorkflowError as e: raise HTTPException(404,detail=str(e)) from e

@router.get("/entities/{entity_type}/{entity_id}/history")
def history(entity_type:ClinicalEntityType,entity_id:str):
    try: return {"results":store.get_history(entity_type,entity_id)}
    except PersistentWorkflowError as e: raise HTTPException(404,detail=str(e)) from e

@router.get("/audit")
def audit(entity_type:ClinicalEntityType|None=None,entity_id:str|None=None): return {"results":store.audit(entity_type,entity_id)}
