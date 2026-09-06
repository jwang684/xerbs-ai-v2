from uuid import uuid4
from fastapi import APIRouter, Header, HTTPException
from app.schemas.integration import Base44GenerateRequest, Base44GenerationResponse
from app.services.integration.base44 import Base44GenerationService, IdempotencyConflictError, GenerationNotFoundError, GenerationRetryError
from app.services.llm.factory import get_provider

router=APIRouter(prefix="/api/v1/integrations/base44",tags=["base44-integration"])
service=Base44GenerationService(get_provider())

@router.post('/generate',response_model=Base44GenerationResponse)
async def generate(req:Base44GenerateRequest,idempotency_key:str|None=Header(default=None,alias='Idempotency-Key'),x_correlation_id:str|None=Header(default=None,alias='X-Correlation-ID')):
    if not idempotency_key: raise HTTPException(400,detail="Idempotency-Key header is required")
    correlation_id=x_correlation_id or f"corr-{uuid4().hex[:16]}"
    try: return await service.generate(req,idempotency_key,correlation_id)
    except IdempotencyConflictError as e: raise HTTPException(409,detail=str(e)) from e

@router.get('/generations/{generation_id}',response_model=Base44GenerationResponse)
def get_generation(generation_id:str):
    try: return service.get(generation_id)
    except GenerationNotFoundError as e: raise HTTPException(404,detail=str(e)) from e

@router.post('/generations/{generation_id}/retry',response_model=Base44GenerationResponse)
async def retry_generation(generation_id:str):
    try: return await service.retry(generation_id)
    except GenerationNotFoundError as e: raise HTTPException(404,detail=str(e)) from e
    except GenerationRetryError as e: raise HTTPException(422,detail=str(e)) from e
