from uuid import uuid4
from fastapi import APIRouter, Header, HTTPException
from app.schemas.integration import Base44GenerateRequest, Base44GenerationResponse
from app.services.integration.base44 import Base44GenerationService, IdempotencyConflictError, GenerationNotFoundError, GenerationRetryError
from app.services.llm.factory import ProviderConfigurationError, get_provider

router=APIRouter(prefix="/api/v1/integrations/base44",tags=["base44-integration"])

# The provider is resolved on first use rather than at import.
#
# Resolving at import meant a misconfigured deployment could only fail by
# refusing to boot, taking /health down with it and leaving the operator
# guessing. Resolving here lets health report the problem and lets /generate
# answer 503 -- an explicit "no inference happened", which the caller can relay
# honestly. It must never degrade into a mock answer instead.
_service: Base44GenerationService | None = None


def _get_service() -> Base44GenerationService:
    global _service
    if _service is None:
        _service = Base44GenerationService(get_provider())
    return _service


@router.post('/generate',response_model=Base44GenerationResponse)
async def generate(req:Base44GenerateRequest,idempotency_key:str|None=Header(default=None,alias='Idempotency-Key'),x_correlation_id:str|None=Header(default=None,alias='X-Correlation-ID')):
    if not idempotency_key: raise HTTPException(400,detail="Idempotency-Key header is required")
    correlation_id=x_correlation_id or f"corr-{uuid4().hex[:16]}"
    try: service=_get_service()
    except ProviderConfigurationError as e: raise HTTPException(503,detail=str(e)) from e
    try: return await service.generate(req,idempotency_key,correlation_id)
    except IdempotencyConflictError as e: raise HTTPException(409,detail=str(e)) from e

@router.get('/generations/{generation_id}',response_model=Base44GenerationResponse)
def get_generation(generation_id:str):
    try: service=_get_service()
    except ProviderConfigurationError as e: raise HTTPException(503,detail=str(e)) from e
    try: return service.get(generation_id)
    except GenerationNotFoundError as e: raise HTTPException(404,detail=str(e)) from e

@router.post('/generations/{generation_id}/retry',response_model=Base44GenerationResponse)
async def retry_generation(generation_id:str):
    try: service=_get_service()
    except ProviderConfigurationError as e: raise HTTPException(503,detail=str(e)) from e
    try: return await service.retry(generation_id)
    except GenerationNotFoundError as e: raise HTTPException(404,detail=str(e)) from e
    except GenerationRetryError as e: raise HTTPException(422,detail=str(e)) from e
