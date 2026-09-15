import asyncio
import json
import logging
from uuid import uuid4
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.integration import Base44GenerateRequest, Base44GenerationResponse
from app.services.integration.base44 import Base44GenerationService, IdempotencyConflictError, GenerationNotFoundError, GenerationRetryError
from app.services.llm.factory import ProviderConfigurationError, get_provider

logger = logging.getLogger(__name__)

# X1D-LEGACYDIAG4.1: a display fragment is capped so a runaway model cannot
# turn the relay into an unbounded firehose. Presentation only.
MAX_DELTA_CHARS = 400

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


@router.post('/generate/stream')
async def generate_stream(req: Base44GenerateRequest,
                          idempotency_key: str | None = Header(default=None, alias='Idempotency-Key'),
                          x_correlation_id: str | None = Header(default=None, alias='X-Correlation-ID')):
    """X1D-LEGACYDIAG4.1: the same generation, reported as it happens.

    LEGACYDIAG4 found that the legacy system's biggest advantage was streaming:
    its first content appeared in about a second, while Xerbs shows nothing for
    4 to 19 seconds and then everything at once. This endpoint closes that gap
    without moving where truth comes from.

    It is emphatically NOT a second inference path. It delegates to the very
    same Base44GenerationService.generate as /generate, so idempotency, the
    request hash, persistence, contract validation, provenance and telemetry
    are the identical code -- a replay still returns the stored row without
    calling a provider, because the short circuit lives in that service and not
    here. What this adds is a queue: display fragments are forwarded as they
    arrive, and the governed response is emitted once, at the end, only if the
    service actually returned one.

    Event contract:
        status  non-clinical progress
        delta   display-only text (the summary field, already consumer-visible)
        final   the governed Base44GenerationResponse, verbatim
        error   a safe failure description, never a provider body or key

    Disconnect: the generation task is deliberately not cancelled. Letting it
    finish keeps persistence and idempotency consistent, so a reconnect finds a
    completed generation rather than a half-written one. A partial stream is
    never a completed diagnosis: nothing downstream reads deltas, and `final`
    is emitted only after the service returns a validated response.
    """
    if not idempotency_key:
        raise HTTPException(400, detail="Idempotency-Key header is required")
    correlation_id = x_correlation_id or f"corr-{uuid4().hex[:16]}"
    try:
        service = _get_service()
    except ProviderConfigurationError as e:
        raise HTTPException(503, detail=str(e)) from e

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def sink(text: str) -> None:
        # Called from the provider read loop. Bounded, and never raises --
        # the provider wraps it too, but a display path should not rely on
        # someone else's try block.
        try:
            if text:
                queue.put_nowait(("delta", str(text)[:MAX_DELTA_CHARS]))
        except Exception:  # noqa: BLE001
            pass

    async def run() -> None:
        try:
            response = await service.generate(req, idempotency_key,
                                              correlation_id,
                                              on_display_text=sink)
            queue.put_nowait(("final", response))
        except IdempotencyConflictError as exc:
            queue.put_nowait(("error", ("IDEMPOTENCY_CONFLICT", str(exc))))
        except ProviderConfigurationError as exc:
            queue.put_nowait(("error", ("PROVIDER_NOT_CONFIGURED", str(exc))))
        except Exception as exc:  # noqa: BLE001
            # The message is class-level only. A provider body can echo part of
            # an API key, so it is logged upstream and never relayed.
            logger.warning("streamed generation failed (%s)", type(exc).__name__)
            queue.put_nowait(("error", ("GENERATION_FAILED",
                                        "Generation failed.")))
        finally:
            queue.put_nowait(None)

    task = loop.create_task(run())

    async def events():
        def frame(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield frame({"type": "status", "stage": "ACCEPTED",
                     "correlation_id": correlation_id})
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                kind, value = item
                if kind == "delta":
                    yield frame({"type": "delta", "text": value})
                elif kind == "final":
                    yield frame({"type": "final",
                                 "generation": json.loads(value.model_dump_json())})
                elif kind == "error":
                    code, detail = value
                    yield frame({"type": "error", "code": code,
                                 "detail": detail})
        finally:
            # Never cancel: see the docstring. The task owns persistence.
            if task.done() and task.exception() is not None:
                logger.warning("streamed generation task ended in %s",
                               type(task.exception()).__name__)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
