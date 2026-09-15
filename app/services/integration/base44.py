from __future__ import annotations
import hashlib, json, time
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import select
from app.db.models import GenerationRequest
from app.db.session import get_session_factory
from app.schemas.integration import Base44GenerateRequest, Base44GenerationResponse
from app.services.recommendation.assembler import RecommendationAssembler
from app.services.llm.provider import LLMProvider
from app.services.telemetry.provider_usage import record_provider_usage
from app.services.telemetry.clarification_rejection import record_rejections

class IdempotencyConflictError(ValueError): pass
class GenerationNotFoundError(ValueError): pass
class GenerationRetryError(ValueError): pass

class Base44GenerationService:
    def __init__(self, provider: LLMProvider):
        self.Session=get_session_factory(); self.provider=provider

    @staticmethod
    def _hash(req:Base44GenerateRequest)->str:
        raw=json.dumps(req.model_dump(mode="json"),sort_keys=True,ensure_ascii=False,separators=(",",":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def _response(self,row:GenerationRequest)->Base44GenerationResponse:
        rec=row.response_snapshot
        return Base44GenerationResponse(
            generation_id=row.id,correlation_id=row.correlation_id,request_id=row.external_request_id or "",
            organization_id=row.organization_id,status=row.status,retry_count=row.retry_count,
            recommendation=rec,error_code=row.error_code,error_message=row.error_message,
        )

    async def generate(self,req:Base44GenerateRequest,idempotency_key:str,correlation_id:str,on_display_text=None)->Base44GenerationResponse:
        request_hash=self._hash(req)
        with self.Session.begin() as s:
            existing=s.scalar(select(GenerationRequest).where(GenerationRequest.idempotency_key==idempotency_key))
            if existing:
                if existing.request_hash!=request_hash: raise IdempotencyConflictError("Idempotency key was already used with a different payload")
                return self._response(existing)
            row=GenerationRequest(
                id=f"gen-{uuid4().hex[:16]}",idempotency_key=idempotency_key,correlation_id=correlation_id,
                external_request_id=req.request_id,organization_id=req.organization_id,status="PENDING",
                request_hash=request_hash,request_snapshot=req.model_dump(mode="json"),created_at=datetime.now(timezone.utc),updated_at=datetime.now(timezone.utc),
            )
            s.add(row)
            generation_id=row.id
        return await self._execute(generation_id,on_display_text)

    async def _execute(self,generation_id:str,on_display_text=None)->Base44GenerationResponse:
        with self.Session() as s:
            row=s.get(GenerationRequest,generation_id)
            if row is None: raise GenerationNotFoundError("Generation not found")
            req=Base44GenerateRequest(**row.request_snapshot)
            row_correlation_id=row.correlation_id
        try:
            intake=req.intake.model_copy(update={"request_id": req.request_id})
            # X1D-TELEMETRY1: capture the raw provider result for accounting.
            # It is recorded after the clinical result is safely persisted, so
            # a telemetry failure can never cost a valid diagnosis.
            captured={}
            started=time.monotonic()
            recommendation=await RecommendationAssembler(self.provider).generate(
                intake, on_provider_result=lambda r: captured.__setitem__("result", r),
                on_clarification_outcomes=lambda o, d: captured.__setitem__(
                    "clarification", (o, d)),
                on_display_text=on_display_text)
            generation_latency_ms=(time.monotonic()-started)*1000.0
            with self.Session.begin() as s:
                row=s.get(GenerationRequest,generation_id)
                row.status="SUCCEEDED"; row.response_snapshot=recommendation.model_dump(mode="json")
                row.error_code=None; row.error_message=None; row.updated_at=datetime.now(timezone.utc)
                response=self._response(row)
            if "result" in captured:
                record_provider_usage(
                    generation_id=generation_id,
                    correlation_id=row_correlation_id,
                    result=captured["result"],
                    generation_latency_ms=generation_latency_ms,
                )
            if "clarification" in captured:
                # X1D-CLARIFY3: why the adaptive proposals were accepted or
                # refused. Fail-open, and recorded only after the clinical
                # result is safely persisted.
                outcomes, discarded = captured["clarification"]
                record_rejections(
                    generation_id=generation_id,
                    correlation_id=row_correlation_id,
                    outcomes=outcomes,
                    malformed_count=discarded,
                )
            return response
        except Exception as exc:
            with self.Session.begin() as s:
                row=s.get(GenerationRequest,generation_id)
                row.status="FAILED"; row.error_code="GENERATION_FAILED"; row.error_message=str(exc)[:1000]; row.updated_at=datetime.now(timezone.utc)
                response=self._response(row)
            return response

    def get(self,generation_id:str)->Base44GenerationResponse:
        with self.Session() as s:
            row=s.get(GenerationRequest,generation_id)
            if row is None: raise GenerationNotFoundError("Generation not found")
            return self._response(row)

    async def retry(self,generation_id:str)->Base44GenerationResponse:
        with self.Session.begin() as s:
            row=s.get(GenerationRequest,generation_id)
            if row is None: raise GenerationNotFoundError("Generation not found")
            if row.status!="FAILED": raise GenerationRetryError("Only FAILED generations can be retried")
            row.status="PENDING"; row.retry_count+=1; row.error_code=None; row.error_message=None; row.updated_at=datetime.now(timezone.utc)
        return await self._execute(generation_id)
