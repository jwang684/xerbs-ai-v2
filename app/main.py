import logging

from fastapi import Depends, FastAPI
from app.api.health import router as health_router
from app.api.recommendations import router as recommendation_router
from app.api.followups import router as followup_router
from app.api.knowledge import router as knowledge_router
from app.api.clinical_knowledge import router as clinical_knowledge_router
from app.api.safety import router as safety_router
from app.api.reasoning import router as reasoning_router
from app.api.interviews import router as interview_router
from app.api.base44 import router as base44_router
from app.core.config import get_settings
from app.core.service_auth import require_service_auth
from app.db.session import init_db
from app.services.knowledge.persistent_seed import seed_legacy_formula_fixtures

settings=get_settings()

# X1D-TELEMETRY1: give application loggers a handler.
#
# uvicorn configures only its own loggers, never the root, so anything this
# application logged below WARNING was silently dropped -- including the
# per-call token/latency/cost record. The rows were being written to
# audit_event correctly; there was simply no way to see them, and
# DATABASE_URL is private, so "no log line" was indistinguishable from
# "no telemetry". basicConfig is a no-op if a handler is already installed.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

init_db(); seed_legacy_formula_fixtures()
app=FastAPI(title=settings.app_name,version=settings.app_version)

# Service authentication (X1D-E2E1).
#
# Attached once here rather than on each router, so a router added later
# cannot be published unprotected by omission. Health is deliberately the
# only unauthenticated router: it must stay reachable for readiness probes
# and returns no sensitive information.
#
# Every other router — including clinical reasoning, the corpus, and safety —
# requires the service credential. This service has no end-user identity and
# must never acquire one; the only caller is xerbs-core.
SERVICE_AUTH = [Depends(require_service_auth)]

app.include_router(health_router)
app.include_router(recommendation_router, dependencies=SERVICE_AUTH)
app.include_router(followup_router, dependencies=SERVICE_AUTH)
app.include_router(knowledge_router, dependencies=SERVICE_AUTH)
app.include_router(clinical_knowledge_router, dependencies=SERVICE_AUTH)
app.include_router(safety_router, dependencies=SERVICE_AUTH)
app.include_router(reasoning_router, dependencies=SERVICE_AUTH)
app.include_router(interview_router, dependencies=SERVICE_AUTH)
app.include_router(base44_router, dependencies=SERVICE_AUTH)
