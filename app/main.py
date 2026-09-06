from fastapi import FastAPI
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
from app.db.session import init_db
from app.services.knowledge.persistent_seed import seed_legacy_formula_fixtures

settings=get_settings()
init_db(); seed_legacy_formula_fixtures()
app=FastAPI(title=settings.app_name,version=settings.app_version)
app.include_router(health_router); app.include_router(recommendation_router); app.include_router(followup_router); app.include_router(knowledge_router); app.include_router(clinical_knowledge_router); app.include_router(safety_router); app.include_router(reasoning_router)
app.include_router(interview_router)
app.include_router(base44_router)
