from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    s = get_settings()
    return {"status": "healthy", "service": s.app_name, "version": s.app_version}
