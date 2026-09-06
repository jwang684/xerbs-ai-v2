from fastapi import APIRouter, HTTPException
from app.schemas.safety import SafetyScreenRequest, RelationshipCreateRequest, SafetyRuleCreateRequest
from app.services.safety.engine import SafetyEngine
router=APIRouter(prefix='/api/v1/safety',tags=['clinical-safety'])
engine=SafetyEngine()

@router.post('/screen')
def screen(req:SafetyScreenRequest): return engine.screen(req)

@router.post('/relationships')
def relationship(req:RelationshipCreateRequest):
    try: return engine.create_relationship(req)
    except ValueError as e: raise HTTPException(422,detail=str(e)) from e

@router.post('/rules')
def rule(req:SafetyRuleCreateRequest):
    try: return engine.create_rule(req)
    except ValueError as e: raise HTTPException(422,detail=str(e)) from e
