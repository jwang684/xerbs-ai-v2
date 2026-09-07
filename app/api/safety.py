from fastapi import APIRouter, HTTPException, Query
from app.schemas.safety import SafetyScreenRequest, RelationshipCreateRequest, SafetyRuleCreateRequest, RelationshipDirection, RelationshipQueryResponse, RelationshipType
from app.services.safety.engine import SafetyEngine
router=APIRouter(prefix='/api/v1/safety',tags=['clinical-safety'])
engine=SafetyEngine()

@router.post('/screen')
def screen(req:SafetyScreenRequest): return engine.screen(req)

@router.post('/relationships')
def relationship(req:RelationshipCreateRequest):
    try: return engine.create_relationship(req)
    except ValueError as e: raise HTTPException(422,detail=str(e)) from e

@router.get('/relationships', response_model=RelationshipQueryResponse)
def list_relationships(
    entity_id:str=Query(min_length=1,description='Entity id whose persisted relationships are returned'),
    relationship_type:RelationshipType|None=Query(default=None),
    direction:RelationshipDirection=Query(default=RelationshipDirection.BOTH),
):
    results=engine.list_relationships(entity_id,relationship_type,direction)
    return RelationshipQueryResponse(entity_id=entity_id,direction=direction,relationship_type=relationship_type,count=len(results),results=results)

@router.post('/rules')
def rule(req:SafetyRuleCreateRequest):
    try: return engine.create_rule(req)
    except ValueError as e: raise HTTPException(422,detail=str(e)) from e
