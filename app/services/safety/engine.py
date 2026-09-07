from uuid import uuid4
from sqlalchemy import or_, select
from app.db.models import ClinicalEntity, ClinicalRelationship, SafetyRule, SourceRegistry
from app.db.session import get_session_factory
from app.schemas.safety import SafetyAssessment, SafetyFinding, SafetyScreenRequest, RelationshipCreateRequest, SafetyRuleCreateRequest, ClinicalRelationshipRecord, RelationshipDirection, RelationshipType

AUTHORIZED={'CLINICAL_REVIEWER','CLINICAL_ADMIN'}

class SafetyEngine:
    def __init__(self): self.Session=get_session_factory()

    def create_relationship(self, req: RelationshipCreateRequest):
        if req.actor_role not in AUTHORIZED: raise ValueError('Reviewer role not authorized')
        with self.Session.begin() as s:
            a=s.get(ClinicalEntity,req.source_entity_id); b=s.get(ClinicalEntity,req.target_entity_id)
            if not a or not b: raise ValueError('Both relationship entities must exist')
            expected=('pattern','formula') if req.relationship_type=='PATTERN_FORMULA' else ('formula','herb')
            if (a.entity_type,b.entity_type)!=expected: raise ValueError('Relationship entity types do not match relationship_type')
            if a.review_status!='REVIEWED' or b.review_status!='REVIEWED': raise ValueError('Relationships require REVIEWED entities')
            if req.source_id and not s.get(SourceRegistry,req.source_id): raise ValueError('Source not found')
            row=ClinicalRelationship(id=f'rel-{uuid4().hex[:16]}',source_entity_id=a.id,target_entity_id=b.id,relationship_type=req.relationship_type,review_status='REVIEWED',source_id=req.source_id,created_by=req.actor_id)
            s.add(row); s.flush(); return {'id':row.id,'source_entity_id':a.id,'target_entity_id':b.id,'relationship_type':row.relationship_type,'review_status':row.review_status}

    def list_relationships(self, entity_id, relationship_type=None, direction=RelationshipDirection.BOTH) -> list[ClinicalRelationshipRecord]:
        """Persisted clinical relationships touching exactly one entity id.

        Pure read of clinical_relationship. Rows are returned as stored, with
        their own review_status exposed; no eligibility rule is applied here so
        ranking and safety keep their existing REVIEWED filters as the single
        source of truth.
        """
        with self.Session() as s:
            stmt=select(ClinicalRelationship)
            if direction==RelationshipDirection.OUTGOING: stmt=stmt.where(ClinicalRelationship.source_entity_id==entity_id)
            elif direction==RelationshipDirection.INCOMING: stmt=stmt.where(ClinicalRelationship.target_entity_id==entity_id)
            else: stmt=stmt.where(or_(ClinicalRelationship.source_entity_id==entity_id,ClinicalRelationship.target_entity_id==entity_id))
            if relationship_type is not None: stmt=stmt.where(ClinicalRelationship.relationship_type==RelationshipType(relationship_type).value)
            rows=s.scalars(stmt.order_by(ClinicalRelationship.created_at,ClinicalRelationship.id)).all()
            ids={x for r in rows for x in (r.source_entity_id,r.target_entity_id)}
            named={e.id:e for e in s.scalars(select(ClinicalEntity).where(ClinicalEntity.id.in_(ids))).all()} if ids else {}
            return [self._relationship_record(r,named,entity_id) for r in rows]

    @staticmethod
    def _relationship_record(r,named,entity_id) -> ClinicalRelationshipRecord:
        src=named.get(r.source_entity_id); tgt=named.get(r.target_entity_id)
        return ClinicalRelationshipRecord(
            id=r.id,
            source_entity_id=r.source_entity_id,
            source_entity_type=src.entity_type if src else None,
            source_entity_name=src.name if src else None,
            target_entity_id=r.target_entity_id,
            target_entity_type=tgt.entity_type if tgt else None,
            target_entity_name=tgt.name if tgt else None,
            relationship_type=RelationshipType(r.relationship_type),
            review_status=r.review_status,
            source_id=r.source_id,
            created_by=r.created_by,
            created_at=r.created_at,
            direction='outgoing' if r.source_entity_id==entity_id else 'incoming',
        )

    def create_rule(self, req: SafetyRuleCreateRequest):
        if req.actor_role not in AUTHORIZED: raise ValueError('Reviewer role not authorized')
        with self.Session.begin() as s:
            e=s.get(ClinicalEntity,req.target_entity_id)
            if not e or e.entity_type not in {'formula','herb'}: raise ValueError('Safety rule target must be a formula or herb')
            if e.review_status!='REVIEWED': raise ValueError('Safety rules require a REVIEWED target')
            if not req.source_id: raise ValueError('Safety rule requires explicit source provenance')
            if not s.get(SourceRegistry,req.source_id): raise ValueError('Source not found')
            row=SafetyRule(id=f'rule-{uuid4().hex[:16]}',target_entity_id=e.id,rule_type=req.rule_type,trigger_term=req.trigger_term.strip(),severity=req.severity,action=req.action,message=req.message,review_status='REVIEWED',source_id=req.source_id,created_by=req.actor_id)
            s.add(row); s.flush(); return {'id':row.id,'target_entity_id':e.id,'rule_type':row.rule_type,'review_status':row.review_status}

    def _herb_ids_for_formula(self,s,formula_id):
        return list(s.scalars(select(ClinicalRelationship.target_entity_id).where(ClinicalRelationship.source_entity_id==formula_id,ClinicalRelationship.relationship_type=='FORMULA_HERB',ClinicalRelationship.review_status=='REVIEWED')).all())

    def screen(self, req: SafetyScreenRequest) -> SafetyAssessment:
        terms=[]
        pc=req.patient_context
        terms += [('MEDICATION',x) for x in pc.medications]
        terms += [('ALLERGY',x) for x in pc.allergies]
        terms += [('DIAGNOSIS',x) for x in pc.diagnoses]
        terms += [('CONSTRAINT',x) for x in req.constraints]
        if pc.pregnancy: terms.append(('PREGNANCY','pregnancy')); terms.append(('PREGNANCY','孕')) if pc.pregnancy else None
        if pc.age is not None: terms.append(('AGE',str(pc.age)))
        findings=[]
        with self.Session() as s:
            target_ids=[]
            if req.formula_id: target_ids.append(req.formula_id); target_ids += self._herb_ids_for_formula(s,req.formula_id)
            if target_ids:
                rules=s.scalars(select(SafetyRule).where(SafetyRule.target_entity_id.in_(target_ids),SafetyRule.review_status=='REVIEWED')).all()
                haystack=' | '.join(v.lower() for _,v in terms if v)
                for r in rules:
                    trig=r.trigger_term.lower().strip()
                    if trig and trig in haystack:
                        findings.append(SafetyFinding(rule_id=r.id,target_entity_id=r.target_entity_id,rule_type=r.rule_type,severity=r.severity,action=r.action,trigger_term=r.trigger_term,message=r.message))
        if not findings: return SafetyAssessment(risk_level='NONE',eligible_for_selection=True,findings=[])
        order={'INFO':0,'WARN':1,'HIGH':2,'CRITICAL':3}; maxsev=max(findings,key=lambda x:order[x.severity]).severity
        risk={'INFO':'LOW','WARN':'MODERATE','HIGH':'HIGH','CRITICAL':'CRITICAL'}[maxsev]
        blocked=any(x.action=='BLOCK' for x in findings)
        return SafetyAssessment(risk_level=risk,eligible_for_selection=not blocked,findings=findings)
