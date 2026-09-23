from uuid import uuid4
from sqlalchemy import or_, select
from app.db.models import (ClinicalEntity, ClinicalRelationship, GovernedObjectReviewEvent,
                           GovernedObjectSource, GovernedObjectVersion, SafetyRule, SourceRegistry)
from app.db.session import get_session_factory
from app.schemas.safety import SafetyAssessment, SafetyFinding, SafetyScreenRequest, RelationshipCreateRequest, SafetyRuleCreateRequest, ClinicalRelationshipRecord, RelationshipDirection, RelationshipType, SafetyRuleRecord, SafetyRuleType
from app.services.governance import identity, lifecycle

#: X1D-AIV2-GOV2-C1: this set no longer authorizes anything.
#:
#: It used to gate creation, and creation used to confer REVIEWED -- so a
#: string in a request body decided whether a clinical relationship became
#: authoritative. Creation now yields DRAFT and confers nothing, so the check
#: survives only as validation of a declared field. Authority over lifecycle
#: promotion lives in xerbs-core, as a verified human attestation.
DECLARED_ACTOR_ROLES={'CLINICAL_REVIEWER','CLINICAL_ADMIN'}
AUTHORIZED=DECLARED_ACTOR_ROLES     # retained name; see above

REL_OBJECT='CLINICAL_RELATIONSHIP'
RULE_OBJECT='SAFETY_RULE'

class SafetyEngine:
    def __init__(self): self.Session=get_session_factory()

    @staticmethod
    def _record_governed_creation(s, *, object_type, object_id, external_id,
                                  snapshot, content_hash, evidence_hash,
                                  author_subject, status):
        """Version snapshot + creation event for a newly authored object.

        Append-only: nothing here is ever rewritten. The event carries
        attestation_id=None because no attestation exists and this service
        cannot produce one.
        """
        s.add(GovernedObjectVersion(
            id=f'gov-{uuid4().hex[:16]}', object_type=object_type,
            object_id=object_id, version=1, external_id=external_id,
            snapshot=snapshot, content_hash=content_hash,
            evidence_hash=evidence_hash, created_by=author_subject))
        s.add(GovernedObjectReviewEvent(
            event_id=f'govevt-{uuid4().hex[:16]}', object_type=object_type,
            object_id=object_id, action='CREATED', actor_subject=author_subject,
            from_status=None, to_status=status, version=1, attestation_id=None))

    def create_relationship(self, req: RelationshipCreateRequest):
        """Create one relationship in DRAFT.

        It used to be created REVIEWED, which is why Production holds an edge
        whose approval nobody ever gave. A relationship now starts
        non-ranking-eligible and can only leave that state through a verified
        human attestation from xerbs-core, which this service cannot mint.
        """
        if req.actor_role not in DECLARED_ACTOR_ROLES: raise ValueError('Reviewer role not authorized')
        author=identity.as_subject(req.actor_id)
        with self.Session.begin() as s:
            a=s.get(ClinicalEntity,req.source_entity_id); b=s.get(ClinicalEntity,req.target_entity_id)
            if not a or not b: raise ValueError('Both relationship entities must exist')
            expected=('pattern','formula') if req.relationship_type=='PATTERN_FORMULA' else ('formula','herb')
            if (a.entity_type,b.entity_type)!=expected: raise ValueError('Relationship entity types do not match relationship_type')
            if a.review_status!='REVIEWED' or b.review_status!='REVIEWED': raise ValueError('Relationships require REVIEWED entities')
            if req.source_id and not s.get(SourceRegistry,req.source_id): raise ValueError('Source not found')

            # Semantic identity and content hash need both endpoints to carry
            # one. Legacy entities do not, and GOV2-C1 does not invent them --
            # such a relationship is still created, just without a
            # cross-service identity, and it can never become eligible because
            # eligibility needs an attestation bound to that identity.
            external_id=content_hash=None
            if a.external_id and b.external_id:
                external_id=identity.relationship_external_id(
                    a.external_id, req.relationship_type, b.external_id)
                content_hash=lifecycle.relationship_content_hash(
                    a.external_id, req.relationship_type, b.external_id)

            evidence=[{'source_id':req.source_id,'source_version':1,'locator':None}] if req.source_id else []
            row=ClinicalRelationship(
                id=f'rel-{uuid4().hex[:16]}', source_entity_id=a.id, target_entity_id=b.id,
                relationship_type=req.relationship_type,
                review_status=lifecycle.INITIAL_STATE, source_id=req.source_id,
                created_by=req.actor_id, external_id=external_id, version=1,
                content_hash=content_hash,
                evidence_hash=lifecycle.evidence_hash(evidence),
                governance_provenance=lifecycle.UNREVIEWED,
                review_attestation_id=None, submitter_subject=None,
                last_material_editor_subject=author)
            s.add(row); s.flush()
            for item in evidence:
                s.add(GovernedObjectSource(
                    object_type=REL_OBJECT, object_id=row.id,
                    source_id=item['source_id'], source_version=1, locator=None))
            self._record_governed_creation(
                s, object_type=REL_OBJECT, object_id=row.id, external_id=external_id,
                snapshot={'source_entity_id':a.id,'target_entity_id':b.id,
                          'relationship_type':row.relationship_type,
                          'source_external_id':a.external_id,
                          'target_external_id':b.external_id},
                content_hash=content_hash, evidence_hash=row.evidence_hash,
                author_subject=author, status=row.review_status)
            return {'id':row.id,'source_entity_id':a.id,'target_entity_id':b.id,
                    'relationship_type':row.relationship_type,'review_status':row.review_status,
                    'external_id':row.external_id,'version':row.version,
                    'content_hash':row.content_hash,'evidence_hash':row.evidence_hash,
                    'governance_provenance':row.governance_provenance,
                    'clinical_ranking_eligible':False}

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
        """Create one safety rule in DRAFT.

        Same defect as relationships, and it mattered more: a safety rule can
        carry action=BLOCK, so "REVIEWED because the constructor said so" meant
        an unreviewed rule could decide what a patient may buy. A rule now
        starts ineffective and stays ineffective until a human decision exists.
        """
        if req.actor_role not in DECLARED_ACTOR_ROLES: raise ValueError('Reviewer role not authorized')
        author=identity.as_subject(req.actor_id)
        with self.Session.begin() as s:
            e=s.get(ClinicalEntity,req.target_entity_id)
            if not e or e.entity_type not in {'formula','herb'}: raise ValueError('Safety rule target must be a formula or herb')
            if e.review_status!='REVIEWED': raise ValueError('Safety rules require a REVIEWED target')
            if not req.source_id: raise ValueError('Safety rule requires explicit source provenance')
            if not s.get(SourceRegistry,req.source_id): raise ValueError('Source not found')

            trigger=req.trigger_term.strip()
            external_id=content_hash=None
            if e.external_id:
                external_id=identity.safety_rule_external_id(
                    e.external_id, req.rule_type, trigger)
                content_hash=lifecycle.safety_rule_content_hash(
                    target_external_id=e.external_id, rule_type=req.rule_type,
                    trigger_term=trigger, severity=req.severity,
                    action=req.action, message=req.message)

            evidence=[{'source_id':req.source_id,'source_version':1,'locator':None}]
            row=SafetyRule(
                id=f'rule-{uuid4().hex[:16]}', target_entity_id=e.id, rule_type=req.rule_type,
                trigger_term=trigger, severity=req.severity, action=req.action,
                message=req.message, review_status=lifecycle.INITIAL_STATE,
                source_id=req.source_id, created_by=req.actor_id,
                external_id=external_id, version=1, content_hash=content_hash,
                evidence_hash=lifecycle.evidence_hash(evidence),
                governance_provenance=lifecycle.UNREVIEWED,
                review_attestation_id=None, submitter_subject=None,
                last_material_editor_subject=author)
            s.add(row); s.flush()
            s.add(GovernedObjectSource(
                object_type=RULE_OBJECT, object_id=row.id, source_id=req.source_id,
                source_version=1, locator=None))
            self._record_governed_creation(
                s, object_type=RULE_OBJECT, object_id=row.id, external_id=external_id,
                snapshot={'target_entity_id':e.id,'target_external_id':e.external_id,
                          'rule_type':row.rule_type,'trigger_term':row.trigger_term,
                          'severity':row.severity,'action':row.action,'message':row.message},
                content_hash=content_hash, evidence_hash=row.evidence_hash,
                author_subject=author, status=row.review_status)
            return {'id':row.id,'target_entity_id':e.id,'rule_type':row.rule_type,
                    'review_status':row.review_status,'external_id':row.external_id,
                    'version':row.version,'content_hash':row.content_hash,
                    'evidence_hash':row.evidence_hash,
                    'governance_provenance':row.governance_provenance,
                    'clinical_ranking_eligible':False}

    def list_safety_rules(self, target_entity_id, rule_type=None) -> list[SafetyRuleRecord]:
        """Persisted safety rules attached to exactly one target entity id.

        Pure read of safety_rule. Rows are returned as stored, with their own
        review_status exposed; no eligibility rule is applied here so screen()
        keeps its existing REVIEWED filter as the single source of truth. An
        empty list means no persisted rule was found - never that the entity is
        safe or eligible for selection.
        """
        with self.Session() as s:
            stmt=select(SafetyRule).where(SafetyRule.target_entity_id==target_entity_id)
            if rule_type is not None: stmt=stmt.where(SafetyRule.rule_type==SafetyRuleType(rule_type).value)
            rows=s.scalars(stmt.order_by(SafetyRule.created_at,SafetyRule.id)).all()
            target=s.get(ClinicalEntity,target_entity_id) if rows else None
            return [self._safety_rule_record(r,target) for r in rows]

    @staticmethod
    def _safety_rule_record(r,target) -> SafetyRuleRecord:
        return SafetyRuleRecord(
            id=r.id,
            target_entity_id=r.target_entity_id,
            target_entity_type=target.entity_type if target else None,
            target_entity_name=target.name if target else None,
            rule_type=SafetyRuleType(r.rule_type),
            trigger_term=r.trigger_term,
            severity=r.severity,
            action=r.action,
            message=r.message,
            review_status=r.review_status,
            source_id=r.source_id,
            created_by=r.created_by,
            created_at=r.created_at,
        )

    @staticmethod
    def _reviewed_evidence_count(s, object_type, object_id) -> int:
        """Attached evidence whose Source is itself REVIEWED.

        Same bar entities have had since Phase 12C-2D3. Relationships and
        safety rules never had it; now they do.
        """
        from sqlalchemy import func
        return s.scalar(
            select(func.count()).select_from(GovernedObjectSource)
            .join(SourceRegistry, SourceRegistry.source_id == GovernedObjectSource.source_id)
            .where(GovernedObjectSource.object_type == object_type,
                   GovernedObjectSource.object_id == object_id,
                   SourceRegistry.review_status == 'REVIEWED')) or 0

    def is_relationship_eligible(self, s, rel) -> bool:
        """THE relationship eligibility question, asked in one place."""
        return lifecycle.is_governed_object_ranking_eligible(
            review_status=rel.review_status,
            governance_provenance=rel.governance_provenance,
            review_attestation_id=rel.review_attestation_id,
            reviewed_evidence_source_count=self._reviewed_evidence_count(
                s, REL_OBJECT, rel.id),
            retired_at=rel.retired_at)

    def is_safety_rule_effective(self, s, rule) -> bool:
        """THE safety-rule effectiveness question, asked in one place.

        A rule that is not effective is simply not applied -- it never relaxes
        an existing finding, so this cannot reduce blocking behaviour for any
        rule that was genuinely reviewed.
        """
        return lifecycle.is_governed_object_ranking_eligible(
            review_status=rule.review_status,
            governance_provenance=rule.governance_provenance,
            review_attestation_id=rule.review_attestation_id,
            reviewed_evidence_source_count=self._reviewed_evidence_count(
                s, RULE_OBJECT, rule.id),
            retired_at=rule.retired_at)

    def _herb_ids_for_formula(self,s,formula_id):
        rows=s.scalars(select(ClinicalRelationship).where(
            ClinicalRelationship.source_entity_id==formula_id,
            ClinicalRelationship.relationship_type=='FORMULA_HERB')).all()
        return [r.target_entity_id for r in rows if self.is_relationship_eligible(s,r)]

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
                candidates=s.scalars(select(SafetyRule).where(SafetyRule.target_entity_id.in_(target_ids))).all()
                rules=[r for r in candidates if self.is_safety_rule_effective(s,r)]
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
