from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from app.db.models import ClinicalEntity, ClinicalEntityVersion, SourceRegistry, EntitySource, IngestionBatch, IngestionItemRow, ReviewEvent, AuditEvent, ClinicalRelationship
from app.db.session import get_session_factory
from app.schemas.clinical_knowledge import ReviewStatus, CorpusStats, SourceRef, SourceConflictDetail, SourceFieldConflict
from app.schemas.clinical_workflow import ClinicalEntityDetail, ClinicalEntityType, IngestionBatchRequest, IngestionBatchResult, ReviewActionRequest, ReviewDecision, WorkflowEvent

# Snapshot keys promoted to dedicated ClinicalEntityDetail fields; everything
# else in the snapshot is the entity's clinical content.
DETAIL_PROMOTED_KEYS = {"entity_type","review_status","version","sources","clinical_ranking_eligible","name","retired_at","superseded_by_id","migration_origin","id","pattern_id","formula_id","herb_id"}


class PersistentWorkflowError(ValueError):
    pass


class SourceConflictError(PersistentWorkflowError):
    """An existing source_id was resubmitted with different canonical metadata.

    Carries the exact per-field disagreement so the caller can correct the
    submission. The persisted Source is never mutated; the batch is rejected.
    """

    def __init__(self, source_id, conflicts):
        self.source_id = source_id
        self.conflicts = conflicts
        super().__init__(f"Source '{source_id}' is already registered with different canonical metadata")

    def to_detail(self) -> SourceConflictDetail:
        return SourceConflictDetail(source_id=self.source_id, message=str(self), conflicting_fields=self.conflicts)


class PersistentClinicalStore:
    def __init__(self, session_factory=None):
        self.Session = session_factory or get_session_factory()

    @staticmethod
    def _clean_payload(payload):
        return {k: v for k, v in payload.items() if k not in {"review_status","version","sources","clinical_ranking_eligible","id","pattern_id","formula_id","herb_id"}}

    def ingest(self, request: IngestionBatchRequest) -> IngestionBatchResult:
        batch_id=f"ing-{uuid4().hex[:16]}"; ids=[]
        with self.Session.begin() as s:
            s.add(IngestionBatch(batch_id=batch_id, submitted_by=request.submitted_by, source_label=request.source_label))
            for i,item in enumerate(request.items):
                clean=self._clean_payload(item.payload); name=str(clean.get("name","")).strip()
                if not name: raise PersistentWorkflowError("payload.name is required")
                prefix={ClinicalEntityType.PATTERN:"pat",ClinicalEntityType.FORMULA:"frm",ClinicalEntityType.HERB:"herb"}[item.entity_type]
                eid=f"{prefix}-{uuid4().hex[:16]}"; ids.append(eid)
                entity=ClinicalEntity(id=eid, entity_type=item.entity_type.value, name=name, review_status="DRAFT", current_version=1, migration_origin=clean.get("migration_origin"))
                s.add(entity)
                self._assert_unique_source_ids(item.sources)
                for src in item.sources:
                    existing=s.get(SourceRegistry, src.source_id)
                    if existing is None:
                        s.add(SourceRegistry(source_id=src.source_id,title=src.title,citation=src.citation,url=src.url,source_type=src.source_type))
                    else:
                        conflicts=self._source_conflicts(existing,src)
                        if conflicts: raise SourceConflictError(src.source_id,conflicts)
                    s.flush(); s.add(EntitySource(entity_id=eid,source_id=src.source_id))
                snap=self._snapshot_dict(item.entity_type.value,eid,clean,item.sources,"DRAFT",1)
                s.add(ClinicalEntityVersion(id=f"ver-{uuid4().hex[:16]}",entity_id=eid,version=1,snapshot=snap,created_by=request.submitted_by))
                s.add(IngestionItemRow(id=f"itm-{uuid4().hex[:16]}",batch_id=batch_id,entity_id=eid,external_id=item.external_id,ordinal=i))
                self._add_event(s,item.entity_type.value,eid,"INGESTED_AS_DRAFT",request.submitted_by,None,"DRAFT",1,None,None,None)
        return IngestionBatchResult(batch_id=batch_id,submitted_by=request.submitted_by,source_label=request.source_label,created_entity_ids=ids,created_at=datetime.now(timezone.utc))

    def submit_for_review(self, entity_type, entity_id, actor, notes=None, expected_version=None):
        with self.Session.begin() as s:
            e=self._get_entity(s,entity_type,entity_id)
            if expected_version is not None and e.current_version != expected_version: raise PersistentWorkflowError("VERSION_CONFLICT")
            if e.review_status not in {"DRAFT","REJECTED"}: raise PersistentWorkflowError(f"Cannot submit {e.review_status} record for review")
            before=e.review_status; e.review_status="IN_REVIEW"; e.current_version += 1
            snap=self._copy_latest_with_status(s,e,"IN_REVIEW")
            self._version(s,e,snap,actor); self._add_event(s,e.entity_type,e.id,"SUBMITTED_FOR_REVIEW",actor,before,e.review_status,e.current_version,notes,None,None)
            return self._serialize(s,e,snap)

    def review(self, entity_type, entity_id, request: ReviewActionRequest):
        with self.Session.begin() as s:
            e=self._get_entity(s,entity_type,entity_id)
            if request.expected_version is not None and e.current_version != request.expected_version: raise PersistentWorkflowError("VERSION_CONFLICT")
            if request.reviewer_role not in {"CLINICAL_REVIEWER","CLINICAL_ADMIN"}: raise PersistentWorkflowError("Reviewer role not authorized")
            if e.review_status != "IN_REVIEW": raise PersistentWorkflowError("Only IN_REVIEW records can receive a review decision")
            before=e.review_status
            if request.decision == ReviewDecision.APPROVE:
                if self._source_count(s,e.id)==0: raise PersistentWorkflowError("Approval requires at least one explicit source")
                e.review_status="REVIEWED"; action="APPROVED"
            elif request.decision == ReviewDecision.REJECT:
                e.review_status="REJECTED"; action="REJECTED"
            else:
                e.review_status="DRAFT"; action="CHANGES_REQUESTED"
            e.current_version += 1; snap=self._copy_latest_with_status(s,e,e.review_status); self._version(s,e,snap,request.reviewer_id)
            self._add_event(s,e.entity_type,e.id,action,request.reviewer_id,before,e.review_status,e.current_version,request.notes,request.reviewer_role,None)
            return self._serialize(s,e,snap)

    def retire(self, entity_type, entity_id, actor, actor_role, notes=None, expected_version=None):
        if actor_role not in {"CLINICAL_REVIEWER","CLINICAL_ADMIN"}: raise PersistentWorkflowError("Reviewer role not authorized")
        with self.Session.begin() as s:
            e=self._get_entity(s,entity_type,entity_id)
            if expected_version is not None and e.current_version != expected_version: raise PersistentWorkflowError("VERSION_CONFLICT")
            if e.review_status == "RETIRED": raise PersistentWorkflowError("Record already retired")
            before=e.review_status; e.review_status="RETIRED"; e.retired_at=datetime.now(timezone.utc); e.current_version += 1
            snap=self._copy_latest_with_status(s,e,"RETIRED"); self._version(s,e,snap,actor)
            self._add_event(s,e.entity_type,e.id,"RETIRED",actor,before,"RETIRED",e.current_version,notes,actor_role,None)
            return self._serialize(s,e,snap)

    def supersede(self, entity_type, entity_id, superseded_by_id, actor, actor_role, expected_version=None):
        if actor_role != "CLINICAL_ADMIN": raise PersistentWorkflowError("CLINICAL_ADMIN role required")
        with self.Session.begin() as s:
            e=self._get_entity(s,entity_type,entity_id); replacement=s.get(ClinicalEntity,superseded_by_id)
            if replacement is None or replacement.entity_type != e.entity_type: raise PersistentWorkflowError("Valid replacement entity required")
            if expected_version is not None and e.current_version != expected_version: raise PersistentWorkflowError("VERSION_CONFLICT")
            e.superseded_by_id=replacement.id; before=e.review_status; e.review_status="RETIRED"; e.retired_at=datetime.now(timezone.utc); e.current_version += 1
            snap=self._copy_latest_with_status(s,e,"RETIRED"); snap["superseded_by_id"]=replacement.id; self._version(s,e,snap,actor)
            self._add_event(s,e.entity_type,e.id,"SUPERSEDED",actor,before,"RETIRED",e.current_version,None,actor_role,{"superseded_by_id":replacement.id})
            return self._serialize(s,e,snap)

    def get_entity_detail(self, entity_type, entity_id) -> ClinicalEntityDetail:
        """Current canonical state of one entity, resolved by exact type + id."""
        with self.Session() as s:
            e=self._get_entity(s,entity_type,entity_id)
            snap=self._latest_snapshot(s,e.id)
            serialized=self._serialize(s,e,snap)
            sources=self._registered_sources(s,e.id)
            return ClinicalEntityDetail(
                entity_id=e.id,
                entity_type=ClinicalEntityType(e.entity_type),
                name=e.name,
                version=e.current_version,
                review_status=ReviewStatus(e.review_status),
                clinical_ranking_eligible=serialized["clinical_ranking_eligible"],
                sources=sources,
                source_count=len(sources),
                content={k:v for k,v in serialized.items() if k not in DETAIL_PROMOTED_KEYS},
                migration_origin=e.migration_origin,
                retired_at=e.retired_at,
                superseded_by_id=e.superseded_by_id,
                created_at=e.created_at,
                updated_at=e.updated_at,
            )

    def get_history(self, entity_type, entity_id):
        with self.Session() as s:
            self._get_entity(s,entity_type,entity_id)
            rows=s.scalars(select(ClinicalEntityVersion).where(ClinicalEntityVersion.entity_id==entity_id).order_by(ClinicalEntityVersion.version)).all()
            return [r.snapshot for r in rows]

    def get_batch(self,batch_id):
        with self.Session() as s:
            b=s.get(IngestionBatch,batch_id)
            if not b: raise PersistentWorkflowError("Ingestion batch not found")
            ids=s.scalars(select(IngestionItemRow.entity_id).where(IngestionItemRow.batch_id==batch_id).order_by(IngestionItemRow.ordinal)).all()
            return IngestionBatchResult(batch_id=b.batch_id,submitted_by=b.submitted_by,source_label=b.source_label,created_entity_ids=list(ids),created_at=b.created_at)

    def audit(self, entity_type=None, entity_id=None):
        with self.Session() as s:
            stmt=select(ReviewEvent).order_by(ReviewEvent.created_at,ReviewEvent.event_id)
            if entity_type is not None: stmt=stmt.where(ReviewEvent.entity_type==entity_type.value)
            if entity_id is not None: stmt=stmt.where(ReviewEvent.entity_id==entity_id)
            rows=s.scalars(stmt).all()
            return [WorkflowEvent(event_id=r.event_id,entity_type=ClinicalEntityType(r.entity_type),entity_id=r.entity_id,action=r.action,actor_id=r.actor_id,from_status=r.from_status,to_status=r.to_status,version=r.version,notes=r.notes,created_at=r.created_at) for r in rows]

    def stats(self):
        with self.Session() as s:
            counts={t:s.scalar(select(func.count()).select_from(ClinicalEntity).where(ClinicalEntity.entity_type==t)) or 0 for t in ["pattern","formula","herb"]}
            eligible={t:s.scalar(select(func.count()).select_from(ClinicalEntity).where(ClinicalEntity.entity_type==t,ClinicalEntity.review_status=="REVIEWED").where(select(func.count()).select_from(EntitySource).where(EntitySource.entity_id==ClinicalEntity.id).correlate(ClinicalEntity).scalar_subquery()>0)) or 0 for t in ["pattern","formula","herb"]}
            return CorpusStats(patterns=counts['pattern'],formulas=counts['formula'],herbs=counts['herb'],ranking_eligible_patterns=eligible['pattern'],ranking_eligible_formulas=eligible['formula'],ranking_eligible_herbs=eligible['herb'])

    def search(self, query, entity_types, reviewed_only=False, limit=20):
        q=query.strip().lower(); out=[]
        with self.Session() as s:
            stmt=select(ClinicalEntity).where(ClinicalEntity.entity_type.in_(entity_types))
            if reviewed_only: stmt=stmt.where(ClinicalEntity.review_status=="REVIEWED")
            for e in s.scalars(stmt).all():
                snap=self._latest_snapshot(s,e.id)
                fields=[snap.get('name',''),*snap.get('aliases',[]),*snap.get('indications',[]),*snap.get('ingredients',[])]
                if any(q in str(v).lower() for v in fields): out.append(self._serialize(s,e,snap))
                if len(out)>=limit: break
        return out


    def eligible_formula_candidates_for_patterns(self, pattern_ids: list[str]):
        pattern_ids=[x for x in pattern_ids if x]
        if not pattern_ids: return []
        scored={}
        with self.Session() as s:
            rels=s.scalars(select(ClinicalRelationship).where(
                ClinicalRelationship.source_entity_id.in_(pattern_ids),
                ClinicalRelationship.relationship_type=="PATTERN_FORMULA",
                ClinicalRelationship.review_status=="REVIEWED"
            )).all()
            for rel in rels:
                formula=s.get(ClinicalEntity,rel.target_entity_id)
                if formula is None or formula.entity_type!="formula" or formula.review_status!="REVIEWED":
                    continue
                if self._source_count(s,formula.id)==0:
                    continue
                snap=self._latest_snapshot(s,formula.id)
                item=scored.setdefault(formula.id,{"count":0,"patterns":[],"snapshot":snap})
                item["count"] += 1
                item["patterns"].append(rel.source_entity_id)
        ranked=sorted(scored.items(),key=lambda kv:(-kv[1]["count"],kv[0]))
        out=[]
        for eid,meta in ranked[:3]:
            snap=meta["snapshot"]
            out.append({
                "formula_id":eid,
                "name":snap.get("name",""),
                "confidence":min(.92,.60+.08*meta["count"]),
                "rationale":f"Reviewed pattern→formula relationship match ({meta['count']} pattern link(s))",
                "ingredients":snap.get("ingredients",[]),
                "safety_flags":list(dict.fromkeys([*snap.get("contraindications",[]),*snap.get("interaction_flags",[]),"PRACTITIONER_REVIEW_REQUIRED"])),
            })
        return out

    def eligible_formula_candidates(self, symptoms, text_input=""):
        normalized={x.strip() for x in symptoms if x and x.strip()}; scored=[]
        with self.Session() as s:
            rows=s.scalars(select(ClinicalEntity).where(ClinicalEntity.entity_type=="formula",ClinicalEntity.review_status=="REVIEWED")).all()
            for e in rows:
                if self._source_count(s,e.id)==0: continue
                snap=self._latest_snapshot(s,e.id); inds=snap.get('indications',[]); matched=[x for x in inds if x in normalized or x in (text_input or '')]
                if matched: scored.append((len(matched),e.id,snap,matched))
        scored.sort(key=lambda x:(-x[0],x[1])); return [{"formula_id":eid,"name":snap.get('name',''),"confidence":min(.85,.40+.10*score),"rationale":f"Reviewed corpus indication overlap: {', '.join(matched)}","ingredients":snap.get('ingredients',[]),"safety_flags":list(dict.fromkeys([*snap.get('contraindications',[]),*snap.get('interaction_flags',[]),"PRACTITIONER_REVIEW_REQUIRED"]))} for score,eid,snap,matched in scored[:3]]

    @staticmethod
    def _canonical(value):
        """Canonical comparison form: trimmed, with empty string treated as absent."""
        if value is None: return None
        text=str(value).strip()
        return text or None

    @classmethod
    def _source_conflicts(cls,existing,submitted) -> list[SourceFieldConflict]:
        """Per-field disagreement between a persisted Source and a resubmission.

        Compares only the fields the registry actually persists. Comparison is
        case-sensitive and whitespace-insensitive; None and '' are equivalent.
        """
        pairs=(("title",existing.title,submitted.title),
               ("citation",existing.citation,submitted.citation),
               ("url",existing.url,submitted.url),
               ("source_type",existing.source_type,submitted.source_type))
        return [SourceFieldConflict(field=f,persisted=cls._canonical(p),submitted=cls._canonical(sub))
                for f,p,sub in pairs if cls._canonical(p)!=cls._canonical(sub)]

    @staticmethod
    def _assert_unique_source_ids(sources):
        seen=set(); duplicates=[]
        for src in sources:
            if src.source_id in seen and src.source_id not in duplicates: duplicates.append(src.source_id)
            seen.add(src.source_id)
        if duplicates: raise PersistentWorkflowError("Duplicate source_id within one ingestion item: "+", ".join(duplicates))

    def _snapshot_dict(self,typ,eid,clean,sources,status,version):
        d={"entity_type":typ,**clean,"review_status":status,"version":version,"sources":[x.model_dump() for x in sources],"clinical_ranking_eligible": status=="REVIEWED" and bool(sources)}
        d[{"pattern":"pattern_id","formula":"formula_id","herb":"herb_id"}[typ]]=eid
        return d
    def _latest_snapshot(self,s,eid): return s.scalar(select(ClinicalEntityVersion.snapshot).where(ClinicalEntityVersion.entity_id==eid).order_by(ClinicalEntityVersion.version.desc()).limit(1)) or {}
    def _copy_latest_with_status(self,s,e,status):
        d=dict(self._latest_snapshot(s,e.id)); d['review_status']=status; d['version']=e.current_version; d['clinical_ranking_eligible']=status=="REVIEWED" and self._source_count(s,e.id)>0; return d
    def _version(self,s,e,snap,actor): s.add(ClinicalEntityVersion(id=f"ver-{uuid4().hex[:16]}",entity_id=e.id,version=e.current_version,snapshot=snap,created_by=actor))
    def _source_count(self,s,eid): return s.scalar(select(func.count()).select_from(EntitySource).where(EntitySource.entity_id==eid)) or 0
    def _registered_sources(self,s,eid):
        rows=s.scalars(select(SourceRegistry).join(EntitySource,EntitySource.source_id==SourceRegistry.source_id).where(EntitySource.entity_id==eid).order_by(SourceRegistry.source_id)).all()
        return [SourceRef(source_id=r.source_id,title=r.title,citation=r.citation,url=r.url,source_type=r.source_type) for r in rows]
    def _serialize(self,s,e,snap):
        d=dict(snap); d['entity_type']=e.entity_type; d['clinical_ranking_eligible']=e.review_status=="REVIEWED" and self._source_count(s,e.id)>0; d['retired_at']=e.retired_at; d['superseded_by_id']=e.superseded_by_id; return d
    def _get_entity(self,s,entity_type,eid):
        e=s.get(ClinicalEntity,eid)
        if not e or e.entity_type != entity_type.value: raise PersistentWorkflowError("Clinical corpus entity not found")
        return e
    def _add_event(self,s,typ,eid,action,actor,frm,to,ver,notes,role,payload):
        rid=f"evt-{uuid4().hex[:16]}"; s.add(ReviewEvent(event_id=rid,entity_id=eid,entity_type=typ,action=action,actor_id=actor,actor_role=role,from_status=frm,to_status=to,version=ver,notes=notes)); s.add(AuditEvent(event_id=f"aud-{uuid4().hex[:16]}",event_type=action,entity_id=eid,actor_id=actor,payload=payload or {"from_status":frm,"to_status":to,"version":ver}))
