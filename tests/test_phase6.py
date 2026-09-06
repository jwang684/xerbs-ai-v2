from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.db.base import Base
from app.db import models  # noqa
from app.schemas.clinical_workflow import ClinicalEntityType, IngestionBatchRequest, IngestionItem, ReviewActionRequest, ReviewDecision
from app.schemas.clinical_knowledge import SourceRef
from app.services.knowledge.persistent_clinical import PersistentClinicalStore, PersistentWorkflowError


def store():
    engine=create_engine("sqlite://",connect_args={"check_same_thread":False},poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return PersistentClinicalStore(sessionmaker(bind=engine,expire_on_commit=False))

def source(): return SourceRef(source_id="src-p6",title="Reviewed source",source_type="CURATED_TEST")

def create_formula(s,name="Persistent Formula",symptom="持久症状",with_source=True):
    b=s.ingest(IngestionBatchRequest(submitted_by="importer",source_label="p6",items=[IngestionItem(entity_type=ClinicalEntityType.FORMULA,payload={"name":name,"indications":[symptom],"ingredients":["Herb A"]},sources=[source()] if with_source else [])]))
    return b.created_entity_ids[0]

def test_persistent_round_trip_across_store_instances():
    engine=create_engine("sqlite://",connect_args={"check_same_thread":False},poolclass=StaticPool); Base.metadata.create_all(engine); Session=sessionmaker(bind=engine,expire_on_commit=False)
    a=PersistentClinicalStore(Session); eid=create_formula(a); a.submit_for_review(ClinicalEntityType.FORMULA,eid,"submit",expected_version=1); a.review(ClinicalEntityType.FORMULA,eid,ReviewActionRequest(reviewer_id="reviewer",reviewer_role="CLINICAL_REVIEWER",decision=ReviewDecision.APPROVE,expected_version=2))
    b=PersistentClinicalStore(Session)
    assert b.eligible_formula_candidates(["持久症状"],"")[0]["name"] == "Persistent Formula"
    assert len(b.get_history(ClinicalEntityType.FORMULA,eid)) == 3

def test_optimistic_concurrency_rejects_stale_write():
    s=store(); eid=create_formula(s)
    try:
        s.submit_for_review(ClinicalEntityType.FORMULA,eid,"submit",expected_version=99); assert False
    except PersistentWorkflowError as e: assert "VERSION_CONFLICT" in str(e)

def test_reviewer_role_enforced():
    s=store(); eid=create_formula(s); s.submit_for_review(ClinicalEntityType.FORMULA,eid,"submit")
    try:
        s.review(ClinicalEntityType.FORMULA,eid,ReviewActionRequest(reviewer_id="x",reviewer_role="IMPORTER",decision=ReviewDecision.APPROVE)); assert False
    except PersistentWorkflowError as e: assert "role" in str(e).lower()

def test_retired_formula_no_longer_ranks():
    s=store(); eid=create_formula(s); s.submit_for_review(ClinicalEntityType.FORMULA,eid,"submit"); s.review(ClinicalEntityType.FORMULA,eid,ReviewActionRequest(reviewer_id="r",decision=ReviewDecision.APPROVE)); assert s.eligible_formula_candidates(["持久症状"],"")
    s.retire(ClinicalEntityType.FORMULA,eid,"r","CLINICAL_REVIEWER")
    assert s.eligible_formula_candidates(["持久症状"],"") == []
    assert s.get_history(ClinicalEntityType.FORMULA,eid)[-1]["review_status"] == "RETIRED"

def test_supersede_requires_admin_and_links_replacement():
    s=store(); old=create_formula(s,"Old Formula","旧症状"); new=create_formula(s,"New Formula","旧症状")
    result=s.supersede(ClinicalEntityType.FORMULA,old,new,"admin","CLINICAL_ADMIN",expected_version=1)
    assert result["review_status"] == "RETIRED" and result["superseded_by_id"] == new

def test_approval_without_source_still_blocked_persistently():
    s=store(); eid=create_formula(s,with_source=False); s.submit_for_review(ClinicalEntityType.FORMULA,eid,"submit")
    try:
        s.review(ClinicalEntityType.FORMULA,eid,ReviewActionRequest(reviewer_id="r",decision=ReviewDecision.APPROVE)); assert False
    except PersistentWorkflowError as e: assert "source" in str(e).lower()
