from sqlalchemy import select
from app.db.models import ClinicalEntity, ClinicalEntityVersion
from app.db.session import get_session_factory
from app.services.knowledge.formulas import LEGACY_FORMULAS
from uuid import uuid4


def seed_legacy_formula_fixtures():
    Session=get_session_factory()
    with Session.begin() as s:
        for row in LEGACY_FORMULAS:
            eid=f"corpus-{row.formula_id}"
            if s.get(ClinicalEntity,eid):
                continue
            e=ClinicalEntity(id=eid,entity_type="formula",name=row.name,review_status="DRAFT",current_version=1,migration_origin="LEGACY_STATIC_DATA_FIXTURE")
            s.add(e)
            snap={"entity_type":"formula","formula_id":eid,"name":row.name,"aliases":[],"pattern_ids":[],"indications":list(row.suitable_symptoms),"ingredients":list(row.ingredients),"contraindications":[],"interaction_flags":[],"review_status":"DRAFT","version":1,"sources":[],"migration_origin":"LEGACY_STATIC_DATA_FIXTURE","clinical_ranking_eligible":False}
            s.add(ClinicalEntityVersion(id=f"ver-{uuid4().hex[:16]}",entity_id=eid,version=1,snapshot=snap,created_by="SYSTEM_MIGRATION"))
