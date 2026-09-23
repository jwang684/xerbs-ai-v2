import os
from tests.governed_fixtures import simulate_core_attestation_for_test
from tests.governed_fixtures import approve_entity_for_test
os.environ.setdefault('LLM_PROVIDER','mock')
from fastapi.testclient import TestClient
from app.main import app

c=TestClient(app)

def ingest_review(entity_type,payload,source_id,name):
    r=c.post('/api/v1/knowledge/clinical/ingest',json={'submitted_by':'phase7','source_label':'phase7','items':[{'entity_type':entity_type,'payload':payload,'sources':[{'source_id':source_id,'title':'Reviewed source','citation':'test source','source_type':'REFERENCE'}]}]})
    assert r.status_code==200, r.text
    eid=r.json()['created_entity_ids'][0]
    r=c.post(f'/api/v1/knowledge/clinical/entities/{entity_type}/{eid}/submit-review',json={'submitted_by':'phase7'}); assert r.status_code==200
    ver=r.json()['version']
    r=approve_entity_for_test(entity_type, eid); assert r.status_code==200, r.text
    return eid

def ensure_source_reviewed(client_, source_id):
    """Idempotently drive a Source to REVIEWED (Phase 12C-2D3A retrieval gate)."""
    base = '/api/v1/knowledge/clinical/sources'
    cur = client_.get(f'{base}/{source_id}').json()
    if cur['review_status'] == 'REVIEWED':
        return
    if cur['review_status'] == 'DRAFT':
        r = client_.post(f'{base}/{source_id}/submit-review', json={'submitted_by': 'curator', 'expected_version': cur['version']})
        assert r.status_code == 200, r.text
        cur = r.json()
    assert cur['review_status'] == 'IN_REVIEW', cur['review_status']
    r = client_.post(f'{base}/{source_id}/review', json={'reviewer_id': 'reviewer', 'reviewer_role': 'CLINICAL_REVIEWER',
                                                         'decision': 'APPROVE', 'expected_version': cur['version']})
    assert r.status_code == 200, r.text


def test_safety_relationship_and_blocking_rule():
    formula=ingest_review('formula',{'name':'Phase7 Formula','indications':['phase7 symptom'],'ingredients':['Phase7 Herb']},'src-p7-f','Phase7 Formula')
    herb=ingest_review('herb',{'name':'Phase7 Herb'},'src-p7-h','Phase7 Herb')
    rr=c.post('/api/v1/safety/relationships',json={'source_entity_id':formula,'target_entity_id':herb,'relationship_type':'FORMULA_HERB','source_id':'src-p7-f','actor_id':'reviewer','actor_role':'CLINICAL_REVIEWER'})
    assert rr.status_code==200, rr.text
    rule=c.post('/api/v1/safety/rules',json={'target_entity_id':herb,'rule_type':'DRUG_INTERACTION','trigger_term':'warfarin','severity':'CRITICAL','action':'BLOCK','message':'Potential reviewed herb-drug interaction','source_id':'src-p7-h','actor_id':'reviewer','actor_role':'CLINICAL_REVIEWER'})
    assert rule.status_code==200, rule.text
    # X1D-AIV2-GOV2-C1: both are created DRAFT now. This test is about
    # blocking behaviour, so stand in for the future core attestation.
    simulate_core_attestation_for_test('CLINICAL_RELATIONSHIP',rr.json()['id'])
    simulate_core_attestation_for_test('SAFETY_RULE',rule.json()['id'])
    s=c.post('/api/v1/safety/screen',json={'formula_id':formula,'formula_name':'Phase7 Formula','ingredients':['Phase7 Herb'],'patient_context':{'medications':['warfarin']}})
    assert s.status_code==200
    assert s.json()['eligible_for_selection'] is False
    assert s.json()['risk_level']=='CRITICAL'

    ensure_source_reviewed(c,'src-p7-f')
    rec=c.post('/api/v1/recommendations/generate',json={'text_input':'phase7 symptom','symptoms':['phase7 symptom'],'patient_context':{'medications':['warfarin']}})
    assert rec.status_code==200, rec.text
    body=rec.json(); assert body['formula_candidates']
    assert body['formula_candidates'][0]['safety_assessment']['eligible_for_selection'] is False
    assert 'SAFETY_BLOCKING_FINDINGS' in body['uncertainty_flags']

def test_safety_rule_requires_source():
    # Existing non-reviewed targets are not acceptable; source is also mandatory.
    r=c.post('/api/v1/safety/rules',json={'target_entity_id':'missing','rule_type':'CONTRAINDICATION','trigger_term':'x','message':'x','actor_id':'r','actor_role':'CLINICAL_REVIEWER'})
    assert r.status_code==422
