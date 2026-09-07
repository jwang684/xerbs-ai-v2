import os
os.environ.setdefault('LLM_PROVIDER','mock')
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app

c=TestClient(app)
BASE='/api/v1/knowledge/clinical'


def ingest(entity_type,payload,sources=None,submitted_by='phase12b1'):
    if sources is None:
        sources=[{'source_id':'src-12b1-'+uuid4().hex[:8],'title':'Reviewed source','citation':'phase12b1 citation','source_type':'REFERENCE'}]
    r=c.post(f'{BASE}/ingest',json={'submitted_by':submitted_by,'source_label':'phase12b1','items':[{'entity_type':entity_type,'payload':payload,'sources':sources}]})
    assert r.status_code==200, r.text
    return r.json()['created_entity_ids'][0]


def approve(entity_type,eid):
    r=c.post(f'{BASE}/entities/{entity_type}/{eid}/submit-review',json={'submitted_by':'phase12b1'})
    assert r.status_code==200, r.text
    ver=r.json()['version']
    r=c.post(f'{BASE}/entities/{entity_type}/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':ver})
    assert r.status_code==200, r.text
    return r.json()['version']


def draft_formula():
    return ingest('formula',{'name':'Phase12B1 Draft Formula','indications':['phase12b1 symptom'],'ingredients':['Phase12B1 Herb'],'contraindications':['pregnancy']})


def test_existing_draft_formula_is_returned_as_draft():
    eid=draft_formula()
    r=c.get(f'{BASE}/entities/formula/{eid}')
    assert r.status_code==200, r.text
    body=r.json()
    assert body['entity_id']==eid
    assert body['entity_type']=='formula'
    assert body['name']=='Phase12B1 Draft Formula'
    assert body['review_status']=='DRAFT'
    assert body['version']==1
    assert body['content']['indications']==['phase12b1 symptom']
    assert body['content']['ingredients']==['Phase12B1 Herb']
    assert body['retired_at'] is None
    assert body['superseded_by_id'] is None


def test_lookup_is_exact_entity_type_plus_entity_id_not_name_search():
    eid=draft_formula()
    # Correct pair resolves.
    assert c.get(f'{BASE}/entities/formula/{eid}').status_code==200
    # Same id under a different entity type must not resolve.
    assert c.get(f'{BASE}/entities/herb/{eid}').status_code==404
    assert c.get(f'{BASE}/entities/pattern/{eid}').status_code==404
    # The name is not an id, and no fuzzy fallback exists.
    assert c.get(f'{BASE}/entities/formula/Phase12B1 Draft Formula').status_code==404
    assert c.get(f'{BASE}/entities/formula/{eid[:-1]}').status_code==404


def test_detail_returns_latest_version_after_governance_transition():
    eid=draft_formula()
    before=c.get(f'{BASE}/entities/formula/{eid}').json()
    assert before['version']==1 and before['review_status']=='DRAFT'
    approved_version=approve('formula',eid)
    after=c.get(f'{BASE}/entities/formula/{eid}').json()
    assert after['review_status']=='REVIEWED'
    assert after['version']==approved_version
    assert after['version']>before['version']


def test_retirement_and_supersession_metadata_is_reported():
    original=draft_formula()
    replacement=draft_formula()
    r=c.post(f'{BASE}/entities/formula/{original}/supersede',json={'actor_id':'admin','actor_role':'CLINICAL_ADMIN','superseded_by_id':replacement})
    assert r.status_code==200, r.text
    body=c.get(f'{BASE}/entities/formula/{original}').json()
    assert body['review_status']=='RETIRED'
    assert body['superseded_by_id']==replacement
    assert body['retired_at'] is not None
    # The replacement carries no retirement metadata.
    other=c.get(f'{BASE}/entities/formula/{replacement}').json()
    assert other['retired_at'] is None
    assert other['superseded_by_id'] is None


def test_sources_are_preserved_on_detail():
    source_id='src-12b1-'+uuid4().hex[:8]
    eid=ingest('herb',{'name':'Phase12B1 Sourced Herb'},sources=[{'source_id':source_id,'title':'Materia Medica','citation':'p. 42','url':'https://example.test/mm','source_type':'REFERENCE'}])
    body=c.get(f'{BASE}/entities/herb/{eid}').json()
    assert body['source_count']==1
    assert len(body['sources'])==1
    src=body['sources'][0]
    assert src['source_id']==source_id
    assert src['title']=='Materia Medica'
    assert src['citation']=='p. 42'
    assert src['url']=='https://example.test/mm'
    assert src['source_type']=='REFERENCE'
    # Sources survive a governance transition.
    approve('herb',eid)
    after=c.get(f'{BASE}/entities/herb/{eid}').json()
    assert after['source_count']==1
    assert after['sources'][0]['source_id']==source_id


def test_ranking_eligibility_is_preserved_not_recomputed_loosely():
    eid=draft_formula()
    assert c.get(f'{BASE}/entities/formula/{eid}').json()['clinical_ranking_eligible'] is False
    approve('formula',eid)
    reviewed=c.get(f'{BASE}/entities/formula/{eid}').json()
    assert reviewed['review_status']=='REVIEWED'
    assert reviewed['source_count']>0
    assert reviewed['clinical_ranking_eligible'] is True
    # Retirement removes eligibility.
    r=c.post(f'{BASE}/entities/formula/{eid}/retire',json={'actor_id':'reviewer','actor_role':'CLINICAL_REVIEWER'})
    assert r.status_code==200, r.text
    assert c.get(f'{BASE}/entities/formula/{eid}').json()['clinical_ranking_eligible'] is False


def test_entity_not_found_returns_404():
    r=c.get(f'{BASE}/entities/formula/frm-does-not-exist')
    assert r.status_code==404
    assert r.json()['detail']=='Clinical corpus entity not found'


def test_invalid_entity_type_is_rejected():
    eid=draft_formula()
    assert c.get(f'{BASE}/entities/potion/{eid}').status_code==422
    assert c.get(f'{BASE}/entities/Formula/{eid}').status_code==422


def test_detail_matches_latest_history_state():
    eid=draft_formula()
    approve('formula',eid)
    detail=c.get(f'{BASE}/entities/formula/{eid}').json()
    hist=c.get(f'{BASE}/entities/formula/{eid}/history').json()['results']
    latest=hist[-1]
    assert len(hist)==detail['version']
    assert latest['version']==detail['version']
    assert latest['review_status']==detail['review_status']
    assert latest['clinical_ranking_eligible']==detail['clinical_ranking_eligible']
    assert latest['formula_id']==detail['entity_id']
    assert latest['name']==detail['name']
    assert latest['entity_type']==detail['entity_type']
    assert {s['source_id'] for s in latest['sources']}=={s['source_id'] for s in detail['sources']}
    for key,value in detail['content'].items():
        assert latest[key]==value


def test_pattern_and_herb_detail_are_supported():
    pattern=ingest('pattern',{'name':'Phase12B1 Pattern','aliases':['p12b1'],'indications':['phase12b1 symptom'],'exclusion_flags':['fever']})
    herb=ingest('herb',{'name':'Phase12B1 Herb','contraindications':['pregnancy'],'interaction_flags':['warfarin']})
    p=c.get(f'{BASE}/entities/pattern/{pattern}').json()
    assert p['entity_type']=='pattern' and p['entity_id']==pattern
    assert p['content']['exclusion_flags']==['fever']
    h=c.get(f'{BASE}/entities/herb/{herb}').json()
    assert h['entity_type']=='herb' and h['entity_id']==herb
    assert h['content']['interaction_flags']==['warfarin']


def test_openapi_exposes_typed_response_schema():
    spec=c.get('/openapi.json').json()
    op=spec['paths']['/api/v1/knowledge/clinical/entities/{entity_type}/{entity_id}']['get']
    ref=op['responses']['200']['content']['application/json']['schema']['$ref']
    assert ref=='#/components/schemas/ClinicalEntityDetail'
    schema=spec['components']['schemas']['ClinicalEntityDetail']
    for field in ['entity_id','entity_type','name','version','review_status','clinical_ranking_eligible','sources','source_count','content','retired_at','superseded_by_id']:
        assert field in schema['properties'], field
