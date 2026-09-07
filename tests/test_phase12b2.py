import os
os.environ.setdefault('LLM_PROVIDER','mock')
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app

c=TestClient(app)
CLINICAL='/api/v1/knowledge/clinical'
REL='/api/v1/safety/relationships'


def reviewed(entity_type,payload):
    """Ingest + approve one entity; relationships require REVIEWED endpoints."""
    source_id='src-12b2-'+uuid4().hex[:8]
    r=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12b2','source_label':'phase12b2','items':[{'entity_type':entity_type,'payload':payload,'sources':[{'source_id':source_id,'title':'Reviewed source','citation':'phase12b2','source_type':'REFERENCE'}]}]})
    assert r.status_code==200, r.text
    eid=r.json()['created_entity_ids'][0]
    r=c.post(f'{CLINICAL}/entities/{entity_type}/{eid}/submit-review',json={'submitted_by':'phase12b2'})
    assert r.status_code==200, r.text
    r=c.post(f'{CLINICAL}/entities/{entity_type}/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':r.json()['version']})
    assert r.status_code==200, r.text
    return eid,source_id


def link(source_entity_id,target_entity_id,relationship_type,source_id=None):
    body={'source_entity_id':source_entity_id,'target_entity_id':target_entity_id,'relationship_type':relationship_type,'actor_id':'reviewer','actor_role':'CLINICAL_REVIEWER'}
    if source_id: body['source_id']=source_id
    r=c.post(REL,json=body)
    assert r.status_code==200, r.text
    return r.json()['id']


def pattern_formula_pair():
    pattern,_=reviewed('pattern',{'name':'Phase12B2 Pattern '+uuid4().hex[:6]})
    formula,src=reviewed('formula',{'name':'Phase12B2 Formula '+uuid4().hex[:6],'ingredients':['Phase12B2 Herb']})
    rel=link(pattern,formula,'PATTERN_FORMULA',src)
    return pattern,formula,rel


def test_pattern_to_formula_lookup():
    pattern,formula,rel=pattern_formula_pair()
    r=c.get(REL,params={'entity_id':pattern})
    assert r.status_code==200, r.text
    body=r.json()
    assert body['entity_id']==pattern
    assert body['count']==1 and len(body['results'])==1
    row=body['results'][0]
    assert row['id']==rel
    assert row['relationship_type']=='PATTERN_FORMULA'
    assert row['source_entity_id']==pattern
    assert row['target_entity_id']==formula
    assert row['source_entity_type']=='pattern'
    assert row['target_entity_type']=='formula'
    assert row['direction']=='outgoing'
    assert row['review_status']=='REVIEWED'
    assert row['created_by']=='reviewer'
    # The formula sees the same persisted row from the other side.
    other=c.get(REL,params={'entity_id':formula}).json()
    assert [x['id'] for x in other['results']]==[rel]
    assert other['results'][0]['direction']=='incoming'


def test_formula_to_herb_lookup():
    formula,src=reviewed('formula',{'name':'Phase12B2 Host Formula '+uuid4().hex[:6],'ingredients':['Phase12B2 Herb']})
    herb,_=reviewed('herb',{'name':'Phase12B2 Herb '+uuid4().hex[:6]})
    rel=link(formula,herb,'FORMULA_HERB',src)
    body=c.get(REL,params={'entity_id':formula,'relationship_type':'FORMULA_HERB'}).json()
    assert body['count']==1
    row=body['results'][0]
    assert row['id']==rel
    assert row['relationship_type']=='FORMULA_HERB'
    assert row['source_entity_id']==formula and row['target_entity_id']==herb
    assert row['source_entity_type']=='formula' and row['target_entity_type']=='herb'
    assert row['direction']=='outgoing'
    herb_side=c.get(REL,params={'entity_id':herb}).json()
    assert [x['id'] for x in herb_side['results']]==[rel]
    assert herb_side['results'][0]['direction']=='incoming'


def test_exact_entity_filtering():
    pattern_a,formula_a,rel_a=pattern_formula_pair()
    pattern_b,formula_b,rel_b=pattern_formula_pair()
    a=c.get(REL,params={'entity_id':pattern_a}).json()
    assert [x['id'] for x in a['results']]==[rel_a]
    b=c.get(REL,params={'entity_id':pattern_b}).json()
    assert [x['id'] for x in b['results']]==[rel_b]
    # No id prefix/substring matching and no name lookup.
    assert c.get(REL,params={'entity_id':pattern_a[:-1]}).json()['count']==0
    assert c.get(REL,params={'entity_id':'Phase12B2 Pattern'}).json()['count']==0


def test_direction_and_type_filters_narrow_exactly():
    formula,src=reviewed('formula',{'name':'Phase12B2 Hub Formula '+uuid4().hex[:6],'ingredients':['Phase12B2 Herb']})
    pattern,_=reviewed('pattern',{'name':'Phase12B2 Hub Pattern '+uuid4().hex[:6]})
    herb,_=reviewed('herb',{'name':'Phase12B2 Hub Herb '+uuid4().hex[:6]})
    incoming=link(pattern,formula,'PATTERN_FORMULA',src)
    outgoing=link(formula,herb,'FORMULA_HERB',src)
    both=c.get(REL,params={'entity_id':formula}).json()
    assert {x['id'] for x in both['results']}=={incoming,outgoing}
    out=c.get(REL,params={'entity_id':formula,'direction':'outgoing'}).json()
    assert [x['id'] for x in out['results']]==[outgoing]
    inc=c.get(REL,params={'entity_id':formula,'direction':'incoming'}).json()
    assert [x['id'] for x in inc['results']]==[incoming]
    typed=c.get(REL,params={'entity_id':formula,'relationship_type':'PATTERN_FORMULA'}).json()
    assert [x['id'] for x in typed['results']]==[incoming]


def test_empty_result_is_an_empty_list_not_an_error():
    lonely,_=reviewed('herb',{'name':'Phase12B2 Unlinked Herb '+uuid4().hex[:6]})
    r=c.get(REL,params={'entity_id':lonely})
    assert r.status_code==200
    body=r.json()
    assert body['count']==0 and body['results']==[]
    assert body['entity_id']==lonely
    # An id that does not exist is an empty query result, not a 404.
    unknown=c.get(REL,params={'entity_id':'frm-does-not-exist'})
    assert unknown.status_code==200
    assert unknown.json()['count']==0
    # A real entity filtered to a type it has no rows for is also empty.
    pattern,_,_=pattern_formula_pair()
    assert c.get(REL,params={'entity_id':pattern,'relationship_type':'FORMULA_HERB'}).json()['count']==0


def test_invalid_filters_and_types_are_rejected():
    pattern,_,_=pattern_formula_pair()
    assert c.get(REL).status_code==422
    assert c.get(REL,params={'entity_id':''}).status_code==422
    assert c.get(REL,params={'entity_id':pattern,'relationship_type':'PATTERN_HERB'}).status_code==422
    assert c.get(REL,params={'entity_id':pattern,'relationship_type':'pattern_formula'}).status_code==422
    assert c.get(REL,params={'entity_id':pattern,'direction':'sideways'}).status_code==422
    assert c.get(REL,params={'entity_id':pattern,'direction':'BOTH'}).status_code==422


def test_read_endpoint_does_not_mutate_relationship_records():
    pattern,formula,rel=pattern_formula_pair()
    before=c.get(REL,params={'entity_id':pattern}).json()
    for _ in range(3):
        c.get(REL,params={'entity_id':pattern})
        c.get(REL,params={'entity_id':formula,'direction':'incoming'})
    after=c.get(REL,params={'entity_id':pattern}).json()
    assert before==after
    assert after['results'][0]['id']==rel
    assert after['count']==1


def test_existing_post_relationship_behavior_is_unchanged():
    formula,src=reviewed('formula',{'name':'Phase12B2 POST Formula '+uuid4().hex[:6],'ingredients':['Phase12B2 Herb']})
    herb,_=reviewed('herb',{'name':'Phase12B2 POST Herb '+uuid4().hex[:6]})
    ok=c.post(REL,json={'source_entity_id':formula,'target_entity_id':herb,'relationship_type':'FORMULA_HERB','source_id':src,'actor_id':'reviewer','actor_role':'CLINICAL_REVIEWER'})
    assert ok.status_code==200, ok.text
    assert set(ok.json())=={'id','source_entity_id','target_entity_id','relationship_type','review_status'}
    assert ok.json()['review_status']=='REVIEWED'
    # Governance still enforced exactly as before.
    unauthorized=c.post(REL,json={'source_entity_id':formula,'target_entity_id':herb,'relationship_type':'FORMULA_HERB','actor_id':'x','actor_role':'PATIENT'})
    assert unauthorized.status_code==422 and 'not authorized' in unauthorized.json()['detail']
    missing=c.post(REL,json={'source_entity_id':'nope','target_entity_id':herb,'relationship_type':'FORMULA_HERB','actor_id':'r','actor_role':'CLINICAL_REVIEWER'})
    assert missing.status_code==422 and 'must exist' in missing.json()['detail']
    mismatched=c.post(REL,json={'source_entity_id':herb,'target_entity_id':formula,'relationship_type':'FORMULA_HERB','actor_id':'r','actor_role':'CLINICAL_REVIEWER'})
    assert mismatched.status_code==422 and 'do not match' in mismatched.json()['detail']
    fresh=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12b2','source_label':'phase12b2','items':[{'entity_type':'herb','payload':{'name':'Phase12B2 Never Reviewed'},'sources':[]}]}).json()['created_entity_ids'][0]
    unreviewed=c.post(REL,json={'source_entity_id':formula,'target_entity_id':fresh,'relationship_type':'FORMULA_HERB','actor_id':'r','actor_role':'CLINICAL_REVIEWER'})
    assert unreviewed.status_code==422 and 'REVIEWED' in unreviewed.json()['detail']


def test_typed_openapi_schema_is_exposed():
    spec=c.get('/openapi.json').json()
    op=spec['paths']['/api/v1/safety/relationships']['get']
    ref=op['responses']['200']['content']['application/json']['schema']['$ref']
    assert ref=='#/components/schemas/RelationshipQueryResponse'
    schema=spec['components']['schemas']['RelationshipQueryResponse']
    for field in ['entity_id','direction','relationship_type','count','results']:
        assert field in schema['properties'], field
    record=spec['components']['schemas']['ClinicalRelationshipRecord']
    for field in ['id','source_entity_id','target_entity_id','relationship_type','review_status','source_id','created_by','created_at','direction']:
        assert field in record['properties'], field
    assert set(spec['components']['schemas']['RelationshipType']['enum'])=={'PATTERN_FORMULA','FORMULA_HERB'}
    # The POST on the same path is still its own operation.
    assert 'post' in spec['paths']['/api/v1/safety/relationships']
