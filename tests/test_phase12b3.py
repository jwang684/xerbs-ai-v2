import os
os.environ.setdefault('LLM_PROVIDER','mock')
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app

c=TestClient(app)
CLINICAL='/api/v1/knowledge/clinical'
RULES='/api/v1/safety/rules'


def reviewed(entity_type,payload):
    """Ingest + approve one entity; safety rules require a REVIEWED target."""
    source_id='src-12b3-'+uuid4().hex[:8]
    r=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12b3','source_label':'phase12b3','items':[{'entity_type':entity_type,'payload':payload,'sources':[{'source_id':source_id,'title':'Reviewed source','citation':'phase12b3','source_type':'REFERENCE'}]}]})
    assert r.status_code==200, r.text
    eid=r.json()['created_entity_ids'][0]
    r=c.post(f'{CLINICAL}/entities/{entity_type}/{eid}/submit-review',json={'submitted_by':'phase12b3'})
    assert r.status_code==200, r.text
    r=c.post(f'{CLINICAL}/entities/{entity_type}/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':r.json()['version']})
    assert r.status_code==200, r.text
    return eid,source_id


def add_rule(target_entity_id,source_id,rule_type='DRUG_INTERACTION',trigger_term='warfarin',severity='CRITICAL',action='BLOCK',message='Reviewed interaction'):
    r=c.post(RULES,json={'target_entity_id':target_entity_id,'rule_type':rule_type,'trigger_term':trigger_term,'severity':severity,'action':action,'message':message,'source_id':source_id,'actor_id':'reviewer','actor_role':'CLINICAL_REVIEWER'})
    assert r.status_code==200, r.text
    return r.json()['id']


def ruled_herb():
    herb,src=reviewed('herb',{'name':'Phase12B3 Herb '+uuid4().hex[:6]})
    rule=add_rule(herb,src)
    return herb,src,rule


def test_entity_specific_lookup():
    herb,src,rule=ruled_herb()
    r=c.get(RULES,params={'target_entity_id':herb})
    assert r.status_code==200, r.text
    body=r.json()
    assert body['target_entity_id']==herb
    assert body['count']==1 and len(body['results'])==1
    row=body['results'][0]
    assert row['id']==rule
    assert row['target_entity_id']==herb
    assert row['target_entity_type']=='herb'
    assert row['rule_type']=='DRUG_INTERACTION'
    assert row['trigger_term']=='warfarin'
    assert row['severity']=='CRITICAL'
    assert row['action']=='BLOCK'
    assert row['message']=='Reviewed interaction'
    assert row['review_status']=='REVIEWED'
    assert row['source_id']==src
    assert row['created_by']=='reviewer'


def test_exact_id_filtering():
    herb_a,_,rule_a=ruled_herb()
    herb_b,_,rule_b=ruled_herb()
    a=c.get(RULES,params={'target_entity_id':herb_a}).json()
    assert [x['id'] for x in a['results']]==[rule_a]
    b=c.get(RULES,params={'target_entity_id':herb_b}).json()
    assert [x['id'] for x in b['results']]==[rule_b]
    # No prefix/substring matching and no name lookup.
    assert c.get(RULES,params={'target_entity_id':herb_a[:-1]}).json()['count']==0
    assert c.get(RULES,params={'target_entity_id':'Phase12B3 Herb'}).json()['count']==0


def test_rule_type_filter_narrows_exactly():
    herb,src=reviewed('herb',{'name':'Phase12B3 Multi Herb '+uuid4().hex[:6]})
    interaction=add_rule(herb,src,rule_type='DRUG_INTERACTION',trigger_term='warfarin')
    pregnancy=add_rule(herb,src,rule_type='PREGNANCY',trigger_term='pregnancy',severity='HIGH',action='WARN',message='Avoid in pregnancy')
    both=c.get(RULES,params={'target_entity_id':herb}).json()
    assert both['count']==2
    assert {x['id'] for x in both['results']}=={interaction,pregnancy}
    only=c.get(RULES,params={'target_entity_id':herb,'rule_type':'PREGNANCY'}).json()
    assert [x['id'] for x in only['results']]==[pregnancy]
    assert only['rule_type']=='PREGNANCY'
    other=c.get(RULES,params={'target_entity_id':herb,'rule_type':'ALLERGY'}).json()
    assert other['count']==0


def test_empty_result_never_asserts_safety():
    herb,_=reviewed('herb',{'name':'Phase12B3 Unruled Herb '+uuid4().hex[:6]})
    r=c.get(RULES,params={'target_entity_id':herb})
    assert r.status_code==200
    body=r.json()
    assert body['count']==0 and body['results']==[]
    # An empty result is absence of evidence, never a safety claim.
    assert body['absence_of_rules_implies_safety'] is False
    assert 'not a safety determination' in body['result_meaning']
    for banned in ['eligible_for_selection','risk_level','findings','safe']:
        assert banned not in body, banned
    # An id that does not exist is an empty query result, not a 404.
    unknown=c.get(RULES,params={'target_entity_id':'herb-does-not-exist'})
    assert unknown.status_code==200
    assert unknown.json()['count']==0
    assert unknown.json()['absence_of_rules_implies_safety'] is False
    # A rule-bearing entity filtered to an unused type is also empty, still not "safe".
    ruled,_,_=ruled_herb()
    narrowed=c.get(RULES,params={'target_entity_id':ruled,'rule_type':'AGE'}).json()
    assert narrowed['count']==0 and narrowed['absence_of_rules_implies_safety'] is False


def test_non_empty_result_also_carries_the_no_safety_claim_guard():
    herb,_,_=ruled_herb()
    body=c.get(RULES,params={'target_entity_id':herb}).json()
    assert body['count']==1
    assert body['absence_of_rules_implies_safety'] is False
    assert 'Persisted safety rules only' in body['result_meaning']


def test_invalid_filters_and_types_are_rejected():
    herb,_,_=ruled_herb()
    assert c.get(RULES).status_code==422
    assert c.get(RULES,params={'target_entity_id':''}).status_code==422
    assert c.get(RULES,params={'target_entity_id':herb,'rule_type':'NOT_A_RULE'}).status_code==422
    assert c.get(RULES,params={'target_entity_id':herb,'rule_type':'drug_interaction'}).status_code==422
    assert c.get(RULES,params={'target_entity_id':herb,'rule_type':'CONTRAINDICATIONS'}).status_code==422


def test_read_endpoint_does_not_mutate_rule_records():
    herb,_,rule=ruled_herb()
    before=c.get(RULES,params={'target_entity_id':herb}).json()
    for _ in range(3):
        c.get(RULES,params={'target_entity_id':herb})
        c.get(RULES,params={'target_entity_id':herb,'rule_type':'DRUG_INTERACTION'})
    after=c.get(RULES,params={'target_entity_id':herb}).json()
    assert before==after
    assert after['count']==1 and after['results'][0]['id']==rule


def test_read_endpoint_does_not_change_screening_behavior():
    formula,src=reviewed('formula',{'name':'Phase12B3 Screen Formula '+uuid4().hex[:6],'ingredients':['Phase12B3 Screen Herb']})
    add_rule(formula,src,rule_type='DRUG_INTERACTION',trigger_term='warfarin',severity='CRITICAL',action='BLOCK',message='Blocking interaction')
    payload={'formula_id':formula,'formula_name':'Phase12B3 Screen Formula','ingredients':['Phase12B3 Screen Herb'],'patient_context':{'medications':['warfarin']}}
    before=c.post('/api/v1/safety/screen',json=payload).json()
    c.get(RULES,params={'target_entity_id':formula})
    after=c.post('/api/v1/safety/screen',json=payload).json()
    assert before==after
    assert after['eligible_for_selection'] is False
    assert after['risk_level']=='CRITICAL'
    # Screening still owns eligibility; the read API never reports it.
    assert 'eligible_for_selection' not in c.get(RULES,params={'target_entity_id':formula}).json()


def test_existing_post_safety_rule_behavior_is_unchanged():
    herb,src=reviewed('herb',{'name':'Phase12B3 POST Herb '+uuid4().hex[:6]})
    ok=c.post(RULES,json={'target_entity_id':herb,'rule_type':'CONTRAINDICATION','trigger_term':'pregnancy','severity':'HIGH','action':'WARN','message':'Reviewed contraindication','source_id':src,'actor_id':'reviewer','actor_role':'CLINICAL_REVIEWER'})
    assert ok.status_code==200, ok.text
    assert set(ok.json())=={'id','target_entity_id','rule_type','review_status'}
    assert ok.json()['review_status']=='REVIEWED'
    # Governance still enforced exactly as before.
    unauthorized=c.post(RULES,json={'target_entity_id':herb,'rule_type':'ALLERGY','trigger_term':'x','message':'x','source_id':src,'actor_id':'x','actor_role':'PATIENT'})
    assert unauthorized.status_code==422 and 'not authorized' in unauthorized.json()['detail']
    no_source=c.post(RULES,json={'target_entity_id':herb,'rule_type':'ALLERGY','trigger_term':'x','message':'x','actor_id':'r','actor_role':'CLINICAL_REVIEWER'})
    assert no_source.status_code==422 and 'source provenance' in no_source.json()['detail']
    bad_source=c.post(RULES,json={'target_entity_id':herb,'rule_type':'ALLERGY','trigger_term':'x','message':'x','source_id':'src-nope','actor_id':'r','actor_role':'CLINICAL_REVIEWER'})
    assert bad_source.status_code==422 and 'Source not found' in bad_source.json()['detail']
    pattern,psrc=reviewed('pattern',{'name':'Phase12B3 Pattern '+uuid4().hex[:6]})
    bad_target=c.post(RULES,json={'target_entity_id':pattern,'rule_type':'ALLERGY','trigger_term':'x','message':'x','source_id':psrc,'actor_id':'r','actor_role':'CLINICAL_REVIEWER'})
    assert bad_target.status_code==422 and 'formula or herb' in bad_target.json()['detail']
    fresh=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12b3','source_label':'phase12b3','items':[{'entity_type':'herb','payload':{'name':'Phase12B3 Never Reviewed'},'sources':[]}]}).json()['created_entity_ids'][0]
    unreviewed=c.post(RULES,json={'target_entity_id':fresh,'rule_type':'ALLERGY','trigger_term':'x','message':'x','source_id':src,'actor_id':'r','actor_role':'CLINICAL_REVIEWER'})
    assert unreviewed.status_code==422 and 'REVIEWED' in unreviewed.json()['detail']


def test_typed_openapi_schema_is_exposed():
    spec=c.get('/openapi.json').json()
    op=spec['paths']['/api/v1/safety/rules']['get']
    ref=op['responses']['200']['content']['application/json']['schema']['$ref']
    assert ref=='#/components/schemas/SafetyRuleQueryResponse'
    schema=spec['components']['schemas']['SafetyRuleQueryResponse']
    for field in ['target_entity_id','rule_type','count','results','absence_of_rules_implies_safety','result_meaning']:
        assert field in schema['properties'], field
    record=spec['components']['schemas']['SafetyRuleRecord']
    for field in ['id','target_entity_id','rule_type','trigger_term','severity','action','message','review_status','source_id','created_by','created_at']:
        assert field in record['properties'], field
    # Existing rule types preserved exactly, none invented.
    assert set(spec['components']['schemas']['SafetyRuleType']['enum'])=={'CONTRAINDICATION','DRUG_INTERACTION','ALLERGY','PREGNANCY','AGE','DIAGNOSIS'}
    # The POST on the same path is still its own operation.
    assert 'post' in spec['paths']['/api/v1/safety/rules']
