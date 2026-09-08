import os
os.environ.setdefault('LLM_PROVIDER','mock')
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app

c=TestClient(app)
CLINICAL='/api/v1/knowledge/clinical'
SOURCES=f'{CLINICAL}/sources'
TAG=uuid4().hex[:8]


def sid(suffix=''):
    return f'src-12c2b-{TAG}-{suffix or uuid4().hex[:6]}'


def source(source_id,title=None,citation='p.1',url='https://example.test/canon',source_type='REFERENCE'):
    ref={'source_id':source_id,'title':title or f'Phase12C2B Text {TAG}','source_type':source_type}
    if citation is not None: ref['citation']=citation
    if url is not None: ref['url']=url
    return ref


def ingest(sources,entity_type='formula',name=None):
    r=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2b','source_label':'phase12c2b',
        'items':[{'entity_type':entity_type,'payload':{'name':name or ('Phase12C2B '+uuid4().hex[:6])},'sources':sources}]})
    assert r.status_code==200, r.text
    return r.json()['created_entity_ids'][0]


def register(source_id,**kw):
    """Create a canonical Source by attaching it to a throwaway entity."""
    return ingest([source(source_id,**kw)])


# --- A, B ------------------------------------------------------------------
def test_a_list_sources():
    s=sid()
    register(s)
    r=c.get(SOURCES,params={'query':f'Phase12C2B Text {TAG}'})
    assert r.status_code==200, r.text
    body=r.json()
    assert body['limit']==20 and body['offset']==0
    assert body['count']>=1
    ids={x['source_id'] for x in body['results']}
    assert s in ids
    row=[x for x in body['results'] if x['source_id']==s][0]
    assert set(row)=={'source_id','title','citation','url','source_type','created_at'}
    assert row['title']==f'Phase12C2B Text {TAG}'
    assert row['citation']=='p.1'
    assert row['url']=='https://example.test/canon'
    assert row['source_type']=='REFERENCE'
    assert row['created_at']


def test_b_empty_list_is_not_an_error():
    r=c.get(SOURCES,params={'query':'no-such-source-text-'+uuid4().hex})
    assert r.status_code==200
    body=r.json()
    assert body['count']==0 and body['results']==[]
    assert body['limit']==20 and body['offset']==0


# --- C, D, E ---------------------------------------------------------------
def test_c_exact_source_id_filter():
    a,b=sid(),sid()
    register(a); register(b)
    body=c.get(SOURCES,params={'source_id':a}).json()
    assert body['count']==1
    assert [x['source_id'] for x in body['results']]==[a]
    # No prefix or substring matching on the id filter.
    assert c.get(SOURCES,params={'source_id':a[:-1]}).json()['count']==0


def test_d_source_type_filter():
    typed=f'PHASE12C2B_TYPE_{TAG}'
    a=sid(); b=sid()
    register(a,source_type=typed)
    register(b,source_type='REFERENCE')
    body=c.get(SOURCES,params={'source_type':typed}).json()
    assert body['count']==1
    assert [x['source_id'] for x in body['results']]==[a]
    assert all(x['source_type']==typed for x in body['results'])
    assert c.get(SOURCES,params={'source_type':'NO_SUCH_TYPE_'+TAG}).json()['count']==0


def test_e_text_query_searches_title_citation_and_url():
    marker=f'Zzmarker{TAG}'
    by_title=sid(); by_citation=sid(); by_url=sid()
    register(by_title,title=f'Title {marker}')
    register(by_citation,citation=f'citation {marker}')
    register(by_url,url=f'https://example.test/{marker}')
    found={x['source_id'] for x in c.get(SOURCES,params={'query':marker,'limit':100}).json()['results']}
    assert {by_title,by_citation,by_url}<=found
    # Case-insensitive.
    lower={x['source_id'] for x in c.get(SOURCES,params={'query':marker.lower(),'limit':100}).json()['results']}
    assert {by_title,by_citation,by_url}<=lower


def test_e2_wildcards_in_query_are_literal_not_patterns():
    s=sid()
    register(s,title=f'Literal 100% {TAG} match')
    # '%' must be escaped, so a bare '%' cannot match everything.
    hits=c.get(SOURCES,params={'query':'%','limit':100}).json()
    assert all('%' in (x['title'] or '')+(x['citation'] or '')+(x['url'] or '') for x in hits['results'])
    exact=c.get(SOURCES,params={'query':f'100% {TAG}','limit':100}).json()
    assert s in {x['source_id'] for x in exact['results']}


# --- F ---------------------------------------------------------------------
def test_f_pagination_limit_and_offset():
    marker=f'Pagemark{TAG}'
    made=sorted(sid(f'page{i}') for i in range(5))
    for x in made: register(x,title=f'{marker} entry')
    first=c.get(SOURCES,params={'query':marker,'limit':2,'offset':0}).json()
    assert first['count']==5 and first['limit']==2 and first['offset']==0
    assert len(first['results'])==2
    second=c.get(SOURCES,params={'query':marker,'limit':2,'offset':2}).json()
    assert second['count']==5 and len(second['results'])==2
    third=c.get(SOURCES,params={'query':marker,'limit':2,'offset':4}).json()
    assert len(third['results'])==1
    paged=[x['source_id'] for x in first['results']+second['results']+third['results']]
    assert paged==made                      # deterministic order, no gaps or repeats
    assert len(set(paged))==5
    past_end=c.get(SOURCES,params={'query':marker,'limit':2,'offset':99}).json()
    assert past_end['count']==5 and past_end['results']==[]


def test_f2_invalid_pagination_is_rejected():
    assert c.get(SOURCES,params={'limit':0}).status_code==422
    assert c.get(SOURCES,params={'limit':101}).status_code==422
    assert c.get(SOURCES,params={'offset':-1}).status_code==422
    assert c.get(SOURCES,params={'source_id':''}).status_code==422


# --- G, H ------------------------------------------------------------------
def test_g_get_one_existing_source():
    s=sid()
    register(s)
    r=c.get(f'{SOURCES}/{s}')
    assert r.status_code==200, r.text
    body=r.json()
    assert body['source_id']==s
    assert set(body)=={'source_id','title','citation','url','source_type','created_at'}
    # Identical to the same row seen through the list endpoint.
    assert body==c.get(SOURCES,params={'source_id':s}).json()['results'][0]


def test_h_nonexistent_source_is_404():
    r=c.get(f'{SOURCES}/src-does-not-exist-{TAG}')
    assert r.status_code==404
    assert r.json()['detail']=='Clinical source not found'


# --- I, J, K, L, M ---------------------------------------------------------
def test_i_reverse_lookup_one_entity():
    s=sid()
    eid=register(s)
    r=c.get(f'{SOURCES}/{s}/entities')
    assert r.status_code==200, r.text
    body=r.json()
    assert body['source_id']==s
    assert body['count']==1 and len(body['results'])==1
    row=body['results'][0]
    assert set(row)=={'entity_id','entity_type','name','current_version','review_status','clinical_ranking_eligible'}
    assert row['entity_id']==eid
    assert row['entity_type']=='formula'
    assert row['current_version']==1
    assert row['review_status']=='DRAFT'
    assert row['clinical_ranking_eligible'] is False


def test_j_reverse_lookup_shared_source_returns_multiple_entities():
    s=sid()
    first=register(s)
    second=ingest([source(s)],entity_type='herb')
    third=ingest([source(s)],entity_type='pattern')
    body=c.get(f'{SOURCES}/{s}/entities').json()
    assert body['count']==3
    assert {x['entity_id'] for x in body['results']}=={first,second,third}
    assert {x['entity_type'] for x in body['results']}=={'formula','herb','pattern'}


def test_k_source_with_zero_entity_references_is_count_zero_not_404():
    """An orphan Source is a real Source with no citations - 200, not 404.

    Ingestion always attaches a Source to an entity, so the unreferenced case is
    seeded directly against the test database.
    """
    from app.db.models import SourceRegistry
    from app.db.session import get_session_factory
    orphan=sid('orphan')
    with get_session_factory().begin() as s:
        s.add(SourceRegistry(source_id=orphan,title=f'Phase12C2B Orphan {TAG}',citation=None,url=None,source_type='REFERENCE'))
    # The Source itself resolves.
    got=c.get(f'{SOURCES}/{orphan}')
    assert got.status_code==200, got.text
    assert got.json()['source_id']==orphan
    # It is listable.
    listed=c.get(SOURCES,params={'source_id':orphan}).json()
    assert listed['count']==1
    # Reverse lookup is an empty result, explicitly not a 404.
    r=c.get(f'{SOURCES}/{orphan}/entities')
    assert r.status_code==200, r.text
    body=r.json()
    assert body['source_id']==orphan
    assert body['count']==0
    assert body['results']==[]


def test_k2_reverse_lookup_reflects_review_state_without_recomputing_eligibility():
    s=sid()
    eid=register(s)
    before=c.get(f'{SOURCES}/{s}/entities').json()['results'][0]
    assert before['review_status']=='DRAFT' and before['clinical_ranking_eligible'] is False
    r=c.post(f'{CLINICAL}/entities/formula/{eid}/submit-review',json={'submitted_by':'phase12c2b'})
    assert r.status_code==200, r.text
    r=c.post(f'{CLINICAL}/entities/formula/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':r.json()['version']})
    assert r.status_code==200, r.text
    after=c.get(f'{SOURCES}/{s}/entities').json()['results'][0]
    assert after['review_status']=='REVIEWED'
    assert after['clinical_ranking_eligible'] is True
    # Matches the entity detail endpoint exactly - one eligibility rule, not two.
    detail=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    assert after['clinical_ranking_eligible']==detail['clinical_ranking_eligible']
    assert after['current_version']==detail['version']
    assert after['name']==detail['name']


def test_l_reverse_lookup_of_nonexistent_source_is_404():
    r=c.get(f'{SOURCES}/src-nope-{TAG}/entities')
    assert r.status_code==404
    assert r.json()['detail']=='Clinical source not found'


def test_m_no_duplicate_entity_results():
    s=sid()
    eid=register(s)
    body=c.get(f'{SOURCES}/{s}/entities').json()
    ids=[x['entity_id'] for x in body['results']]
    assert len(ids)==len(set(ids))
    assert ids==[eid]
    # An entity citing several sources still appears once per source lookup.
    multi_a, multi_b = sid(), sid()
    both=ingest([source(multi_a),source(multi_b)])
    for other in (multi_a,multi_b):
        rows=c.get(f'{SOURCES}/{other}/entities').json()['results']
        assert [x['entity_id'] for x in rows]==[both]


def test_m2_reverse_lookup_does_not_mix_relationship_or_safety_rule_sources():
    """entity_source references only - clinical_relationship/safety_rule excluded."""
    s=sid()
    formula=ingest([source(s)])
    herb=ingest([source(s)],entity_type='herb')
    for eid,typ in ((formula,'formula'),(herb,'herb')):
        r=c.post(f'{CLINICAL}/entities/{typ}/{eid}/submit-review',json={'submitted_by':'phase12c2b'})
        c.post(f'{CLINICAL}/entities/{typ}/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':r.json()['version']})
    before={x['entity_id'] for x in c.get(f'{SOURCES}/{s}/entities').json()['results']}
    # A relationship and a safety rule both cite the same source_id.
    rel=c.post('/api/v1/safety/relationships',json={'source_entity_id':formula,'target_entity_id':herb,'relationship_type':'FORMULA_HERB','source_id':s,'actor_id':'reviewer','actor_role':'CLINICAL_REVIEWER'})
    assert rel.status_code==200, rel.text
    rule=c.post('/api/v1/safety/rules',json={'target_entity_id':herb,'rule_type':'ALLERGY','trigger_term':'x','message':'x','source_id':s,'actor_id':'reviewer','actor_role':'CLINICAL_REVIEWER'})
    assert rule.status_code==200, rule.text
    after=c.get(f'{SOURCES}/{s}/entities').json()
    assert {x['entity_id'] for x in after['results']}==before
    assert after['count']==2


# --- N ---------------------------------------------------------------------
def test_n_typed_openapi_schemas_and_routes():
    spec=c.get('/openapi.json').json()
    paths=spec['paths']
    assert paths[SOURCES]['get']['responses']['200']['content']['application/json']['schema']['$ref']=='#/components/schemas/SourceListResponse'
    assert paths[SOURCES+'/{source_id}']['get']['responses']['200']['content']['application/json']['schema']['$ref']=='#/components/schemas/SourceRecord'
    assert paths[SOURCES+'/{source_id}/entities']['get']['responses']['200']['content']['application/json']['schema']['$ref']=='#/components/schemas/SourceEntitiesResponse'
    comp=spec['components']['schemas']
    assert set(comp['SourceRecord']['properties'])=={'source_id','title','citation','url','source_type','created_at'}
    for f in ['count','limit','offset','results']:
        assert f in comp['SourceListResponse']['properties'], f
    for f in ['source_id','count','results']:
        assert f in comp['SourceEntitiesResponse']['properties'], f
    assert set(comp['SourceEntityRef']['properties'])=={'entity_id','entity_type','name','current_version','review_status','clinical_ranking_eligible'}
    params={p['name'] for p in paths[SOURCES]['get']['parameters']}
    assert params=={'source_id','source_type','query','limit','offset'}
    # These endpoints are read-only: no write verbs on any source path.
    for p in (SOURCES, SOURCES+'/{source_id}', SOURCES+'/{source_id}/entities'):
        assert set(paths[p])=={'get'}, (p,set(paths[p]))


# --- O ---------------------------------------------------------------------
def test_o_get_operations_cause_no_writes():
    s=sid()
    eid=register(s)
    stats_before=c.get(f'{CLINICAL}/stats').json()
    detail_before=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    history_before=c.get(f'{CLINICAL}/entities/formula/{eid}/history').json()
    audit_before=c.get(f'{CLINICAL}/audit').json()
    source_before=c.get(f'{SOURCES}/{s}').json()
    for _ in range(3):
        c.get(SOURCES,params={'limit':100})
        c.get(f'{SOURCES}/{s}')
        c.get(f'{SOURCES}/{s}/entities')
        c.get(f'{SOURCES}/src-missing-{TAG}')
        c.get(f'{SOURCES}/src-missing-{TAG}/entities')
    assert c.get(f'{CLINICAL}/stats').json()==stats_before
    assert c.get(f'{CLINICAL}/entities/formula/{eid}').json()==detail_before
    assert c.get(f'{CLINICAL}/entities/formula/{eid}/history').json()==history_before
    assert c.get(f'{CLINICAL}/audit').json()==audit_before      # no audit events created
    assert c.get(f'{SOURCES}/{s}').json()==source_before        # source metadata untouched


# --- P ---------------------------------------------------------------------
def test_p_phase12c2a_conflict_behavior_still_enforced():
    s=sid()
    register(s)
    conflict=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2b','source_label':'phase12c2b','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2B Conflict'},'sources':[source(s,title='Different Title')]}]})
    assert conflict.status_code==409, conflict.text
    assert conflict.json()['detail']['error']=='SOURCE_METADATA_CONFLICT'
    # The canonical source the read API serves is unchanged.
    assert c.get(f'{SOURCES}/{s}').json()['title']==f'Phase12C2B Text {TAG}'
    # And the rejected entity never became a reference.
    assert c.get(f'{SOURCES}/{s}/entities').json()['count']==1
    dup=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2b','source_label':'phase12c2b','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2B Dup'},'sources':[source(sid()),]*2}]})
    assert dup.status_code==422, dup.text
