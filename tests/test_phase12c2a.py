import os
os.environ.setdefault('LLM_PROVIDER','mock')
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app

c=TestClient(app)
CLINICAL='/api/v1/knowledge/clinical'


def sid():
    return 'src-12c2a-'+uuid4().hex[:10]


def ingest(sources,entity_type='formula',name=None,payload=None):
    """Raw ingest call so conflict/validation status codes stay observable."""
    body={'name':name or ('Phase12C2A '+uuid4().hex[:6]),**(payload or {})}
    return c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2a','source_label':'phase12c2a',
        'items':[{'entity_type':entity_type,'payload':body,'sources':sources}]})


def source(source_id,title='Canonical Text',citation='p.1',url='https://example.test/canon',source_type='REFERENCE'):
    ref={'source_id':source_id,'title':title,'source_type':source_type}
    if citation is not None: ref['citation']=citation
    if url is not None: ref['url']=url
    return ref


def registered(source_id):
    """Canonical persisted Source, read back through the entity detail API."""
    r=ingest([source(source_id)])
    assert r.status_code==200, r.text
    return r.json()['created_entity_ids'][0]


# --- A ---------------------------------------------------------------------
def test_a_new_source_id_is_created_normally():
    s=sid()
    r=ingest([source(s)])
    assert r.status_code==200, r.text
    eid=r.json()['created_entity_ids'][0]
    detail=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    assert detail['source_count']==1
    got=detail['sources'][0]
    assert got['source_id']==s
    assert got['title']=='Canonical Text'
    assert got['citation']=='p.1'
    assert got['url']=='https://example.test/canon'
    assert got['source_type']=='REFERENCE'


# --- B ---------------------------------------------------------------------
def test_b_identical_metadata_is_reused_not_duplicated():
    s=sid()
    first=registered(s)
    again=ingest([source(s)])
    assert again.status_code==200, again.text
    second=again.json()['created_entity_ids'][0]
    assert second!=first
    # Same canonical Source reused by both entities, no duplicate row.
    for eid in (first,second):
        d=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
        assert d['source_count']==1
        assert d['sources'][0]['source_id']==s
        assert d['sources'][0]['title']=='Canonical Text'


def test_b2_whitespace_only_difference_is_still_identical():
    s=sid()
    registered(s)
    r=ingest([source(s,title='  Canonical Text  ',citation=' p.1 ')])
    assert r.status_code==200, r.text


# --- C, D, E, F ------------------------------------------------------------
def _conflict(field,**override):
    s=sid()
    registered(s)
    r=ingest([source(s,**override)])
    assert r.status_code==409, r.text
    detail=r.json()['detail']
    assert detail['error']=='SOURCE_METADATA_CONFLICT'
    assert detail['source_id']==s
    assert 'canonical metadata' in detail['message']
    fields={x['field'] for x in detail['conflicting_fields']}
    assert fields=={field}, fields
    return s,detail


def test_c_conflicting_title_is_409():
    s,detail=_conflict('title',title='A Completely Different Title')
    row=[x for x in detail['conflicting_fields'] if x['field']=='title'][0]
    assert row['persisted']=='Canonical Text'
    assert row['submitted']=='A Completely Different Title'


def test_d_conflicting_citation_is_409():
    s,detail=_conflict('citation',citation='p.999')
    row=[x for x in detail['conflicting_fields'] if x['field']=='citation'][0]
    assert row['persisted']=='p.1' and row['submitted']=='p.999'


def test_e_conflicting_url_is_409():
    s,detail=_conflict('url',url='https://example.test/other')
    row=[x for x in detail['conflicting_fields'] if x['field']=='url'][0]
    assert row['persisted']=='https://example.test/canon'
    assert row['submitted']=='https://example.test/other'


def test_f_conflicting_source_type_is_409():
    s,detail=_conflict('source_type',source_type='FABRICATED')
    row=[x for x in detail['conflicting_fields'] if x['field']=='source_type'][0]
    assert row['persisted']=='REFERENCE' and row['submitted']=='FABRICATED'


def test_f2_multiple_conflicting_fields_are_all_reported():
    s=sid()
    registered(s)
    r=ingest([source(s,title='Other',citation='p.2',url='https://example.test/x',source_type='OTHER')])
    assert r.status_code==409, r.text
    fields={x['field'] for x in r.json()['detail']['conflicting_fields']}
    assert fields=={'title','citation','url','source_type'}


def test_f3_dropping_an_optional_field_is_a_conflict_not_a_silent_discard():
    # Omitted citation/url must not silently erase persisted provenance.
    s=sid()
    registered(s)
    r=ingest([source(s,citation=None,url=None)])
    assert r.status_code==409, r.text
    fields={x['field'] for x in r.json()['detail']['conflicting_fields']}
    assert fields=={'citation','url'}
    assert all(x['submitted'] is None for x in r.json()['detail']['conflicting_fields'])


# --- G ---------------------------------------------------------------------
def test_g_conflict_leaves_canonical_source_unchanged():
    s=sid()
    first=registered(s)
    before=c.get(f'{CLINICAL}/entities/formula/{first}').json()['sources'][0]
    assert ingest([source(s,title='Hostile Overwrite',citation='p.666',source_type='FABRICATED')]).status_code==409
    after=c.get(f'{CLINICAL}/entities/formula/{first}').json()['sources'][0]
    assert after==before
    assert after['title']=='Canonical Text'
    assert after['citation']=='p.1'
    assert after['source_type']=='REFERENCE'


# --- H ---------------------------------------------------------------------
def test_h_conflict_creates_no_entity_version_or_attachment():
    s=sid()
    registered(s)
    before=c.get(f'{CLINICAL}/stats').json()
    r=ingest([source(s,title='Rejected Submission')],name='Phase12C2A Must Not Exist')
    assert r.status_code==409
    assert c.get(f'{CLINICAL}/stats').json()==before
    # The rejected entity was never created, so it is not searchable.
    found=c.post(f'{CLINICAL}/search',json={'query':'Phase12C2A Must Not Exist','limit':100}).json()['results']
    assert found==[]


def test_h2_conflict_rolls_back_the_whole_batch_no_partial_write():
    good=sid(); bad=sid()
    registered(bad)
    before=c.get(f'{CLINICAL}/stats').json()
    r=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2a','source_label':'phase12c2a','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2A Batch Good'},'sources':[source(good)]},
        {'entity_type':'formula','payload':{'name':'Phase12C2A Batch Bad'},'sources':[source(bad,title='Conflicting')]},
    ]})
    assert r.status_code==409, r.text
    # Neither item persisted, and the first item's brand-new source was not created.
    assert c.get(f'{CLINICAL}/stats').json()==before
    assert c.post(f'{CLINICAL}/search',json={'query':'Phase12C2A Batch','limit':100}).json()['results']==[]
    fresh=ingest([source(good,title='Now Free To Define')])
    assert fresh.status_code==200, fresh.text


def test_h3_conflict_within_one_batch_is_detected():
    s=sid()
    r=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2a','source_label':'phase12c2a','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2A Intra A'},'sources':[source(s,title='First Definition')]},
        {'entity_type':'formula','payload':{'name':'Phase12C2A Intra B'},'sources':[source(s,title='Second Definition')]},
    ]})
    assert r.status_code==409, r.text
    assert r.json()['detail']['source_id']==s


# --- I ---------------------------------------------------------------------
def test_i_duplicate_source_id_in_one_item_is_clean_422_not_500():
    s=sid()
    before=c.get(f'{CLINICAL}/stats').json()
    r=ingest([source(s),source(s)])
    assert r.status_code==422, r.text
    assert 'Duplicate source_id' in r.json()['detail']
    assert s in r.json()['detail']
    # No partial write.
    assert c.get(f'{CLINICAL}/stats').json()==before


def test_i2_duplicate_detection_precedes_registry_write():
    s=sid()
    assert ingest([source(s),source(s)]).status_code==422
    # The source_id is still unclaimed, so it can be defined freely afterwards.
    ok=ingest([source(s,title='Defined Later')])
    assert ok.status_code==200, ok.text


def test_i3_same_source_id_across_separate_items_is_allowed():
    s=sid()
    r=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2a','source_label':'phase12c2a','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2A Share A'},'sources':[source(s)]},
        {'entity_type':'herb','payload':{'name':'Phase12C2A Share B'},'sources':[source(s)]},
    ]})
    assert r.status_code==200, r.text
    a,b=r.json()['created_entity_ids']
    assert c.get(f'{CLINICAL}/entities/formula/{a}').json()['sources'][0]['source_id']==s
    assert c.get(f'{CLINICAL}/entities/herb/{b}').json()['sources'][0]['source_id']==s


# --- J, K: snapshot integrity ---------------------------------------------
def _canonical_fields(ref):
    return {k:ref.get(k) for k in ('source_id','title','citation','url','source_type')}


def test_j_entity_detail_sourceref_matches_canonical_registry():
    s=sid()
    eid=registered(s)
    detail=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    assert _canonical_fields(detail['sources'][0])==_canonical_fields(source(s))


def test_k_version_history_sourceref_matches_canonical_registry():
    s=sid()
    first=registered(s)
    # A second entity reusing the same source must snapshot canonical metadata.
    second=ingest([source(s)]).json()['created_entity_ids'][0]
    detail=c.get(f'{CLINICAL}/entities/formula/{second}').json()
    history=c.get(f'{CLINICAL}/entities/formula/{second}/history').json()['results']
    canonical=_canonical_fields(detail['sources'][0])
    for version in history:
        for ref in version['sources']:
            assert _canonical_fields(ref)==canonical, version


def test_k2_snapshot_cannot_diverge_from_registry_across_governance_transitions():
    s=sid()
    eid=registered(s)
    r=c.post(f'{CLINICAL}/entities/formula/{eid}/submit-review',json={'submitted_by':'phase12c2a'})
    assert r.status_code==200, r.text
    r=c.post(f'{CLINICAL}/entities/formula/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':r.json()['version']})
    assert r.status_code==200, r.text
    detail=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    history=c.get(f'{CLINICAL}/entities/formula/{eid}/history').json()['results']
    canonical=_canonical_fields(detail['sources'][0])
    assert len(history)>=3
    for version in history:
        for ref in version['sources']:
            assert _canonical_fields(ref)==canonical


def test_k3_the_phase12c1_divergence_defect_is_closed():
    """The exact 12C-1 reproduction: same source_id, different submitted metadata."""
    s=sid()
    first=registered(s)
    rejected=ingest([source(s,title='SNAPSHOT CLAIM',citation='p.999',source_type='FABRICATED')])
    assert rejected.status_code==409, rejected.text
    # No entity exists whose history asserts the rejected metadata.
    detail=c.get(f'{CLINICAL}/entities/formula/{first}').json()
    history=c.get(f'{CLINICAL}/entities/formula/{first}/history').json()['results']
    for version in history:
        for ref in version['sources']:
            assert ref['title']!='SNAPSHOT CLAIM'
            assert ref['title']==detail['sources'][0]['title']


def test_openapi_documents_the_typed_conflict_response():
    spec=c.get('/openapi.json').json()
    op=spec['paths'][f'{CLINICAL}/ingest']['post']
    assert '409' in op['responses']
    ref=op['responses']['409']['content']['application/json']['schema']['$ref']
    assert ref=='#/components/schemas/SourceConflictDetail'
    schema=spec['components']['schemas']['SourceConflictDetail']
    for field in ['error','source_id','message','conflicting_fields']:
        assert field in schema['properties'], field
    conflict=spec['components']['schemas']['SourceFieldConflict']
    assert set(conflict['properties']['field']['enum'])=={'title','citation','url','source_type'}
