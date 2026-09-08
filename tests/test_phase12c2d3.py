import os
os.environ.setdefault('LLM_PROVIDER','mock')
from uuid import uuid4

from fastapi.testclient import TestClient
from app.main import app

c=TestClient(app)
CLINICAL='/api/v1/knowledge/clinical'
SOURCES=f'{CLINICAL}/sources'
TAG=uuid4().hex[:8]


def sid():
    return f'src-12c2d3-{TAG}-'+uuid4().hex[:6]


def ingest(source_ids,entity_type='formula',name=None):
    """Create an entity citing the given (already-registered or new) sources."""
    refs=[{'source_id':x,'title':f'Text {x}','citation':'p.1','source_type':'REFERENCE'} for x in source_ids]
    r=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2d3','source_label':'phase12c2d3','items':[
        {'entity_type':entity_type,'payload':{'name':name or ('Phase12C2D3 '+uuid4().hex[:6]),
                                              'indications':[f'phase12c2d3 symptom {TAG}']},'sources':refs}]})
    assert r.status_code==200, r.text
    return r.json()['created_entity_ids'][0]


def approve_entity(eid,entity_type='formula'):
    r=c.post(f'{CLINICAL}/entities/{entity_type}/{eid}/submit-review',json={'submitted_by':'phase12c2d3'})
    assert r.status_code==200, r.text
    r=c.post(f'{CLINICAL}/entities/{entity_type}/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':r.json()['version']})
    assert r.status_code==200, r.text
    return r.json()


def in_review_entity(eid,entity_type='formula'):
    r=c.post(f'{CLINICAL}/entities/{entity_type}/{eid}/submit-review',json={'submitted_by':'phase12c2d3'})
    assert r.status_code==200, r.text
    return r.json()


def reject_entity(eid,entity_type='formula'):
    v=in_review_entity(eid,entity_type)['version']
    r=c.post(f'{CLINICAL}/entities/{entity_type}/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'REJECT','expected_version':v})
    assert r.status_code==200, r.text
    return r.json()


def drive_source(source_id,to):
    """Advance a Source to a target status from wherever it currently is."""
    cur=c.get(f'{SOURCES}/{source_id}').json()
    if cur['review_status']==to: return cur
    if cur['review_status']=='DRAFT':
        r=c.post(f'{SOURCES}/{source_id}/submit-review',json={'submitted_by':'curator','expected_version':cur['version']})
        assert r.status_code==200, r.text
        cur=r.json()
        if to=='IN_REVIEW': return cur
    assert cur['review_status']=='IN_REVIEW', cur['review_status']
    decision={'REVIEWED':'APPROVE','REJECTED':'REJECT','DRAFT':'REQUEST_CHANGES'}[to]
    r=c.post(f'{SOURCES}/{source_id}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':decision,'expected_version':cur['version']})
    assert r.status_code==200, r.text
    return r.json()


def eligible(eid,entity_type='formula'):
    return c.get(f'{CLINICAL}/entities/{entity_type}/{eid}').json()['clinical_ranking_eligible']


def entity_with(source_status,approve=True):
    s=sid()
    eid=ingest([s])
    drive_source(s,source_status)
    if approve: approve_entity(eid)
    return eid,s


# --- A-E: entity REVIEWED, varying single Source status --------------------
def test_a_reviewed_entity_with_no_source_is_not_eligible():
    """The entity approval gate still requires a source, so the zero-source
    case is reached by approving with a source, which is itself DRAFT."""
    eid=ingest([])
    detail=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    assert detail['source_count']==0
    # Cannot even be approved without a source; it is certainly not eligible.
    v=in_review_entity(eid)['version']
    r=c.post(f'{CLINICAL}/entities/formula/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':v})
    assert r.status_code==409
    assert eligible(eid) is False


def test_b_reviewed_entity_with_draft_source_is_not_eligible():
    eid,s=entity_with('DRAFT')
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='DRAFT'
    assert c.get(f'{CLINICAL}/entities/formula/{eid}').json()['review_status']=='REVIEWED'
    assert eligible(eid) is False


def test_c_reviewed_entity_with_in_review_source_is_not_eligible():
    eid,s=entity_with('IN_REVIEW')
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='IN_REVIEW'
    assert eligible(eid) is False


def test_d_reviewed_entity_with_rejected_source_is_not_eligible():
    eid,s=entity_with('REJECTED')
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='REJECTED'
    assert eligible(eid) is False


def test_e_reviewed_entity_with_reviewed_source_is_eligible():
    eid,s=entity_with('REVIEWED')
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='REVIEWED'
    assert c.get(f'{CLINICAL}/entities/formula/{eid}').json()['review_status']=='REVIEWED'
    assert eligible(eid) is True


# --- F, G, H: entity not REVIEWED -------------------------------------------
def test_f_draft_entity_with_reviewed_source_is_not_eligible():
    s=sid(); eid=ingest([s]); drive_source(s,'REVIEWED')
    assert c.get(f'{CLINICAL}/entities/formula/{eid}').json()['review_status']=='DRAFT'
    assert eligible(eid) is False


def test_g_in_review_entity_with_reviewed_source_is_not_eligible():
    s=sid(); eid=ingest([s]); drive_source(s,'REVIEWED')
    assert in_review_entity(eid)['review_status']=='IN_REVIEW'
    assert eligible(eid) is False


def test_h_rejected_entity_with_reviewed_source_is_not_eligible():
    s=sid(); eid=ingest([s]); drive_source(s,'REVIEWED')
    assert reject_entity(eid)['review_status']=='REJECTED'
    assert eligible(eid) is False


# --- I, J: multiple Sources --------------------------------------------------
def test_i_multiple_sources_one_reviewed_is_eligible():
    a,b=sid(),sid()
    eid=ingest([a,b])
    drive_source(b,'REVIEWED')          # a stays DRAFT
    approve_entity(eid)
    assert c.get(f'{SOURCES}/{a}').json()['review_status']=='DRAFT'
    assert c.get(f'{SOURCES}/{b}').json()['review_status']=='REVIEWED'
    assert c.get(f'{CLINICAL}/entities/formula/{eid}').json()['source_count']==2
    assert eligible(eid) is True         # one qualifying Source is enough


def test_j_multiple_sources_none_reviewed_is_not_eligible():
    a,b,d=sid(),sid(),sid()
    eid=ingest([a,b,d])
    drive_source(a,'IN_REVIEW'); drive_source(b,'REJECTED')   # d stays DRAFT
    approve_entity(eid)
    assert c.get(f'{CLINICAL}/entities/formula/{eid}').json()['source_count']==3
    assert eligible(eid) is False


def test_i2_not_every_source_must_be_reviewed():
    reviewed,*others=[sid() for _ in range(4)]
    eid=ingest([reviewed,*others])
    drive_source(reviewed,'REVIEWED')
    drive_source(others[0],'REJECTED')
    drive_source(others[1],'IN_REVIEW')
    approve_entity(eid)
    assert eligible(eid) is True


# --- K, L: governance transitions drive derived eligibility ------------------
def test_k_source_approval_flips_eligibility_false_to_true():
    s=sid(); eid=ingest([s])
    approve_entity(eid)
    assert eligible(eid) is False                       # Source DRAFT
    drive_source(s,'IN_REVIEW')
    assert eligible(eid) is False                       # still not REVIEWED
    drive_source(s,'REVIEWED')
    assert eligible(eid) is True                        # derived from live state


def test_l_request_changes_and_rejection_never_grant_eligibility():
    s=sid(); eid=ingest([s])
    approve_entity(eid)
    cur=c.get(f'{SOURCES}/{s}').json()
    r=c.post(f'{SOURCES}/{s}/submit-review',json={'submitted_by':'curator','expected_version':cur['version']})
    r=c.post(f'{SOURCES}/{s}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'REQUEST_CHANGES','expected_version':r.json()['version']})
    assert r.status_code==200 and r.json()['review_status']=='DRAFT'
    assert eligible(eid) is False
    other=sid(); eid2=ingest([other]); approve_entity(eid2)
    drive_source(other,'REJECTED')
    assert eligible(eid2) is False


# --- M, N: API surfaces ------------------------------------------------------
def test_m_entity_detail_returns_new_eligibility():
    eid_true,_=entity_with('REVIEWED')
    eid_false,_=entity_with('DRAFT')
    assert c.get(f'{CLINICAL}/entities/formula/{eid_true}').json()['clinical_ranking_eligible'] is True
    assert c.get(f'{CLINICAL}/entities/formula/{eid_false}').json()['clinical_ranking_eligible'] is False


def test_n_clinical_search_returns_new_eligibility():
    eid_true,_=entity_with('REVIEWED')
    eid_false,_=entity_with('DRAFT')
    rows=c.post(f'{CLINICAL}/search',json={'query':f'phase12c2d3 symptom {TAG}','limit':100}).json()['results']
    by_id={r['formula_id']:r for r in rows if 'formula_id' in r}
    assert by_id[eid_true]['clinical_ranking_eligible'] is True
    assert by_id[eid_false]['clinical_ranking_eligible'] is False


def test_n2_source_reverse_lookup_uses_the_same_rule():
    eid,s=entity_with('REVIEWED')
    row=[x for x in c.get(f'{SOURCES}/{s}/entities').json()['results'] if x['entity_id']==eid][0]
    assert row['clinical_ranking_eligible'] is True
    assert row['clinical_ranking_eligible']==c.get(f'{CLINICAL}/entities/formula/{eid}').json()['clinical_ranking_eligible']
    eid2,s2=entity_with('DRAFT')
    row2=[x for x in c.get(f'{SOURCES}/{s2}/entities').json()['results'] if x['entity_id']==eid2][0]
    assert row2['clinical_ranking_eligible'] is False


def test_n3_corpus_stats_counts_by_the_same_rule():
    before=c.get(f'{CLINICAL}/stats').json()['ranking_eligible_formulas']
    eid_false,_=entity_with('DRAFT')
    assert c.get(f'{CLINICAL}/stats').json()['ranking_eligible_formulas']==before
    eid_true,_=entity_with('REVIEWED')
    assert c.get(f'{CLINICAL}/stats').json()['ranking_eligible_formulas']==before+1


# --- O, P: earlier phases unchanged ------------------------------------------
def test_o_phase12c2a_conflict_behavior_unchanged():
    s=sid()
    ingest([s])
    bad=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2d3','source_label':'phase12c2d3','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2D3 Conflict'},
         'sources':[{'source_id':s,'title':'Conflicting Title','citation':'p.1','source_type':'REFERENCE'}]}]})
    assert bad.status_code==409
    assert bad.json()['detail']['error']=='SOURCE_METADATA_CONFLICT'
    ref={'source_id':s,'title':f'Text {s}','citation':'p.1','source_type':'REFERENCE'}
    ok=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2d3','source_label':'phase12c2d3','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2D3 Reuse'},'sources':[ref]}]})
    assert ok.status_code==200, ok.text


def test_p_phase12c2d2_source_governance_unchanged():
    s=sid(); ingest([s])
    cur=c.get(f'{SOURCES}/{s}').json()
    assert cur['review_status']=='DRAFT' and cur['version']==1
    assert c.post(f'{SOURCES}/{s}/review',json={'reviewer_id':'r','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':1}).status_code==409
    assert c.post(f'{SOURCES}/{s}/submit-review',json={'submitted_by':'x','expected_version':99}).status_code==409
    v=c.post(f'{SOURCES}/{s}/submit-review',json={'submitted_by':'curator','expected_version':1}).json()['version']
    assert c.post(f'{SOURCES}/{s}/review',json={'reviewer_id':'r','reviewer_role':'PATIENT','decision':'APPROVE','expected_version':v}).status_code==409
    assert c.post(f'{SOURCES}/{s}/review',json={'reviewer_id':'r','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':v}).status_code==200
    hist=c.get(f'{SOURCES}/{s}/reviews').json()
    assert [x['action'] for x in hist['results']]==['SUBMITTED_FOR_REVIEW','APPROVED']
    assert c.get(f'{SOURCES}/{s}/audit').json()['count']==2


# --- Q, R, S: nothing auto-promoted ------------------------------------------
def test_q_no_source_is_auto_promoted_by_entity_approval():
    s=sid(); eid=ingest([s])
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='DRAFT'
    approve_entity(eid)
    after=c.get(f'{SOURCES}/{s}').json()
    assert after['review_status']=='DRAFT'          # entity approval never promotes a Source
    assert after['version']==1
    assert c.get(f'{SOURCES}/{s}/reviews').json()['count']==0


def test_r_no_clinical_entity_is_auto_promoted_by_source_approval():
    s=sid(); eid=ingest([s])
    before=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    assert before['review_status']=='DRAFT'
    drive_source(s,'REVIEWED')
    after=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    assert after['review_status']=='DRAFT'          # Source approval never promotes an entity
    assert after['version']==before['version']
    assert after['clinical_ranking_eligible'] is False


def test_s_legacy_draft_source_does_not_qualify():
    """Legacy ingestion-created Sources stay DRAFT and confer no eligibility."""
    s=sid(); eid=ingest([s])
    approve_entity(eid)
    detail=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    assert detail['review_status']=='REVIEWED'
    assert detail['source_count']==1
    assert detail['clinical_ranking_eligible'] is False
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='DRAFT'


def test_s2_legacy_seeded_formulas_are_not_eligible():
    rows=c.post(f'{CLINICAL}/search',json={'query':'汤','entity_types':['formula'],'limit':100}).json()['results']
    legacy=[r for r in rows if r.get('migration_origin')=='LEGACY_STATIC_DATA_FIXTURE']
    assert all(r['clinical_ranking_eligible'] is False for r in legacy)


# --- version history consistency ---------------------------------------------
def test_version_snapshot_records_eligibility_by_the_same_rule():
    s=sid(); eid=ingest([s])
    drive_source(s,'REVIEWED')
    approve_entity(eid)
    detail=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    history=c.get(f'{CLINICAL}/entities/formula/{eid}/history').json()['results']
    assert detail['clinical_ranking_eligible'] is True
    assert history[-1]['clinical_ranking_eligible'] is True
    assert history[0]['clinical_ranking_eligible'] is False      # DRAFT at v1
