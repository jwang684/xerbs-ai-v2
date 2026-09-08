import os
os.environ.setdefault('LLM_PROVIDER','mock')
import sqlite3
import subprocess
import sys
from uuid import uuid4

from fastapi.testclient import TestClient
from app.main import app

c=TestClient(app)
CLINICAL='/api/v1/knowledge/clinical'
SOURCES=f'{CLINICAL}/sources'
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG=uuid4().hex[:8]

CANON={'source_id':None,'title':'Phase12C2D2 Canonical','citation':'p.1','url':'https://example.test/d2','source_type':'REFERENCE'}


def sid():
    return f'src-12c2d2-{TAG}-'+uuid4().hex[:6]


def create(source_id=None,actor_id='curator',**over):
    body={**{k:v for k,v in CANON.items() if k!='source_id'},'source_id':source_id or sid(),'actor_id':actor_id,**over}
    return c.post(SOURCES,json=body)


def created(**over):
    r=create(**over)
    assert r.status_code==201, r.text
    return r.json()['source_id'],r.json()


def submit(source_id,version,actor='curator'):
    return c.post(f'{SOURCES}/{source_id}/submit-review',json={'submitted_by':actor,'expected_version':version})


def review(source_id,version,decision='APPROVE',role='CLINICAL_REVIEWER',actor='reviewer',notes=None):
    body={'reviewer_id':actor,'reviewer_role':role,'decision':decision,'expected_version':version}
    if notes: body['notes']=notes
    return c.post(f'{SOURCES}/{source_id}/review',json=body)


def in_review():
    s,body=created()
    r=submit(s,body['version'])
    assert r.status_code==200, r.text
    return s,r.json()['version']


# --- A, B, C ---------------------------------------------------------------
def test_a_create_source_is_draft_version_one():
    r=create()
    assert r.status_code==201, r.text
    b=r.json()
    assert b['review_status']=='DRAFT'
    assert b['version']==1
    assert b['created_by']=='curator'
    assert b['reviewed_by'] is None
    assert b['created_at'] and b['updated_at']
    assert b['title']=='Phase12C2D2 Canonical' and b['citation']=='p.1'
    assert b['url']=='https://example.test/d2' and b['source_type']=='REFERENCE'


def test_b_duplicate_creation_is_clean_409():
    s,_=created()
    again=create(source_id=s,title='Different Title')
    assert again.status_code==409, again.text
    detail=again.json()['detail']
    assert detail['error']=='SOURCE_ALREADY_EXISTS'
    assert detail['source_id']==s


def test_c_failed_creation_does_not_alter_existing_source():
    s,_=created()
    # Compare persisted reads on both sides: SQLite drops tzinfo, so the POST
    # body renders the same instant differently from a subsequent GET.
    before=c.get(f'{SOURCES}/{s}').json()
    assert create(source_id=s,title='Hostile',citation='p.999',source_type='FABRICATED',actor_id='attacker').status_code==409
    after=c.get(f'{SOURCES}/{s}').json()
    assert after==before
    assert after['title']=='Phase12C2D2 Canonical'
    assert after['created_by']=='curator'
    # And no extra governance events were recorded.
    assert c.get(f'{SOURCES}/{s}/reviews').json()['count']==1


# --- D, E, F, G, L ---------------------------------------------------------
def test_d_draft_to_in_review():
    s,body=created()
    r=submit(s,body['version'])
    assert r.status_code==200, r.text
    assert r.json()['review_status']=='IN_REVIEW'
    assert r.json()['version']==2
    # Canonical bibliographic metadata untouched by the transition.
    assert r.json()['title']=='Phase12C2D2 Canonical' and r.json()['citation']=='p.1'


def test_e_in_review_to_reviewed():
    s,v=in_review()
    r=review(s,v,'APPROVE')
    assert r.status_code==200, r.text
    b=r.json()
    assert b['review_status']=='REVIEWED'
    assert b['version']==3
    assert b['reviewed_by']=='reviewer'


def test_f_in_review_to_rejected():
    s,v=in_review()
    r=review(s,v,'REJECT')
    assert r.status_code==200, r.text
    assert r.json()['review_status']=='REJECTED'
    assert r.json()['reviewed_by']=='reviewer'


def test_g_in_review_to_draft_via_request_changes():
    s,v=in_review()
    r=review(s,v,'REQUEST_CHANGES')
    assert r.status_code==200, r.text
    b=r.json()
    assert b['review_status']=='DRAFT'
    assert b['version']==3
    # Not a completed review, so reviewed_by stays unset.
    assert b['reviewed_by'] is None
    # It can be resubmitted from DRAFT.
    assert submit(s,b['version']).status_code==200


def test_l_every_successful_transition_increments_version():
    s,body=created()
    assert body['version']==1
    v=submit(s,1).json()['version']; assert v==2
    v=review(s,v,'REQUEST_CHANGES').json()['version']; assert v==3
    v=submit(s,v).json()['version']; assert v==4
    v=review(s,v,'APPROVE').json()['version']; assert v==5


# --- H, I: illegal transitions ---------------------------------------------
def test_h_illegal_draft_to_reviewed_is_rejected():
    s,body=created()
    for decision in ('APPROVE','REJECT','REQUEST_CHANGES'):
        r=review(s,body['version'],decision)
        assert r.status_code==409, (decision,r.text)
        assert 'DRAFT source' in r.json()['detail']
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='DRAFT'


def test_i_illegal_reviewed_transitions_are_rejected():
    s,v=in_review()
    v=review(s,v,'APPROVE').json()['version']
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='REVIEWED'
    # REVIEWED -> IN_REVIEW and REVIEWED -> DRAFT are both refused.
    assert submit(s,v).status_code==409
    for decision in ('APPROVE','REJECT','REQUEST_CHANGES'):
        assert review(s,v,decision).status_code==409
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='REVIEWED'


def test_i2_rejected_is_terminal_in_this_phase():
    s,v=in_review()
    v=review(s,v,'REJECT').json()['version']
    assert submit(s,v).status_code==409
    assert review(s,v,'APPROVE').status_code==409
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='REJECTED'


# --- J, K ------------------------------------------------------------------
def test_j_stale_expected_version_is_rejected_without_writing():
    s,body=created()
    assert submit(s,99).status_code==409
    assert submit(s,99).json()['detail']=='VERSION_CONFLICT'
    unchanged=c.get(f'{SOURCES}/{s}').json()
    assert unchanged['version']==1 and unchanged['review_status']=='DRAFT'
    v=submit(s,1).json()['version']
    assert review(s,1,'APPROVE').status_code==409          # now stale
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='IN_REVIEW'
    assert review(s,v,'APPROVE').status_code==200


def test_j2_expected_version_is_required():
    s,_=created()
    assert c.post(f'{SOURCES}/{s}/submit-review',json={'submitted_by':'x'}).status_code==422
    assert c.post(f'{SOURCES}/{s}/review',json={'reviewer_id':'r','decision':'APPROVE'}).status_code==422


def test_k_unauthorized_reviewer_role_is_rejected():
    s,v=in_review()
    bad=review(s,v,'APPROVE',role='PATIENT')
    assert bad.status_code==409, bad.text
    assert 'not authorized' in bad.json()['detail']
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='IN_REVIEW'
    assert c.get(f'{SOURCES}/{s}/reviews').json()['count']==2      # no event written
    assert review(s,v,'APPROVE',role='CLINICAL_ADMIN').status_code==200


def test_k2_invalid_decision_is_rejected():
    s,v=in_review()
    assert c.post(f'{SOURCES}/{s}/review',json={'reviewer_id':'r','reviewer_role':'CLINICAL_REVIEWER','decision':'RETIRE','expected_version':v}).status_code==422


# --- M, N ------------------------------------------------------------------
def test_m_source_review_event_written_exactly_once_per_transition():
    s,body=created()
    assert c.get(f'{SOURCES}/{s}/reviews').json()['count']==1       # CREATED
    v=submit(s,1).json()['version']
    v=review(s,v,'APPROVE').json()['version']
    hist=c.get(f'{SOURCES}/{s}/reviews').json()
    assert hist['count']==3
    assert [x['action'] for x in hist['results']]==['CREATED','SUBMITTED_FOR_REVIEW','APPROVED']
    assert [x['from_status'] for x in hist['results']]==[None,'DRAFT','IN_REVIEW']
    assert [x['to_status'] for x in hist['results']]==['DRAFT','IN_REVIEW','REVIEWED']
    assert [x['version'] for x in hist['results']]==[1,2,3]
    assert hist['results'][2]['actor_role']=='CLINICAL_REVIEWER'


def test_n_audit_event_source_id_written_exactly_once_per_transition():
    s,_=created()
    v=submit(s,1).json()['version']
    review(s,v,'APPROVE')
    audit=c.get(f'{SOURCES}/{s}/audit').json()
    assert audit['count']==3
    assert [x['event_type'] for x in audit['results']]==['SOURCE_CREATED','SOURCE_SUBMITTED_FOR_REVIEW','SOURCE_APPROVED']
    assert all(x['source_id']==s for x in audit['results'])
    ids=[x['event_id'] for x in audit['results']]
    assert len(ids)==len(set(ids))
    assert audit['results'][2]['payload']=={'from_status':'IN_REVIEW','to_status':'REVIEWED','version':3}


# --- O: namespace isolation -------------------------------------------------
def test_o_source_audit_does_not_collide_with_same_valued_entity_id():
    """A Source whose source_id equals a real clinical entity id stays separate."""
    r=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2d2','source_label':'phase12c2d2','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2D2 Collide'},
         'sources':[{'source_id':sid(),'title':'x','source_type':'REFERENCE'}]}]})
    assert r.status_code==200, r.text
    entity_id=r.json()['created_entity_ids'][0]
    # Register a Source using the entity's own id as its source_id.
    assert create(source_id=entity_id).status_code==201
    v=submit(entity_id,1).json()['version']
    review(entity_id,v,'APPROVE')
    # Source audit returns only Source events.
    src_audit=c.get(f'{SOURCES}/{entity_id}/audit').json()
    assert src_audit['count']==3
    assert all(x['event_type'].startswith('SOURCE_') for x in src_audit['results'])
    assert all(x['source_id']==entity_id for x in src_audit['results'])
    # The clinical entity's own audit trail is untouched by any of it.
    ent_audit=c.get(f'{CLINICAL}/audit',params={'entity_id':entity_id}).json()['results']
    assert [x['action'] for x in ent_audit]==['INGESTED_AS_DRAFT']
    assert all(not x['action'].startswith('SOURCE_') for x in ent_audit)


def test_o2_audit_rows_keep_entity_id_null_for_source_events():
    from app.db.models import AuditEvent
    from app.db.session import get_session_factory
    from sqlalchemy import select
    s,_=created()
    submit(s,1)
    with get_session_factory()() as db:
        rows=db.scalars(select(AuditEvent).where(AuditEvent.source_id==s)).all()
        assert rows
        assert all(r.entity_id is None for r in rows)
        assert all(r.source_id==s for r in rows)


# --- P, Q, R, S -------------------------------------------------------------
def test_p_source_review_history_endpoint():
    s,_=created()
    r=c.get(f'{SOURCES}/{s}/reviews')
    assert r.status_code==200, r.text
    body=r.json()
    assert body['source_id']==s and body['count']==1
    row=body['results'][0]
    assert set(row)=={'event_id','source_id','action','actor_id','actor_role','from_status','to_status','version','notes','created_at'}
    assert row['action']=='CREATED' and row['actor_id']=='curator'


def test_p2_review_notes_are_persisted():
    s,v=in_review()
    review(s,v,'REJECT',notes='Citation could not be verified')
    last=c.get(f'{SOURCES}/{s}/reviews').json()['results'][-1]
    assert last['notes']=='Citation could not be verified'


def test_q_source_audit_history_endpoint():
    s,_=created()
    r=c.get(f'{SOURCES}/{s}/audit')
    assert r.status_code==200, r.text
    body=r.json()
    assert body['source_id']==s and body['count']==1
    assert set(body['results'][0])=={'event_id','event_type','source_id','actor_id','payload','created_at'}


def test_r_zero_event_source_returns_empty_history_not_404():
    """A legacy ingestion-created Source has no governance events yet."""
    s=sid()
    r=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2d2','source_label':'phase12c2d2','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2D2 Ingested'},
         'sources':[{'source_id':s,'title':'Ingested','source_type':'REFERENCE'}]}]})
    assert r.status_code==200, r.text
    reviews=c.get(f'{SOURCES}/{s}/reviews')
    assert reviews.status_code==200
    assert reviews.json()['count']==0 and reviews.json()['results']==[]
    audit=c.get(f'{SOURCES}/{s}/audit')
    assert audit.status_code==200
    assert audit.json()['count']==0 and audit.json()['results']==[]


def test_s_nonexistent_source_history_is_404():
    missing=f'src-missing-{TAG}'
    assert c.get(f'{SOURCES}/{missing}/reviews').status_code==404
    assert c.get(f'{SOURCES}/{missing}/audit').status_code==404
    assert c.get(f'{SOURCES}/{missing}/reviews').json()['detail']=='Clinical source not found'


# --- T ---------------------------------------------------------------------
def test_t_source_list_and_detail_expose_governance_fields():
    expected={'source_id','title','citation','url','source_type','review_status','version','created_by','reviewed_by','created_at','updated_at'}
    s,_=created()
    detail=c.get(f'{SOURCES}/{s}').json()
    assert set(detail)==expected
    listed=c.get(SOURCES,params={'source_id':s}).json()['results'][0]
    assert set(listed)==expected
    assert listed==detail
    spec=c.get('/openapi.json').json()
    assert set(spec['components']['schemas']['SourceRecord']['properties'])==expected
    for name in ('SourceCreateRequest','SourceReviewActionRequest','SourceSubmitReviewRequest',
                 'SourceReviewHistoryResponse','SourceAuditHistoryResponse','SourceAlreadyExistsDetail','SourceReviewDecision'):
        assert name in spec['components']['schemas'], name
    assert set(spec['components']['schemas']['SourceReviewDecision']['enum'])=={'APPROVE','REJECT','REQUEST_CHANGES'}


# --- U, V, W: ingestion compatibility ---------------------------------------
def _ingest(source_ref,name=None):
    return c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2d2','source_label':'phase12c2d2','items':[
        {'entity_type':'formula','payload':{'name':name or ('Phase12C2D2 '+uuid4().hex[:6])},'sources':[source_ref]}]})


def test_u_phase12c2a_identical_reuse_still_works():
    s=sid()
    ref={'source_id':s,'title':'Reused Text','citation':'p.5','source_type':'REFERENCE'}
    assert _ingest(ref).status_code==200
    assert _ingest(dict(ref)).status_code==200
    assert c.get(f'{SOURCES}/{s}/entities').json()['count']==2


def test_v_phase12c2a_conflicting_reuse_is_still_409():
    s=sid()
    ref={'source_id':s,'title':'Original','citation':'p.5','source_type':'REFERENCE'}
    assert _ingest(ref).status_code==200
    bad=_ingest({**ref,'title':'Conflicting'})
    assert bad.status_code==409, bad.text
    assert bad.json()['detail']['error']=='SOURCE_METADATA_CONFLICT'
    assert c.get(f'{SOURCES}/{s}').json()['title']=='Original'


def test_w_ingestion_created_source_remains_draft():
    s=sid()
    assert _ingest({'source_id':s,'title':'Ingested Draft','source_type':'REFERENCE'}).status_code==200
    body=c.get(f'{SOURCES}/{s}').json()
    assert body['review_status']=='DRAFT'
    assert body['version']==1
    assert body['created_by'] is None          # ingestion does not set a curator
    assert body['reviewed_by'] is None
    # Ingestion did not route through POST /sources, so no CREATED event exists.
    assert c.get(f'{SOURCES}/{s}/reviews').json()['count']==0


def test_w2_ingestion_can_reuse_a_reviewed_source():
    s,_=created()
    v=submit(s,1).json()['version']
    assert review(s,v,'APPROVE').status_code==200
    ref={k:v2 for k,v2 in CANON.items() if k!='source_id'}
    ok=_ingest({'source_id':s,**ref})
    assert ok.status_code==200, ok.text
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='REVIEWED'


# --- X: eligibility unchanged ------------------------------------------------
def test_x_clinical_ranking_eligible_is_unchanged_by_source_governance():
    """Intentionally temporary: a DRAFT Source still permits entity eligibility."""
    s=sid()
    r=_ingest({'source_id':s,'title':'Draft Source','source_type':'REFERENCE'})
    eid=r.json()['created_entity_ids'][0]
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='DRAFT'
    v=c.post(f'{CLINICAL}/entities/formula/{eid}/submit-review',json={'submitted_by':'phase12c2d2'}).json()['version']
    ok=c.post(f'{CLINICAL}/entities/formula/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':v})
    assert ok.status_code==200, ok.text
    detail=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    assert detail['review_status']=='REVIEWED'
    assert detail['clinical_ranking_eligible'] is True      # Source is still DRAFT
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='DRAFT'
    assert c.get(f'{SOURCES}/{s}/entities').json()['results'][0]['clinical_ranking_eligible'] is True


# --- Y ---------------------------------------------------------------------
def test_y_existing_clinical_audit_behavior_is_unchanged():
    before=c.get(f'{CLINICAL}/audit').json()['results']
    s,_=created()
    v=submit(s,1).json()['version']
    review(s,v,'APPROVE')
    after=c.get(f'{CLINICAL}/audit').json()['results']
    # Source governance adds nothing to the clinical entity audit stream.
    assert after==before
    r=_ingest({'source_id':sid(),'title':'y','source_type':'REFERENCE'})
    eid=r.json()['created_entity_ids'][0]
    rows=c.get(f'{CLINICAL}/audit',params={'entity_id':eid}).json()['results']
    assert len(rows)==1
    assert set(rows[0])>={'event_id','entity_type','entity_id','action','actor_id','from_status','to_status','version'}
    assert rows[0]['action']=='INGESTED_AS_DRAFT'


# --- Z ---------------------------------------------------------------------
def _alembic(db_path,*args):
    env={**os.environ,'DATABASE_URL':f'sqlite:///{db_path}','LLM_PROVIDER':'mock'}
    p=subprocess.run([sys.executable,'-m','alembic',*args],cwd=REPO,env=env,capture_output=True,text=True)
    assert p.returncode==0, f'alembic {args} failed:\n{p.stdout}\n{p.stderr}'
    return p.stdout


def test_z_migration_from_0005_to_head_succeeds(tmp_path):
    db=tmp_path/'d2.db'
    _alembic(db,'upgrade','0005_phase12c2d1')
    con=sqlite3.connect(db)
    cols_before={r[1] for r in con.execute('PRAGMA table_info(audit_event)')}
    con.execute("INSERT INTO audit_event (event_id,event_type,entity_id,actor_id,payload,created_at)"
                " VALUES ('aud-legacy','INGESTED_AS_DRAFT','frm-legacy','someone','{}','2020-01-01 00:00:00')")
    con.commit(); con.close()
    assert 'source_id' not in cols_before
    _alembic(db,'upgrade','head')
    assert '0006_phase12c2d2' in _alembic(db,'current')
    con=sqlite3.connect(db)
    try:
        cols={r[1]:r[2] for r in con.execute('PRAGMA table_info(audit_event)')}
        assert cols['source_id']=='VARCHAR(128)'
        assert cols['entity_id']=='VARCHAR(64)'              # deliberately not widened
        legacy=con.execute("SELECT entity_id,source_id FROM audit_event WHERE event_id='aud-legacy'").fetchone()
        assert legacy==('frm-legacy',None)                   # existing rows stay valid
        idx={r[1] for r in con.execute('PRAGMA index_list(audit_event)')}
        assert 'ix_audit_event_source_id' in idx
    finally: con.close()


def test_z2_downgrade_reverses_0006(tmp_path):
    db=tmp_path/'d2down.db'
    _alembic(db,'upgrade','head')
    _alembic(db,'downgrade','0005_phase12c2d1')
    con=sqlite3.connect(db)
    try:
        assert 'source_id' not in {r[1] for r in con.execute('PRAGMA table_info(audit_event)')}
    finally: con.close()
