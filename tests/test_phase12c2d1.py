import os
os.environ.setdefault('LLM_PROVIDER','mock')
import sqlite3
import subprocess
import sys
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from app.main import app

c=TestClient(app)
CLINICAL='/api/v1/knowledge/clinical'
SOURCES=f'{CLINICAL}/sources'
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG=uuid4().hex[:8]

LEGACY_SOURCE='src-legacy-12c2d1'
LEGACY_ENTITY='frm-legacy-12c2d1'


def _alembic(db_path,*args):
    """Run alembic in a subprocess so the app's cached settings are not reused."""
    env={**os.environ,'DATABASE_URL':f'sqlite:///{db_path}','LLM_PROVIDER':'mock'}
    p=subprocess.run([sys.executable,'-m','alembic',*args],cwd=REPO,env=env,capture_output=True,text=True)
    assert p.returncode==0, f'alembic {args} failed:\n{p.stdout}\n{p.stderr}'
    return p.stdout


@pytest.fixture(scope='module')
def migrated(tmp_path_factory):
    """A DB stopped at 0004, seeded with legacy rows, then upgraded to head.

    This is the authoritative legacy-migration fixture: the rows exist *before*
    the 12C-2D1 columns do.
    """
    db=tmp_path_factory.mktemp('mig')/'legacy.db'
    _alembic(db,'upgrade','0004_phase10')
    con=sqlite3.connect(db)
    con.execute("INSERT INTO source_registry (source_id,title,citation,url,source_type,created_at)"
                " VALUES (?,?,?,?,?,'2020-01-01 00:00:00')",
                (LEGACY_SOURCE,'Legacy Text','p.7','https://example.test/legacy','REFERENCE'))
    con.execute("INSERT INTO clinical_entity (id,entity_type,name,review_status,current_version,created_at,updated_at)"
                " VALUES (?,'formula','Legacy Formula','DRAFT',1,'2020-01-01 00:00:00','2020-01-01 00:00:00')",
                (LEGACY_ENTITY,))
    con.execute("INSERT INTO entity_source (entity_id,source_id) VALUES (?,?)",(LEGACY_ENTITY,LEGACY_SOURCE))
    con.commit(); con.close()
    _alembic(db,'upgrade','head')
    return db


def _row(db,sql,args=()):
    con=sqlite3.connect(db)
    con.row_factory=sqlite3.Row
    try: return dict(con.execute(sql,args).fetchone())
    finally: con.close()


def _cols(db,table):
    con=sqlite3.connect(db)
    try: return {r[1]:{'type':r[2],'notnull':bool(r[3]),'default':r[4]} for r in con.execute(f'PRAGMA table_info({table})')}
    finally: con.close()


# --- A, B, E, F, L ---------------------------------------------------------
def test_a_existing_source_migrates_as_draft(migrated):
    row=_row(migrated,'SELECT * FROM source_registry WHERE source_id=?',(LEGACY_SOURCE,))
    assert row['review_status']=='DRAFT'


def test_b_existing_source_version_is_one(migrated):
    assert _row(migrated,'SELECT version FROM source_registry WHERE source_id=?',(LEGACY_SOURCE,))['version']==1


def test_e_created_by_is_nullable_for_legacy(migrated):
    assert _row(migrated,'SELECT created_by FROM source_registry WHERE source_id=?',(LEGACY_SOURCE,))['created_by'] is None
    assert _cols(migrated,'source_registry')['created_by']['notnull'] is False


def test_f_reviewed_by_is_nullable(migrated):
    assert _row(migrated,'SELECT reviewed_by FROM source_registry WHERE source_id=?',(LEGACY_SOURCE,))['reviewed_by'] is None
    assert _cols(migrated,'source_registry')['reviewed_by']['notnull'] is False


def test_l_no_existing_source_becomes_reviewed_automatically(migrated):
    con=sqlite3.connect(migrated)
    try:
        assert con.execute("SELECT COUNT(*) FROM source_registry WHERE review_status='REVIEWED'").fetchone()[0]==0
        assert con.execute("SELECT COUNT(*) FROM source_registry WHERE review_status!='DRAFT'").fetchone()[0]==0
    finally: con.close()


def test_updated_at_is_backfilled_from_created_at(migrated):
    row=_row(migrated,'SELECT created_at,updated_at FROM source_registry WHERE source_id=?',(LEGACY_SOURCE,))
    assert row['updated_at'] is not None
    assert row['updated_at']==row['created_at']


def test_review_status_and_version_are_not_null(migrated):
    cols=_cols(migrated,'source_registry')
    assert cols['review_status']['notnull'] is True
    assert cols['version']['notnull'] is True


# --- C, D ------------------------------------------------------------------
def test_c_existing_entity_source_pins_version_one(migrated):
    row=_row(migrated,'SELECT * FROM entity_source WHERE entity_id=? AND source_id=?',(LEGACY_ENTITY,LEGACY_SOURCE))
    assert row['source_version']==1
    assert _cols(migrated,'entity_source')['source_version']['notnull'] is True


def test_d_locator_is_nullable_and_on_entity_source_not_source_registry(migrated):
    row=_row(migrated,'SELECT locator FROM entity_source WHERE entity_id=?',(LEGACY_ENTITY,))
    assert row['locator'] is None
    assert _cols(migrated,'entity_source')['locator']['notnull'] is False
    assert 'locator' not in _cols(migrated,'source_registry')


def test_entity_source_composite_identity_is_unchanged(migrated):
    con=sqlite3.connect(migrated)
    try:
        pk=[r[1] for r in con.execute('PRAGMA table_info(entity_source)') if r[5]]
        assert sorted(pk)==['entity_id','source_id']
    finally: con.close()


# --- G, H ------------------------------------------------------------------
def test_g_source_review_event_schema_and_fk(migrated):
    cols=_cols(migrated,'source_review_event')
    assert set(cols)=={'event_id','source_id','action','actor_id','actor_role','from_status','to_status','version','notes','created_at'}
    for required in ('event_id','source_id','action','actor_id','version','created_at'):
        assert cols[required]['notnull'] is True, required
    for optional in ('actor_role','from_status','to_status','notes'):
        assert cols[optional]['notnull'] is False, optional
    assert cols['source_id']['type']=='VARCHAR(128)'      # matches source_registry.source_id width
    con=sqlite3.connect(migrated)
    try:
        fks=con.execute('PRAGMA foreign_key_list(source_review_event)').fetchall()
        assert len(fks)==1
        assert fks[0][2]=='source_registry' and fks[0][3]=='source_id' and fks[0][4]=='source_id'
        assert fks[0][6]=='RESTRICT'
    finally: con.close()


def test_g2_source_review_event_fk_is_enforced(migrated):
    con=sqlite3.connect(migrated)
    con.execute('PRAGMA foreign_keys=ON')
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO source_review_event (event_id,source_id,action,actor_id,version,created_at)"
                        " VALUES ('evt-x','src-does-not-exist','APPROVED','r',1,'2020-01-01 00:00:00')")
        con.rollback()
        # A real source_id is accepted.
        con.execute("INSERT INTO source_review_event (event_id,source_id,action,actor_id,version,created_at)"
                    " VALUES ('evt-ok',?,'APPROVED','r',1,'2020-01-01 00:00:00')",(LEGACY_SOURCE,))
        con.rollback()
    finally: con.close()


def test_h_existing_clinical_review_event_fk_remains_intact(migrated):
    con=sqlite3.connect(migrated)
    try:
        fks=con.execute('PRAGMA foreign_key_list(review_event)').fetchall()
        assert len(fks)==1
        assert fks[0][2]=='clinical_entity' and fks[0][4]=='id' and fks[0][6]=='RESTRICT'
        cols=_cols(migrated,'review_event')
        assert cols['entity_id']['type']=='VARCHAR(64)' and cols['entity_id']['notnull'] is True
    finally: con.close()


def test_audit_event_entity_id_is_never_widened(migrated):
    """Phase 12C-2D2 added a structured source_id column instead of widening
    entity_id. entity_id must stay String(64) and stay entity-only."""
    cols=_cols(migrated,'audit_event')
    assert set(cols)=={'event_id','event_type','entity_id','source_id','actor_id','payload','created_at'}
    assert cols['entity_id']['type']=='VARCHAR(64)'      # never widened
    assert cols['source_id']['type']=='VARCHAR(128)'     # structured Source identity
    assert cols['source_id']['notnull'] is False
    con=sqlite3.connect(migrated)
    try:
        assert con.execute('PRAGMA foreign_key_list(audit_event)').fetchall()==[]
    finally: con.close()


# --- M ---------------------------------------------------------------------
def test_m_clean_alembic_upgrade_base_to_head(tmp_path):
    db=tmp_path/'clean.db'
    _alembic(db,'upgrade','head')
    assert '0006_phase12c2d2' in _alembic(db,'current')
    con=sqlite3.connect(db)
    try:
        tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally: con.close()
    assert 'source_review_event' in tables
    assert {'source_registry','entity_source','review_event','audit_event'}<=tables
    assert {'review_status','version','created_by','reviewed_by','updated_at'}<=set(_cols(db,'source_registry'))
    assert {'source_version','locator'}<=set(_cols(db,'entity_source'))


def test_m2_downgrade_reverses_the_migration(tmp_path):
    db=tmp_path/'down.db'
    _alembic(db,'upgrade','head')
    _alembic(db,'downgrade','0004_phase10')
    assert set(_cols(db,'source_registry'))=={'source_id','title','citation','url','source_type','created_at'}
    assert set(_cols(db,'entity_source'))=={'entity_id','source_id'}
    con=sqlite3.connect(db)
    try:
        tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally: con.close()
    assert 'source_review_event' not in tables


# --- I: read API contract preserved ---------------------------------------
def _seed_source():
    s=f'src-12c2d1-{TAG}-'+uuid4().hex[:6]
    r=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2d1','source_label':'phase12c2d1','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2D1 '+uuid4().hex[:6]},
         'sources':[{'source_id':s,'title':f'Phase12C2D1 Text {TAG}','citation':'p.1','url':'https://example.test/c','source_type':'REFERENCE'}]}]})
    assert r.status_code==200, r.text
    return s,r.json()['created_entity_ids'][0]


def test_i_source_read_api_contract():
    """Phase 12C-2D2 intentionally added governance state to SourceRecord;
    every original canonical field is preserved alongside it."""
    s,eid=_seed_source()
    one=c.get(f'{SOURCES}/{s}')
    assert one.status_code==200, one.text
    assert set(one.json())=={'source_id','title','citation','url','source_type','review_status','version','created_by','reviewed_by','created_at','updated_at'}
    assert {'source_id','title','citation','url','source_type','created_at'} <= set(one.json())
    listed=c.get(SOURCES,params={'source_id':s})
    assert listed.status_code==200
    body=listed.json()
    assert set(body)=={'count','limit','offset','results'}
    assert set(body['results'][0])=={'source_id','title','citation','url','source_type','review_status','version','created_by','reviewed_by','created_at','updated_at'}
    ents=c.get(f'{SOURCES}/{s}/entities')
    assert ents.status_code==200
    assert set(ents.json())=={'source_id','count','results'}
    assert set(ents.json()['results'][0])=={'entity_id','entity_type','name','current_version','review_status','clinical_ranking_eligible'}
    assert c.get(f'{SOURCES}/src-missing-{TAG}').status_code==404
    assert c.get(f'{SOURCES}/src-missing-{TAG}/entities').status_code==404


def test_i2_openapi_source_schemas():
    spec=c.get('/openapi.json').json()
    assert set(spec['components']['schemas']['SourceRecord']['properties'])=={'source_id','title','citation','url','source_type','review_status','version','created_by','reviewed_by','created_at','updated_at'}
    # Still no bibliographic fields and no SourceType enum in this line of work.
    for absent in ('authors','publisher','publication_year','edition','language','doi','isbn','pmid'):
        assert absent not in spec['components']['schemas']['SourceRecord']['properties'], absent
    assert 'SourceType' not in spec['components']['schemas']
    for p in (SOURCES+'/{source_id}', SOURCES+'/{source_id}/entities'):
        assert set(spec['paths'][p])=={'get'}


# --- J: 12C-2A gate ---------------------------------------------------------
def test_j_phase12c2a_conflict_gate_still_enforced():
    s,_=_seed_source()
    conflict=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2d1','source_label':'phase12c2d1','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2D1 Conflict'},
         'sources':[{'source_id':s,'title':'Different Title','citation':'p.1','url':'https://example.test/c','source_type':'REFERENCE'}]}]})
    assert conflict.status_code==409, conflict.text
    assert conflict.json()['detail']['error']=='SOURCE_METADATA_CONFLICT'
    dup_id=f'src-12c2d1-dup-{TAG}-'+uuid4().hex[:6]
    ref={'source_id':dup_id,'title':'Dup','source_type':'REFERENCE'}
    dup=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2d1','source_label':'phase12c2d1','items':[
        {'entity_type':'formula','payload':{'name':'Phase12C2D1 Dup'},'sources':[ref,ref]}]})
    assert dup.status_code==422, dup.text


# --- K: eligibility ---------------------------------------------------------
def test_k_clinical_ranking_eligibility_requires_a_reviewed_source():
    s,eid=_seed_source()
    detail=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    assert detail['review_status']=='DRAFT'
    assert detail['clinical_ranking_eligible'] is False
    r=c.post(f'{CLINICAL}/entities/formula/{eid}/submit-review',json={'submitted_by':'phase12c2d1'})
    assert r.status_code==200, r.text
    r=c.post(f'{CLINICAL}/entities/formula/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':r.json()['version']})
    assert r.status_code==200, r.text
    after=c.get(f'{CLINICAL}/entities/formula/{eid}').json()
    # Phase 12C-2D3: an entity_source reference alone is no longer enough; the
    # referenced Source must itself be REVIEWED.
    assert after['review_status']=='REVIEWED'
    assert after['source_count']==1
    assert after['clinical_ranking_eligible'] is False
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='DRAFT'
    assert c.get(f'{SOURCES}/{s}/entities').json()['results'][0]['clinical_ranking_eligible'] is False
