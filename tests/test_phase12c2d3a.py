import os
os.environ.setdefault('LLM_PROVIDER','mock')
from uuid import uuid4

from fastapi.testclient import TestClient
from app.main import app
from app.services.knowledge.persistent_clinical import PersistentClinicalStore

c=TestClient(app)
CLINICAL='/api/v1/knowledge/clinical'
SOURCES=f'{CLINICAL}/sources'
TAG=uuid4().hex[:8]
store=PersistentClinicalStore()


def sid():
    return f'src-12c2d3a-{TAG}-'+uuid4().hex[:6]


def symptom():
    return f'phase12c2d3a symptom {uuid4().hex[:8]}'


def ingest(source_ids,entity_type='formula',indications=None,name=None):
    refs=[{'source_id':x,'title':f'Text {x}','citation':'p.1','source_type':'REFERENCE'} for x in source_ids]
    payload={'name':name or ('Phase12C2D3A '+uuid4().hex[:6])}
    if indications is not None: payload['indications']=indications
    if entity_type=='formula': payload.setdefault('ingredients',['Phase12C2D3A Herb'])
    r=c.post(f'{CLINICAL}/ingest',json={'submitted_by':'phase12c2d3a','source_label':'phase12c2d3a',
        'items':[{'entity_type':entity_type,'payload':payload,'sources':refs}]})
    assert r.status_code==200, r.text
    return r.json()['created_entity_ids'][0]


def approve_entity(eid,entity_type='formula'):
    r=c.post(f'{CLINICAL}/entities/{entity_type}/{eid}/submit-review',json={'submitted_by':'phase12c2d3a'})
    assert r.status_code==200, r.text
    r=c.post(f'{CLINICAL}/entities/{entity_type}/{eid}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':'APPROVE','expected_version':r.json()['version']})
    assert r.status_code==200, r.text


def drive_source(source_id,to):
    cur=c.get(f'{SOURCES}/{source_id}').json()
    if cur['review_status']==to: return cur
    if cur['review_status']=='DRAFT':
        r=c.post(f'{SOURCES}/{source_id}/submit-review',json={'submitted_by':'curator','expected_version':cur['version']})
        assert r.status_code==200, r.text
        cur=r.json()
        if to=='IN_REVIEW': return cur
    decision={'REVIEWED':'APPROVE','REJECTED':'REJECT','DRAFT':'REQUEST_CHANGES'}[to]
    r=c.post(f'{SOURCES}/{source_id}/review',json={'reviewer_id':'reviewer','reviewer_role':'CLINICAL_REVIEWER','decision':decision,'expected_version':cur['version']})
    assert r.status_code==200, r.text
    return r.json()


def reviewed_formula(source_status,indications):
    """A REVIEWED formula whose single Source is driven to source_status."""
    s=sid()
    eid=ingest([s],indications=indications)
    if source_status!='DRAFT': drive_source(s,source_status)
    approve_entity(eid)
    return eid,s


def retrieved(indications,eid):
    return eid in {x['formula_id'] for x in store.eligible_formula_candidates(indications,'')}


# --- A-F: eligible_formula_candidates() -------------------------------------
def test_a_reviewed_formula_with_no_source_is_not_retrieved():
    sym=symptom()
    eid=ingest([],indications=[sym])
    # It cannot even be approved without a source, so it is certainly not retrieved.
    assert retrieved([sym],eid) is False
    assert c.get(f'{CLINICAL}/entities/formula/{eid}').json()['source_count']==0


def test_b_reviewed_formula_with_draft_source_is_not_retrieved():
    sym=symptom()
    eid,s=reviewed_formula('DRAFT',[sym])
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='DRAFT'
    assert c.get(f'{CLINICAL}/entities/formula/{eid}').json()['review_status']=='REVIEWED'
    assert retrieved([sym],eid) is False


def test_c_reviewed_formula_with_in_review_source_is_not_retrieved():
    sym=symptom()
    eid,s=reviewed_formula('IN_REVIEW',[sym])
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='IN_REVIEW'
    assert retrieved([sym],eid) is False


def test_d_reviewed_formula_with_rejected_source_is_not_retrieved():
    sym=symptom()
    eid,s=reviewed_formula('REJECTED',[sym])
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='REJECTED'
    assert retrieved([sym],eid) is False


def test_e_reviewed_formula_with_reviewed_source_is_retrieved():
    sym=symptom()
    eid,s=reviewed_formula('REVIEWED',[sym])
    assert c.get(f'{SOURCES}/{s}').json()['review_status']=='REVIEWED'
    assert retrieved([sym],eid) is True


def test_f_multiple_sources_with_one_reviewed_is_retrieved():
    sym=symptom()
    a,b=sid(),sid()
    eid=ingest([a,b],indications=[sym])
    drive_source(b,'REVIEWED')          # a stays DRAFT
    approve_entity(eid)
    assert c.get(f'{SOURCES}/{a}').json()['review_status']=='DRAFT'
    assert retrieved([sym],eid) is True


def test_f2_multiple_sources_none_reviewed_is_not_retrieved():
    sym=symptom()
    a,b=sid(),sid()
    eid=ingest([a,b],indications=[sym])
    drive_source(a,'IN_REVIEW'); drive_source(b,'REJECTED')
    approve_entity(eid)
    assert retrieved([sym],eid) is False


# --- G: canonical rule parity for eligible_formula_candidates() -------------
def test_g_eligible_formula_candidates_follows_canonical_rule():
    sym=symptom()
    ineligible,_=reviewed_formula('DRAFT',[sym])
    eligible,_=reviewed_formula('REVIEWED',[sym])
    found={x['formula_id'] for x in store.eligible_formula_candidates([sym],'')}
    assert eligible in found
    assert ineligible not in found
    # Retrieval agrees with what the API reports for the same entities.
    for eid in (eligible,ineligible):
        api=c.get(f'{CLINICAL}/entities/formula/{eid}').json()['clinical_ranking_eligible']
        assert api==(eid in found), eid


# --- H: canonical rule parity for the pattern->formula path -----------------
def _linked_pattern(formula_id,formula_source):
    """A REVIEWED pattern with a REVIEWED PATTERN_FORMULA link to the formula."""
    ps=sid()
    pattern=ingest([ps],entity_type='pattern')
    drive_source(ps,'REVIEWED')
    approve_entity(pattern,'pattern')
    r=c.post('/api/v1/safety/relationships',json={'source_entity_id':pattern,'target_entity_id':formula_id,
        'relationship_type':'PATTERN_FORMULA','source_id':formula_source,'actor_id':'reviewer','actor_role':'CLINICAL_REVIEWER'})
    assert r.status_code==200, r.text
    return pattern


def test_h_pattern_formula_retrieval_follows_canonical_rule():
    sym=symptom()
    eid,s=reviewed_formula('DRAFT',[sym])
    pattern=_linked_pattern(eid,s)
    # Relationship is REVIEWED and the formula is REVIEWED, but its Source is DRAFT.
    assert store.eligible_formula_candidates_for_patterns([pattern])==[]
    drive_source(s,'REVIEWED')
    got=store.eligible_formula_candidates_for_patterns([pattern])
    assert [x['formula_id'] for x in got]==[eid]
    assert 'pattern→formula' in got[0]['rationale']


def test_h2_pattern_formula_retrieval_with_rejected_source_returns_nothing():
    sym=symptom()
    eid,s=reviewed_formula('DRAFT',[sym])
    pattern=_linked_pattern(eid,s)
    drive_source(s,'REJECTED')
    assert store.eligible_formula_candidates_for_patterns([pattern])==[]


# --- I: transition flips retrieval ------------------------------------------
def test_i_source_approval_flips_retrieval_false_to_true():
    sym=symptom()
    eid,s=reviewed_formula('DRAFT',[sym])
    assert retrieved([sym],eid) is False
    drive_source(s,'IN_REVIEW')
    assert retrieved([sym],eid) is False
    drive_source(s,'REVIEWED')
    assert retrieved([sym],eid) is True


# --- J, K: RecommendationAssembler end to end -------------------------------
def generate(sym):
    r=c.post('/api/v1/recommendations/generate',json={'text_input':sym,'symptoms':[sym],'patient_context':{}})
    assert r.status_code==200, r.text
    return r.json()


def test_j_assembler_cannot_receive_draft_only_sourced_formula():
    sym=symptom()
    eid,s=reviewed_formula('DRAFT',[sym])
    body=generate(sym)
    assert eid not in {x['formula_id'] for x in body['formula_candidates']}
    # Nothing governed was retrievable, so no corpus-backed retrieval was claimed.
    assert 'REVIEWED_CLINICAL_CORPUS' not in body['uncertainty_flags']


def test_k_assembler_receives_reviewed_sourced_formula():
    sym=symptom()
    eid,s=reviewed_formula('REVIEWED',[sym])
    body=generate(sym)
    ids={x['formula_id'] for x in body['formula_candidates']}
    assert eid in ids, body['uncertainty_flags']
    assert 'REVIEWED_CLINICAL_CORPUS' in body['uncertainty_flags']
    assert 'AI_FORMULA_RANKING_NOT_USED' in body['uncertainty_flags']
    assert body['requires_practitioner_review'] is True


def test_k2_same_formula_appears_only_after_its_source_is_approved():
    sym=symptom()
    eid,s=reviewed_formula('DRAFT',[sym])
    assert eid not in {x['formula_id'] for x in generate(sym)['formula_candidates']}
    drive_source(s,'REVIEWED')
    assert eid in {x['formula_id'] for x in generate(sym)['formula_candidates']}


# --- L: model-candidate suppression unchanged --------------------------------
def test_l_model_generated_candidate_suppression_is_unchanged():
    """A symptom with no eligible corpus must never yield a selectable formula."""
    sym=symptom()
    body=generate(sym)
    assert body['formula_candidates']==[]
    flags=set(body['uncertainty_flags'])
    # Either the model proposed something and it was suppressed, or nothing existed.
    assert flags & {'MODEL_FORMULA_CANDIDATES_SUPPRESSED_UNVERIFIED_CORPUS','NO_FORMULA_CANDIDATE'}
    if 'MODEL_FORMULA_CANDIDATES_SUPPRESSED_UNVERIFIED_CORPUS' in flags:
        assert 'REVIEWED_CLINICAL_CORPUS_REQUIRED_FOR_FORMULA_SELECTION' in flags
        assert 'NO_VERIFIED_FORMULA_CANDIDATE' in flags


def test_l2_draft_sourced_formula_does_not_become_a_model_backed_candidate():
    sym=symptom()
    eid,s=reviewed_formula('DRAFT',[sym])
    body=generate(sym)
    assert body['formula_candidates']==[]
    assert eid not in str(body['formula_candidates'])


# --- M: 12C-2D3 API eligibility unchanged ------------------------------------
def test_m_entity_detail_and_search_eligibility_unchanged():
    sym=symptom()
    eligible,_=reviewed_formula('REVIEWED',[sym])
    ineligible,_=reviewed_formula('DRAFT',[sym])
    assert c.get(f'{CLINICAL}/entities/formula/{eligible}').json()['clinical_ranking_eligible'] is True
    assert c.get(f'{CLINICAL}/entities/formula/{ineligible}').json()['clinical_ranking_eligible'] is False
    rows={r['formula_id']:r for r in c.post(f'{CLINICAL}/search',json={'query':sym,'limit':100}).json()['results'] if 'formula_id' in r}
    assert rows[eligible]['clinical_ranking_eligible'] is True
    assert rows[ineligible]['clinical_ranking_eligible'] is False


def test_m2_retrieval_and_reported_eligibility_can_no_longer_disagree():
    """The exact bypass this phase closes: reported flag vs actual retrieval."""
    sym=symptom()
    for status in ('DRAFT','IN_REVIEW','REJECTED','REVIEWED'):
        eid,s=reviewed_formula(status,[sym])
        reported=c.get(f'{CLINICAL}/entities/formula/{eid}').json()['clinical_ranking_eligible']
        actually_retrieved=retrieved([sym],eid)
        assert reported==actually_retrieved, (status,reported,actually_retrieved)
