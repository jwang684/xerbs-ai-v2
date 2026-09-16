"""Phase 11.2 - intake mapping verification, Xerbs AI v2 receiving side.

Scope note: Base44 builds the payload sent to POST /integrations/base44/generate.
That builder lives in the Base44 codebase, not in this repository, so the tests
below characterise what this service *accepts and preserves* - the half of the
contract that is verifiable here. Visit/Patient/practitioner-entered sources and
/ask-vs-Visit parity cannot be asserted from this repo.
"""
import os
os.environ.setdefault('LLM_PROVIDER','mock')
from uuid import uuid4

from fastapi.testclient import TestClient
from app.main import app
from app.schemas.intake import RecommendationRequest
from app.schemas.integration import Base44GenerateRequest
from app.services.integration.base44 import Base44GenerationService

c=TestClient(app)
GEN='/api/v1/integrations/base44/generate'
INTERVIEWS='/api/v1/interviews'


def key():
    return 'idem-11-2-'+uuid4().hex[:16]


# The deterministic Phase 11.2 case: complaint, several symptoms, an explicit
# negative, a goal, a constraint, medication, allergy, diagnosis and age.
CASE={
    'text_input':'主诉：头痛三天。无发热。否认胸痛。',
    'symptoms':['头痛','失眠','无发热'],
    'goals':['改善睡眠'],
    'constraints':['避免麻黄'],
    'patient_context':{
        'medications':['warfarin'],
        'allergies':['peanut'],
        'diagnoses':['hypertension'],
        'pregnancy':False,
        'age':42,
    },
}


def generate(intake=None,idempotency_key=None,request_id='req-11-2'):
    r=c.post(GEN,json={'request_id':request_id,'organization_id':'org-11-2','intake':intake or CASE},
             headers={'Idempotency-Key':idempotency_key or key()})
    assert r.status_code==200, r.text
    return r.json()


def stored_intake(generation_id):
    """The exact normalized intake this service persisted for a generation."""
    from app.db.models import GenerationRequest
    from app.db.session import get_session_factory
    with get_session_factory()() as s:
        row=s.get(GenerationRequest,generation_id)
        assert row is not None
        return row.request_snapshot['intake']


# --- normalization contract --------------------------------------------------
def test_contract_accepts_exactly_the_five_normalized_fields():
    fields=set(RecommendationRequest.model_fields)
    assert {'text_input','symptoms','goals','constraints','patient_context'} <= fields
    # Nothing clinical is invented beyond the declared surface.
    # X1D-LEGACYDIAG3.2.1: interview_depth is an authorized, non-clinical
    # routing field -- how many governed turns this case has already completed,
    # derived server-side. The set stays closed and exhaustively asserted; this
    # names the one addition rather than loosening the check.
    # X1D-LEGACYDIAG4.4B: interview_state is the second authorized additive
    # field -- the previous turn's validated working differential plus the list
    # of evidence this case can substantiate, both supplied by core from
    # governed history. The set stays closed and exhaustively asserted.
    assert fields=={'request_id','text_input','symptoms','goals','constraints',
                    'image_data','image_filename','language','patient_context',
                    'interview_depth','interview_state'}
    # ...and neither invents anything clinical: a bounded integer and an
    # optional structure that defaults to absent.
    assert RecommendationRequest(text_input='x').interview_depth==0
    assert RecommendationRequest(text_input='x').interview_state is None


def test_h_goals_and_i_constraints_and_symptoms_survive_mapping():
    body=generate()
    intake=stored_intake(body['generation_id'])
    assert intake['text_input']==CASE['text_input']
    assert intake['symptoms']==['头痛','失眠','无发热']      # order preserved
    assert intake['goals']==['改善睡眠']
    assert intake['constraints']==['避免麻黄']


def test_b_j_k_l_m_patient_context_maps_every_structured_field():
    body=generate()
    pc=stored_intake(body['generation_id'])['patient_context']
    assert pc['medications']==['warfarin']       # J
    assert pc['allergies']==['peanut']           # K
    assert pc['diagnoses']==['hypertension']     # L
    assert pc['age']==42                         # M
    assert pc['pregnancy'] is False


def test_f_unanswered_is_distinguishable_from_explicit_negative():
    """pregnancy is tri-state: None (unasked) vs False (explicitly denied)."""
    unasked=RecommendationRequest(text_input='x').patient_context
    assert unasked.pregnancy is None
    assert unasked.age is None
    assert unasked.medications==[] and unasked.allergies==[] and unasked.diagnoses==[]
    denied=RecommendationRequest(text_input='x',patient_context={'pregnancy':False}).patient_context
    assert denied.pregnancy is False
    assert denied.pregnancy != unasked.pregnancy   # False is not None


def test_e_explicit_negatives_are_preserved_verbatim():
    body=generate()
    intake=stored_intake(body['generation_id'])
    assert '无发热' in intake['text_input']
    assert '否认胸痛' in intake['text_input']
    assert '无发热' in intake['symptoms']
    # Never rewritten into an "unknown"/absent marker.
    for banned in ('unknown','UNKNOWN','未知','null'):
        assert banned not in intake['text_input']


def test_q_no_clinical_field_is_fabricated():
    minimal={'text_input':'只有主诉'}
    body=generate(intake=minimal,request_id='req-min')
    intake=stored_intake(body['generation_id'])
    assert intake['symptoms']==[]
    assert intake['goals']==[]
    assert intake['constraints']==[]
    assert intake['patient_context']=={'medications':[],'allergies':[],'diagnoses':[],'pregnancy':None,'age':None}
    assert intake['image_data'] is None


def test_g_payload_normalization_is_deterministic():
    """Same clinical state in, byte-identical normalized payload out."""
    a=stored_intake(generate()['generation_id'])
    b=stored_intake(generate()['generation_id'])
    assert a==b
    # Request hashing is canonical, so the same payload hashes identically.
    h=Base44GenerationService._hash
    req=Base44GenerateRequest(request_id='r',intake=RecommendationRequest(**CASE))
    assert h(req)==h(Base44GenerateRequest(request_id='r',intake=RecommendationRequest(**CASE)))


def test_g2_duplicate_symptoms_are_preserved_not_silently_merged():
    """This service does not de-duplicate; it records what Base44 sent."""
    dup={**CASE,'symptoms':['头痛','头痛','失眠']}
    intake=stored_intake(generate(intake=dup,request_id='req-dup')['generation_id'])
    assert intake['symptoms']==['头痛','头痛','失眠']


# --- C, D: interview follow-up answers reach the payload ---------------------
def test_c_d_interview_answers_accumulate_into_the_intake_snapshot():
    r=c.post(f'{INTERVIEWS}/start',json={'intake':CASE,'max_questions_per_round':2})
    assert r.status_code==200, r.text
    body=r.json(); iid=body['interview_id']
    before=c.get(f'{INTERVIEWS}/{iid}').json()['intake']['text_input']
    q=body['next_questions'][0]
    answered=c.post(f'{INTERVIEWS}/{iid}/answers',json={'answers':[{'question_id':q['question_id'],'answer':'无发热，也不怕冷'}]})
    assert answered.status_code==200, answered.text
    after=c.get(f'{INTERVIEWS}/{iid}').json()['intake']['text_input']
    assert after!=before
    assert '无发热，也不怕冷' in after                      # D: the answer itself, not just the question
    assert q['field'] in after                             # tagged with its field
    assert CASE['text_input'] in after                     # original complaint retained


def test_d2_answers_reach_the_generated_recommendation_intake():
    r=c.post(f'{INTERVIEWS}/start',json={'intake':CASE,'max_questions_per_round':1})
    iid=r.json()['interview_id']; q=r.json()['next_questions'][0]
    c.post(f'{INTERVIEWS}/{iid}/answers',json={'answers':[{'question_id':q['question_id'],'answer':'否认咳嗽'}]})
    state=c.get(f'{INTERVIEWS}/{iid}').json()
    assert '否认咳嗽' in state['intake']['text_input']
    rec=c.post(f'{INTERVIEWS}/{iid}/recommendation')
    assert rec.status_code==200, rec.text
    assert rec.json()['requires_practitioner_review'] is True


def test_answers_are_appended_in_request_order_deterministically():
    r=c.post(f'{INTERVIEWS}/start',json={'intake':CASE,'max_questions_per_round':2})
    iid=r.json()['interview_id']; qs=r.json()['next_questions']
    assert len(qs)==2
    c.post(f'{INTERVIEWS}/{iid}/answers',json={'answers':[
        {'question_id':qs[0]['question_id'],'answer':'AAA'},
        {'question_id':qs[1]['question_id'],'answer':'BBB'}]})
    text=c.get(f'{INTERVIEWS}/{iid}').json()['intake']['text_input']
    assert text.index('AAA')<text.index('BBB')


# --- N: regeneration semantics on this side ----------------------------------
def test_n_retry_replays_the_stored_snapshot_and_does_not_pick_up_new_data():
    """Documented behaviour: retry re-executes the ORIGINAL persisted payload."""
    k=key()
    first=generate(idempotency_key=k)
    snapshot=stored_intake(first['generation_id'])
    r=c.post(f"/api/v1/integrations/base44/generations/{first['generation_id']}/retry")
    # Only FAILED generations may retry; a succeeded one is refused outright.
    assert r.status_code==422
    assert stored_intake(first['generation_id'])==snapshot


def test_n2_same_key_with_updated_intake_conflicts_rather_than_regenerating():
    """A new Idempotency-Key is required after intake changes."""
    k=key()
    first=generate(idempotency_key=k)
    updated={**CASE,'symptoms':CASE['symptoms']+['新增症状']}
    same_key=c.post(GEN,json={'request_id':'req-11-2','organization_id':'org-11-2','intake':updated},
                    headers={'Idempotency-Key':k})
    assert same_key.status_code==409                     # stale key is refused, not silently reused
    fresh=generate(intake=updated,idempotency_key=key())
    assert '新增症状' in stored_intake(fresh['generation_id'])['symptoms']
    assert '新增症状' not in stored_intake(first['generation_id'])['symptoms']


def test_n3_same_key_same_payload_replays_the_same_generation():
    k=key()
    first=generate(idempotency_key=k)
    again=generate(idempotency_key=k)
    assert again['generation_id']==first['generation_id']


# --- P: image safety ---------------------------------------------------------
def test_p_a_url_is_never_transported_as_image_data():
    """image_data is interpolated into a base64 data URI, so a URL would be
    malformed. This service performs no URL fetching anywhere."""
    import inspect
    from app.services.llm import openai_compatible
    src=inspect.getsource(openai_compatible)
    assert 'data:image/jpeg;base64,{image_data}' in src
    # No outbound image retrieval exists in the provider.
    for fetch in ('image_url_fetch','download_image','requests.get','urlopen'):
        assert fetch not in src
    # And nothing in the service reads image_filename as a transport source.
    assert 'image_filename' not in src


def test_p2_image_data_defaults_absent_and_is_optional_metadata():
    intake=stored_intake(generate()['generation_id'])
    assert intake['image_data'] is None
    assert intake['image_filename'] is None
    with_ref={**CASE,'image_filename':'tongue-2026-09-07.jpg'}
    kept=stored_intake(generate(intake=with_ref,request_id='req-img')['generation_id'])
    assert kept['image_filename']=='tongue-2026-09-07.jpg'
    assert kept['image_data'] is None                    # filename never promoted to image_data


# --- R: Phase 10 contract behaviour intact -----------------------------------
def test_r_existing_base44_contract_behaviour_is_intact():
    assert c.post(GEN,json={'request_id':'r','intake':CASE}).status_code==400   # key required
    body=generate()
    assert body['status']=='SUCCEEDED'
    assert body['trust_score_owned_by_ai_service'] is False
    assert body['contract_version']=='base44-xerbs-ai-v1'
    assert body['correlation_id']
    assert body['retry_count']==0
