from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.reasoning import ReasoningResponse, PatternAssessment, MissingInformation, EvidenceItem
from app.services.reasoning.convergence import PatternConvergenceEngine
from app.services.reasoning.contradictions import ContradictionEngine

c=TestClient(app)

def test_pattern_convergence_rewards_stability_and_verified_pattern():
    prev=ReasoningResponse(pattern_assessments=[PatternAssessment(name='测试证型',model_confidence=.8,corpus_match=True)],missing_information=[])
    cur=ReasoningResponse(pattern_assessments=[PatternAssessment(name='测试证型',model_confidence=.85,corpus_match=True)],missing_information=[])
    metrics=PatternConvergenceEngine().evaluate(cur,prev)
    assert metrics.pattern_stability==1.0
    assert metrics.verified_pattern_strength==.85
    assert metrics.score>=.8
    assert metrics.stable_pattern_names==['测试证型']


def test_pattern_convergence_penalizes_missing_information_and_instability():
    prev=ReasoningResponse(pattern_assessments=[PatternAssessment(name='证型A',model_confidence=.8)],missing_information=[])
    cur=ReasoningResponse(pattern_assessments=[PatternAssessment(name='证型B',model_confidence=.8)],missing_information=[MissingInformation(field='duration',reason='missing',priority='HIGH')])
    metrics=PatternConvergenceEngine().evaluate(cur,prev)
    assert metrics.pattern_stability==0
    assert metrics.score<.5
    assert set(metrics.changed_pattern_names)=={'证型A','证型B'}


def test_explicit_patient_negation_becomes_auditable_contradiction():
    reasoning=ReasoningResponse(pattern_assessments=[PatternAssessment(name='证型A',model_confidence=.8,supporting_evidence=[EvidenceItem(text='患者发热，头痛',source='model_reasoning')])])
    out=ContradictionEngine().annotate(reasoning,'患者明确无发热')
    assert out.pattern_assessments[0].contradictions
    assert 'PATTERN_CONTRADICTIONS_PRESENT' in out.uncertainty_flags
    assert out.ready_for_formula_retrieval is False


def test_interview_exposes_convergence_breakdown():
    r=c.post('/api/v1/interviews/start',json={'intake':{'text_input':'头痛','symptoms':['头痛']},'max_questions_per_round':2})
    assert r.status_code==200, r.text
    body=r.json()
    assert body['convergence'] is not None
    assert body['convergence_score']==body['convergence']['score']
    assert 'evidence_sufficiency' in body['convergence']


def test_base44_contract_requires_idempotency_key():
    key=str(uuid4())
    payload={'organization_id':'org-test','request_id':'req-'+key,'intake':{'text_input':'头痛','symptoms':['头痛']}}
    r=c.post('/api/v1/integrations/base44/generate',json=payload)
    assert r.status_code==400


def test_base44_contract_is_idempotent_and_correlated():
    key='idem-'+uuid4().hex
    corr='corr-'+uuid4().hex
    payload={'organization_id':'org-test','request_id':'req-'+uuid4().hex,'intake':{'text_input':'头痛','symptoms':['头痛']}}
    headers={'Idempotency-Key':key,'X-Correlation-ID':corr}
    r1=c.post('/api/v1/integrations/base44/generate',json=payload,headers=headers)
    assert r1.status_code==200, r1.text
    b1=r1.json(); assert b1['status']=='SUCCEEDED'
    assert b1['correlation_id']==corr
    assert b1['request_id']==payload['request_id']
    assert b1['trust_score_owned_by_ai_service'] is False
    assert b1['recommendation']['requires_practitioner_review'] is True
    r2=c.post('/api/v1/integrations/base44/generate',json=payload,headers=headers)
    assert r2.status_code==200
    assert r2.json()['generation_id']==b1['generation_id']
    assert c.get('/api/v1/integrations/base44/generations/'+b1['generation_id']).status_code==200


def test_base44_idempotency_key_reuse_with_different_payload_conflicts():
    key='idem-'+uuid4().hex
    headers={'Idempotency-Key':key}
    p1={'request_id':'req-a-'+uuid4().hex,'intake':{'text_input':'头痛','symptoms':['头痛']}}
    p2={'request_id':'req-b-'+uuid4().hex,'intake':{'text_input':'咳嗽','symptoms':['咳嗽']}}
    assert c.post('/api/v1/integrations/base44/generate',json=p1,headers=headers).status_code==200
    assert c.post('/api/v1/integrations/base44/generate',json=p2,headers=headers).status_code==409


def test_succeeded_generation_cannot_be_retried():
    key='idem-'+uuid4().hex
    payload={'request_id':'req-'+uuid4().hex,'intake':{'text_input':'头痛','symptoms':['头痛']}}
    b=c.post('/api/v1/integrations/base44/generate',json=payload,headers={'Idempotency-Key':key}).json()
    r=c.post('/api/v1/integrations/base44/generations/'+b['generation_id']+'/retry')
    assert r.status_code==422
