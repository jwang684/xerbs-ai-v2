from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from app.main import app
from app.db.base import Base
from app.db.models import ClinicalRelationship
from app.schemas.clinical_workflow import ClinicalEntityType, IngestionBatchRequest, IngestionItem, ReviewActionRequest, ReviewDecision
from app.schemas.clinical_knowledge import SourceRef
from app.services.knowledge.persistent_clinical import PersistentClinicalStore

c=TestClient(app)

def test_adaptive_interview_selects_high_information_questions_and_persists():
    r=c.post('/api/v1/interviews/start',json={'intake':{'text_input':'头痛','symptoms':['头痛']},'max_questions_per_round':2})
    assert r.status_code==200, r.text
    body=r.json(); assert body['turn_count']==0
    assert len(body['next_questions'])==2
    assert {q['field'] for q in body['next_questions']}=={'duration','temperature'}
    iid=body['interview_id']
    answers=[{'question_id':q['question_id'],'answer':'3天' if q['field']=='duration' else '无发热，也不怕冷'} for q in body['next_questions']]
    r2=c.post(f'/api/v1/interviews/{iid}/answers',json={'answers':answers,'max_questions_per_round':2})
    assert r2.status_code==200, r2.text
    body2=r2.json(); assert body2['turn_count']==1
    assert not ({'duration','temperature'} & {q['field'] for q in body2['next_questions']})
    r3=c.get(f'/api/v1/interviews/{iid}')
    assert r3.status_code==200 and r3.json()['turn_count']==1


def test_interview_rejects_unknown_question_id():
    r=c.post('/api/v1/interviews/start',json={'intake':{'text_input':'咳嗽','symptoms':['咳嗽']}})
    iid=r.json()['interview_id']
    bad=c.post(f'/api/v1/interviews/{iid}/answers',json={'answers':[{'question_id':'field:not-asked','answer':'x'}]})
    assert bad.status_code==422


def test_interview_can_generate_recommendation_from_accumulated_intake():
    r=c.post('/api/v1/interviews/start',json={'intake':{'text_input':'测试症状','symptoms':['测试症状']}})
    iid=r.json()['interview_id']
    rec=c.post(f'/api/v1/interviews/{iid}/recommendation')
    assert rec.status_code==200, rec.text
    assert rec.json()['requires_practitioner_review'] is True
    state=c.get(f'/api/v1/interviews/{iid}').json()
    assert state['status']=='COMPLETED'


def _reviewed(store, typ, name, source_id, **payload):
    req=IngestionBatchRequest(submitted_by='phase9',source_label='phase9',items=[IngestionItem(entity_type=typ,payload={'name':name,**payload},sources=[SourceRef(source_id=source_id,title='phase9 source',source_type='TEST')])])
    eid=store.ingest(req).created_entity_ids[0]
    store.submit_for_review(typ,eid,'phase9')
    store.review(typ,eid,ReviewActionRequest(reviewer_id='reviewer',reviewer_role='CLINICAL_REVIEWER',decision=ReviewDecision.APPROVE))
    return eid


def test_pattern_formula_relationship_retrieval_is_deterministic():
    engine=create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool)
    Base.metadata.create_all(engine); Session=sessionmaker(bind=engine,expire_on_commit=False)
    store=PersistentClinicalStore(Session)
    pid=_reviewed(store,ClinicalEntityType.PATTERN,'Phase9 Pattern','p9-pattern')
    fid=_reviewed(store,ClinicalEntityType.FORMULA,'Phase9 Formula','p9-formula',ingredients=['Herb X'])
    with Session.begin() as s:
        s.add(ClinicalRelationship(id='rel-phase9',source_entity_id=pid,target_entity_id=fid,relationship_type='PATTERN_FORMULA',review_status='REVIEWED',source_id=None,created_by='reviewer'))
    found=store.eligible_formula_candidates_for_patterns([pid])
    assert found and found[0]['formula_id']==fid
    assert 'pattern→formula' in found[0]['rationale']

def test_interview_recomputes_model_patterns_after_answer():
    import asyncio
    from app.services.llm.provider import LLMProvider, ProviderResult
    from app.services.interview.engine import AdaptiveInterviewEngine
    from app.schemas.intake import RecommendationRequest
    from app.schemas.interview import InterviewAnswerRequest, InterviewAnswer

    class ScriptedProvider(LLMProvider):
        async def generate_recommendation(self, **kwargs):
            has_duration='3天' in kwargs['text_input']
            return ProviderResult(
                summary='scripted',
                pattern_hypotheses=[{'name':'测试证型','confidence':0.7,'reasoning':'病程已补充'}] if has_duration else [],
                provider='scripted',model='scripted-v1'
            )

    engine=AdaptiveInterviewEngine(ScriptedProvider())
    state=asyncio.run(engine.start(RecommendationRequest(text_input='头痛',symptoms=['头痛']),2))
    duration=next(q for q in state.next_questions if q.field=='duration')
    updated=asyncio.run(engine.answer(state.interview_id,InterviewAnswerRequest(answers=[InterviewAnswer(question_id=duration.question_id,answer='3天')])))
    assert any(p.name=='测试证型' for p in updated.reasoning.pattern_assessments)
