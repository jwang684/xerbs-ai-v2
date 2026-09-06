from fastapi.testclient import TestClient
from app.main import app
from app.services.reasoning.engine import DiagnosticReasoningEngine
from app.schemas.intake import RecommendationRequest

client=TestClient(app)

def test_reasoning_detects_missing_information():
    r=DiagnosticReasoningEngine().analyze(RecommendationRequest(text_input="头痛",symptoms=["头痛"]))
    assert "MISSING_CLINICAL_INFORMATION" in r.uncertainty_flags
    assert any(x.field=="duration" for x in r.missing_information)
    assert r.ready_for_formula_retrieval is False

def test_reasoning_endpoint_contract():
    r=client.post('/api/v1/reasoning/analyze',json={"text_input":"咳嗽三天，发热","symptoms":["咳嗽","发热"]})
    assert r.status_code==200
    data=r.json(); assert data['structured_symptoms'][0]['name']=='咳嗽'; assert isinstance(data['followup_questions'],list)

def test_recommendation_contains_reasoning_and_no_silent_formula():
    r=client.post('/api/v1/recommendations/generate',json={"text_input":"头痛","symptoms":["头痛"]})
    assert r.status_code==200
    data=r.json(); assert data['reasoning'] is not None; assert data['formula_candidates']==[]; assert 'NO_FORMULA_CANDIDATE' in data['uncertainty_flags']
