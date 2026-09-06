from fastapi.testclient import TestClient
from app.main import app
from app.services.knowledge.formulas import FormulaKnowledgeService

client = TestClient(app)


def test_followup_extraction_returns_ten_questions():
    r = client.post('/api/v1/followups/generate', json={
        'request_id': 'req-followup-1', 'symptoms': ['头痛'], 'language': 'zh'
    })
    assert r.status_code == 200
    body = r.json()
    assert body['request_id'] == 'req-followup-1'
    assert body['source'] == 'legacy-ten-questions-v1'
    assert len(body['questions']) == 10


def test_formula_bridge_explicit_match_only():
    service = FormulaKnowledgeService()
    matches = service.rank_explicit_matches(['发热', '咽痛'])
    assert matches
    assert matches[0]['name'] == '银翘散'
    assert 'LEGACY_FIXTURE_ONLY' in matches[0]['safety_flags']


def test_formula_bridge_no_arbitrary_default():
    service = FormulaKnowledgeService()
    assert service.rank_explicit_matches(['完全未知症状']) == []


def test_legacy_bridge_is_not_used_by_phase4_recommendation_pipeline():
    r = client.post('/api/v1/recommendations/generate', json={
        'request_id': 'req-bridge-1',
        'text_input': '发热伴咽痛',
        'symptoms': ['发热', '咽痛']
    })
    assert r.status_code == 200
    body = r.json()
    assert body['formula_candidates'] == []
    assert 'LEGACY_KNOWLEDGE_BRIDGE' not in body['uncertainty_flags']
    assert body['model_confidence'] == 0.0
