from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'healthy'
    assert body['service'] == 'Xerbs AI v2'


def test_generate_contract_offline():
    response = client.post('/api/v1/recommendations/generate', json={
        'request_id': 'req-test-1',
        'text_input': 'offline contract test',
        'symptoms': ['example'],
        'goals': [],
        'constraints': [],
    })
    assert response.status_code == 200
    body = response.json()
    assert body['request_id'] == 'req-test-1'
    assert body['status'] == 'DRAFT_AI_RECOMMENDATION'
    assert body['requires_practitioner_review'] is True
    assert body['provenance']['provider'] == 'mock'
    assert body['model_confidence'] == 0.0
    assert body['formula_candidates'] == []
    assert 'MOCK_PROVIDER' in body['uncertainty_flags']


def test_rejects_empty_text():
    response = client.post('/api/v1/recommendations/generate', json={'text_input': ''})
    assert response.status_code == 422
