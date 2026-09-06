from fastapi.testclient import TestClient

from app.main import app
from app.services.knowledge.tse_repository import TSEKnowledgeRepository

client = TestClient(app)


def test_tse_snapshot_loaded_and_gated():
    repo = TSEKnowledgeRepository()
    stats = repo.stats()
    assert stats["formula_rows"] == 1025
    assert stats["product_rows"] == 548
    assert stats["clinical_ranking_eligible_rows"] == 0
    assert stats["corpus_policy"] == "CATALOG_METADATA_ONLY_UNTIL_CLINICAL_CURATION"


def test_tse_code_lookup_preserves_catalog_identity():
    repo = TSEKnowledgeRepository()
    item = repo.get_formula("60082117")
    assert item is not None
    assert item.name == "桑葉"
    assert item.source_catalog == "TSE-Herb"
    assert item.clinical_ranking_eligible is False


def test_catalog_search_by_name_is_deterministic():
    repo = TSEKnowledgeRepository()
    results = repo.search_catalog("桑葉", limit=5)
    assert results
    assert any(x["tse_code"] == "60082117" for x in results)
    assert all(x["clinical_ranking_eligible"] is False for x in results)


def test_symptom_retrieval_is_explicitly_nonclinical():
    repo = TSEKnowledgeRepository()
    results = repo.retrieve_symptom_metadata(["发热"], limit=5)
    assert results
    assert all(x["clinical_ranking_eligible"] is False for x in results)
    assert all("LEGACY_TSE_METADATA_UNVERIFIED" in x["safety_flags"] for x in results)


def test_unknown_symptom_does_not_default():
    repo = TSEKnowledgeRepository()
    assert repo.retrieve_symptom_metadata(["完全不存在的症状xyz"], limit=5) == []


def test_tse_api_stats_and_search():
    r = client.get('/api/v1/knowledge/tse/stats')
    assert r.status_code == 200
    assert r.json()['formula_rows'] == 1025
    r = client.post('/api/v1/knowledge/tse/search', json={'query': '桑葉', 'limit': 5})
    assert r.status_code == 200
    assert r.json()['results']


def test_tse_symptom_api_cannot_claim_recommendation():
    r = client.post('/api/v1/knowledge/tse/symptom-metadata', json={'symptoms': ['发热'], 'limit': 3})
    assert r.status_code == 200
    body = r.json()
    assert body['policy'] == 'RETRIEVAL_ONLY_NOT_CLINICAL_RECOMMENDATION'
    assert all(x['clinical_ranking_eligible'] is False for x in body['results'])


def test_tse_catalog_cannot_create_recommendation_candidate():
    r = client.post('/api/v1/recommendations/generate', json={
        'request_id': 'req-tse-map-1',
        'text_input': '发热伴咽痛',
        'symptoms': ['发热', '咽痛']
    })
    assert r.status_code == 200
    body = r.json()
    assert body['formula_candidates'] == []
    assert 'LEGACY_KNOWLEDGE_BRIDGE' not in body['uncertainty_flags']
