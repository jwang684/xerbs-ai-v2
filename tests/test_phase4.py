from fastapi.testclient import TestClient

from app.main import app
from app.services.knowledge.clinical_corpus import ClinicalKnowledgeCorpus

client = TestClient(app)


def test_corpus_starts_with_no_ranking_eligible_records():
    repo = ClinicalKnowledgeCorpus()
    stats = repo.stats()
    assert stats.formulas == 3
    assert stats.ranking_eligible_formulas == 0
    assert stats.ranking_eligible_patterns == 0
    assert stats.ranking_eligible_herbs == 0


def test_legacy_formula_is_searchable_but_not_eligible():
    r = client.post('/api/v1/knowledge/clinical/search', json={"query": "银翘散"})
    assert r.status_code == 200
    rows = r.json()['results']
    assert len(rows) == 1
    assert rows[0]['review_status'] == 'DRAFT'
    assert rows[0]['clinical_ranking_eligible'] is False
    assert rows[0]['migration_origin'] == 'LEGACY_STATIC_DATA_FIXTURE'


def test_reviewed_only_search_excludes_legacy_drafts():
    r = client.post('/api/v1/knowledge/clinical/search', json={"query": "银翘散", "reviewed_only": True})
    assert r.status_code == 200
    assert r.json()['results'] == []


def test_governance_endpoint_declares_review_gate():
    r = client.get('/api/v1/knowledge/clinical/governance')
    assert r.status_code == 200
    body = r.json()
    assert body['draft_records_may_rank'] is False
    assert body['legacy_tse_symptom_metadata_may_rank'] is False
    assert body['trust_score_owned_by_ai_service'] is False


def test_review_gate_prevents_draft_formula_ranking():
    repo = ClinicalKnowledgeCorpus()
    assert repo.eligible_formula_candidates(["发热", "咽痛"], "") == []
