from fastapi.testclient import TestClient

from app.main import app
from app.schemas.clinical_workflow import (
    ClinicalEntityType,
    IngestionBatchRequest,
    IngestionItem,
    ReviewActionRequest,
    ReviewDecision,
)
from app.schemas.clinical_knowledge import SourceRef
from app.services.knowledge.clinical_corpus import ClinicalKnowledgeCorpus
from app.services.knowledge.clinical_workflow import ClinicalCorpusWorkflow, ClinicalWorkflowError

client = TestClient(app)


def _source():
    return SourceRef(source_id="src-test", title="Curated test source", citation="test citation", source_type="CURATED_TEST")


def test_ingestion_forces_draft_and_records_history():
    repo = ClinicalKnowledgeCorpus()
    flow = ClinicalCorpusWorkflow(repo)
    result = flow.ingest(IngestionBatchRequest(
        submitted_by="importer",
        source_label="test",
        items=[IngestionItem(
            entity_type=ClinicalEntityType.FORMULA,
            payload={"name": "Test Formula", "indications": ["test symptom"], "review_status": "REVIEWED", "version": 99},
            sources=[_source()],
        )],
    ))
    entity_id = result.created_entity_ids[0]
    row = next(x for x in repo.formulas if x.formula_id == entity_id)
    assert row.review_status.value == "DRAFT"
    assert row.version == 1
    assert row.clinical_ranking_eligible is False
    assert len(flow.get_history(ClinicalEntityType.FORMULA, entity_id)) == 1


def test_approval_requires_source():
    repo = ClinicalKnowledgeCorpus()
    flow = ClinicalCorpusWorkflow(repo)
    batch = flow.ingest(IngestionBatchRequest(
        submitted_by="importer",
        source_label="test",
        items=[IngestionItem(entity_type=ClinicalEntityType.FORMULA, payload={"name": "No Source Formula"})],
    ))
    entity_id = batch.created_entity_ids[0]
    flow.submit_for_review(ClinicalEntityType.FORMULA, entity_id, "submitter")
    try:
        flow.review(ClinicalEntityType.FORMULA, entity_id, ReviewActionRequest(reviewer_id="reviewer", decision=ReviewDecision.APPROVE))
        assert False, "expected ClinicalWorkflowError"
    except ClinicalWorkflowError as exc:
        assert "source" in str(exc).lower()


def test_reviewed_formula_becomes_ranking_eligible():
    repo = ClinicalKnowledgeCorpus()
    flow = ClinicalCorpusWorkflow(repo)
    batch = flow.ingest(IngestionBatchRequest(
        submitted_by="importer",
        source_label="test",
        items=[IngestionItem(
            entity_type=ClinicalEntityType.FORMULA,
            payload={"name": "Reviewed Formula", "indications": ["测试症状"], "ingredients": ["Test Herb"]},
            sources=[_source()],
        )],
    ))
    entity_id = batch.created_entity_ids[0]
    flow.submit_for_review(ClinicalEntityType.FORMULA, entity_id, "submitter")
    reviewed = flow.review(ClinicalEntityType.FORMULA, entity_id, ReviewActionRequest(reviewer_id="reviewer", decision=ReviewDecision.APPROVE, notes="approved for test"))
    assert reviewed["review_status"] == "REVIEWED"
    assert reviewed["clinical_ranking_eligible"] is True
    assert reviewed["version"] == 3
    matches = repo.eligible_formula_candidates(["测试症状"], "")
    assert matches[0]["name"] == "Reviewed Formula"
    assert len(flow.get_history(ClinicalEntityType.FORMULA, entity_id)) == 3
    assert [x.action for x in flow.audit(entity_id=entity_id)] == ["INGESTED_AS_DRAFT", "SUBMITTED_FOR_REVIEW", "APPROVED"]


def test_request_changes_returns_record_to_draft():
    repo = ClinicalKnowledgeCorpus()
    flow = ClinicalCorpusWorkflow(repo)
    batch = flow.ingest(IngestionBatchRequest(
        submitted_by="importer",
        source_label="test",
        items=[IngestionItem(entity_type=ClinicalEntityType.HERB, payload={"name": "Test Herb"}, sources=[_source()])],
    ))
    entity_id = batch.created_entity_ids[0]
    flow.submit_for_review(ClinicalEntityType.HERB, entity_id, "submitter")
    row = flow.review(ClinicalEntityType.HERB, entity_id, ReviewActionRequest(reviewer_id="reviewer", decision=ReviewDecision.REQUEST_CHANGES))
    assert row["review_status"] == "DRAFT"
    assert row["clinical_ranking_eligible"] is False


def test_phase5_api_ingest_review_history_round_trip():
    ingest = client.post('/api/v1/knowledge/clinical/ingest', json={
        "submitted_by": "api-importer",
        "source_label": "phase5-api-test",
        "items": [{
            "entity_type": "formula",
            "payload": {"name": "API Reviewed Formula", "indications": ["API症状"], "ingredients": ["API Herb"]},
            "sources": [{"source_id": "api-source", "title": "API source", "source_type": "CURATED_TEST"}]
        }]
    })
    assert ingest.status_code == 200
    entity_id = ingest.json()["created_entity_ids"][0]

    submit = client.post(f'/api/v1/knowledge/clinical/entities/formula/{entity_id}/submit-review', json={"submitted_by": "api-submit"})
    assert submit.status_code == 200
    assert submit.json()["review_status"] == "IN_REVIEW"

    approve = client.post(f'/api/v1/knowledge/clinical/entities/formula/{entity_id}/review', json={"reviewer_id": "api-reviewer", "decision": "APPROVE"})
    assert approve.status_code == 200
    assert approve.json()["review_status"] == "REVIEWED"
    # Since Phase 12C-2D3 ranking also requires a REVIEWED Source; "api-source"
    # is only ingested, so it stays DRAFT and the entity does not rank.
    assert approve.json()["clinical_ranking_eligible"] is False

    history = client.get(f'/api/v1/knowledge/clinical/entities/formula/{entity_id}/history')
    assert history.status_code == 200
    assert len(history.json()["results"]) == 3


def ensure_source_reviewed(client_, source_id):
    """Idempotently drive a Source to REVIEWED (Phase 12C-2D3A retrieval gate)."""
    base = '/api/v1/knowledge/clinical/sources'
    cur = client_.get(f'{base}/{source_id}').json()
    if cur['review_status'] == 'REVIEWED':
        return
    if cur['review_status'] == 'DRAFT':
        r = client_.post(f'{base}/{source_id}/submit-review', json={'submitted_by': 'curator', 'expected_version': cur['version']})
        assert r.status_code == 200, r.text
        cur = r.json()
    assert cur['review_status'] == 'IN_REVIEW', cur['review_status']
    r = client_.post(f'{base}/{source_id}/review', json={'reviewer_id': 'reviewer', 'reviewer_role': 'CLINICAL_REVIEWER',
                                                         'decision': 'APPROVE', 'expected_version': cur['version']})
    assert r.status_code == 200, r.text


def test_reviewed_api_formula_enters_recommendation_pipeline():
    ingest = client.post('/api/v1/knowledge/clinical/ingest', json={
        "submitted_by": "pipeline-importer",
        "source_label": "phase5-pipeline-test",
        "items": [{
            "entity_type": "formula",
            "payload": {"name": "Pipeline Formula", "indications": ["管线症状"], "ingredients": ["Pipeline Herb"]},
            "sources": [{"source_id": "pipeline-source", "title": "Pipeline source", "source_type": "CURATED_TEST"}]
        }]
    })
    entity_id = ingest.json()["created_entity_ids"][0]
    client.post(f'/api/v1/knowledge/clinical/entities/formula/{entity_id}/submit-review', json={"submitted_by": "pipeline-submit"})
    client.post(f'/api/v1/knowledge/clinical/entities/formula/{entity_id}/review', json={"reviewer_id": "pipeline-reviewer", "decision": "APPROVE"})
    ensure_source_reviewed(client, "pipeline-source")

    response = client.post('/api/v1/recommendations/generate', json={
        "request_id": "phase5-pipeline-request",
        "text_input": "患者报告管线症状",
        "symptoms": ["管线症状"],
        "goals": [],
        "constraints": [],
        "language": "zh-CN"
    })
    assert response.status_code == 200
    body = response.json()
    assert body["formula_candidates"][0]["name"] == "Pipeline Formula"
    assert "REVIEWED_CLINICAL_CORPUS" in body["uncertainty_flags"]
    assert body["requires_practitioner_review"] is True
