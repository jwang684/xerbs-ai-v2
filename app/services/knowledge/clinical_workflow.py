from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.clinical_knowledge import (
    ClinicalFormulaRecord,
    HerbRecord,
    PatternRecord,
    ReviewStatus,
)
from app.schemas.clinical_workflow import (
    ClinicalEntityType,
    IngestionBatchRequest,
    IngestionBatchResult,
    ReviewActionRequest,
    ReviewDecision,
    WorkflowEvent,
)
from app.services.knowledge.clinical_corpus import ClinicalKnowledgeCorpus


class ClinicalWorkflowError(ValueError):
    pass


class ClinicalCorpusWorkflow:
    """Import -> normalize -> draft -> review -> approve/reject workflow.

    This is intentionally persistence-agnostic in Phase 5. History and audit are
    append-only in process memory; a later persistence phase can map the same
    contract to PostgreSQL without changing API semantics.
    """

    def __init__(self, corpus: ClinicalKnowledgeCorpus):
        self.corpus = corpus
        self.batches: dict[str, IngestionBatchResult] = {}
        self.history: dict[tuple[str, str], list[dict]] = {}
        self.events: list[WorkflowEvent] = []

    def ingest(self, request: IngestionBatchRequest) -> IngestionBatchResult:
        batch_id = f"ing-{uuid4().hex[:16]}"
        ids: list[str] = []
        for item in request.items:
            record = self._normalize(item.entity_type, item.payload, item.sources)
            self._append_record(item.entity_type, record)
            entity_id = self._entity_id(item.entity_type, record)
            ids.append(entity_id)
            self._snapshot(item.entity_type, record)
            self._event(item.entity_type, entity_id, "INGESTED_AS_DRAFT", request.submitted_by, None, ReviewStatus.DRAFT.value, record.version)

        result = IngestionBatchResult(
            batch_id=batch_id,
            submitted_by=request.submitted_by,
            source_label=request.source_label,
            created_entity_ids=ids,
            created_at=datetime.now(timezone.utc),
        )
        self.batches[batch_id] = result
        return result

    def submit_for_review(self, entity_type: ClinicalEntityType, entity_id: str, actor: str, notes: str | None = None) -> dict:
        record = self._get(entity_type, entity_id)
        if record.review_status not in {ReviewStatus.DRAFT, ReviewStatus.REJECTED}:
            raise ClinicalWorkflowError(f"Cannot submit {record.review_status.value} record for review")
        before = record.review_status.value
        record.review_status = ReviewStatus.IN_REVIEW
        record.version += 1
        self._snapshot(entity_type, record)
        self._event(entity_type, entity_id, "SUBMITTED_FOR_REVIEW", actor, before, record.review_status.value, record.version, notes)
        return self._serialize(entity_type, record)

    def review(self, entity_type: ClinicalEntityType, entity_id: str, request: ReviewActionRequest) -> dict:
        record = self._get(entity_type, entity_id)
        if record.review_status != ReviewStatus.IN_REVIEW:
            raise ClinicalWorkflowError("Only IN_REVIEW records can receive a review decision")

        before = record.review_status.value
        if request.decision == ReviewDecision.APPROVE:
            if not record.sources:
                raise ClinicalWorkflowError("Approval requires at least one explicit source")
            record.review_status = ReviewStatus.REVIEWED
            action = "APPROVED"
        elif request.decision == ReviewDecision.REJECT:
            record.review_status = ReviewStatus.REJECTED
            action = "REJECTED"
        else:
            record.review_status = ReviewStatus.DRAFT
            action = "CHANGES_REQUESTED"

        record.version += 1
        self._snapshot(entity_type, record)
        self._event(entity_type, entity_id, action, request.reviewer_id, before, record.review_status.value, record.version, request.notes)
        return self._serialize(entity_type, record)

    def get_history(self, entity_type: ClinicalEntityType, entity_id: str) -> list[dict]:
        self._get(entity_type, entity_id)
        return deepcopy(self.history.get((entity_type.value, entity_id), []))

    def get_batch(self, batch_id: str) -> IngestionBatchResult:
        if batch_id not in self.batches:
            raise ClinicalWorkflowError("Ingestion batch not found")
        return self.batches[batch_id]

    def audit(self, entity_type: ClinicalEntityType | None = None, entity_id: str | None = None) -> list[WorkflowEvent]:
        rows = self.events
        if entity_type is not None:
            rows = [x for x in rows if x.entity_type == entity_type]
        if entity_id is not None:
            rows = [x for x in rows if x.entity_id == entity_id]
        return rows

    def _normalize(self, entity_type: ClinicalEntityType, payload: dict, sources: list):
        # Workflow-owned fields cannot be injected by ingestion payload.
        clean = {k: v for k, v in payload.items() if k not in {"review_status", "version", "sources", "clinical_ranking_eligible"}}
        if entity_type == ClinicalEntityType.PATTERN:
            return PatternRecord(
                pattern_id=f"pat-{uuid4().hex[:16]}",
                name=self._required_name(clean),
                aliases=clean.get("aliases", []),
                indications=clean.get("indications", []),
                exclusion_flags=clean.get("exclusion_flags", []),
                sources=sources,
                review_status=ReviewStatus.DRAFT,
                version=1,
            )
        if entity_type == ClinicalEntityType.HERB:
            return HerbRecord(
                herb_id=f"herb-{uuid4().hex[:16]}",
                name=self._required_name(clean),
                aliases=clean.get("aliases", []),
                contraindications=clean.get("contraindications", []),
                interaction_flags=clean.get("interaction_flags", []),
                sources=sources,
                review_status=ReviewStatus.DRAFT,
                version=1,
            )
        return ClinicalFormulaRecord(
            formula_id=f"frm-{uuid4().hex[:16]}",
            name=self._required_name(clean),
            aliases=clean.get("aliases", []),
            pattern_ids=clean.get("pattern_ids", []),
            indications=clean.get("indications", []),
            ingredients=clean.get("ingredients", []),
            contraindications=clean.get("contraindications", []),
            interaction_flags=clean.get("interaction_flags", []),
            sources=sources,
            review_status=ReviewStatus.DRAFT,
            version=1,
            migration_origin=clean.get("migration_origin"),
        )

    @staticmethod
    def _required_name(payload: dict) -> str:
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ClinicalWorkflowError("payload.name is required")
        return name

    def _append_record(self, entity_type: ClinicalEntityType, record):
        if entity_type == ClinicalEntityType.PATTERN:
            self.corpus.patterns.append(record)
        elif entity_type == ClinicalEntityType.HERB:
            self.corpus.herbs.append(record)
        else:
            self.corpus.formulas.append(record)

    def _get(self, entity_type: ClinicalEntityType, entity_id: str):
        collection = {
            ClinicalEntityType.PATTERN: self.corpus.patterns,
            ClinicalEntityType.FORMULA: self.corpus.formulas,
            ClinicalEntityType.HERB: self.corpus.herbs,
        }[entity_type]
        id_field = {
            ClinicalEntityType.PATTERN: "pattern_id",
            ClinicalEntityType.FORMULA: "formula_id",
            ClinicalEntityType.HERB: "herb_id",
        }[entity_type]
        for record in collection:
            if getattr(record, id_field) == entity_id:
                return record
        raise ClinicalWorkflowError("Clinical corpus entity not found")

    @staticmethod
    def _entity_id(entity_type: ClinicalEntityType, record) -> str:
        return getattr(record, {ClinicalEntityType.PATTERN: "pattern_id", ClinicalEntityType.FORMULA: "formula_id", ClinicalEntityType.HERB: "herb_id"}[entity_type])

    def _snapshot(self, entity_type: ClinicalEntityType, record):
        entity_id = self._entity_id(entity_type, record)
        self.history.setdefault((entity_type.value, entity_id), []).append(deepcopy(self._serialize(entity_type, record)))

    def _event(self, entity_type, entity_id, action, actor, from_status, to_status, version, notes=None):
        self.events.append(WorkflowEvent(
            event_id=f"evt-{uuid4().hex[:16]}",
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            actor_id=actor,
            from_status=from_status,
            to_status=to_status,
            version=version,
            notes=notes,
        ))

    @staticmethod
    def _serialize(entity_type: ClinicalEntityType, record) -> dict:
        return {"entity_type": entity_type.value, **record.model_dump(), "clinical_ranking_eligible": record.clinical_ranking_eligible}
