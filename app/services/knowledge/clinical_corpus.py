from functools import lru_cache

from app.schemas.clinical_knowledge import (
    ClinicalFormulaRecord,
    CorpusStats,
    HerbRecord,
    PatternRecord,
    ReviewStatus,
)
from app.services.knowledge.formulas import LEGACY_FORMULAS


class ClinicalKnowledgeCorpus:
    """Versioned, review-gated clinical knowledge corpus.

    Legacy formula fixtures remain DRAFT migration records. Only REVIEWED records
    with explicit sources can participate in deterministic clinical ranking.
    """

    def __init__(self):
        self.patterns: list[PatternRecord] = []
        self.herbs: list[HerbRecord] = []
        self.formulas: list[ClinicalFormulaRecord] = [
            ClinicalFormulaRecord(
                formula_id=f"corpus-{row.formula_id}",
                name=row.name,
                indications=list(row.suitable_symptoms),
                ingredients=list(row.ingredients),
                review_status=ReviewStatus.DRAFT,
                version=1,
                sources=[],
                migration_origin="LEGACY_STATIC_DATA_FIXTURE",
            )
            for row in LEGACY_FORMULAS
        ]

    def stats(self) -> CorpusStats:
        return CorpusStats(
            patterns=len(self.patterns),
            formulas=len(self.formulas),
            herbs=len(self.herbs),
            ranking_eligible_patterns=sum(1 for x in self.patterns if x.clinical_ranking_eligible),
            ranking_eligible_formulas=sum(1 for x in self.formulas if x.clinical_ranking_eligible),
            ranking_eligible_herbs=sum(1 for x in self.herbs if x.clinical_ranking_eligible),
        )

    def search(self, query: str, entity_types: list[str], reviewed_only: bool = False, limit: int = 20) -> list[dict]:
        q = query.strip().lower()
        rows: list[dict] = []

        def allowed(item) -> bool:
            return (not reviewed_only) or item.review_status == ReviewStatus.REVIEWED

        if "pattern" in entity_types:
            for x in self.patterns:
                if allowed(x) and self._matches(q, [x.name, *x.aliases, *x.indications]):
                    rows.append({"entity_type": "pattern", **x.model_dump(), "clinical_ranking_eligible": x.clinical_ranking_eligible})
        if "formula" in entity_types:
            for x in self.formulas:
                if allowed(x) and self._matches(q, [x.name, *x.aliases, *x.indications, *x.ingredients]):
                    rows.append({"entity_type": "formula", **x.model_dump(), "clinical_ranking_eligible": x.clinical_ranking_eligible})
        if "herb" in entity_types:
            for x in self.herbs:
                if allowed(x) and self._matches(q, [x.name, *x.aliases]):
                    rows.append({"entity_type": "herb", **x.model_dump(), "clinical_ranking_eligible": x.clinical_ranking_eligible})
        return rows[:limit]

    def eligible_formula_candidates(self, symptoms: list[str], text_input: str = "") -> list[dict]:
        normalized = {s.strip() for s in symptoms if s and s.strip()}
        haystack = text_input or ""
        scored: list[tuple[int, ClinicalFormulaRecord, list[str]]] = []
        for formula in self.formulas:
            if not formula.clinical_ranking_eligible:
                continue
            matched = [x for x in formula.indications if x in normalized or x in haystack]
            if matched:
                scored.append((len(matched), formula, matched))
        scored.sort(key=lambda x: (-x[0], x[1].formula_id))
        return [
            {
                "formula_id": f.formula_id,
                "name": f.name,
                "confidence": min(0.85, 0.40 + 0.10 * score),
                "rationale": f"Reviewed corpus indication overlap: {', '.join(matched)}",
                "ingredients": f.ingredients,
                "safety_flags": list(dict.fromkeys([*f.contraindications, *f.interaction_flags, "PRACTITIONER_REVIEW_REQUIRED"])),
            }
            for score, f, matched in scored[:3]
        ]

    @staticmethod
    def _matches(query: str, fields: list[str]) -> bool:
        if not query:
            return False
        return any(query in str(v).lower() for v in fields)


@lru_cache
def get_clinical_corpus() -> ClinicalKnowledgeCorpus:
    """Process-local corpus singleton.

    Phase 5 uses this to ensure ingestion/review and recommendation retrieval see
    the same state. Production persistence should replace this with PostgreSQL.
    """
    return ClinicalKnowledgeCorpus()
