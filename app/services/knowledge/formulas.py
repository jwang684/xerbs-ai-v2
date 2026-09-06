from dataclasses import dataclass


@dataclass(frozen=True)
class FormulaRecord:
    formula_id: str
    name: str
    ingredients: tuple[str, ...]
    efficacy: str
    suitable_symptoms: tuple[str, ...]


# Extracted from the legacy StaticDataService. Kept only as a tiny offline
# regression fixture; this is not intended to be the production TSE corpus.
LEGACY_FORMULAS: tuple[FormulaRecord, ...] = (
    FormulaRecord(
        formula_id="legacy-1",
        name="银翘散",
        ingredients=("金银花", "连翘", "薄荷", "荆芥", "桔梗", "甘草", "竹叶", "牛蒡子"),
        efficacy="清热解毒，疏风散热",
        suitable_symptoms=("发热", "头痛", "咽痛", "咳嗽", "风热感冒"),
    ),
    FormulaRecord(
        formula_id="legacy-2",
        name="麻黄汤",
        ingredients=("麻黄", "桂枝", "杏仁", "甘草"),
        efficacy="发汗解表，宣肺平喘",
        suitable_symptoms=("恶寒发热", "头痛", "身痛", "喘咳", "风寒感冒"),
    ),
    FormulaRecord(
        formula_id="legacy-3",
        name="藿香正气散",
        ingredients=("藿香", "紫苏叶", "白芷", "桔梗", "白术", "陈皮", "厚朴", "甘草"),
        efficacy="解表化湿，理气和中",
        suitable_symptoms=("胃肠不适", "恶心", "腹泻", "头重", "暑湿感冒"),
    ),
)


class FormulaKnowledgeService:
    """Deterministic legacy-fixture matcher.

    Important difference from v1: if there is no explicit symptom overlap,
    return no formula instead of silently returning the first three formulas.
    """

    def rank_explicit_matches(self, symptoms: list[str], text_input: str = "") -> list[dict]:
        normalized = {s.strip() for s in symptoms if s and s.strip()}
        haystack = text_input or ""
        scored: list[tuple[int, FormulaRecord, list[str]]] = []
        for formula in LEGACY_FORMULAS:
            matched = [s for s in formula.suitable_symptoms if s in normalized or s in haystack]
            if matched:
                scored.append((len(matched), formula, matched))
        scored.sort(key=lambda x: (-x[0], x[1].formula_id))
        return [
            {
                "formula_id": f.formula_id,
                "name": f.name,
                "confidence": min(0.75, 0.35 + 0.10 * score),
                "rationale": f"Legacy fixture explicit symptom overlap: {', '.join(matched)}",
                "ingredients": list(f.ingredients),
                "safety_flags": ["LEGACY_FIXTURE_ONLY", "PRACTITIONER_REVIEW_REQUIRED"],
            }
            for score, f, matched in scored[:3]
        ]
