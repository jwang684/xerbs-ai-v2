from __future__ import annotations
from app.schemas.reasoning import EvidenceItem, ReasoningResponse

CLINICAL_TERMS=("发热","怕冷","恶寒","怕热","头痛","咳嗽","食欲差","便秘","腹泻","失眠")
NEGATED_FORMS={
    "发热": ("无发热","没有发热","否认发热","不发热"),
    "怕冷": ("不怕冷","无怕冷","没有怕冷"),
    "恶寒": ("无恶寒","没有恶寒","不恶寒"),
    "怕热": ("不怕热","无怕热","没有怕热"),
    "头痛": ("无头痛","没有头痛","不头痛"),
    "咳嗽": ("无咳嗽","没有咳嗽","不咳嗽"),
    "食欲差": ("食欲正常","胃口正常"),
    "便秘": ("无便秘","没有便秘","大便正常"),
    "腹泻": ("无腹泻","没有腹泻","大便正常"),
    "失眠": ("无失眠","没有失眠","睡眠正常"),
}

class ContradictionEngine:
    """Conservative contradiction annotator based only on explicit patient text."""
    def annotate(self, reasoning: ReasoningResponse, patient_text: str) -> ReasoningResponse:
        text=patient_text or ""
        for assessment in reasoning.pattern_assessments:
            model_text=" ".join(x.text for x in assessment.supporting_evidence)
            for term in CLINICAL_TERMS:
                if term not in model_text:
                    continue
                if any(form in text for form in NEGATED_FORMS.get(term,())):
                    message=f"Patient report explicitly contradicts model evidence: {term}"
                    if message not in {x.text for x in assessment.contradictions}:
                        assessment.contradictions.append(EvidenceItem(text=message,source="deterministic_contradiction_check"))
        if any(x.contradictions for x in reasoning.pattern_assessments):
            if "PATTERN_CONTRADICTIONS_PRESENT" not in reasoning.uncertainty_flags:
                reasoning.uncertainty_flags.append("PATTERN_CONTRADICTIONS_PRESENT")
            reasoning.ready_for_formula_retrieval=False
        return reasoning
