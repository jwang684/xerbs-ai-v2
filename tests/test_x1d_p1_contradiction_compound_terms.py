"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-ACCELERATION-P1: compound signs are not the bare term.

Observed on Staging: a patient wrote "没有发热怕冷" (no fever or chills) while the model's
evidence cited "手心脚心发热" (hot palms and soles, a yin-deficiency sign). The substring
check read the evidence as "fever" and the patient text as denying fever, flagged a
contradiction and forced ready_for_formula_retrieval=False -- vetoing a correct pattern.
"""

from app.schemas.reasoning import EvidenceItem, PatternAssessment, ReasoningResponse
from app.services.reasoning.contradictions import ContradictionEngine


def _reasoning(evidence: str, ready: bool = True) -> ReasoningResponse:
    return ReasoningResponse(
        pattern_assessments=[PatternAssessment(
            pattern_id="p-1", name="阴虚火旺", model_confidence=0.9, corpus_match=True,
            supporting_evidence=[EvidenceItem(text=evidence, source="model_reasoning")])],
        ready_for_formula_retrieval=ready,
    )


def _annotate(evidence: str, patient_text: str) -> ReasoningResponse:
    return ContradictionEngine().annotate(_reasoning(evidence), patient_text)


def test_hot_palms_and_soles_is_not_contradicted_by_denying_fever():
    r = _annotate("失眠多梦、手心脚心发热、舌红少苔，提示阴虚内热。",
                  "晚上难以入睡，手心脚心发热。没有发热怕冷。")
    assert r.pattern_assessments[0].contradictions == []
    assert "PATTERN_CONTRADICTIONS_PRESENT" not in r.uncertainty_flags
    assert r.ready_for_formula_retrieval is True


def test_every_listed_compound_form_is_excluded():
    for phrase in ("手足心发热", "五心发热", "手心发热", "脚心发热", "足心发热", "掌心发热"):
        r = _annotate(f"症见{phrase}，舌红。", "没有发热。")
        assert r.pattern_assessments[0].contradictions == [], phrase
        assert r.ready_for_formula_retrieval is True, phrase


def test_a_genuine_fever_claim_is_still_contradicted():
    r = _annotate("外感风热，发热恶寒并见，咽痛。", "没有发热，也不怕冷。")
    assert [x.text for x in r.pattern_assessments[0].contradictions] == \
        ["Patient report explicitly contradicts model evidence: 发热"]
    assert "PATTERN_CONTRADICTIONS_PRESENT" in r.uncertainty_flags
    assert r.ready_for_formula_retrieval is False


def test_fever_alongside_a_compound_sign_is_still_contradicted():
    # The compound is removed, the bare "发热" that remains is still checked.
    r = _annotate("手心脚心发热，且午后发热。", "没有发热。")
    assert r.pattern_assessments[0].contradictions
    assert r.ready_for_formula_retrieval is False


def test_other_terms_are_unchanged():
    r = _annotate("心烦失眠，入睡困难。", "睡眠正常，没有失眠。")
    assert [x.text for x in r.pattern_assessments[0].contradictions] == \
        ["Patient report explicitly contradicts model evidence: 失眠"]
    assert r.ready_for_formula_retrieval is False


def test_no_negation_means_no_contradiction():
    r = _annotate("外感风热，发热恶寒。", "发热两天，咽痛。")
    assert r.pattern_assessments[0].contradictions == []
    assert r.ready_for_formula_retrieval is True
