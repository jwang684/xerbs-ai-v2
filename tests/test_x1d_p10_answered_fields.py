"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P10: earlier answers count; 不清楚 is answered, not missing.

P9 (real Staging patient): sleep and stool were asked, answered 不清楚, and asked again
two turns later -- the deterministic check only read the current turn's prose, which
no longer repeated them. The structured answers core already carries in
interview_state are now consumed, and an explicit unknown is its own state.
"""

import pytest

from app.schemas.intake import RecommendationRequest
from app.schemas.reasoning import InterviewCarryState, ResolvableEvidence
from app.services.reasoning.engine import (ANSWERED_UNKNOWN, DiagnosticReasoningEngine,
                                           carried_answer_states, is_unknown_answer)

COMPLAINT = "发病两天，恶寒，发热，无汗，身痛头痛，鼻流清涕。舌质淡红，苔薄而润，脉浮紧。"


class _NoCorpus:
    def match_reviewed_patterns(self, *a, **k):
        return []


class _Resolver:
    def match_pattern(self, *a, **k):
        return []


def _engine():
    return DiagnosticReasoningEngine(_NoCorpus(), _Resolver())


def _request(text=COMPLAINT, answers=None):
    carry = None
    if answers is not None:
        carry = InterviewCarryState(resolvable_evidence=ResolvableEvidence(
            answers=[{"turn_id": t, "question_field": f, "answer": a} for t, f, a in answers]))
    return RecommendationRequest(text_input=text, symptoms=[], interview_state=carry)


def _missing(r):
    return {m.field: m for m in r.missing_information}


def test_turn_one_asks_the_unanswered_basic_fields():
    r = _engine().analyze(_request())
    assert set(r.followup_questions) == {"近期食欲和进食情况如何？", "大便情况如何？", "近期睡眠情况如何？"}
    assert all(m.answer_status == "UNANSWERED" for m in r.missing_information)


def test_unknown_is_answered_and_is_not_asked_again_in_a_later_turn():
    # Turn 3's own text repeats nothing from turn 2 -- exactly the P9 shape.
    r = _engine().analyze(_request(
        text=COMPLAINT + "。补充信息：这次发病前有没有受凉、淋雨或吹风？ 否",
        answers=[(1, "sleep", "不清楚"), (1, "stool", "不清楚"), (2, "cause", "否")]))
    assert "近期睡眠情况如何？" not in r.followup_questions
    assert "大便情况如何？" not in r.followup_questions
    m = _missing(r)
    # still clinically unknown: kept in missing_information, marked as answered-unknown
    assert m["sleep"].answer_status == ANSWERED_UNKNOWN and m["stool"].answer_status == ANSWERED_UNKNOWN
    assert "BASIC_FIELD_ANSWERED_UNKNOWN" in r.uncertainty_flags
    # appetite was never answered: still asked
    assert r.followup_questions == ["近期食欲和进食情况如何？"]


def test_a_known_answer_from_an_earlier_turn_satisfies_the_field():
    r = _engine().analyze(_request(answers=[(1, "sleep", "入睡困难，易醒"), (1, "stool", "偏干")]))
    m = _missing(r)
    assert "sleep" not in m and "stool" not in m
    assert r.followup_questions == ["近期食欲和进食情况如何？"]


def test_a_high_priority_unknown_is_not_reasked_but_still_blocks_retrieval():
    r = _engine().analyze(_request(text="头痛，鼻塞", answers=[(1, "duration", "不清楚")]))
    m = _missing(r)
    assert m["duration"].priority == "HIGH" and m["duration"].answer_status == ANSWERED_UNKNOWN
    assert "症状持续多久？何时开始？" not in r.followup_questions
    assert r.ready_for_formula_retrieval is False           # threshold unchanged


def test_the_latest_answer_for_a_field_wins():
    states = carried_answer_states(_request(answers=[(1, "sleep", "不清楚"), (2, "sleep", "失眠多梦")]))
    assert states == {"sleep": "KNOWN"}


@pytest.mark.parametrize("answer,unknown", [
    ("不清楚", True), (" 不知道。", True), ("不确定", True), ("说不清", True), ("Unknown", True),
    ("不清楚是不是发热", False), ("否", False), ("一般", False), ("", False)])
def test_unknown_is_matched_exactly_not_as_a_substring(answer, unknown):
    assert is_unknown_answer(answer) is unknown


@pytest.mark.parametrize("carry", [None, "junk"])
def test_missing_or_malformed_history_means_ask(carry):
    req = _request()
    if carry == "junk":
        object.__setattr__(req, "interview_state", "junk")
    r = _engine().analyze(req)
    assert "近期睡眠情况如何？" in r.followup_questions


def test_answers_without_a_field_or_value_are_ignored():
    r = _engine().analyze(_request(answers=[(1, "", "不清楚"), (1, "sleep", "")]))
    assert "近期睡眠情况如何？" in r.followup_questions
