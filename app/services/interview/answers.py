"""X1D-P19B-FIX: the patient's answers from earlier rounds of this case.

core sends each turn's text with only THIS round's answers in it. Earlier
answers live in core's governed history and arrive here structured, in
``interview_state.resolvable_evidence.answers`` -- scoped by core to one user
and one case, de-duplicated with the latest answer per field winning, and
never containing an unanswered question.

Staging records #313-#315 showed what happens when nothing reads them: a
round-1 answer (出汗 否 / 流清涕 / 咽痛 是) was UNKNOWN again in round 3, so the
domain was re-asked, and the full-reasoning call -- the one whose output has
consequence -- never saw it at all.

Two readers, both pure:

  * answered_domains: which coverage domains the patient has explicitly
    answered, so routing and question selection stop treating them as open;
  * patient_answers: the answers themselves, field identity + patient value,
    for the full-reasoning prompt.

Rules that do not bend:

  * only explicit answers count; an unanswered question is absent, never "no";
  * the question WORDING is not carried, only the field it was asked under
    (ResolvableEvidence's provenance rule: model-written text never becomes a
    patient fact);
  * condition-scope answers (cough_primary, ...) are governance inputs to the
    applicability gate, not interview evidence. They are excluded here, so the
    gate stays independent of anything the model reads or writes.
"""

from __future__ import annotations

from typing import Any, List, Set

from app.services.clarification.coverage import domain_for_field
from app.services.governance.applicability import CONDITION_CONTEXT_FIELDS

_CONDITION_SCOPE_FIELDS = frozenset(CONDITION_CONTEXT_FIELDS.values())
_MAX_ANSWERS = 40
_MAX_ANSWER_CHARS = 200


def _raw_answers(request: Any) -> list:
    carry = getattr(request, "interview_state", None)
    evidence = getattr(carry, "resolvable_evidence", None) if carry is not None else None
    answers = getattr(evidence, "answers", None) if evidence is not None else None
    return answers if isinstance(answers, list) else []


def patient_answers(request: Any) -> List[dict]:
    """Explicit answers from this case, latest per field, condition-scope excluded."""
    latest: dict = {}
    try:
        for item in _raw_answers(request):
            if not isinstance(item, dict):
                continue
            field = str(item.get("question_field") or "").strip().lower()
            answer = str(item.get("answer") or "").strip()
            if not field or not answer or field in _CONDITION_SCOPE_FIELDS:
                continue
            latest.pop(field, None)              # latest answer wins, keeps order
            latest[field] = {"question_field": field,
                             "domain": domain_for_field(field),
                             "answer": answer[:_MAX_ANSWER_CHARS]}
    except Exception:  # noqa: BLE001 - missing history means "ask", never a crash
        return []
    return list(latest.values())[-_MAX_ANSWERS:]


def answered_domains(request: Any) -> Set[str]:
    """Coverage domains with an explicit answer in this case's history."""
    return {a["domain"] for a in patient_answers(request) if a.get("domain")}
