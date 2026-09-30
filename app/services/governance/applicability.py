"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P18: structured, deterministic applicability.

Some authoritative sources state a pattern -> formula relationship only for a
defined patient condition. 《咳嗽中医诊疗专家共识意见（2021）》 §1.1 is the first:
it applies to patients "以咳嗽为主要或唯一症状者". Such a relationship must not
produce a formula for a patient whose condition has not been established.

The contract is deliberately tiny:

    content["applicability"] = {"condition": "COUGH_PRIMARY"}

on a governed pattern or formula. It lives in the entity content, so it is part
of the entity's content hash: a verified object cannot later be broadened
without becoming a new version that its verification no longer covers.

Evaluation reads only an explicit, patient-answered context value:

    condition_context = {"cough_primary": "YES" | "NO" | "UNKNOWN"}

    YES                      -> satisfied
    NO / UNKNOWN / malformed -> rejected (the patient answered; never re-asked)
    absent                   -> missing (a fixed question may be asked once)
    unsupported/malformed applicability -> fail closed

Nothing here reads free text, syndrome names, model reasoning or keywords.
Absent applicability means "not condition-scoped": existing content (the
influenza 风寒束表 -> 麻黄汤加味 path) is unaffected. Pure; no I/O.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

COUGH_PRIMARY = "COUGH_PRIMARY"

#: The only supported condition, and the patient-context field that answers it.
CONDITION_CONTEXT_FIELDS = {COUGH_PRIMARY: "cough_primary"}

YES, NO, UNKNOWN = "YES", "NO", "UNKNOWN"

NOT_SCOPED = "NOT_SCOPED"
SATISFIED = "SCOPE_SATISFIED"
REJECTED = "SCOPE_REJECTED"
MISSING = "SCOPE_CONTEXT_MISSING"
UNSUPPORTED = "SCOPE_UNSUPPORTED_FAIL_CLOSED"

# Engineering/governance flags (not clinical judgments).
FLAG_CONTEXT_REQUIRED = "CONDITION_SCOPE_CONTEXT_REQUIRED"
FLAG_NOT_SATISFIED = "CONDITION_SCOPE_NOT_SATISFIED"
FLAG_UNSUPPORTED = "CONDITION_SCOPE_UNSUPPORTED_FAIL_CLOSED"
FLAG_SATISFIED = "CONDITION_SCOPE_SATISFIED"

_ALLOWED_KEYS = frozenset({"condition"})


def read_condition(content: Any):
    """(state, condition) for one entity's content.

    state is NOT_SCOPED when the content carries no applicability, UNSUPPORTED
    for anything that is not exactly {"condition": <supported>}.
    """
    if not isinstance(content, Mapping) or "applicability" not in content:
        return NOT_SCOPED, None
    a = content.get("applicability")
    if not isinstance(a, Mapping) or set(a.keys()) != _ALLOWED_KEYS:
        return UNSUPPORTED, None
    condition = a.get("condition")
    if not isinstance(condition, str) or condition not in CONDITION_CONTEXT_FIELDS:
        return UNSUPPORTED, None
    return "SCOPED", condition


def evaluate(contents: Iterable[Any], condition_context: Any):
    """Outcome for a relationship whose endpoints have these contents.

    Returns (outcome, conditions, missing_fields). Every endpoint's condition
    must be satisfied; any unsupported applicability fails the whole link.
    """
    conditions = []
    for content in contents:
        state, condition = read_condition(content)
        if state == UNSUPPORTED:
            return UNSUPPORTED, (), ()
        if condition and condition not in conditions:
            conditions.append(condition)
    if not conditions:
        return NOT_SCOPED, (), ()
    ctx = condition_context if isinstance(condition_context, Mapping) else {}
    missing, rejected = [], False
    for condition in conditions:
        field = CONDITION_CONTEXT_FIELDS[condition]
        if field not in ctx or ctx.get(field) is None:
            missing.append(field)
        elif ctx.get(field) != YES:
            rejected = True
    if rejected:
        return REJECTED, tuple(conditions), ()
    if missing:
        return MISSING, tuple(conditions), tuple(missing)
    return SATISFIED, tuple(conditions), ()


def eligible(outcome: str) -> bool:
    return outcome in (NOT_SCOPED, SATISFIED)
