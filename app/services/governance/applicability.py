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

X1D-P19E adds an ordered list form, {"conditions": [C1, C2]}, and one
provisional condition (the P19C draft exclusion question). Provisional
conditions fail closed unless provisional rules are enabled in-process; no
deployment enables them.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

COUGH_PRIMARY = "COUGH_PRIMARY"
# X1D-P19E: the P19C draft exclusion question (§1.1 / §4.2.3 features), held as
# a PROVISIONAL ENGINEERING RULE. It is not clinically approved; see
# provisional_rules_enabled().
COUGH_EXCLUSION_FEATURES_NONE = "COUGH_EXCLUSION_FEATURES_NONE"

# X1D-P19R: the same question as an APPROVED, versioned condition. It may only
# appear inside its contract (see APPLICABILITY_CONTRACTS), never on its own.
COUGH_EXCLUSION_FEATURES_NONE_P19C_V1 = "COUGH_EXCLUSION_FEATURES_NONE_P19C_V1"

#: Supported conditions and the patient-context field that answers each.
CONDITION_CONTEXT_FIELDS = {
    COUGH_PRIMARY: "cough_primary",
    COUGH_EXCLUSION_FEATURES_NONE: "cough_exclusion_features",
    COUGH_EXCLUSION_FEATURES_NONE_P19C_V1: "cough_exclusion_features",
}
YES, NO, UNKNOWN = "YES", "NO", "UNKNOWN"
NONE_REPORTED, PRESENT = "NONE", "PRESENT"
#: The single context value that satisfies each condition. Anything else rejects.
SATISFYING_VALUE = {COUGH_PRIMARY: YES, COUGH_EXCLUSION_FEATURES_NONE: NONE_REPORTED,
                    COUGH_EXCLUSION_FEATURES_NONE_P19C_V1: NONE_REPORTED}

# ----------------------------------------------------------------------
# X1D-P19R: approved, versioned applicability contracts.
#
# A contract pins the exact ordered conditions and the reviewed document they
# were approved from (by SHA-256). Content opts in with
#     {"conditions": [...], "contract_version": "<id>"}
# and the conditions must equal the contract's exactly. A contract is usable
# only while ``active``; an inactive or unknown contract fails closed and makes
# the content's pattern invisible to matching (pattern_matchable).
#
# X1D-P19C-V1 was approved by the project owner's instruction ("原稿无修改通过").
# The packet's §6 approval record (reviewer name, qualification, date,
# signature) is not filled in, so the contract stays inactive until that
# record is completed and a reviewed commit activates it. Activation is a code
# change: no setting, variable or request can turn it on.
# ----------------------------------------------------------------------
APPROVED_APPLICABILITY_CONTRACT = "APPROVED_APPLICABILITY_CONTRACT"
APPROVAL_RECORD_INCOMPLETE = "APPROVAL_RECORD_INCOMPLETE"
APPROVAL_RECORD_COMPLETE = "APPROVAL_RECORD_COMPLETE"

APPLICABILITY_CONTRACTS = {
    "X1D-P19C-V1": {
        "conditions": (COUGH_PRIMARY, COUGH_EXCLUSION_FEATURES_NONE_P19C_V1),
        "source": "《咳嗽中医诊疗专家共识意见（2021）》 中医杂志 62(16) DOI 10.13288/j.11-2166/r.2021.16.018 §1.1 §4.2.3",
        "review_packet": "X1D-P19C-cough-consensus-2021-applicability-review.md",
        "review_packet_sha256": "82db5e2a57a47cf2aac0e42d5d2dffe623afd846859cbd820049a2ccd27a1b5e",
        "approval_basis": "OWNER_INSTRUCTION",
        "approval_record": APPROVAL_RECORD_INCOMPLETE,
        "active": False,
    },
}
#: Conditions that are valid only inside a contract.
CONTRACT_ONLY_CONDITIONS = frozenset({COUGH_EXCLUSION_FEATURES_NONE_P19C_V1})


def contract_active(version) -> bool:
    c = APPLICABILITY_CONTRACTS.get(version) if isinstance(version, str) else None
    return bool(c) and c.get("active") is True and c.get("approval_record") == APPROVAL_RECORD_COMPLETE


def contract_for(conditions):
    """The contract whose ordered conditions are exactly these, or None."""
    t = tuple(conditions or ())
    return next((k for k, c in APPLICABILITY_CONTRACTS.items() if c["conditions"] == t), None)


def contract_for_set(conditions):
    """The contract whose conditions are exactly this set (order-free), or None."""
    want = set(conditions or ())
    return next((k for k, c in APPLICABILITY_CONTRACTS.items() if set(c["conditions"]) == want), None)

#: Conditions whose rule is not clinically approved.
PROVISIONAL_CONDITIONS = frozenset({COUGH_EXCLUSION_FEATURES_NONE})
PROVISIONAL_ENGINEERING_RULE = "PROVISIONAL_ENGINEERING_RULE"
PENDING_CLINICAL_REVIEW = "PENDING_CLINICAL_REVIEW"

# X1D-P19E isolation: a code constant, deliberately NOT bound to settings,
# environment variables, request fields or any patient input. With it False
# (the only value any deployment ships), a provisional condition evaluates as
# UNSUPPORTED: the link fails closed and no question is asked. Only in-process
# test code may switch it on.
_PROVISIONAL_ENGINEERING_RULES_ENABLED = False


def provisional_rules_enabled() -> bool:
    return _PROVISIONAL_ENGINEERING_RULES_ENABLED is True


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
FLAG_PROVISIONAL = "PROVISIONAL_ENGINEERING_RULE_APPLIED"


def read_conditions(content: Any):
    """(state, conditions) for one entity's content.

    Accepted shapes, exactly: {"condition": C} (P18) or {"conditions": [C, ...]}
    -- a non-empty ordered list of distinct supported conditions. Anything else
    is UNSUPPORTED. No applicability key at all is NOT_SCOPED.
    """
    if not isinstance(content, Mapping) or "applicability" not in content:
        return NOT_SCOPED, ()
    a = content.get("applicability")
    if not isinstance(a, Mapping):
        return UNSUPPORTED, ()
    contract = None
    if set(a.keys()) == {"condition"}:
        items = [a.get("condition")]
    elif set(a.keys()) == {"conditions"} and isinstance(a.get("conditions"), list):
        items = list(a.get("conditions"))
    elif set(a.keys()) == {"conditions", "contract_version"} and isinstance(a.get("conditions"), list):
        items, contract = list(a.get("conditions")), a.get("contract_version")
        if not isinstance(contract, str) or contract not in APPLICABILITY_CONTRACTS                 or tuple(items) != APPLICABILITY_CONTRACTS[contract]["conditions"]:
            return UNSUPPORTED, ()
    else:
        return UNSUPPORTED, ()
    if not items or len(set(map(repr, items))) != len(items):
        return UNSUPPORTED, ()
    for c in items:
        if not isinstance(c, str) or c not in CONDITION_CONTEXT_FIELDS:
            return UNSUPPORTED, ()
        if c in CONTRACT_ONLY_CONDITIONS and contract is None:
            return UNSUPPORTED, ()
    return "SCOPED", tuple(items)


def pattern_matchable(content: Any) -> bool:
    """Whether a pattern with this content may match a model hypothesis at all.

    False while its applicability depends on something deliberately switched
    off -- an inactive or unknown contract, or a provisional rule while
    provisional rules are disabled. Such content is then exactly as if it had
    never been ingested: no match, no readiness, no question, no candidate.
    Malformed applicability stays matchable (P18): its links fail closed.
    """
    if not isinstance(content, Mapping) or "applicability" not in content:
        return True
    a = content.get("applicability")
    if isinstance(a, Mapping) and "contract_version" in a and not contract_active(a.get("contract_version")):
        return False
    state, items = read_conditions(content)
    if state == "SCOPED" and is_provisional(items) and not provisional_rules_enabled():
        return False
    if state == "SCOPED" and any(c in CONTRACT_ONLY_CONDITIONS for c in items)             and not contract_active(contract_for(items)):
        return False
    return True


def is_provisional(conditions: Iterable[str]) -> bool:
    return any(c in PROVISIONAL_CONDITIONS for c in conditions or ())


def evaluate(contents: Iterable[Any], condition_context: Any):
    """Outcome for a relationship whose endpoints have these contents.

    Returns (outcome, conditions, missing_fields). Conditions are evaluated in
    order: the first one without an answer is the only one reported missing,
    so a later question is never asked before an earlier one is satisfied.
    Any unsupported applicability -- or any provisional condition while
    provisional rules are disabled -- fails the whole link closed.
    """
    conditions = []
    for content in contents:
        state, items = read_conditions(content)
        if state == UNSUPPORTED:
            return UNSUPPORTED, (), ()
        for c in items:
            if c not in conditions:
                conditions.append(c)
    if not conditions:
        return NOT_SCOPED, (), ()
    if is_provisional(conditions) and not provisional_rules_enabled():
        return UNSUPPORTED, tuple(conditions), ()
    if any(c in CONTRACT_ONLY_CONDITIONS for c in conditions) and not contract_active(contract_for(conditions)):
        return UNSUPPORTED, tuple(conditions), ()
    ctx = condition_context if isinstance(condition_context, Mapping) else {}
    for condition in conditions:
        field = CONDITION_CONTEXT_FIELDS[condition]
        if field not in ctx or ctx.get(field) is None:
            return MISSING, tuple(conditions), (field,)
        if ctx.get(field) != SATISFYING_VALUE[condition]:
            return REJECTED, tuple(conditions), ()
    return SATISFIED, tuple(conditions), ()


def eligible(outcome: str) -> bool:
    return outcome in (NOT_SCOPED, SATISFIED)
