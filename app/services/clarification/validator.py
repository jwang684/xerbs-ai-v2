"""X1D-CLARIFY1: the boundary between what the model proposes and what a patient sees.

Why this exists
---------------
DiagnosticReasoningEngine asks five fixed questions -- duration, temperature,
appetite, stool, sleep -- chosen by keyword marker matching. That mechanism is
deterministic and stays. But a patient reporting "头痛、咳嗽、发热3天" may need
clarification that no five-field checklist can anticipate: whether the cough
produces sputum, what colour it is, whether there is aversion to cold. Those
questions depend on the complaint.

So the model is allowed to propose. It is not allowed to ask.

Everything it proposes passes through this module first, and this module is
deterministic: fixed limits, a fixed answer-type allowlist, a fixed field
grammar, and a fixed list of things a clarification question may never request.
A proposal is data to be checked, never an instruction to be followed -- which
matters because the proposals are derived from patient-supplied text, and that
text can contain anything at all.

What it deliberately does not do
--------------------------------
No second model call to judge the first. A validator that needs an LLM to
decide whether output is safe has not removed the problem, only moved it.

No authority. A proposal cannot name a formula, set a recommendation state, a
safety verdict or purchase eligibility; fields that would touch those are
rejected outright rather than sanitised, because a question that wants to write
a safety verdict is not a question.

Failure direction
-----------------
Rejection is always safe: the deterministic questions still work, and the
governed result is unaffected. So every check drops the offending proposal and
continues rather than raising -- a malformed question must never cost a patient
a valid diagnosis.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# At most this many adaptive questions reach a patient in one turn. The
# deterministic engine's own questions are counted separately and take
# precedence; the combined visible total is bounded by the caller.
MAX_PROPOSALS_PER_TURN = 3

MAX_QUESTION_CHARS = 120
MAX_FIELD_CHARS = 40

# Known controls only. The frontend maps this enum to fixed widgets; it never
# renders model-supplied markup and never builds a control from arbitrary text.
ANSWER_TYPES = ("yes_no", "single_choice", "number", "short_text")
DEFAULT_ANSWER_TYPE = "short_text"

PRIORITIES = ("high", "medium", "low")
DEFAULT_PRIORITY = "medium"

MAX_CHOICES = 6
MAX_CHOICE_CHARS = 24

# A field identifier is an internal key, not free text: lower-case ASCII words
# joined by underscores. This excludes paths, HTML, script, SQL, JSON pointers
# and prompt fragments by construction rather than by blocklist.
FIELD_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,38}[a-z0-9]$")

# Fields that would carry clinical or commercial authority. Rejected rather
# than sanitised: a question asking to set these is not a clarification.
AUTHORITY_FIELDS = frozenset({
    "formula_id", "formula", "formula_candidates", "recommendation_state",
    "safety_verdict", "safety", "consumer_purchasable", "purchasable",
    "eligibility", "trust_score", "trustscore", "approved", "approval",
    "corpus_status", "reviewed", "price", "payment", "order",
})

# Things a clarification question may never ask a patient for. Matched against
# the question text, case-folded.
PROHIBITED_SUBSTRINGS = (
    "密码", "口令", "password", "passwd",
    "账号", "账户", "account number", "credential", "登录凭据",
    "api key", "api_key", "token", "密钥", "secret",
    "信用卡", "银行卡", "credit card", "bank", "cvv", "支付", "payment",
    "身份证", "社保", "ssn", "passport", "护照", "证件号",
    "system prompt", "系统提示", "提示词", "prompt",
    "忽略", "ignore previous", "ignore the above", "disregard",
    "剂量", "dosage", "自行服用", "self-prescribe",
    "购买", "下单", "purchase", "checkout",
    "内部", "internal", "数据库", "database",
)


# ======================================================================
# X1D-CLARIFY3: why a proposal was refused
# ======================================================================
#
# CLARIFY2 staging produced a sparse complaint where the model proposed
# questions and every one of them was rejected. The turn recovered -- the
# coverage floor supplied two 十问歌 questions -- but nothing recorded which
# rule had fired, so the cause could not be established from the record.
#
# These codes are that record. They are a *report* of the decision the rules
# below already make, not a new set of rules: every code corresponds to a
# branch that existed before this phase, in the order it already executed.
#
# One rejection yields exactly one primary reason: the first rule to fire.
# Later rules are not evaluated, exactly as before, so a code names the rule
# that actually stopped the proposal rather than every rule it might also have
# failed.

REASON_NOT_AN_OBJECT = "NOT_AN_OBJECT"
REASON_FIELD_MISSING = "FIELD_MISSING"
REASON_FIELD_TOO_LONG = "FIELD_TOO_LONG"
REASON_FIELD_PATTERN = "FIELD_PATTERN"
REASON_FIELD_AUTHORITY = "FIELD_AUTHORITY"
REASON_FIELD_ALREADY_KNOWN = "FIELD_ALREADY_KNOWN"
REASON_FIELD_DUPLICATE = "FIELD_DUPLICATE"
REASON_QUESTION_MISSING = "QUESTION_MISSING"
REASON_QUESTION_TOO_LONG = "QUESTION_TOO_LONG"
REASON_QUESTION_PROHIBITED = "QUESTION_PROHIBITED"

REJECTION_REASONS = (
    REASON_NOT_AN_OBJECT,
    REASON_FIELD_MISSING,
    REASON_FIELD_TOO_LONG,
    REASON_FIELD_PATTERN,
    REASON_FIELD_AUTHORITY,
    REASON_FIELD_ALREADY_KNOWN,
    REASON_FIELD_DUPLICATE,
    REASON_QUESTION_MISSING,
    REASON_QUESTION_TOO_LONG,
    REASON_QUESTION_PROHIBITED,
)

# Adjustments the validator makes to a proposal it nonetheless accepts. These
# are NOT rejections and never were: an unusable control is downgraded to the
# least powerful one rather than costing the patient a useful question.
# Recorded because "accepted after being changed" and "accepted as proposed"
# are different facts, and only one of them suggests a contract problem.
NORMALISATION_ANSWER_TYPE_DEFAULTED = "ANSWER_TYPE_DEFAULTED"
NORMALISATION_PRIORITY_DEFAULTED = "PRIORITY_DEFAULTED"
NORMALISATION_CHOICES_INSUFFICIENT = "CHOICES_INSUFFICIENT_DOWNGRADED"
NORMALISATION_CHOICES_DROPPED = "CHOICES_DROPPED_NOT_CHOICE_TYPE"
NORMALISATION_CHOICES_TRUNCATED = "CHOICES_TRUNCATED"

NORMALISATION_CODES = (
    NORMALISATION_ANSWER_TYPE_DEFAULTED,
    NORMALISATION_PRIORITY_DEFAULTED,
    NORMALISATION_CHOICES_INSUFFICIENT,
    NORMALISATION_CHOICES_DROPPED,
    NORMALISATION_CHOICES_TRUNCATED,
)

# How a field identifier may be recorded.
#
# A field that passes FIELD_PATTERN is by construction [a-z][a-z0-9_]{1,38}
# [a-z0-9] -- a machine identifier, not prose, so it is legible in a log and
# safe there. Anything that failed the pattern is arbitrary model output
# derived from patient language and is hashed instead, on exactly the
# KNOWLEDGE1B principle: keep the frequency signal, store none of the content.
FIELD_FORM_ASCII_SNAKE = "ASCII_SNAKE"
FIELD_FORM_ASCII_OTHER = "ASCII_OTHER"
FIELD_FORM_NON_ASCII = "NON_ASCII"
FIELD_FORM_EMPTY = "EMPTY"
FIELD_FORM_NOT_A_STRING = "NOT_A_STRING"


def classify_field(raw: Any) -> dict:
    """Describe a field identifier in a form that is safe to record.

    Returns the identifier itself only when it is a valid snake_case key.
    Anything else is reported by shape and digest, never by value.
    """
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        return {"field_form": FIELD_FORM_NOT_A_STRING, "field": None}

    text = raw.strip()
    if not text:
        return {"field_form": FIELD_FORM_EMPTY, "field": None}

    normalised = text.lower().replace("-", "_").replace(" ", "_")
    if FIELD_PATTERN.match(normalised) and len(normalised) <= MAX_FIELD_CHARS:
        return {"field_form": FIELD_FORM_ASCII_SNAKE, "field": normalised}

    form = (FIELD_FORM_ASCII_OTHER if text.isascii()
            else FIELD_FORM_NON_ASCII)
    return {
        "field_form": form,
        "field": None,
        "field_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "field_length": len(text),
    }


@dataclass(frozen=True)
class ProposalOutcome:
    """What the validator decided about one proposal, and why.

    ``question`` is the accepted question or None; the accept/reject decision
    is exactly what it was before CLARIFY3, and the rest is description.
    """

    index: int
    accepted: bool
    reason_code: str | None = None
    normalisations: tuple[str, ...] = ()
    field_form: str = FIELD_FORM_NOT_A_STRING
    field: str | None = None
    field_hash: str | None = None
    answer_type: str | None = None
    question_chars: int = 0
    question: "ClarificationQuestion | None" = None

    def as_record(self) -> dict:
        """The safe, machine-readable record of this decision.

        An explicit allowlist. The question text is never included in any
        form: the reason code, the control type and the length are enough to
        explain a rejection, and none of them carries what the patient said.
        """
        record = {
            "index": self.index,
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "field_form": self.field_form,
            "answer_type": self.answer_type,
            "question_chars": self.question_chars,
        }
        if self.field is not None:
            record["field"] = self.field
        if self.field_hash is not None:
            record["field_hash"] = self.field_hash
        if self.normalisations:
            record["normalisations"] = list(self.normalisations)
        return record


@dataclass(frozen=True)
class ClarificationQuestion:
    """A question that has passed validation and may be shown to a patient."""

    field: str
    question: str
    answer_type: str
    priority: str
    choices: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        payload = {
            "field": self.field,
            "question": self.question,
            "answer_type": self.answer_type,
            "priority": self.priority,
            "source": "xerbs-ai-v2-adaptive",
        }
        if self.choices:
            payload["choices"] = list(self.choices)
        return payload


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalise_field(raw: Any) -> str | None:
    field = _text(raw).lower().replace("-", "_").replace(" ", "_")
    if not field or len(field) > MAX_FIELD_CHARS:
        return None
    if not FIELD_PATTERN.match(field):
        return None
    if field in AUTHORITY_FIELDS:
        return None
    return field


def _normalise_choices(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    out: list[str] = []
    for item in raw[:MAX_CHOICES]:
        choice = _text(item)
        if choice and len(choice) <= MAX_CHOICE_CHARS:
            out.append(choice)
    return tuple(dict.fromkeys(out))


def _is_prohibited(question: str) -> bool:
    lowered = question.lower()
    return any(bad in lowered for bad in PROHIBITED_SUBSTRINGS)


def _field_rejection(raw_field: Any) -> str | None:
    """Which field rule rejects this identifier, in the order they execute."""
    text = _text(raw_field)
    if not text:
        return REASON_FIELD_MISSING
    normalised = text.lower().replace("-", "_").replace(" ", "_")
    if len(normalised) > MAX_FIELD_CHARS:
        return REASON_FIELD_TOO_LONG
    if not FIELD_PATTERN.match(normalised):
        return REASON_FIELD_PATTERN
    if normalised in AUTHORITY_FIELDS:
        return REASON_FIELD_AUTHORITY
    return None


def validate_proposal_detailed(
    raw: Any,
    known_fields: Iterable[str] = (),
    *,
    index: int = 0,
    already_known: Iterable[str] | None = None,
) -> ProposalOutcome:
    """Validate one proposal and report the decision.

    ``known_fields`` is the set that blocks a proposal, exactly as before.
    ``already_known`` is the subset of it supplied by the caller rather than
    accumulated from earlier proposals in this batch, and exists only so a
    field the deterministic engine already asks about can be told apart from a
    field this same batch already used. It changes nothing about the decision.

    Defaulting it to None rather than the empty set matters: called on its own
    there is no batch, so every blocking field came from the caller, and
    reporting those as in-batch duplicates would be simply wrong.
    """
    if not isinstance(raw, dict):
        return ProposalOutcome(index=index, accepted=False,
                               reason_code=REASON_NOT_AN_OBJECT)

    described = classify_field(raw.get("field"))
    question_text = _text(raw.get("question"))
    shape = {
        "field_form": described["field_form"],
        "field": described.get("field"),
        "field_hash": described.get("field_hash"),
        "question_chars": len(question_text),
    }

    reason = _field_rejection(raw.get("field"))
    if reason is not None:
        return ProposalOutcome(index=index, accepted=False,
                               reason_code=reason, **shape)

    field_value = _normalise_field(raw.get("field"))
    blocked = {str(f).lower() for f in known_fields}
    if field_value in blocked:
        prior = {str(f).lower() for f in
                 (known_fields if already_known is None else already_known)}
        return ProposalOutcome(
            index=index, accepted=False,
            reason_code=(REASON_FIELD_ALREADY_KNOWN if field_value in prior
                         else REASON_FIELD_DUPLICATE),
            **shape)

    if not question_text:
        return ProposalOutcome(index=index, accepted=False,
                               reason_code=REASON_QUESTION_MISSING, **shape)
    if len(question_text) > MAX_QUESTION_CHARS:
        return ProposalOutcome(index=index, accepted=False,
                               reason_code=REASON_QUESTION_TOO_LONG, **shape)
    if _is_prohibited(question_text):
        return ProposalOutcome(index=index, accepted=False,
                               reason_code=REASON_QUESTION_PROHIBITED, **shape)

    applied: list[str] = []

    answer_type = _text(raw.get("answer_type")).lower()
    if answer_type not in ANSWER_TYPES:
        # Unknown control: fall back to the least powerful one rather than
        # dropping a possibly useful question, and never invent a widget.
        answer_type = DEFAULT_ANSWER_TYPE
        applied.append(NORMALISATION_ANSWER_TYPE_DEFAULTED)

    priority = _text(raw.get("priority")).lower()
    if priority not in PRIORITIES:
        priority = DEFAULT_PRIORITY
        applied.append(NORMALISATION_PRIORITY_DEFAULTED)

    offered = raw.get("choices")
    choices = _normalise_choices(offered)
    if isinstance(offered, (list, tuple)) and len(offered) > len(choices):
        applied.append(NORMALISATION_CHOICES_TRUNCATED)
    if answer_type == "single_choice" and len(choices) < 2:
        # A choice control with nothing to choose between is not usable.
        answer_type = DEFAULT_ANSWER_TYPE
        choices = ()
        applied.append(NORMALISATION_CHOICES_INSUFFICIENT)
    if answer_type != "single_choice":
        if choices:
            applied.append(NORMALISATION_CHOICES_DROPPED)
        choices = ()

    shape["answer_type"] = answer_type
    return ProposalOutcome(
        index=index, accepted=True, normalisations=tuple(applied),
        question=ClarificationQuestion(
            field=field_value, question=question_text,
            answer_type=answer_type, priority=priority, choices=choices),
        **shape)


def validate_proposal(raw: Any, known_fields: Iterable[str] = ()) -> ClarificationQuestion | None:
    """Validate one proposal. Returns None for anything that does not pass.

    Expressed on top of validate_proposal_detailed so there is one decision
    procedure rather than two that could drift apart.
    """
    return validate_proposal_detailed(raw, known_fields).question


def validate_proposals(
    raw: Any,
    known_fields: Iterable[str] = (),
    limit: int = MAX_PROPOSALS_PER_TURN,
) -> list[ClarificationQuestion]:
    """Validate a batch, deduplicate by field, order by priority, and cap it.

    Ordering is deterministic: the model's stated priority is honoured only
    after being normalised into a three-value enum, and ties keep the model's
    original order so the result is reproducible.
    """
    if not isinstance(raw, (list, tuple)):
        return []

    return [outcome.question
            for outcome in validate_proposals_detailed(
                raw, known_fields=known_fields, limit=limit)
            if outcome.accepted]


def validate_proposals_detailed(
    raw: Any,
    known_fields: Iterable[str] = (),
    limit: int = MAX_PROPOSALS_PER_TURN,
) -> list[ProposalOutcome]:
    """Validate a batch and report every decision, accepted and refused.

    The accepted outcomes come back in the same order validate_proposals
    produces, and the refused ones follow in proposal order. A proposal
    dropped by the ``limit`` is reported as accepted-but-not-returned by being
    absent from the accepted prefix, which keeps this function a description
    of the existing behaviour rather than a second policy.
    """
    if not isinstance(raw, (list, tuple)):
        return []

    supplied = {str(f).lower() for f in known_fields}
    seen: set[str] = set(supplied)
    outcomes: list[ProposalOutcome] = []
    for index, item in enumerate(raw):
        outcome = validate_proposal_detailed(
            item, known_fields=seen, index=index, already_known=supplied)
        if outcome.accepted and outcome.question is not None:
            seen.add(outcome.question.field)
        outcomes.append(outcome)

    accepted = [o for o in outcomes if o.accepted]
    rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(
        enumerate(accepted),
        key=lambda pair: (rank.get(pair[1].question.priority, 1), pair[0]),
    )
    kept = [o for _, o in ordered][:max(0, limit)]
    return kept + [o for o in outcomes if not o.accepted]


def summarise_outcomes(outcomes: Sequence[ProposalOutcome]) -> dict:
    """Counts only. Safe to put in a flag or a log line."""
    reasons: dict[str, int] = {}
    normalisations: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.reason_code:
            reasons[outcome.reason_code] = reasons.get(outcome.reason_code, 0) + 1
        for code in outcome.normalisations:
            normalisations[code] = normalisations.get(code, 0) + 1
    return {
        "proposed": len(outcomes),
        "accepted": sum(1 for o in outcomes if o.accepted),
        "rejected": sum(1 for o in outcomes if not o.accepted),
        "reason_counts": dict(sorted(reasons.items())),
        "normalisation_counts": dict(sorted(normalisations.items())),
    }


def describe_policy() -> dict:
    """Operational description of the limits, for logs and tests."""
    return {
        "max_proposals_per_turn": MAX_PROPOSALS_PER_TURN,
        "max_question_chars": MAX_QUESTION_CHARS,
        "answer_types": list(ANSWER_TYPES),
        "priorities": list(PRIORITIES),
        "max_field_chars": MAX_FIELD_CHARS,
        "field_pattern": FIELD_PATTERN.pattern,
        "rejection_reasons": list(REJECTION_REASONS),
        "normalisation_codes": list(NORMALISATION_CODES),
    }
