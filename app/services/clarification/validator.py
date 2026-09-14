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

import re
from dataclasses import dataclass
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


def validate_proposal(raw: Any, known_fields: Iterable[str] = ()) -> ClarificationQuestion | None:
    """Validate one proposal. Returns None for anything that does not pass."""
    if not isinstance(raw, dict):
        return None

    field = _normalise_field(raw.get("field"))
    if field is None or field in {str(f).lower() for f in known_fields}:
        return None

    question = _text(raw.get("question"))
    if not question or len(question) > MAX_QUESTION_CHARS:
        return None
    if _is_prohibited(question):
        return None

    answer_type = _text(raw.get("answer_type")).lower()
    if answer_type not in ANSWER_TYPES:
        # Unknown control: fall back to the least powerful one rather than
        # dropping a possibly useful question, and never invent a widget.
        answer_type = DEFAULT_ANSWER_TYPE

    priority = _text(raw.get("priority")).lower()
    if priority not in PRIORITIES:
        priority = DEFAULT_PRIORITY

    choices = _normalise_choices(raw.get("choices"))
    if answer_type == "single_choice" and len(choices) < 2:
        # A choice control with nothing to choose between is not usable.
        answer_type = DEFAULT_ANSWER_TYPE
        choices = ()
    if answer_type != "single_choice":
        choices = ()

    return ClarificationQuestion(
        field=field, question=question, answer_type=answer_type,
        priority=priority, choices=choices,
    )


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

    seen: set[str] = {str(f).lower() for f in known_fields}
    accepted: list[ClarificationQuestion] = []
    for index, item in enumerate(raw):
        question = validate_proposal(item, known_fields=seen)
        if question is None:
            continue
        seen.add(question.field)
        accepted.append(question)

    rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(
        enumerate(accepted),
        key=lambda pair: (rank.get(pair[1].priority, 1), pair[0]),
    )
    return [q for _, q in ordered][:max(0, limit)]


def describe_policy() -> dict:
    """Operational description of the limits, for logs and tests."""
    return {
        "max_proposals_per_turn": MAX_PROPOSALS_PER_TURN,
        "max_question_chars": MAX_QUESTION_CHARS,
        "answer_types": list(ANSWER_TYPES),
        "priorities": list(PRIORITIES),
    }
