"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P15: the one deterministic primary-hypothesis selector.

Why this exists. Retrieval used to pool the governed pattern of EVERY
hypothesis, so a lower-ranked 风寒束表 could fetch 麻黄汤加味 for a case whose
top diagnosis was wind-heat (P14: 18 of 71 Staging trace turns). The reasoning
contract does not say whether a secondary hypothesis is a concurrent pattern
or merely a differential, so only the top diagnosis may drive formula
retrieval -- and "top" must be provable, not assumed.

Primary is established only when every hypothesis is well formed (a name and a
numeric confidence in [0, 1]) and exactly one has the strictly highest
confidence. List position is NOT a fallback: a tie, a missing or invalid
confidence, or a malformed entry fails closed. Position is still reported, as
provenance.

Pure function; used by the reasoning engine, the interview engine and the
coverage harness alike, so there is one definition of "primary".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

# Stable, versioned identifier stamped on every pattern-derived candidate.
RETRIEVAL_POLICY = "PRIMARY_ANCHORED_FAIL_CLOSED_V1"

PRIMARY_SELECTED = "PRIMARY_SELECTED"
NO_PATTERN_HYPOTHESES = "NO_PATTERN_HYPOTHESES"
AMBIGUOUS_PRIMARY_HYPOTHESIS = "AMBIGUOUS_PRIMARY_HYPOTHESIS"
INVALID_PRIMARY_HYPOTHESIS = "INVALID_PRIMARY_HYPOTHESIS"

# Engineering/governance explanations, not clinical judgments.
PRIMARY_PATTERN_NOT_IN_GOVERNED_CORPUS = "PRIMARY_PATTERN_NOT_IN_GOVERNED_CORPUS"
SECONDARY_GOVERNED_MATCH_NOT_USED = "SECONDARY_GOVERNED_MATCH_NOT_USED"


@dataclass(frozen=True)
class PrimarySelection:
    outcome: str
    index: int | None = None          # 0-based list position of the primary
    name: str | None = None
    confidence: float | None = None

    @property
    def selected(self) -> bool:
        return self.outcome == PRIMARY_SELECTED

    @property
    def rank(self) -> int | None:
        """1-based list position, as provenance."""
        return None if self.index is None else self.index + 1


def valid_confidence(value: Any) -> float | None:
    """A real number in [0, 1], or None. Strings and booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if math.isnan(v) or math.isinf(v) or not 0.0 <= v <= 1.0:
        return None
    return v


def _as_dict(h):
    if isinstance(h, dict):
        return h
    if hasattr(h, "model_dump"):
        return h.model_dump()
    return None


def select_primary(hypotheses: Iterable[Any] | None) -> PrimarySelection:
    items = list(hypotheses or [])
    if not items:
        return PrimarySelection(NO_PATTERN_HYPOTHESES)
    parsed = []
    for h in items:
        d = _as_dict(h)
        if d is None:
            return PrimarySelection(INVALID_PRIMARY_HYPOTHESIS)
        name = str(d.get("name") or "").strip()
        conf = valid_confidence(d.get("confidence"))
        if not name or conf is None:
            return PrimarySelection(INVALID_PRIMARY_HYPOTHESIS)
        parsed.append((name, conf))
    top = max(c for _, c in parsed)
    winners = [i for i, (_, c) in enumerate(parsed) if c == top]
    if len(winners) != 1:
        return PrimarySelection(AMBIGUOUS_PRIMARY_HYPOTHESIS)
    i = winners[0]
    return PrimarySelection(PRIMARY_SELECTED, i, parsed[i][0], parsed[i][1])


def primary_governed_assessment(assessments):
    """The selected primary's assessment when it matched an eligible governed pattern, else None."""
    return next((a for a in assessments or []
                 if getattr(a, "is_primary", False) and a.corpus_match and a.pattern_id), None)


def derived_from(assessment) -> dict:
    """Server-side provenance for a candidate retrieved through this assessment."""
    return {
        "pattern_id": assessment.pattern_id,
        "pattern_name": assessment.governed_pattern_name,
        "hypothesis_rank": assessment.hypothesis_rank,
        "hypothesis_name": assessment.name,
        "model_confidence": assessment.model_confidence,
        "match_mechanism": assessment.match_mechanism,
    }
