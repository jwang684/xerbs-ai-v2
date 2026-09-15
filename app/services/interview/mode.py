"""X1D-LEGACYDIAG3.2: which kind of inference this turn needs.

Why the obvious rule is wrong
-----------------------------
The natural routing rule is "incomplete case -> interview mode". It cannot be
used, and the reason is worth stating before the code.

``ready_for_formula_retrieval`` requires a pattern assessment that matched the
reviewed corpus. CORPUS1 measured that corpus: it holds zero pattern entities.
So corpus_match is always False, readiness is always False, and by that
definition *every* case is currently incomplete. Routing on it would send all
traffic to interview mode, and the ClinicalReasoningEnvelope that LEGACYDIAG2
exists to retain would silently stop being produced -- a real loss of clinical
record, hidden behind a latency win.

What is used instead
--------------------
Two deterministic signals available *before* any provider call, so the routing
decision never depends on the model:

  * whether the deterministic engine's marker matching still reports
    HIGH-priority missing information, and whether CLARIFY2 coverage still has
    a material domain unknown -- both computed from accumulated patient text;

  * the turn number, which guarantees an exit. At the depth where core stops
    asking adaptive questions, this stops interviewing too and the full
    contract runs. A case therefore always ends with full reasoning rather
    than interviewing forever.

The model's own ``information_sufficient`` is advisory and is deliberately not
an input here. A model that decides when questioning ends decides when
questioning ends in its own favour, and there would be no boundary outside it.

What this module cannot do
--------------------------
Choose a formula, verify anything, grant eligibility, or make something
purchasable. It selects between two prompts. Everything downstream -- corpus
governance, SafetyEngine, the purchase chokepoint -- is untouched by the
choice and decides exactly what it decided before.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from app.services.clarification.coverage import assess_coverage

INTERVIEW = "INTERVIEW"
FULL_REASONING = "FULL_REASONING"

# The turn at which interviewing stops regardless of coverage. Mirrors core's
# clarification_policy.MAX_CLARIFICATION_TURNS: past that depth core shows no
# further adaptive questions, so continuing to interview would spend a call
# producing questions nobody will be asked.
MAX_INTERVIEW_TURNS = 3

# Reasons, for the record. A turn should never be routed without one.
REASON_PROVIDER_UNSUPPORTED = "PROVIDER_DOES_NOT_SUPPORT_INTERVIEW"
REASON_DEPTH_REACHED = "INTERVIEW_DEPTH_REACHED"
REASON_MATERIALLY_MISSING = "INFORMATION_MATERIALLY_MISSING"
REASON_COVERAGE_SUFFICIENT = "COVERAGE_SUFFICIENT"


def provider_supports_interview(provider: Any) -> bool:
    """Whether this provider implements the small contract.

    Checked rather than assumed: generate_interview is concrete on the base
    class and raises, so a provider that has not implemented it must not be
    routed to interview mode.
    """
    method = getattr(type(provider), "generate_interview", None)
    base = getattr(__import__("app.services.llm.provider", fromlist=["LLMProvider"]),
                   "LLMProvider").generate_interview
    return method is not None and method is not base


def decide_mode(
    *,
    accumulated_text: str,
    missing_information: Optional[Sequence[Any]] = None,
    turn_count: int = 1,
    supports_interview: bool = True,
) -> Tuple[str, str]:
    """Return (mode, reason). Deterministic, and never consults the model.

    ``missing_information`` is the deterministic engine's own output, passed in
    rather than recomputed so there is one marker-matching implementation.
    """
    if not supports_interview:
        return FULL_REASONING, REASON_PROVIDER_UNSUPPORTED

    try:
        turn = int(turn_count or 1)
    except (TypeError, ValueError):
        turn = 1
    if turn >= MAX_INTERVIEW_TURNS:
        return FULL_REASONING, REASON_DEPTH_REACHED

    has_high = any(
        str(getattr(item, "priority", "") or "").upper() == "HIGH"
        for item in (missing_information or [])
    )
    coverage = assess_coverage(accumulated_text or "")

    if has_high or coverage.unknown_material():
        return INTERVIEW, REASON_MATERIALLY_MISSING
    return FULL_REASONING, REASON_COVERAGE_SUFFICIENT


def describe_policy() -> dict:
    """Operational description. No clinical content, no configuration values."""
    return {
        "modes": [INTERVIEW, FULL_REASONING],
        "max_interview_turns": MAX_INTERVIEW_TURNS,
        "reasons": [REASON_PROVIDER_UNSUPPORTED, REASON_DEPTH_REACHED,
                    REASON_MATERIALLY_MISSING, REASON_COVERAGE_SUFFICIENT],
    }
