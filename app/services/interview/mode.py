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

  * the governed interview depth, which guarantees an exit. It is counted by
    core from its own trace rows -- same user, same case, excluding this
    submission's own idempotency key so a replay does not advance it -- and
    arrives through the typed intake. A case therefore always reaches full
    reasoning rather than interviewing forever.

    The browser also sends a turn_count on the form. It is not this value and
    must never be: handing the depth limit to the party being limited is not a
    limit. core derives the number and the browser cannot reach it.

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

# How many interview turns a case may have, in total.
#
# The unit is deliberate and worth stating once: interview_depth counts the
# governed turns this case has ALREADY COMPLETED, not including the submission
# being routed. So turn 1 arrives as depth 0 and turn 4 as depth 3, and
# "depth >= MAX_INTERVIEW_TURNS" reads exactly as "three interview turns have
# happened; that is enough".
#
#   turn 1  depth 0  ->  INTERVIEW
#   turn 2  depth 1  ->  INTERVIEW
#   turn 3  depth 2  ->  INTERVIEW
#   turn 4  depth 3  ->  FULL_REASONING / INTERVIEW_DEPTH_REACHED
#
# Mirrors core's clarification_policy.MAX_CLARIFICATION_TURNS: past that depth
# core shows no further adaptive questions anyway, so continuing to interview
# would spend a call producing questions nobody will be asked.
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
    interview_depth: int = 0,
    supports_interview: bool = True,
) -> Tuple[str, str]:
    """Return (mode, reason). Deterministic, and never consults the model.

    ``missing_information`` is the deterministic engine's own output, passed in
    rather than recomputed so there is one marker-matching implementation.

    ``interview_depth`` is the number of governed turns this case has already
    completed, excluding the submission being routed. See MAX_INTERVIEW_TURNS
    above for the turn-by-turn table.

    The depth check comes before the coverage check on purpose. Once the
    interview budget is spent, the answer is "stop asking and reason with what
    we have" regardless of how much is still unknown -- which is exactly the
    case coverage would otherwise keep sending back to interview mode forever.
    """
    if not supports_interview:
        return FULL_REASONING, REASON_PROVIDER_UNSUPPORTED

    try:
        depth = int(interview_depth or 0)
    except (TypeError, ValueError):
        depth = 0
    if depth >= MAX_INTERVIEW_TURNS:
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
        "depth_unit": "governed turns already completed, excluding this one",
        "reasons": [REASON_PROVIDER_UNSUPPORTED, REASON_DEPTH_REACHED,
                    REASON_MATERIALLY_MISSING, REASON_COVERAGE_SUFFICIENT],
    }
