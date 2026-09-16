"""X1D-LEGACYDIAG4.6-R1: where the working differential stops being one.

The 4.6 staging acceptance could not tell three very different situations
apart, because all three look identical from outside:

  * the model emitted no working_differential at all;
  * it emitted one and validation rejected every hypothesis;
  * it emitted a good one and every rule passed silently.

validate_state returns ``(None, [])`` for the first and ``(state, [])`` for
the third, so the DIFFERENTIAL_* uncertainty flags -- which exist only to say
that a RULE FIRED -- are empty in both. This module says which one happened.

Strictly observational. It reads values the turn has already computed and
writes one log line. It returns None, is called for its side effect only, and
every caller wraps it so that a diagnostic can never affect a clinical turn.

Non-clinical by construction: counts, booleans, canonical domain keys, reason
codes, the budget, and the selected domain keys. No patient text, no raw model
output, no pattern names, no question wording, no credentials.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

EVENT = "DIFFERENTIAL_PIPELINE_DIAGNOSTIC"

# The boundary the diagnostic is trying to locate, most upstream first. The
# first one that fails is the answer.
STAGE_NO_EMISSION = "NO_MODEL_EMISSION"
STAGE_PARSE_REJECTED = "PARSE_OR_VALIDATION_REJECTED"
STAGE_NO_LIVE = "NO_LIVE_HYPOTHESIS"
STAGE_NO_REQUIRED = "NO_REQUIRED_DOMAINS"
STAGE_NO_CANDIDATES = "NO_GOVERNED_CANDIDATES"
STAGE_REACHED = "REACHED_PREFERENCE_SELECTION"


def _domains(values: Iterable[Any]) -> List[str]:
    return sorted({str(v) for v in (values or []) if v})


def first_failing_stage(emitted: bool, validated: bool, live: int,
                        required: int, governed: int) -> str:
    """The most upstream boundary that is empty. Pure function."""
    if not emitted:
        return STAGE_NO_EMISSION
    if not validated:
        return STAGE_PARSE_REJECTED
    if live <= 0:
        return STAGE_NO_LIVE
    if required <= 0:
        return STAGE_NO_REQUIRED
    if governed <= 0:
        return STAGE_NO_CANDIDATES
    return STAGE_REACHED


def emit(
    *,
    generation_id: str = "",
    correlation_id: str = "",
    raw_state: Any = None,
    validated: Any = None,
    notes: Optional[Iterable[str]] = None,
    required_domains: Optional[Iterable[str]] = None,
    governed_domains: Optional[Iterable[str]] = None,
    adaptive_total: int = 0,
    adaptive_budget: int = 0,
    selected_domains: Optional[Iterable[str]] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    suppressed_by_already: Optional[Iterable[str]] = None,
) -> None:
    """Record one turn's differential pipeline. Never raises."""
    try:
        emitted = isinstance(raw_state, dict) and bool(raw_state)
        raw_hypotheses = 0
        if emitted:
            raw_hypotheses = len([h for h in (raw_state.get("hypotheses") or [])
                                  if isinstance(h, dict)])

        hypotheses = list(getattr(validated, "hypotheses", None) or [])
        live = [h for h in hypotheses
                if getattr(h, "standing", None) in ("PRIMARY_WORKING",
                                                    "PLAUSIBLE")]
        discriminators = [d for h in hypotheses
                          for d in (getattr(h, "unresolved_discriminators", None)
                                    or [])]
        discriminating = [d for d in discriminators
                          if isinstance(d, dict) and d.get("discriminating")]

        required = _domains(required_domains)
        governed = _domains(governed_domains)
        selected = _domains(selected_domains)

        # X1D-LEGACYDIAG4.6-R2: surplus must be counted AFTER the adaptive
        # score floor, not before it.
        #
        # R1 counted candidates as they were built, and reported that the
        # preference layer "had a choice" on turns where every 4.6-added
        # candidate had already been dropped for scoring below the floor. A
        # pool of four that becomes three before ranking leaves nothing to
        # arbitrate, and the old field said the opposite.
        rows = [c for c in (candidates or []) if isinstance(c, dict)]
        eligible = [c for c in rows if c.get("above_floor")]
        surplus = max(0, len(eligible) - int(adaptive_budget or 0))
        if not rows:  # no per-candidate detail supplied; fall back to counts
            surplus = max(0, int(adaptive_total or 0) - int(adaptive_budget or 0))

        record: Dict[str, Any] = {
            "generation_id": generation_id,
            "correlation_id": correlation_id,
            "stage": first_failing_stage(
                emitted, validated is not None, len(live), len(required),
                len(governed)),
            "model_emitted": emitted,
            "raw_hypotheses": raw_hypotheses,
            "validated": validated is not None,
            "validated_hypotheses": len(hypotheses),
            "live_hypotheses": len(live),
            "discriminators": len(discriminators),
            "discriminating": len(discriminating),
            "reason_codes": sorted(set(notes or [])),
            "required_domains": required,
            "required_count": len(required),
            "governed_candidate_domains": governed,
            "governed_candidate_count": len(governed),
            "adaptive_candidates": int(adaptive_total or 0),
            "adaptive_budget": int(adaptive_budget or 0),
            "adaptive_surplus": surplus,
            "preference_had_a_choice": bool(surplus > 0 and required),
            # The whole adaptive pool, one row per candidate. Provenance,
            # canonical domain, tier, preference key, score and outcome --
            # no question text, no patient text, no model prose.
            "candidates": rows,
            "candidates_total": len(rows),
            "candidates_above_floor": len(eligible),
            "candidates_below_floor": sorted(
                str(c.get("domain")) for c in rows if not c.get("above_floor")),
            # Domains a governed candidate was NOT generated for because a
            # model proposal had already taken that domain.
            "governed_suppressed_by_model": _domains(suppressed_by_already),
            "selected_domains": selected,
            "selected_from_required": sorted(set(selected) & set(required)),
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("%s %s", EVENT,
                    json.dumps(record, ensure_ascii=False, sort_keys=True))
    except Exception:  # noqa: BLE001 - a diagnostic never breaks a turn
        logger.debug("%s emission failed", EVENT, exc_info=True)
