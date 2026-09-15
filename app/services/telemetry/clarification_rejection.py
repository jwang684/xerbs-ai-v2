"""X1D-CLARIFY3: record why adaptive clarification proposals were refused.

What this is for
----------------
CLARIFY2 staging produced a sparse respiratory complaint where the model
proposed clarification questions and the validator rejected every one of them.
The turn recovered correctly -- the coverage floor supplied two 十问歌
questions -- but nothing in the record said which rule had fired, so the cause
could not be established afterwards. A validator that silently discards model
output is a validator nobody can debug or tune.

This records the decision. It adds no rule and reverses none: the accept and
reject outcomes are exactly what they were, and this is their description.

Privacy
-------
The question text is never recorded, in any form. A rejection is explained by
the reason code, the control type and the character count, and none of those
carries what the patient said.

Field identifiers are treated on the KNOWLEDGE1B principle. One that passed
FIELD_PATTERN is by construction a snake_case machine key of at most 40 ASCII
characters -- legible and safe. One that failed is arbitrary model output
derived from patient language, so it is recorded as a digest and a length
instead, which keeps a recurring problem visible as recurring without storing
its content.

Nothing else is recorded: no complaint, no observations, no answers, no
prompt, no system prompt, no model response body, no user id, no credential.

Failure direction
-----------------
Fail-open, like TELEMETRY1 and the KNOWLEDGE1B gap recorder. If this cannot be
written the patient's diagnosis is still correct and still governed, and
trading a real clinical result for an operational row would be the wrong way
round. Every path swallows its own exception and logs.

This module is write-only with respect to the clinical path. It exposes no
reader the pipeline could consult, and nothing recorded here may influence
pattern hypotheses, formula selection, corpus resolution, safety, eligibility
or recommendation state.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from app.db.models import AuditEvent
from app.db.session import get_session_factory
from app.services.clarification.validator import (
    ProposalOutcome,
    summarise_outcomes,
)

logger = logging.getLogger(__name__)

EVENT_TYPE = "CLARIFICATION_REJECTION"
ACTOR = "xerbs-ai-v2"

# A turn cannot produce many proposals, but the model supplies the list, so the
# record is bounded here rather than trusted to be small.
MAX_RECORDED_OUTCOMES = 12


def build_rejection_payload(
    *,
    generation_id: str,
    correlation_id: Optional[str],
    outcomes: Sequence[ProposalOutcome],
    malformed_count: int = 0,
) -> Dict[str, Any]:
    """Assemble the record. Pure, so the privacy properties are testable.

    ``malformed_count`` is the number of proposals the provider discarded
    before the validator saw them, because they were not JSON objects. Without
    it, "the model proposed nothing" and "the model proposed several things of
    the wrong shape" are the same observation.
    """
    summary = summarise_outcomes(outcomes)
    return {
        "generation_id": generation_id,
        "correlation_id": correlation_id,
        "proposed": summary["proposed"],
        "accepted": summary["accepted"],
        "rejected": summary["rejected"],
        "malformed_discarded_by_provider": int(malformed_count or 0),
        "reason_counts": summary["reason_counts"],
        "normalisation_counts": summary["normalisation_counts"],
        # Per-proposal detail, each an explicit allowlist built by
        # ProposalOutcome.as_record. No question text in any form.
        "outcomes": [o.as_record() for o in outcomes[:MAX_RECORDED_OUTCOMES]],
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }


def record_rejections(
    *,
    generation_id: str,
    correlation_id: Optional[str],
    outcomes: Sequence[ProposalOutcome],
    malformed_count: int = 0,
) -> bool:
    """Persist one record. Returns whether it was written. Never raises."""
    try:
        if not outcomes and not malformed_count:
            return False

        payload = build_rejection_payload(
            generation_id=generation_id,
            correlation_id=correlation_id,
            outcomes=outcomes,
            malformed_count=malformed_count,
        )
        session_factory = get_session_factory()
        with session_factory.begin() as session:
            session.add(AuditEvent(
                event_id=f"evt-{uuid4().hex[:16]}",
                event_type=EVENT_TYPE,
                entity_id=None,    # clinical namespace: deliberately unused
                source_id=None,    # Source namespace: deliberately unused
                actor_id=ACTOR,
                payload=payload,
            ))
        # Emitted as well, so the reasons are visible operationally before any
        # admin surface exists -- which is the whole point of the phase.
        logger.info("CLARIFICATION_REJECTION %s",
                    json.dumps(payload, ensure_ascii=False))
        return True
    except Exception as exc:  # noqa: BLE001 - telemetry must not break clinical work
        logger.warning(
            "clarification rejection telemetry not recorded for generation "
            "%s (%s: %s)", generation_id, type(exc).__name__, exc)
        return False


def outcome_flags(outcomes: Sequence[ProposalOutcome],
                  malformed_count: int = 0) -> List[str]:
    """Governance flags carrying the same counts.

    core already persists uncertainty_flags into its trace, so putting the
    counts here makes a turn explicable from core's own record without core
    changing. Bounded by construction: the counts are small integers and the
    reason codes are a fixed enum.
    """
    summary = summarise_outcomes(outcomes)
    flags = [
        "CLARIFICATION_MODEL_PROPOSED_%d" % summary["proposed"],
        "CLARIFICATION_VALIDATOR_ACCEPTED_%d" % summary["accepted"],
    ]
    if malformed_count:
        flags.append("CLARIFICATION_PROPOSALS_MALFORMED_%d" % malformed_count)
    for code, count in summary["reason_counts"].items():
        flags.append("CLARIFICATION_REJECTED_%s_%d" % (code, count))
    return flags
