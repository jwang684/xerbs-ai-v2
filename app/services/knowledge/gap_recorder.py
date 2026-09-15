"""X1D-KNOWLEDGE1B: record which governed claims the local corpus could not answer.

Why measure before retrieving
-----------------------------
CORPUS1 established that representative cases stop at
PATTERN_NOT_VERIFIED_IN_REVIEWED_CORPUS because the corpus holds zero pattern
entities. The obvious next step is to go and fetch the missing knowledge. The
cheaper step first is to find out *which* knowledge is actually missing, and
how often, because that decides what is worth researching and in what order.

So this records the gap and nothing else. No retrieval, no evidence, no
verification, no new state a claim could occupy. Nothing here can influence
formula eligibility, safety, or purchase: it is write-only with respect to the
clinical path, in the same way TELEMETRY1 is.

Where it is stored, and why no migration
----------------------------------------
audit_event already holds arbitrary JSON payloads, and TELEMETRY1 already uses
it for operational records. Gaps go there as event_type=KNOWLEDGE_GAP, with
entity_id and source_id left NULL -- entity_id is the clinical entity
namespace, and a gap is by definition about an entity that does not exist, so
putting anything there would be worse than leaving it empty.

Privacy
-------
The subject of a gap is a clinical concept such as 风热犯表, which is the
abstract term KNOWLEDGE1A identified as safe to carry outside the patient
context. It is not patient text and not an identifier.

But the subject originates as model output derived from patient language, so
it is not trusted to be a concept just because it should be. Real pattern
names are short; anything long is more likely to be echoed patient text, and
is stored as a hash instead of plainly. That keeps the frequency count usable
-- the same unexpected string still aggregates -- without putting free text
into an operational log.

Nothing else about the patient is recorded: no complaint, no observations, no
answers, no user id, no case id, no credential.

Failure direction
-----------------
Fail-open, deliberately. A governed clinical result that is otherwise correct
must not be lost because an analytics row could not be written.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from sqlalchemy import select

from app.db.models import AuditEvent
from app.db.session import get_session_factory

logger = logging.getLogger(__name__)

EVENT_TYPE = "KNOWLEDGE_GAP"
ACTOR = "xerbs-ai-v2"

# Claim kinds this phase can observe. Deliberately one entry: the resolver
# currently sits in front of exactly one lookup, and naming claim kinds the
# code cannot yet produce would overstate what is measured.
CLAIM_PATTERN_ENTITY = "PATTERN_ENTITY"

# Real pattern names are a handful of characters. Longer strings are more
# likely to be echoed patient language than a clinical concept.
MAX_PLAIN_SUBJECT_CHARS = 24


def normalise_subject(raw: Any) -> Dict[str, Any]:
    """Return how the subject should be recorded: plainly, or hashed.

    Hashing rather than dropping keeps the frequency signal: the same
    unexpected string aggregates to the same digest, so a recurring gap is
    still visible as recurring without its content being stored.
    """
    text = str(raw or "").strip()
    if not text:
        return {"subject": None, "subject_form": "EMPTY"}
    if len(text) <= MAX_PLAIN_SUBJECT_CHARS:
        return {"subject": text, "subject_form": "CONCEPT"}
    return {
        "subject": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "subject_form": "HASHED_OVERLONG",
        "subject_length": len(text),
    }


def build_gap_payload(
    *,
    claim_type: str,
    subject: Any,
    entity_types: List[str],
    reviewed_only: bool,
) -> Dict[str, Any]:
    """Assemble the record. Pure, so the privacy properties are testable."""
    payload: Dict[str, Any] = {
        "claim_type": claim_type,
        "entity_types": sorted(str(t) for t in entity_types or []),
        "reviewed_only": bool(reviewed_only),
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(normalise_subject(subject))
    return payload


def record_gap(
    *,
    claim_type: str,
    subject: Any,
    entity_types: List[str],
    reviewed_only: bool = True,
) -> bool:
    """Persist one gap. Returns whether it was written. Never raises."""
    try:
        payload = build_gap_payload(
            claim_type=claim_type, subject=subject,
            entity_types=entity_types, reviewed_only=reviewed_only)
        session_factory = get_session_factory()
        with session_factory.begin() as session:
            session.add(AuditEvent(
                event_id=f"evt-{uuid4().hex[:16]}",
                event_type=EVENT_TYPE,
                entity_id=None,   # the entity does not exist; that is the point
                source_id=None,
                actor_id=ACTOR,
                payload=payload,
            ))
        logger.info("KNOWLEDGE_GAP %s subject_form=%s",
                    payload["claim_type"], payload["subject_form"])
        return True
    except Exception as exc:  # noqa: BLE001 - analytics must not break clinical work
        logger.warning("knowledge gap not recorded (%s: %s)",
                       type(exc).__name__, exc)
        return False


def summarise_gaps(limit: int = 20) -> Dict[str, Any]:
    """Baseline KPI: which claims the local corpus cannot answer, and how often.

    Deliberately does not report LOCAL_REVIEWED_HIT_RATE. A rate needs hits in
    the denominator, and recording every successful lookup would add rows to
    establish a number CORPUS1 already measured directly: with zero pattern
    entities the reviewed-pattern hit rate is zero. Hit recording belongs with
    the first phase that can actually move it.
    """
    try:
        session_factory = get_session_factory()
        with session_factory() as session:
            rows = list(session.scalars(
                select(AuditEvent).where(AuditEvent.event_type == EVENT_TYPE)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("gap summary unavailable (%s: %s)",
                       type(exc).__name__, exc)
        return {"available": False, "total_gaps": 0, "top_subjects": []}

    subjects = Counter()
    by_claim = Counter()
    by_form = Counter()
    for row in rows:
        payload = row.payload if isinstance(row.payload, dict) else {}
        by_claim[payload.get("claim_type")] += 1
        by_form[payload.get("subject_form")] += 1
        subject = payload.get("subject")
        if subject:
            subjects[subject] += 1

    return {
        "available": True,
        "total_gaps": len(rows),
        "distinct_subjects": len(subjects),
        "by_claim_type": dict(by_claim),
        "by_subject_form": dict(by_form),
        "top_subjects": subjects.most_common(limit),
    }
