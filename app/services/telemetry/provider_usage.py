"""X1D-TELEMETRY1: record what one real LLM call cost, and nothing else.

Why this is fail-open while everything around it fails closed
-------------------------------------------------------------
The clinical pipeline refuses rather than guesses: a provider failure becomes
AI_SERVICE_UNAVAILABLE, never a fabricated recommendation. Telemetry is the
opposite kind of thing. If the token counts cannot be written, the patient's
diagnosis is still correct and still governed -- turning that into a failure
would trade a real clinical result for a missing accounting row. So every path
here swallows its own exceptions and logs.

The one rule that must never bend: nothing recorded here may flow back into
pattern hypotheses, formula selection, corpus resolution, safety, eligibility
or recommendation state. This module is write-only with respect to the clinical
path and exposes no reader that the pipeline could consult.

Where it is stored, and why no migration was needed
---------------------------------------------------
audit_event already exists as a generic event log with a free-form JSON
payload. Telemetry goes there as event_type=LLM_PROVIDER_USAGE.

entity_id and source_id are left NULL on purpose. entity_id is the clinical
entity namespace, and that column's own comment records that Source identity
was given a separate column rather than sharing it. Putting generation ids
there would repeat exactly the mistake that comment documents, so the
generation and correlation ids live inside payload instead.

The cost of that choice is honest: there is no index on generation_id, so
lookups are JSON queries. For "persist now, present later" that is acceptable;
a dedicated column is a migration to propose separately if telemetry is ever
queried at volume.

What is never stored
--------------------
No API key, no Authorization header, no raw request or response body, no
prompt text, no patient identifiers, no JWT, no service token, no private
hostname. Token counts, latency, model name and a cost estimate are enough to
answer every operational question this is for, and none of them is sensitive.
"""

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.db.models import AuditEvent
from app.db.session import get_session_factory
from app.services.llm.pricing import (
    PRICING_SOURCE,
    PRICING_VERSION,
    estimate_cost_usd,
)
from app.services.llm.provider import ProviderResult

logger = logging.getLogger(__name__)

EVENT_TYPE = "LLM_PROVIDER_USAGE"
ACTOR = "xerbs-ai-v2"


def build_usage_payload(
    *,
    generation_id: str,
    correlation_id: str | None,
    result: ProviderResult,
    generation_latency_ms: float | None = None,
) -> dict:
    """Assemble the telemetry record. Pure -- no I/O, so it is easy to test.

    provider_latency_ms and generation_latency_ms are deliberately separate
    fields. The first is the external LLM call alone; the second is the whole
    governed generation including corpus resolution and persistence. Collapsing
    them would make the provider look slower than it is and make a regression in
    either one invisible.
    """
    usage = result.usage
    prompt = usage.prompt_tokens if usage else None
    completion = usage.completion_tokens if usage else None
    total = usage.total_tokens if usage else None
    cached = usage.cached_input_tokens if usage else None
    reasoning = usage.reasoning_tokens if usage else None

    cost = estimate_cost_usd(
        model=result.model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_input_tokens=cached,
    )

    return {
        "generation_id": generation_id,
        "correlation_id": correlation_id,
        "provider": result.provider,
        "model": result.model,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_input_tokens": cached,
        "reasoning_tokens": reasoning,
        "provider_latency_ms": (
            round(result.provider_latency_ms, 3)
            if result.provider_latency_ms is not None else None),
        "generation_latency_ms": (
            round(generation_latency_ms, 3)
            if generation_latency_ms is not None else None),
        "estimated_cost_usd": cost,
        "pricing_version": PRICING_VERSION if cost is not None else None,
        "pricing_source": PRICING_SOURCE if cost is not None else None,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }


def record_provider_usage(
    *,
    generation_id: str,
    correlation_id: str | None,
    result: ProviderResult,
    generation_latency_ms: float | None = None,
) -> bool:
    """Persist one usage record. Returns whether it was written.

    Never raises. The caller is on the clinical path and must not be able to
    fail because accounting did.
    """
    try:
        payload = build_usage_payload(
            generation_id=generation_id,
            correlation_id=correlation_id,
            result=result,
            generation_latency_ms=generation_latency_ms,
        )
        session_factory = get_session_factory()
        with session_factory.begin() as session:
            session.add(AuditEvent(
                event_id=f"evt-{uuid4().hex[:16]}",
                event_type=EVENT_TYPE,
                entity_id=None,      # clinical namespace: deliberately unused
                source_id=None,      # Source namespace: deliberately unused
                actor_id=ACTOR,
                payload=payload,
            ))
        # Also emit it, so the numbers are visible operationally before any
        # admin surface exists. Safe by construction: the payload is an
        # explicit allowlist of counts, latency, model and cost -- no key, no
        # prompt, no patient material (see the telemetry tests).
        logger.info("LLM_PROVIDER_USAGE %s", json.dumps(payload, ensure_ascii=False))
        return True
    except Exception as exc:  # noqa: BLE001 - telemetry must not break clinical work
        logger.warning(
            "provider usage telemetry not recorded for generation %s (%s: %s)",
            generation_id, type(exc).__name__, exc)
        return False
