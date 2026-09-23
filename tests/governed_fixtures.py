"""Test-only helpers for governed objects (X1D-AIV2-GOV2-C1).

Why this exists
---------------
Before GOV2-C1, creating a relationship or a safety rule made it REVIEWED and
immediately effective, so a test that wanted to exercise *downstream* behaviour
-- pattern->formula retrieval, safety screening, blocking -- could just create
one. That shortcut was the defect: authority came from a literal in the source.

Creation now yields DRAFT, and the only route to REVIEWED is a verified human
clinical review attestation from xerbs-core, which this service cannot produce
and AI-GOV2 has not yet wired up. Tests that are really about retrieval or
screening still need an object in the reviewed state, so they use the helper
below.

What this is NOT
----------------
It is not a bypass, and it adds no capability to the application. It writes
rows directly with a test session, exactly as a test fixture may; no
application code path can reach it, and importing it from ``app/`` would be a
mistake worth failing a review over. The attestation ids it writes are
unmistakably synthetic.

Think of it as standing in for the future AI-GOV2 integration: "assume core
has already attested this object, now assert what retrieval does".
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db.models import ClinicalRelationship, GovernedObjectSource, SafetyRule
from app.db.session import get_session_factory
from app.services.governance import lifecycle

#: Prefix that makes every fixture-written attestation identifiable at a glance
#: and greppable in any database it ever reaches.
SYNTHETIC_ATTESTATION_PREFIX = "att-SYNTHETIC-TEST-"


def _synthetic_attestation_id() -> str:
    return SYNTHETIC_ATTESTATION_PREFIX + uuid.uuid4().hex[:12]


def simulate_core_attestation_for_test(object_type: str, object_id: str,
                                       *, ensure_reviewed_evidence: bool = True):
    """Put one governed object into the state a verified core attestation
    would leave it in.

    ``ensure_reviewed_evidence`` also promotes the object's attached Sources to
    REVIEWED, because eligibility requires at least one -- the same bar
    clinical entities have had since Phase 12C-2D3.
    """
    Session = get_session_factory()
    with Session.begin() as s:
        model = {"CLINICAL_RELATIONSHIP": ClinicalRelationship,
                 "SAFETY_RULE": SafetyRule}[object_type]
        row = s.get(model, object_id)
        assert row is not None, "%s %s does not exist" % (object_type, object_id)

        if ensure_reviewed_evidence:
            from app.db.models import SourceRegistry
            source_ids = list(s.scalars(select(GovernedObjectSource.source_id).where(
                GovernedObjectSource.object_type == object_type,
                GovernedObjectSource.object_id == object_id)).all())
            for sid in source_ids:
                src = s.get(SourceRegistry, sid)
                if src is not None:
                    src.review_status = "REVIEWED"

        row.review_status = lifecycle.REVIEWED
        row.governance_provenance = lifecycle.ATTESTED
        row.review_attestation_id = _synthetic_attestation_id()
        return row.review_attestation_id
