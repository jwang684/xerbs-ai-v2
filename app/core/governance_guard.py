"""Governance mutation authorization (X1D-E2E1 Step 5).

The problem
-----------
Clinical review authority in this service is asserted by the caller. Both
``ReviewActionRequest.reviewer_role`` and ``SourceReviewActionRequest``
default to ``"CLINICAL_REVIEWER"``, and the enforcement sites only check that
the submitted string is one of two accepted values. So any caller could
become a clinical reviewer by saying it was one, and ``reviewer_id`` — the
value written into the audit trail — is likewise an arbitrary string. There
is no self-review guard either: the same identity may ingest, submit and
approve the same record.

That is not an authorization model. It is a field in a JSON body.

What this does
--------------
Service authentication proves the caller is xerbs-core; it does not make
xerbs-core a qualified clinical reviewer. Those are different claims, and the
second one cannot be established until this service has real human reviewer
identity.

So external governance mutation is refused by default. Corpus content is
instead established by the controlled bootstrap, which runs in-process
against the governance service, records the same audit events, and is not
reachable from the network.

Setting ``ALLOW_EXTERNAL_GOVERNANCE_MUTATION=true`` re-opens the endpoints for
a deliberately chosen environment. It is off by default, so a staging
deployment that simply does not set it is closed.

Read-only governance endpoints stay reachable to an authenticated service
caller: they expose review state and provenance, which is what xerbs-core
needs in order to explain why a formula was eligible.
"""

from __future__ import annotations

from fastapi import HTTPException

from app.core.config import get_settings


def require_governance_mutation_enabled() -> None:
    """Refuse corpus/governance mutation unless explicitly enabled.

    Fails closed: the default configuration rejects every mutation.
    """
    if not get_settings().allow_external_governance_mutation:
        raise HTTPException(
            status_code=403,
            detail=(
                "External clinical governance mutation is disabled. "
                "reviewer_role is self-asserted by the request body and cannot "
                "establish clinical review authority; use the controlled corpus "
                "bootstrap instead."
            ),
        )
