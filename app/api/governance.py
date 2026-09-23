"""Attested human review endpoint (X1D-AIV2-ATTEST1).

The only route in this service that can move a governed clinical object to
REVIEWED. It does so by verifying a human decision that already happened in
xerbs-core, never by being told that one did.

What the request may say
------------------------
Which object, which attestation, which version it expects, and the usual
correlation/idempotency context. That is all. It may **not** carry
``reviewer_id``, ``reviewer_role``, ``reviewed_by`` or ``approved_by``: those
would be self-assertions, and self-assertion is the failure this whole
programme exists to undo. Reviewer identity comes from core's record, fetched
over an authenticated internal call.

Service authentication is not human approval. A valid credential proves the
caller is xerbs-core; the attestation lookup is what proves a person decided.
"""

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.services.governance.attested_review import (OBJECT_TYPES,
                                                     AttestedReviewError,
                                                     attested_review_service)

router = APIRouter(prefix="/api/v1/governance", tags=["governance"])


class AttestedReviewRequest(BaseModel):
    """Deliberately minimal. Every field here is a reference, not a claim."""

    attestation_id: str = Field(min_length=1, max_length=64)
    expected_version: int | None = None


#: Refusal reasons mapped to status codes. Distinct codes matter: "this
#: attestation does not exist" and "it exists and was rejected" are different
#: facts, and an operator reading a log should not have to guess which.
_STATUS = {
    "ATTESTATION_NOT_FOUND": 422,
    "ATTESTATION_RESPONSE_MALFORMED": 502,
    "ATTESTATION_VERIFICATION_UNAVAILABLE": 503,
    "ATTESTATION_VERIFICATION_UNAUTHORIZED": 502,
    "ATTESTATION_REVOKED": 409,
    "ATTESTATION_SUPERSEDED": 409,
    "ATTESTATION_NOT_ACTIVE": 409,
    "ALREADY_APPROVED_BY_DIFFERENT_ATTESTATION": 409,
    "VERSION_CONFLICT": 409,
    "OBJECT_NOT_FOUND": 404,
    "OBJECT_NOT_IN_REVIEW": 409,
    "EVIDENCE_REQUIRED": 409,
    "SEMANTIC_IDENTITY_MISSING": 409,
    "INVALID_TRANSITION": 409,
    "UNKNOWN_OBJECT_TYPE": 422,
    "LOCAL_APPROVAL_NOT_FOUND": 404,
    "ATTESTATION_STILL_ACTIVE": 409,
    "RE_APPROVAL_AFTER_REVOCATION_UNSUPPORTED": 409,
    "SUBJECT_MISMATCH": 422,
}


class SubmitForReviewRequest(BaseModel):
    """Who is submitting, as a machine principal. Confers nothing."""

    submitted_by: str = Field(min_length=1, max_length=128)


class ReconcileRequest(BaseModel):
    """Bounded, operator-initiated reconciliation. Names no decision."""

    limit: int | None = Field(default=None, ge=1, le=1000)


@router.get("/pending-review")
def pending_review_queue(
    object_type: str | None = None,
    limit: int = 200,
):
    """Governed objects awaiting a human decision. Read-only.

    X1D-CORE-ATTEST-B1. xerbs-core's admin console reads this to build a
    review worklist. It is served over the existing service credential and is
    never reachable from a browser, because the browser has no such token --
    core proxies it behind its own authenticated admin session.

    Objects that cannot be attested at all are returned WITH their reason
    rather than filtered out. A queue that silently hides the legacy rows
    would look complete while being partial.
    """
    try:
        return attested_review_service.pending_queue(
            object_types=[object_type] if object_type else None,
            limit=max(1, min(int(limit), 1000)))
    except AttestedReviewError as exc:
        raise HTTPException(_STATUS.get(exc.code, 422),
                            detail={"code": exc.code, "message": exc.message}) from exc


@router.get("/pending-review/{object_type}/{object_id}")
def pending_review_detail(object_type: str, object_id: str):
    """Full evidence and provenance for one object. Read-only.

    This is what makes review review: content, semantic identity, version,
    both hashes, every attached source with its own review state, the author /
    submitter / last-material-editor subjects with whether each is HUMAN,
    MACHINE or UNKNOWN, the full governance history from both event streams,
    and an explicit list of what is missing.
    """
    try:
        return attested_review_service.binding_detail(
            object_type=object_type, object_id=object_id)
    except AttestedReviewError as exc:
        raise HTTPException(_STATUS.get(exc.code, 422),
                            detail={"code": exc.code, "message": exc.message}) from exc


@router.post("/reconcile")
def reconcile(request: ReconcileRequest,
              correlation_id: str | None = Header(default=None,
                                                  alias="X-Correlation-ID")):
    """Run the existing reconciliation once, on demand.

    X1D-CORE-ATTEST-B1 exposes the entry point that X1D-AIV2-GOVCLOSURE1
    implemented and left uncallable. No new algorithm: this calls
    ``reconcile_attested_approvals`` and returns exactly what it reports.

    Deliberately not scheduled. There is no cron, no background loop and no
    recurring automation in this repository, and adding one is a separate
    authorization. Idempotent, mints nothing, and fails closed per attestation
    when core cannot be reached -- an unreachable attestation is reported
    UNCHECKED, never assumed still-approved and never revoked on a guess.
    """
    result = attested_review_service.reconcile_attested_approvals(
        limit=request.limit)
    result["correlation_id"] = correlation_id
    return result


@router.post("/{object_type}/{object_id}/submit-review")
def submit_for_review(object_type: str, object_id: str,
                      request: SubmitForReviewRequest):
    """DRAFT -> IN_REVIEW.

    Relationships and safety rules are created DRAFT and had no route to
    IN_REVIEW at all, which made the attested chain unreachable. This is a
    locally-authorisable transition: submitting confers no authority, and
    lifecycle.next_state still refuses APPROVE from every state.
    """
    if object_type not in OBJECT_TYPES:
        raise HTTPException(422, detail="unknown object_type %r" % object_type)
    try:
        return attested_review_service.submit(
            object_type=object_type, object_id=object_id,
            submitted_by=request.submitted_by)
    except AttestedReviewError as exc:
        raise HTTPException(_STATUS.get(exc.code, 422),
                            detail={"code": exc.code, "message": exc.message}) from exc


@router.post("/{object_type}/{object_id}/attested-review")
def attested_review(
    object_type: str,
    object_id: str,
    request: AttestedReviewRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
):
    if object_type not in OBJECT_TYPES:
        raise HTTPException(422, detail="unknown object_type %r" % object_type)
    try:
        return attested_review_service.approve(
            object_type=object_type,
            object_id=object_id,
            attestation_id=request.attestation_id,
            expected_version=request.expected_version,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )
    except AttestedReviewError as exc:
        # Everything else is a binding mismatch: the attestation is real but
        # does not describe the object this service stores. 422, with the code.
        raise HTTPException(_STATUS.get(exc.code, 422),
                            detail={"code": exc.code, "message": exc.message}) from exc


class RevokeRequest(BaseModel):
    """Names an attestation. Nothing here is taken as proof of anything."""

    attestation_id: str = Field(min_length=1, max_length=64)


@router.post("/attested-review/revoke")
def revoke_attested_review(
    request: RevokeRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
):
    """Withdraw a verified approval after confirming it with xerbs-core.

    Core tells this service that a human decision was withdrawn; this service
    goes and checks. A caller holding the service credential can therefore ask
    for a re-check of any attestation, and cannot un-approve one that core
    still stands behind -- the endpoint refuses with ATTESTATION_STILL_ACTIVE.

    Not addressed by object: the attestation id is the thing being withdrawn,
    and the local approval event is what says which object it touched. Naming
    the object in the request would invite the two to disagree.
    """
    try:
        return attested_review_service.revoke(
            attestation_id=request.attestation_id,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key)
    except AttestedReviewError as exc:
        raise HTTPException(_STATUS.get(exc.code, 422),
                            detail={"code": exc.code, "message": exc.message}) from exc
