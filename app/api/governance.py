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
