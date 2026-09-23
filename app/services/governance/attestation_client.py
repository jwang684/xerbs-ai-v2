"""Authoritative attestation lookup against xerbs-core (X1D-AIV2-ATTEST1).

Why a lookup and not the pushed payload
---------------------------------------
xerbs-core sends ``attestation_id`` plus the binding it claims. Trusting that
payload would mean a leaked ``AI_V2_SERVICE_TOKEN`` is enough to approve any
clinical object: the holder just posts a plausible-looking body. Asking core
removes that. Minting an attestation needs an authenticated admin session with
``clinical:review`` going through a SECURITY-checked governance function; a
fabricated id simply fails to resolve.

Fail closed, in every direction
-------------------------------
Unconfigured, unreachable, slow, non-200, malformed, or missing a field: all of
them raise. None of them is ever read as approval. The cost of that choice is
that an approval cannot be recorded while core is down — which is the correct
trade, because the alternative is recording approvals nobody made.
"""

from __future__ import annotations

from typing import Any, Mapping

import httpx

from app.core.config import get_settings

#: Fields the verification path requires. A response missing any of them is
#: malformed and refused rather than partially trusted.
REQUIRED_FIELDS = (
    "attestation_id", "decision", "target_system", "target_environment",
    "governance_object_type", "semantic_object_id", "object_version",
    "content_hash", "reviewer_subject", "is_active", "is_revoked",
    "is_superseded",
)

LOOKUP_PATH = "/internal/governance/clinical-attestations/%s"


class AttestationLookupError(Exception):
    """Verification could not be completed. Never means "approved"."""

    def __init__(self, message: str, *, code: str):
        super().__init__(message)
        self.message = message
        self.code = code


class AttestationNotFound(AttestationLookupError):
    """core has no such attestation."""

    def __init__(self, attestation_id: str):
        super().__init__("attestation %s not found in xerbs-core" % attestation_id,
                         code="ATTESTATION_NOT_FOUND")


class CoreAttestationClient:
    """Read-only client for core's internal governance lookup."""

    def __init__(self, base_url: str | None = None, token: str | None = None,
                 timeout: float | None = None):
        s = get_settings()
        self._base_url = (base_url if base_url is not None else s.core_base_url)
        self._token = (token if token is not None else s.core_internal_service_token)
        self._timeout = timeout if timeout is not None else s.core_lookup_timeout_seconds

    def configured(self) -> bool:
        return bool((self._base_url or "").strip()) and bool((self._token or "").strip())

    def fetch(self, attestation_id: str, *, correlation_id: str | None = None) -> Mapping[str, Any]:
        """Return core's authoritative record, or raise.

        The credential is sent as a bearer token and never logged. The returned
        mapping is validated for shape only; whether it *matches the object*
        is decided by the caller, which is the half core cannot do.
        """
        if not self.configured():
            raise AttestationLookupError(
                "verification against xerbs-core is not configured; refusing to "
                "record a human approval that cannot be verified",
                code="ATTESTATION_VERIFICATION_UNAVAILABLE")

        url = self._base_url.rstrip("/") + (LOOKUP_PATH % attestation_id)
        headers = {"Authorization": "Bearer %s" % self._token,
                   "Accept": "application/json"}
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id

        try:
            response = httpx.get(url, headers=headers, timeout=self._timeout)
        except httpx.HTTPError as exc:
            # Transport failure. Deliberately not retried here: a retry loop
            # would turn "core is down" into "core is down for longer", and the
            # caller can retry the whole idempotent operation.
            raise AttestationLookupError(
                "xerbs-core attestation lookup failed: %s" % type(exc).__name__,
                code="ATTESTATION_VERIFICATION_UNAVAILABLE") from exc

        if response.status_code == 404:
            raise AttestationNotFound(attestation_id)
        if response.status_code in (401, 403):
            raise AttestationLookupError(
                "xerbs-core refused the internal service credential",
                code="ATTESTATION_VERIFICATION_UNAUTHORIZED")
        if response.status_code != 200:
            raise AttestationLookupError(
                "xerbs-core attestation lookup returned HTTP %d" % response.status_code,
                code="ATTESTATION_VERIFICATION_UNAVAILABLE")

        try:
            payload = response.json()
        except ValueError as exc:
            raise AttestationLookupError(
                "xerbs-core attestation lookup returned a non-JSON body",
                code="ATTESTATION_RESPONSE_MALFORMED") from exc

        if not isinstance(payload, dict):
            raise AttestationLookupError(
                "xerbs-core attestation lookup returned %s, expected an object"
                % type(payload).__name__,
                code="ATTESTATION_RESPONSE_MALFORMED")

        missing = [f for f in REQUIRED_FIELDS if f not in payload]
        if missing:
            raise AttestationLookupError(
                "xerbs-core attestation response is missing %s" % ", ".join(missing),
                code="ATTESTATION_RESPONSE_MALFORMED")

        if payload["attestation_id"] != attestation_id:
            raise AttestationLookupError(
                "xerbs-core returned a different attestation than requested",
                code="ATTESTATION_RESPONSE_MALFORMED")

        return payload
