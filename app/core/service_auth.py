"""Service-to-service authentication (X1D-E2E1).

Why this exists
---------------
Until now every endpoint in this service was reachable without any
credential, governance mutations included: an unauthenticated POST could
ingest a clinical entity, submit it for review, and approve it, simply by
putting ``reviewer_role: "CLINICAL_REVIEWER"`` in the request body. That is
acceptable for a laptop and unacceptable anywhere reachable by other people.

This service has no notion of an end user and must never acquire one. It is
called by xerbs-core, server to server. So the credential is a single service
secret supplied by the environment, and the caller it authenticates is
"xerbs-core", not a person.

The consumer's browser must never hold this secret. The call path is

    browser -> xerbs-core (consumer session) -> xerbs-ai-v2 (service secret)

and the two credentials are deliberately unrelated: a consumer token is
useless here, and this secret is never sent to a browser.

Fail closed
-----------
If ``SERVICE_AUTH_TOKEN`` is not configured, protected endpoints refuse every
request rather than falling open. An unconfigured deployment is a broken
deployment, not an open one. The one exception is an explicitly declared
local development mode, which must be set deliberately and is reported by
the health endpoint so it cannot be mistaken for a hardened deployment.

Comparison uses ``secrets.compare_digest`` so a wrong token cannot be
recovered by timing the response. The secret is never logged, never echoed
in an error body, and never returned by any endpoint.
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request

from app.core.config import get_settings

_BEARER = "bearer"


def service_auth_enabled() -> bool:
    """True when a service secret is configured."""
    return bool((get_settings().service_auth_token or "").strip())


def _dev_mode_allows_anonymous() -> bool:
    """Anonymous access is only tolerable when explicitly asked for.

    Requires BOTH an unconfigured secret AND allow_insecure_local=True, so
    simply forgetting to set the secret never opens the service up.
    """
    s = get_settings()
    return (not service_auth_enabled()) and bool(s.allow_insecure_local)


def require_service_auth(request: Request) -> str:
    """Authenticate the calling service.

    Returns the caller label for audit use. Raises 401 otherwise.

    The header is read from the request rather than declared as a ``Header``
    parameter deliberately. A declared parameter is published into every
    route's OpenAPI parameter list, which changes the documented API surface
    of endpoints that have nothing to do with authentication. Reading it here
    keeps the published schema identical to before this dependency existed.

    The trade-off is that the credential requirement is not advertised in
    OpenAPI; that is acceptable because this schema describes an internal
    service-to-service API and docs are disabled outside development.
    """
    settings = get_settings()
    authorization = request.headers.get("authorization")

    if _dev_mode_allows_anonymous():
        # Explicit local development. Health reports this state.
        return "local-insecure"

    expected = (settings.service_auth_token or "").strip()
    if not expected:
        # Fail closed: no secret configured means no access, not open access.
        raise HTTPException(
            status_code=401,
            detail="Service authentication is not configured; refusing request.",
        )

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing service credential.")

    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != _BEARER:
        raise HTTPException(status_code=401, detail="Malformed service credential.")

    # Constant-time comparison; never echo either value back to the caller.
    if not secrets.compare_digest(parts[1].strip(), expected):
        raise HTTPException(status_code=401, detail="Invalid service credential.")

    return settings.service_caller_name
