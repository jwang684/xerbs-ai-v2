"""Test environment for the existing suite (X1D-E2E1).

X1D-E2E1 added two fail-closed defaults to this service:

  * every API router except health now requires a service credential;
  * external clinical governance mutation is refused unless explicitly
    enabled, because ``reviewer_role`` arrives in the request body and so
    cannot establish clinical review authority.

The 233 pre-existing tests were written before either existed. They drive the
app in-process with ``TestClient`` and call ingest/review freely, so without
this file they would all fail on the new defaults — which would say nothing
about whether they still describe correct behaviour.

This file therefore restores the environment those tests assume, and only
that: anonymous local access and governance mutation enabled. It does not
weaken the shipped defaults, which stay closed.

The new defaults are covered separately by
``tests/test_x1d_e2e1_service_auth.py``, which configures a real credential
and asserts that anonymous and wrong-credential callers are refused and that
governance mutation is closed unless deliberately enabled. Keeping the two
concerns in separate files means the old tests never quietly mask the new
guarantees.
"""

import os

# Must be set before app.main is imported, which pytest does when the first
# test module imports it. conftest is loaded first, so this is the right place.
os.environ.setdefault("LLM_PROVIDER", "mock")

# Assigned, not setdefault, and SERVICE_AUTH_TOKEN is explicitly cleared.
#
# Settings also reads a local .env file, so a developer who has configured a
# real service token there would otherwise turn authentication on for the
# whole suite and every pre-existing test would fail with 401 — which says
# nothing about whether those tests still describe correct behaviour. The test
# environment is therefore stated outright here rather than inherited.
#
# Environment variables take precedence over .env in pydantic-settings, so an
# empty value here reliably neutralises a configured token for tests only.
os.environ["SERVICE_AUTH_TOKEN"] = ""
os.environ["ALLOW_INSECURE_LOCAL"] = "true"
os.environ["ALLOW_EXTERNAL_GOVERNANCE_MUTATION"] = "true"

# X1D-PIPE1: the mock provider is now permitted only where a non-clinical
# answer is the expected outcome. State that this is such a place, for the same
# reason the values above are stated rather than inherited: a local .env with
# ENVIRONMENT=staging would otherwise turn every mock-backed test into a
# configuration error, which says nothing about whether those tests still
# describe correct behaviour.
#
# The guard itself is covered by tests/test_x1d_pipe1_provider_policy.py, which
# sets the environment explicitly per case and never relies on this default.
os.environ["ENVIRONMENT"] = "test"
