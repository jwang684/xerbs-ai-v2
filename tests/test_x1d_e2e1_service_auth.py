"""X1D-E2E1: service authentication and governance authorization.

``tests/conftest.py`` restores the permissive environment the pre-existing 233
tests were written against, so those tests keep describing what they always
described. This file deliberately does the opposite: it configures a real
credential and asserts the shipped defaults — anonymous access refused,
governance mutation closed — so the old tests can never quietly mask the new
guarantees.

The settings object is ``lru_cache``d, so each test sets the environment and
clears the cache before building its own ``TestClient``.
"""

import importlib
import os

import pytest
from fastapi.testclient import TestClient

TOKEN = "x1d-e2e1-test-service-token-not-a-real-secret"

GENERATE = "/api/v1/integrations/base44/generate"
INGEST = "/api/v1/knowledge/clinical/ingest"
STATS = "/api/v1/knowledge/clinical/stats"


def _client(**env):
    """Build an app client with an explicit service-auth environment."""
    # Set explicitly rather than popping: a popped variable falls back to .env,
    # where a developer may have a real token configured, which would silently
    # change what the test is exercising. The booleans get "false" rather than
    # "" because pydantic cannot parse an empty string as a bool.
    os.environ["SERVICE_AUTH_TOKEN"] = ""
    os.environ["ALLOW_INSECURE_LOCAL"] = "false"
    os.environ["ALLOW_EXTERNAL_GOVERNANCE_MUTATION"] = "false"
    os.environ.update({k: v for k, v in env.items() if v is not None})

    from app.core import config

    config.get_settings.cache_clear()

    import app.main

    importlib.reload(app.main)
    return TestClient(app.main.app)


@pytest.fixture(autouse=True)
def _restore_env():
    """Leave the permissive test environment as conftest set it."""
    yield
    os.environ["ALLOW_INSECURE_LOCAL"] = "true"
    os.environ["ALLOW_EXTERNAL_GOVERNANCE_MUTATION"] = "true"
    # Blank rather than pop: Settings also reads .env, where a developer may
    # have a real token configured. Popping would let that value reappear and
    # switch authentication on for every test that runs after this file.
    os.environ["SERVICE_AUTH_TOKEN"] = ""
    from app.core import config

    config.get_settings.cache_clear()
    import app.main

    importlib.reload(app.main)


def _gen_body():
    return {"request_id": "r1", "intake": {"text_input": "发热恶寒"}}


def _gen_headers(token=None):
    h = {"Idempotency-Key": "e2e1-" + (token or "anon")[:16]}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


class TestServiceAuthentication:
    def test_missing_credential_is_refused(self):
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        r = c.post(GENERATE, json=_gen_body(), headers=_gen_headers())
        assert r.status_code == 401, r.text

    def test_wrong_credential_is_refused(self):
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        r = c.post(GENERATE, json=_gen_body(), headers=_gen_headers("not-the-token"))
        assert r.status_code == 401, r.text

    def test_malformed_scheme_is_refused(self):
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        r = c.post(
            GENERATE, json=_gen_body(),
            headers={"Idempotency-Key": "e2e1-mal", "Authorization": TOKEN},
        )
        assert r.status_code == 401, r.text

    def test_valid_credential_is_accepted(self):
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        r = c.post(GENERATE, json=_gen_body(), headers=_gen_headers(TOKEN))
        assert r.status_code == 200, r.text

    def test_unconfigured_secret_fails_closed(self):
        """No secret and no explicit local mode means no access, not open access."""
        c = _client()  # neither token nor insecure-local
        r = c.post(GENERATE, json=_gen_body(), headers=_gen_headers())
        assert r.status_code == 401, r.text

    def test_read_only_endpoints_also_require_credential(self):
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        assert c.get(STATS).status_code == 401
        assert c.get(STATS, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200

    def test_health_stays_reachable_without_credential(self):
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        r = c.get("/health")
        assert r.status_code == 200
        # Health must not leak the credential or configuration secrets
        assert TOKEN not in r.text

    def test_credential_is_never_echoed_in_error_bodies(self):
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        r = c.post(GENERATE, json=_gen_body(), headers=_gen_headers("wrong-value"))
        assert TOKEN not in r.text
        assert "wrong-value" not in r.text


class TestGovernanceAuthorization:
    """reviewer_role is self-asserted, so it cannot confer review authority."""

    def test_ingest_refused_when_governance_closed(self):
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        r = c.post(
            INGEST,
            json={
                "submitted_by": "attacker",
                "source_label": "x",
                "items": [{"entity_type": "formula", "payload": {"name": "attacker-formula"}}],
            },
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert r.status_code == 403, r.text

    def test_self_asserted_reviewer_role_cannot_approve(self):
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        r = c.post(
            "/api/v1/knowledge/clinical/entities/formula/frm-anything/review",
            json={
                "reviewer_id": "attacker",
                "reviewer_role": "CLINICAL_REVIEWER",
                "decision": "APPROVE",
            },
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert r.status_code == 403, r.text

    def test_self_asserted_admin_role_cannot_approve_a_source(self):
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        r = c.post(
            "/api/v1/knowledge/clinical/sources/any-source/review",
            json={
                "reviewer_id": "attacker",
                "reviewer_role": "CLINICAL_ADMIN",
                "decision": "APPROVE",
                "expected_version": 1,
            },
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert r.status_code == 403, r.text

    def test_relationship_creation_refused(self):
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        r = c.post(
            "/api/v1/safety/relationships",
            json={
                "source_entity_id": "a",
                "target_entity_id": "b",
                "relationship_type": "PATTERN_FORMULA",
                "actor_id": "attacker",
            },
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert r.status_code == 403, r.text

    def test_service_auth_alone_does_not_confer_review_authority(self):
        """A valid service credential proves the caller is xerbs-core.

        It does not make xerbs-core a qualified clinical reviewer — those are
        different claims, and only the first one can be proved here.
        """
        c = _client(SERVICE_AUTH_TOKEN=TOKEN)
        assert c.post(GENERATE, json=_gen_body(),
                      headers=_gen_headers(TOKEN)).status_code == 200
        assert c.post(INGEST,
                      json={"submitted_by": "x", "source_label": "y",
                            "items": [{"entity_type": "formula", "payload": {"name": "z"}}]},
                      headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 403


class TestGoldenCorpusBootstrap:
    def test_bootstrap_is_idempotent_and_reaches_eligibility(self, tmp_path):
        """The bootstrap must drive the real lifecycle and be safe to re-run."""
        os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path.as_posix()}/boot.db"
        from app.core import config

        config.get_settings.cache_clear()

        import app.db.session as session_mod

        importlib.reload(session_mod)
        session_mod.init_db()

        import app.services.knowledge.persistent_clinical as pc

        importlib.reload(pc)
        import app.bootstrap.golden_corpus as gc

        importlib.reload(gc)

        first = gc.bootstrap_golden_corpus(pc.PersistentClinicalStore())
        assert first["clinical_ranking_eligible"] is True, first
        assert first["source_status_after"] == "REVIEWED"
        assert first["formula_status_after"] == "REVIEWED"
        assert "ingested_formula_as_draft" in first["actions"]

        second = gc.bootstrap_golden_corpus(pc.PersistentClinicalStore())
        assert second["entity_id"] == first["entity_id"], "re-run must not duplicate"
        assert second["clinical_ranking_eligible"] is True
        assert "already_eligible" in second["actions"]

        os.environ.pop("DATABASE_URL", None)
        config.get_settings.cache_clear()

    def test_bootstrap_embeds_no_credential(self):
        import app.bootstrap.golden_corpus as gc

        src = open(gc.__file__, encoding="utf-8").read()
        assert "eyJ" not in src
        assert "SERVICE_AUTH_TOKEN" not in src

    def test_bootstrap_does_not_fetch_at_deploy_time(self):
        """No web scraping, no runtime search, no model inference."""
        import app.bootstrap.golden_corpus as gc

        src = open(gc.__file__, encoding="utf-8").read()
        for forbidden in ("requests.", "httpx.", "urllib.request", "openai", "web_search"):
            assert forbidden not in src, forbidden
