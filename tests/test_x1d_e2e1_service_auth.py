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
from uuid import uuid4

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
    """Headers for one generate call, with an idempotency key unique to it.

    X1D-TESTISO1. This key used to be a constant: "e2e1-" + the first 16
    characters of whatever token the case passed. The suite runs against the
    developer database in the repo root -- conftest sets the environment but
    says nothing about DATABASE_URL, so every run reads and writes
    ./xerbs_ai_v2.db -- and a constant key means every run after the first
    collides with a row the first one left behind.

    That is not hypothetical. The stored row dates from 2026-09-13; the intake
    schema has since gained interview_depth and interview_state, so the
    request hash it was written with no longer matches the hash of the same
    body today. generate() compared the two, found them different, and raised
    IdempotencyConflictError -- a 409 where the test asserts 200, permanently,
    on any machine whose database holds that row.

    The endpoint was right every time. A key that is reused with a changed
    payload SHOULD conflict; that is the guarantee it exists to provide. The
    test was wrong to reuse one. A fresh key per call means each case asserts
    what it means to assert -- that a valid credential is accepted -- and
    nothing about what a previous run happened to leave in a database.

    The key also no longer carries the first 16 characters of the credential
    into a persisted row, which a test token made harmless and which was never
    a good shape to keep.
    """
    h = {"Idempotency-Key": "e2e1-%s" % uuid4().hex[:16]}
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


_ABSENT = object()


@pytest.fixture
def _bootstrap_store(tmp_path):
    """A private database for the bootstrap test, built by injection.

    X1D-TESTISO1. The bootstrap test must not write the golden-corpus
    lifecycle into whatever database the rest of the suite is using, so it
    needs one of its own. It used to get that by setting DATABASE_URL and
    reloading app.db.session, persistent_clinical and golden_corpus -- with
    the cleanup written as four statements at the end of the test body, so it
    ran only if every assertion above it had passed.

    Restoring that afterwards is not actually possible. importlib.reload
    re-executes a module in its existing namespace, so PersistentWorkflowError
    becomes a NEW class object while app/api/clinical_knowledge.py keeps
    catching the one it imported at startup. The store then raises a class no
    `except` clause matches, and endpoints that should answer 404 or 409
    answer 500 instead. That is what turned one failure in
    test_phase12c2d2.py into twelve when this file ran first, and no amount of
    reloading afterwards repairs it: each reload mints another class.

    So nothing global is touched. PersistentClinicalStore already accepts a
    session factory, and bootstrap_golden_corpus works entirely through the
    store it is given, so the test can be handed its own engine and leave
    DATABASE_URL, the lru_cache'd engine and factory, and every class identity
    exactly as it found them. The teardown asserts precisely that -- not that
    global state was put back, but that it was never moved.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.bootstrap.golden_corpus as gc
    import app.db.session as session_mod
    import app.services.knowledge.persistent_clinical as pc
    from app.db import models  # noqa: F401 - registers the tables
    from app.db.base import Base

    before_env = os.environ.get("DATABASE_URL", _ABSENT)
    before_url = str(session_mod.get_engine().url)
    before_error = pc.PersistentWorkflowError
    before_store = pc.PersistentClinicalStore

    engine = create_engine(
        "sqlite:///%s/boot.db" % tmp_path.as_posix(),
        connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    try:
        yield gc, pc, factory
    finally:
        engine.dispose()
        assert os.environ.get("DATABASE_URL", _ABSENT) == before_env, (
            "DATABASE_URL was modified")
        assert str(session_mod.get_engine().url) == before_url, (
            "the process-wide engine was moved")
        assert pc.PersistentWorkflowError is before_error, (
            "persistent_clinical was reloaded; handlers now mismatch")
        assert pc.PersistentClinicalStore is before_store, (
            "persistent_clinical was reloaded")


class TestGoldenCorpusBootstrap:
    def test_bootstrap_drives_the_lifecycle_and_stops_before_approval(self, _bootstrap_store):
        """The bootstrap drives the real lifecycle, is safe to re-run, and
        stops at IN_REVIEW.

        It used to approve the formula itself and this test asserted the
        resulting eligibility. X1D-AIV2-ATTEST1 closed that: a machine
        asserting clinical review authority is the failure the whole programme
        exists to undo, and golden_corpus.py's own docstring already said the
        human decision is external. So a freshly bootstrapped environment now
        holds an IN_REVIEW formula that is NOT ranking-eligible -- which is the
        honest state, because nobody has approved it there.
        """
        gc, pc, factory = _bootstrap_store

        first = gc.bootstrap_golden_corpus(
            pc.PersistentClinicalStore(session_factory=factory))
        assert first["source_status_after"] == "REVIEWED"
        assert first["formula_status_after"] == "IN_REVIEW"
        assert first["clinical_ranking_eligible"] is False, first
        assert "ingested_formula_as_draft" in first["actions"]
        assert "formula_awaiting_human_attestation" in first["actions"]

        second = gc.bootstrap_golden_corpus(
            pc.PersistentClinicalStore(session_factory=factory))
        assert second["entity_id"] == first["entity_id"], "re-run must not duplicate"
        assert second["formula_status_after"] == "IN_REVIEW"
        assert second["clinical_ranking_eligible"] is False

    def test_bootstrap_cannot_approve_anything(self, _bootstrap_store):
        """No bootstrap path may confer human clinical approval."""
        gc, pc, factory = _bootstrap_store
        gc.bootstrap_golden_corpus(pc.PersistentClinicalStore(session_factory=factory))
        src = open(gc.__file__, encoding="utf-8").read()
        # Code only. The comment explaining what was removed legitimately names
        # the call, and a test that forbids explaining a removal would be a
        # test against documentation.
        code = chr(10).join(l for l in src.splitlines()
                            if not l.strip().startswith("#"))
        assert "store.review(" not in code
        assert "decision=ReviewDecision.APPROVE" not in code
        from app.services.governance.identity import (
            PRINCIPALS_THAT_MAY_HUMAN_APPROVE)
        assert PRINCIPALS_THAT_MAY_HUMAN_APPROVE == frozenset()

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
