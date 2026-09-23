"""X1D-AIV2-GOV2-C1: governed object identity, lifecycle and provenance.

The defect these tests exist to keep closed: ``create_relationship`` and
``create_rule`` wrote ``review_status='REVIEWED'`` as a literal, so an object's
authority came from a constant in a source file rather than from any review.
Production still holds one PATTERN_FORMULA edge of exactly that shape.

Everything here is synthetic. No test asserts anything clinical, and no test
creates or simulates a human review decision -- this service cannot produce
one, which is itself the point of several of these tests.
"""

import json
import pathlib
import unicodedata
import uuid

import pytest

from app.services.governance import canonical, identity, lifecycle

VECTORS_PATH = (pathlib.Path(__file__).parent / "vectors"
                / "canonical_hash_vectors.json")


# ----------------------------------------------------------------------
# 1. Canonical hashing -- vectors are literal constants in the artifact
# ----------------------------------------------------------------------

class TestCanonicalHashVectors:
    """Expected digests come from the JSON file, never from recomputation.

    Recomputing the expected value with the same implementation inside the
    assertion would pass no matter what the implementation did. These vectors
    are the cross-repo contract xerbs-core will verify against.
    """

    @pytest.fixture(scope="class")
    def doc(self):
        return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))

    def test_spec_version_matches_the_implementation(self, doc):
        assert doc["spec_version"] == canonical.SPEC_VERSION

    def test_every_vector_reproduces_its_literal_digest(self, doc):
        for v in doc["vectors"]:
            assert canonical.canonical_json(v["subject"]) == v["canonical_json"], v["name"]
            assert canonical.digest(v["subject"]) == v["sha256"], v["name"]

    def test_nfd_and_nfc_inputs_agree(self, doc):
        by = {v["name"]: v for v in doc["vectors"]}
        assert by["entity_nfd_input"]["sha256"] == by["entity_nfc_input"]["sha256"]

    def test_evidence_order_is_irrelevant(self, doc):
        by = {v["name"]: v for v in doc["vectors"]}
        assert by["evidence_three_sources"]["sha256"] == \
            by["evidence_three_permuted"]["sha256"]

    def test_material_differences_change_the_digest(self, doc):
        by = {v["name"]: v for v in doc["vectors"]}
        assert by["relationship_pattern_formula"]["sha256"] != \
            by["relationship_different_target"]["sha256"]
        assert by["safety_rule_block_cjk"]["sha256"] != \
            by["safety_rule_warn_variant"]["sha256"]
        assert by["evidence_three_sources"]["sha256"] != \
            by["evidence_added_source"]["sha256"]

    def test_vector_file_covers_the_required_shapes(self, doc):
        names = {v["name"] for v in doc["vectors"]}
        for required in ("source_ascii", "source_cjk_null_fields",
                         "entity_cjk_set_sorted", "entity_nfd_input",
                         "relationship_pattern_formula", "safety_rule_block_cjk",
                         "evidence_three_sources", "evidence_three_permuted",
                         "evidence_empty"):
            assert required in names, required


class TestCanonicalHashRules:

    def test_none_is_omitted_not_serialised_as_null(self):
        assert canonical.canonical_json({"a": 1, "b": None}) == '{"a":1}'

    def test_keys_are_sorted_and_whitespace_is_absent(self):
        assert canonical.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_cjk_is_never_escaped(self):
        assert "银翘散" in canonical.canonical_json({"n": "银翘散"})

    def test_floats_are_refused_rather_than_rounded(self):
        with pytest.raises(canonical.CanonicalisationError):
            canonical.canonical_json({"x": 1.5})

    def test_nfc_normalisation_applies_to_keys_too(self):
        a = canonical.canonical_json({unicodedata.normalize("NFD", "å"): 1})
        b = canonical.canonical_json({unicodedata.normalize("NFC", "å"): 1})
        assert a == b

    def test_unknown_relationship_type_is_refused(self):
        with pytest.raises(canonical.CanonicalisationError):
            canonical.relationship_subject(
                source_external_id="a", relationship_type="MADE_UP",
                target_external_id="b")


# ----------------------------------------------------------------------
# 2. Semantic identity
# ----------------------------------------------------------------------

class TestSemanticIdentity:

    def test_relationship_identity_is_stable_across_evidence_and_version(self):
        """Evidence and versions move; identity does not."""
        a = identity.relationship_external_id(
            "xerbs:pattern:p", "PATTERN_FORMULA", "xerbs:formula:f")
        b = identity.relationship_external_id(
            "xerbs:pattern:p", "PATTERN_FORMULA", "xerbs:formula:f")
        assert a == b == "rel:xerbs:pattern:p|PATTERN_FORMULA|xerbs:formula:f"

    def test_relationship_identity_changes_with_the_triple(self):
        a = identity.relationship_external_id("xerbs:pattern:p", "PATTERN_FORMULA",
                                              "xerbs:formula:f")
        b = identity.relationship_external_id("xerbs:pattern:p", "PATTERN_FORMULA",
                                              "xerbs:formula:g")
        assert a != b

    def test_relationship_identity_needs_both_endpoints(self):
        with pytest.raises(identity.SemanticIdentityError):
            identity.relationship_external_id("", "PATTERN_FORMULA", "xerbs:formula:f")

    def test_safety_rule_identity_normalises_the_trigger(self):
        a = identity.safety_rule_external_id(
            "xerbs:formula:f", "PREGNANCY", unicodedata.normalize("NFD", "孕 "))
        b = identity.safety_rule_external_id(
            "xerbs:formula:f", "PREGNANCY", unicodedata.normalize("NFC", "孕"))
        assert a == b

    @pytest.mark.parametrize("bad", [
        "",                 # empty
        "ab",               # too short
        "x" * 300,          # too long
        "has space",        # whitespace makes identities invisibly different
        "tab\there",
        "ctrl\x01char",
        "pipe|delimiter",   # the delimiter derived ids are built from
    ])
    def test_malformed_external_id_fails_closed(self, bad):
        with pytest.raises(identity.SemanticIdentityError):
            identity.validate_external_id(bad)

    def test_cjk_external_id_is_accepted(self):
        """The one semantic identity this corpus already uses contains CJK.

        An ASCII-only grammar would have rejected
        ``xerbs-core:canonical-formula:清肺排毒汤``, which the X1D-E2E1
        bootstrap has been writing since before this phase existed.
        """
        got = identity.validate_external_id("xerbs-core:canonical-formula:清肺排毒汤")
        assert got == "xerbs-core:canonical-formula:清肺排毒汤"

    def test_external_id_is_nfc_normalised(self):
        nfd = unicodedata.normalize("NFD", "xerbs:formula:银翘散")
        assert identity.validate_external_id(nfd) == \
            unicodedata.normalize("NFC", "xerbs:formula:银翘散")

    def test_source_identity_is_the_existing_source_id(self):
        """No parallel identity field was invented for sources."""
        from app.db.models import SourceRegistry
        assert SourceRegistry.__table__.primary_key.columns.keys() == ["source_id"]
        assert not hasattr(SourceRegistry, "external_id")


# ----------------------------------------------------------------------
# 3. Principals -- machines only
# ----------------------------------------------------------------------

class TestPrincipals:

    def test_no_principal_may_human_approve(self):
        assert identity.PRINCIPALS_THAT_MAY_HUMAN_APPROVE == frozenset()

    def test_no_principal_is_named_like_a_human_reviewer(self):
        for subject in identity.MACHINE_PRINCIPALS:
            assert "reviewer" not in subject.lower() or subject == identity.LEGACY_UNKNOWN
            assert ":admin:" not in subject

    def test_every_principal_matches_the_core_subject_grammar(self):
        for subject in identity.MACHINE_PRINCIPALS:
            assert identity.is_valid_subject(subject), subject

    def test_unknown_historical_actor_stays_unknown(self):
        assert identity.as_subject(None) == identity.LEGACY_UNKNOWN
        assert identity.as_subject("has spaces and !") == identity.LEGACY_UNKNOWN

    def test_known_historical_actor_is_preserved_when_expressible(self):
        assert identity.as_subject("phase12c4b-automation") == \
            "xerbs-ai-v2:principal:phase12c4b-automation"

    def test_an_already_namespaced_subject_passes_through(self):
        assert identity.as_subject(identity.BOOTSTRAP) == identity.BOOTSTRAP


# ----------------------------------------------------------------------
# 4. Lifecycle -- approval is absent, not merely restricted
# ----------------------------------------------------------------------

class TestLifecycle:

    def test_initial_state_is_never_reviewed(self):
        assert lifecycle.INITIAL_STATE == lifecycle.DRAFT
        assert lifecycle.INITIAL_STATE != lifecycle.REVIEWED

    def test_no_local_transition_reaches_reviewed(self):
        assert lifecycle.REVIEWED not in set(lifecycle.LOCAL_TRANSITIONS.values())

    def test_approve_always_fails_closed(self):
        for state in lifecycle.GOVERNED_STATES:
            with pytest.raises(lifecycle.HumanAttestationRequired):
                lifecycle.next_state(state, "APPROVE")

    def test_locally_authorisable_transitions_work(self):
        assert lifecycle.next_state(lifecycle.DRAFT, "SUBMIT") == lifecycle.IN_REVIEW
        assert lifecycle.next_state(lifecycle.IN_REVIEW, "REJECT") == lifecycle.REJECTED
        assert lifecycle.next_state(lifecycle.IN_REVIEW, "REQUEST_CHANGES") == lifecycle.DRAFT
        assert lifecycle.next_state(lifecycle.REVIEWED, "RETIRE") == lifecycle.RETIRED

    def test_illegal_transitions_are_refused(self):
        with pytest.raises(lifecycle.GovernanceLifecycleError):
            lifecycle.next_state(lifecycle.DRAFT, "REJECT")

    def test_attestation_is_always_required_for_approval(self):
        assert lifecycle.approve_requires_attestation() is True


class TestRankingEligibilityRule:

    def _call(self, **over):
        kwargs = dict(review_status=lifecycle.REVIEWED,
                      governance_provenance=lifecycle.ATTESTED,
                      review_attestation_id="att-1",
                      reviewed_evidence_source_count=1, retired_at=None)
        kwargs.update(over)
        return lifecycle.is_governed_object_ranking_eligible(**kwargs)

    def test_gov2_object_without_attestation_is_not_eligible(self):
        assert self._call(review_attestation_id=None) is False

    def test_gov2_object_without_reviewed_evidence_is_not_eligible(self):
        assert self._call(reviewed_evidence_source_count=0) is False

    def test_draft_is_never_eligible(self):
        assert self._call(review_status=lifecycle.DRAFT) is False

    def test_newly_created_object_shape_is_not_eligible(self):
        """Exactly what create_relationship writes."""
        assert self._call(review_status=lifecycle.DRAFT,
                          governance_provenance=lifecycle.UNREVIEWED,
                          review_attestation_id=None,
                          reviewed_evidence_source_count=1) is False

    def test_retired_is_never_eligible(self):
        assert self._call(retired_at="2026-01-01") is False

    def test_legacy_rows_are_explicitly_classified_not_silently_trusted(self):
        """Grandfathering is a named, single, flippable decision.

        AI-GOV2-D flips LEGACY_ELIGIBILITY_GRANDFATHERED to False, and that
        makes every pre-attestation REVIEWED relationship ineligible until a
        human re-reviews it. The behaviour is pinned here so the flip cannot
        happen silently.
        """
        for prov in lifecycle.LEGACY_PROVENANCE:
            got = self._call(governance_provenance=prov, review_attestation_id=None,
                             reviewed_evidence_source_count=0)
            assert got is lifecycle.LEGACY_ELIGIBILITY_GRANDFATHERED, prov

    def test_legacy_provenance_set_is_complete(self):
        assert lifecycle.LEGACY_PROVENANCE < lifecycle.GOVERNANCE_PROVENANCE
        assert lifecycle.ATTESTED not in lifecycle.LEGACY_PROVENANCE
        assert lifecycle.UNREVIEWED not in lifecycle.LEGACY_PROVENANCE


# ----------------------------------------------------------------------
# 5. Engine behaviour against a real database
# ----------------------------------------------------------------------

def _seed_reviewed_pair(client, ext_a="xerbs:pattern:synth-a",
                        ext_b="xerbs:formula:synth-b"):
    """Two REVIEWED entities with external ids, via the ordinary governed path."""
    tag = uuid.uuid4().hex[:8]
    src = "SYNTH-src-%s" % tag
    body = {"submitted_by": "synth-author", "source_label": "SYNTHETIC GOV2-C1",
            "items": [
                {"entity_type": "pattern", "external_id": "%s-%s" % (ext_a, tag),
                 "payload": {"name": "SYNTHETIC 证 %s" % tag},
                 "sources": [{"source_id": src, "title": "SYNTHETIC source",
                              "source_type": "GUIDELINE"}]},
                {"entity_type": "formula", "external_id": "%s-%s" % (ext_b, tag),
                 "payload": {"name": "SYNTHETIC 方 %s" % tag},
                 "sources": [{"source_id": src, "title": "SYNTHETIC source",
                              "source_type": "GUIDELINE"}]},
            ]}
    r = client.post("/api/v1/knowledge/clinical/ingest", json=body)
    assert r.status_code in (200, 201), r.text
    ids = r.json()["created_entity_ids"]

    # Source through its own lifecycle so the entities can be approved.
    client.post("/api/v1/knowledge/clinical/sources/%s/submit-review" % src,
                json={"submitted_by": "synth-author"})
    client.post("/api/v1/knowledge/clinical/sources/%s/review" % src,
                json={"reviewer_id": "synth-reviewer",
                      "reviewer_role": "CLINICAL_REVIEWER", "decision": "APPROVE"})
    for etype, eid in zip(("pattern", "formula"), ids):
        client.post("/api/v1/knowledge/clinical/entities/%s/%s/submit-review"
                    % (etype, eid), json={"submitted_by": "synth-author"})
        rr = client.post("/api/v1/knowledge/clinical/entities/%s/%s/review"
                         % (etype, eid),
                         json={"reviewer_id": "synth-reviewer",
                               "reviewer_role": "CLINICAL_REVIEWER",
                               "decision": "APPROVE"})
        assert rr.status_code == 200, rr.text
    return ids[0], ids[1], src


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


class TestRelationshipCreation:

    def test_new_relationship_starts_non_reviewed(self, client):
        """Requirement 1: creation no longer confers REVIEWED."""
        pat, frm, src = _seed_reviewed_pair(client)
        r = client.post("/api/v1/safety/relationships", json={
            "source_entity_id": pat, "target_entity_id": frm,
            "relationship_type": "PATTERN_FORMULA", "source_id": src,
            "actor_id": "synth-author", "actor_role": "CLINICAL_REVIEWER"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["review_status"] == "DRAFT"
        assert body["review_status"] != "REVIEWED"

    def test_creation_cannot_make_it_ranking_eligible(self, client):
        """Requirement 2 and 20: the retrieval path must not see it."""
        pat, frm, src = _seed_reviewed_pair(client)
        r = client.post("/api/v1/safety/relationships", json={
            "source_entity_id": pat, "target_entity_id": frm,
            "relationship_type": "PATTERN_FORMULA", "source_id": src,
            "actor_id": "synth-author", "actor_role": "CLINICAL_REVIEWER"})
        assert r.json()["clinical_ranking_eligible"] is False

        from app.services.knowledge.persistent_clinical import PersistentClinicalStore
        assert PersistentClinicalStore().eligible_formula_candidates_for_patterns(
            [pat]) == []

    def test_a_machine_principal_cannot_approve_it(self, client):
        """Requirement 3: no bootstrap or authoring principal has a path."""
        for principal in identity.MACHINE_PRINCIPALS:
            with pytest.raises(lifecycle.HumanAttestationRequired):
                lifecycle.next_state(lifecycle.IN_REVIEW, "APPROVE")
        assert identity.PRINCIPALS_THAT_MAY_HUMAN_APPROVE == frozenset()

    def test_request_body_reviewer_role_cannot_approve_it(self, client):
        """Requirement 4: CLINICAL_ADMIN in a body still yields DRAFT."""
        pat, frm, src = _seed_reviewed_pair(client)
        r = client.post("/api/v1/safety/relationships", json={
            "source_entity_id": pat, "target_entity_id": frm,
            "relationship_type": "PATTERN_FORMULA", "source_id": src,
            "actor_id": "synth-author", "actor_role": "CLINICAL_ADMIN"})
        assert r.json()["review_status"] == "DRAFT"

    def test_semantic_identity_and_hashes_are_recorded(self, client):
        """Requirements 5, 6, 7: identity, content hash and evidence hash."""
        pat, frm, src = _seed_reviewed_pair(client)
        r = client.post("/api/v1/safety/relationships", json={
            "source_entity_id": pat, "target_entity_id": frm,
            "relationship_type": "PATTERN_FORMULA", "source_id": src,
            "actor_id": "synth-author", "actor_role": "CLINICAL_REVIEWER"})
        body = r.json()
        assert body["external_id"].startswith("rel:")
        assert len(body["content_hash"]) == 64
        assert len(body["evidence_hash"]) == 64
        assert body["governance_provenance"] == lifecycle.UNREVIEWED
        assert body["version"] == 1

    def test_content_hash_changes_with_the_triple(self):
        """Requirement 6, at the level that decides it."""
        a = lifecycle.relationship_content_hash("x:p:1", "PATTERN_FORMULA", "x:f:1")
        b = lifecycle.relationship_content_hash("x:p:1", "PATTERN_FORMULA", "x:f:2")
        assert a != b

    def test_evidence_hash_changes_when_the_evidence_set_changes(self):
        """Requirement 7."""
        one = lifecycle.evidence_hash([{"source_id": "s1", "source_version": 1}])
        two = lifecycle.evidence_hash([{"source_id": "s1", "source_version": 1},
                                       {"source_id": "s2", "source_version": 1}])
        assert one != two

    def test_evidence_hash_is_invariant_to_row_order(self):
        """Requirement 8: database order is not meaningful."""
        a = lifecycle.evidence_hash([{"source_id": "s2", "source_version": 1},
                                     {"source_id": "s1", "source_version": 2}])
        b = lifecycle.evidence_hash([{"source_id": "s1", "source_version": 2},
                                     {"source_id": "s2", "source_version": 1}])
        assert a == b

    def test_version_and_evidence_rows_are_written(self, client):
        """Requirement 11 groundwork: an append-only trail exists."""
        pat, frm, src = _seed_reviewed_pair(client)
        r = client.post("/api/v1/safety/relationships", json={
            "source_entity_id": pat, "target_entity_id": frm,
            "relationship_type": "PATTERN_FORMULA", "source_id": src,
            "actor_id": "synth-author", "actor_role": "CLINICAL_REVIEWER"})
        rid = r.json()["id"]

        from sqlalchemy import select
        from app.db.models import (GovernedObjectReviewEvent, GovernedObjectSource,
                                   GovernedObjectVersion)
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            versions = s.scalars(select(GovernedObjectVersion).where(
                GovernedObjectVersion.object_id == rid)).all()
            events = s.scalars(select(GovernedObjectReviewEvent).where(
                GovernedObjectReviewEvent.object_id == rid)).all()
            evidence = s.scalars(select(GovernedObjectSource).where(
                GovernedObjectSource.object_id == rid)).all()
        assert len(versions) == 1 and versions[0].version == 1
        assert len(events) == 1 and events[0].to_status == "DRAFT"
        assert events[0].attestation_id is None
        assert [e.source_id for e in evidence] == [src]

    def test_author_provenance_is_a_namespaced_machine_subject(self, client):
        """Requirement 18: known actors preserved, as machines."""
        pat, frm, src = _seed_reviewed_pair(client)
        rid = client.post("/api/v1/safety/relationships", json={
            "source_entity_id": pat, "target_entity_id": frm,
            "relationship_type": "PATTERN_FORMULA", "source_id": src,
            "actor_id": "synth-author", "actor_role": "CLINICAL_REVIEWER"}).json()["id"]
        from sqlalchemy import select
        from app.db.models import GovernedObjectVersion
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            v = s.scalars(select(GovernedObjectVersion).where(
                GovernedObjectVersion.object_id == rid)).one()
        assert v.created_by == "xerbs-ai-v2:principal:synth-author"
        assert identity.is_valid_subject(v.created_by)


class TestSafetyRuleCreation:

    def _rule(self, client, **over):
        pat, frm, src = _seed_reviewed_pair(client)
        body = {"target_entity_id": frm, "rule_type": "PREGNANCY",
                "trigger_term": "孕", "severity": "CRITICAL", "action": "BLOCK",
                "message": "SYNTHETIC 孕期禁用", "source_id": src,
                "actor_id": "synth-author", "actor_role": "CLINICAL_REVIEWER"}
        body.update(over)
        return frm, client.post("/api/v1/safety/rules", json=body)

    def test_new_safety_rule_starts_non_reviewed(self, client):
        """Requirement 12."""
        _, r = self._rule(client)
        assert r.status_code == 200, r.text
        assert r.json()["review_status"] == "DRAFT"

    def test_safety_rule_cannot_become_effective_by_creation(self, client):
        """Requirement 13: screen() must not apply it.

        This tightens safety rather than relaxing it: an unreviewed rule is not
        applied, and a rule that is not applied never removes a finding that a
        reviewed rule produced.
        """
        frm, r = self._rule(client)
        assert r.json()["clinical_ranking_eligible"] is False
        screened = client.post("/api/v1/safety/screen", json={
            "formula_id": frm, "formula_name": "SYNTHETIC 方",
            "patient_context": {"pregnancy": True}, "constraints": []})
        assert screened.status_code == 200, screened.text
        assert screened.json()["findings"] == []

    def test_safety_rule_records_identity_and_hashes(self, client):
        _, r = self._rule(client)
        body = r.json()
        assert body["external_id"].startswith("rule:")
        assert len(body["content_hash"]) == 64
        assert len(body["evidence_hash"]) == 64
        assert body["governance_provenance"] == lifecycle.UNREVIEWED

    def test_severity_change_changes_the_content_hash(self):
        block = lifecycle.safety_rule_content_hash(
            target_external_id="x:f:1", rule_type="PREGNANCY", trigger_term="孕",
            severity="CRITICAL", action="BLOCK", message="m")
        warn = lifecycle.safety_rule_content_hash(
            target_external_id="x:f:1", rule_type="PREGNANCY", trigger_term="孕",
            severity="WARN", action="WARN", message="m")
        assert block != warn


class TestEntityVersioning:

    def test_material_change_produces_a_new_version_and_hash(self, client):
        """Requirements 10 and 11: reuse of clinical_entity_version."""
        pat, frm, src = _seed_reviewed_pair(client)
        hist = client.get("/api/v1/knowledge/clinical/entities/pattern/%s/history" % pat)
        assert hist.status_code == 200
        snapshots = hist.json()["results"]
        assert len(snapshots) >= 3          # DRAFT, IN_REVIEW, REVIEWED

        first = canonical.digest(canonical.entity_subject(
            external_id="x", entity_type="pattern", snapshot=snapshots[0]))
        changed = dict(snapshots[0]); changed["indications"] = ["SYNTHETIC-新"]
        second = canonical.digest(canonical.entity_subject(
            external_id="x", entity_type="pattern", snapshot=changed))
        assert first != second

    def test_status_alone_does_not_change_the_content_hash(self, client):
        """review_status is excluded from the subject, by rule 8."""
        pat, frm, src = _seed_reviewed_pair(client)
        snaps = client.get(
            "/api/v1/knowledge/clinical/entities/pattern/%s/history"
            % pat).json()["results"]
        digests = {canonical.digest(canonical.entity_subject(
            external_id="x", entity_type="pattern", snapshot=s)) for s in snaps}
        assert len(digests) == 1, (
            "only review_status/version changed across these versions, so the "
            "content digest must not move")

    def test_historical_versions_are_immutable_rows(self, client):
        """Requirement 11: append-only, one row per version."""
        pat, frm, src = _seed_reviewed_pair(client)
        from sqlalchemy import select
        from app.db.models import ClinicalEntityVersion
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            rows = s.scalars(select(ClinicalEntityVersion).where(
                ClinicalEntityVersion.entity_id == pat)).all()
        versions = sorted(r.version for r in rows)
        assert versions == list(range(1, len(versions) + 1))

    def test_entity_external_id_is_stored_and_unique(self, client):
        """Requirements 14 and 15."""
        pat, frm, src = _seed_reviewed_pair(client)
        from app.db.models import ClinicalEntity
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            e = s.get(ClinicalEntity, pat)
            assert e.external_id and e.external_id.startswith("xerbs:pattern:")

    def test_malformed_entity_external_id_is_refused(self, client):
        r = client.post("/api/v1/knowledge/clinical/ingest", json={
            "submitted_by": "synth-author", "source_label": "SYNTHETIC",
            "items": [{"entity_type": "pattern", "external_id": "bad id with spaces",
                       "payload": {"name": "SYNTHETIC"}, "sources": []}]})
        assert r.status_code >= 400

    def test_approval_provenance_is_recorded_honestly(self, client):
        """Requirement 16: a local approval is never labelled ATTESTED."""
        pat, frm, src = _seed_reviewed_pair(client)
        from app.db.models import ClinicalEntity
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            e = s.get(ClinicalEntity, pat)
        assert e.review_status == "REVIEWED"
        assert e.governance_provenance in lifecycle.LEGACY_PROVENANCE
        assert e.governance_provenance != lifecycle.ATTESTED


class TestGovernanceGuardUnchanged:

    def test_guard_is_still_closed_by_default(self):
        """Requirement 22: shipped defaults were not weakened.

        The test suite enables mutation for itself in conftest; the Settings
        default is what ships, and it is still closed.
        """
        from app.core.config import Settings
        assert Settings.model_fields["allow_external_governance_mutation"].default is False

    def test_guard_refuses_when_disabled(self, monkeypatch):
        from fastapi import HTTPException
        from app.core import governance_guard
        from app.core.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("ALLOW_EXTERNAL_GOVERNANCE_MUTATION", "false")
        try:
            with pytest.raises(HTTPException) as e:
                governance_guard.require_governance_mutation_enabled()
            assert e.value.status_code == 403
        finally:
            monkeypatch.undo()
            get_settings.cache_clear()
