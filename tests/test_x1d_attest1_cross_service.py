"""X1D-CROSSSERVICE-AI-GOV2-ATTEST1: cross-service attestation verification.

The property under test, stated once: **a service credential proves the caller
is xerbs-core; only a verified core attestation proves a human approved.**

Every negative case below exists because some earlier version of this system
would have said yes to it. The positive case runs the whole chain locally --
DRAFT, submit, attestation, authoritative lookup, recomputation, transition,
durable event -- against synthetic objects only. Nothing here touches a
deployed environment, and no real clinical object is approved.
"""

import uuid

import pytest

from app.services.governance import canonical, identity, lifecycle
from app.services.governance.attestation_client import (AttestationLookupError,
                                                        AttestationNotFound,
                                                        CoreAttestationClient)
from app.services.governance.attested_review import (APPROVED_BY_ATTESTATION,
                                                     AttestedReviewError,
                                                     AttestedReviewService,
                                                     has_verified_attested_approval)
from tests.governed_fixtures import (StubCoreAttestationClient, attested_approve,
                                     core_attestation_for, local_binding_for)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def _governed_pair(client):
    """Two REVIEWED entities and a relationship submitted for review."""
    tag = uuid.uuid4().hex[:8]
    src = "SYNTH-attest-src-%s" % tag
    body = {"submitted_by": "synth-author", "source_label": "SYNTHETIC ATTEST1",
            "items": [
                {"entity_type": "pattern", "external_id": "xerbs:pattern:synth-%s" % tag,
                 "payload": {"name": "SYNTHETIC 证 %s" % tag},
                 "sources": [{"source_id": src, "title": "SYNTHETIC source",
                              "source_type": "GUIDELINE"}]},
                {"entity_type": "formula", "external_id": "xerbs:formula:synth-%s" % tag,
                 "payload": {"name": "SYNTHETIC 方 %s" % tag},
                 "sources": [{"source_id": src, "title": "SYNTHETIC source",
                              "source_type": "GUIDELINE"}]},
            ]}
    ids = client.post("/api/v1/knowledge/clinical/ingest", json=body).json()["created_entity_ids"]
    from tests.governed_fixtures import drive_source_to_reviewed
    drive_source_to_reviewed(client, src)
    from tests.governed_fixtures import submit_and_attest_entity
    for etype, eid in zip(("pattern", "formula"), ids):
        submit_and_attest_entity(etype, eid)

    rel = client.post("/api/v1/safety/relationships", json={
        "source_entity_id": ids[0], "target_entity_id": ids[1],
        "relationship_type": "PATTERN_FORMULA", "source_id": src,
        "actor_id": "synth-author", "actor_role": "CLINICAL_REVIEWER"}).json()
    client.post("/api/v1/governance/CLINICAL_RELATIONSHIP/%s/submit-review" % rel["id"],
                json={"submitted_by": "synth-submitter"})
    return ids[0], ids[1], src, rel["id"]


# ----------------------------------------------------------------------
# The positive chain
# ----------------------------------------------------------------------

class TestEndToEndChain:

    def test_full_chain_draft_to_reviewed(self, client):
        """DRAFT -> IN_REVIEW -> attestation -> verified -> REVIEWED."""
        _pat, _frm, _src, rel = _governed_pair(client)
        result, record, _ = attested_approve("CLINICAL_RELATIONSHIP", rel)

        assert result["review_status"] == lifecycle.REVIEWED
        assert result["review_attestation_id"] == record["attestation_id"]
        assert result["idempotent_replay"] is False

        from app.db.models import ClinicalRelationship
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            row = s.get(ClinicalRelationship, rel)
            assert row.review_status == "REVIEWED"
            assert row.governance_provenance == lifecycle.ATTESTED
            assert row.review_attestation_id == record["attestation_id"]
            assert row.reviewed_at is not None
            # 17: the durable local event, referencing the attestation.
            assert has_verified_attested_approval(
                s, "CLINICAL_RELATIONSHIP", rel, row.version,
                record["attestation_id"]) is True

    def test_review_event_records_the_external_subject_as_external(self, client):
        """13: the reviewer is cached for audit, never as a local identity."""
        from sqlalchemy import select
        from app.db.models import GovernedObjectReviewEvent
        from app.db.session import get_session_factory
        _pat, _frm, _src, rel = _governed_pair(client)
        _result, record, _ = attested_approve("CLINICAL_RELATIONSHIP", rel)
        with get_session_factory()() as s:
            ev = s.scalars(select(GovernedObjectReviewEvent).where(
                GovernedObjectReviewEvent.object_id == rel,
                GovernedObjectReviewEvent.action == APPROVED_BY_ATTESTATION)).one()
        assert ev.attestation_id == record["attestation_id"]
        assert ev.actor_subject == record["reviewer_subject"]
        assert ev.actor_subject.startswith("xerbs-core:admin:")
        assert "asserted by core, not" in ev.notes
        assert "corr-SYNTHETIC" in ev.notes

    def test_eligibility_follows_only_after_evidence_is_reviewed(self, client):
        """Approval is necessary, not sufficient: evidence rules still apply."""
        _pat, frm, _src, rel = _governed_pair(client)
        attested_approve("CLINICAL_RELATIONSHIP", rel)
        from app.services.knowledge.persistent_clinical import PersistentClinicalStore
        got = PersistentClinicalStore().eligible_formula_candidates_for_patterns([_pat])
        assert [g["formula_id"] for g in got] == [frm]


# ----------------------------------------------------------------------
# Negative cases -- the numbered list from the phase brief
# ----------------------------------------------------------------------

class TestNegativeCrossService:

    def _rel(self, client):
        return _governed_pair(client)[3]

    def test_1_service_token_alone_cannot_approve(self, client):
        """The endpoint is service-authenticated, and still refuses."""
        rel = self._rel(client)
        r = client.post("/api/v1/governance/CLINICAL_RELATIONSHIP/%s/attested-review" % rel,
                        json={"attestation_id": "att-does-not-exist"})
        assert r.status_code in (422, 503)
        from app.db.models import ClinicalRelationship
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            assert s.get(ClinicalRelationship, rel).review_status == "IN_REVIEW"

    def test_2_nonexistent_attestation_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel,
                             raises=AttestationNotFound("att-nope"))
        assert e.value.code == "ATTESTATION_NOT_FOUND"

    @pytest.mark.parametrize("decision,code", [
        ("REJECTED", "DECISION_NOT_APPROVED"),
        ("CHANGES_REQUESTED", "DECISION_NOT_APPROVED"),
    ])
    def test_3_4_non_approval_decisions_rejected(self, client, decision, code):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel, decision=decision)
        assert e.value.code == code

    def test_5_revoked_attestation_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel,
                             is_revoked=True, revoked_at="2026-01-02T00:00:00+00:00")
        assert e.value.code == "ATTESTATION_REVOKED"

    def test_5b_superseded_attestation_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel, is_superseded=True,
                             superseded_by_attestation_id="att-SYNTHETIC-TEST-newer")
        assert e.value.code == "ATTESTATION_SUPERSEDED"

    def test_6_wrong_target_environment_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel,
                             target_environment="production")
        assert e.value.code == "TARGET_ENVIRONMENT_MISMATCH"

    def test_7_wrong_target_system_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel,
                             target_system="some-other-service")
        assert e.value.code == "TARGET_SYSTEM_MISMATCH"

    def test_8_wrong_object_type_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel,
                             governance_object_type="SAFETY_RULE")
        assert e.value.code == "OBJECT_TYPE_MISMATCH"

    def test_9_wrong_semantic_object_id_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel,
                             semantic_object_id="rel:someone:else|PATTERN_FORMULA|x")
        assert e.value.code == "SUBJECT_MISMATCH"

    def test_10_wrong_object_version_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel, object_version=99)
        assert e.value.code == "VERSION_MISMATCH"

    def test_11_wrong_content_hash_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel, content_hash="b" * 64)
        assert e.value.code == "CONTENT_HASH_MISMATCH"

    def test_12_wrong_evidence_hash_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel, evidence_hash="c" * 64)
        assert e.value.code == "EVIDENCE_HASH_MISMATCH"

    def test_13_author_mismatch_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel,
                             author_subject="xerbs-ai-v2:principal:someone-else")
        assert e.value.code == "AUTHOR_SUBJECT_MISMATCH"

    def test_14_submitter_mismatch_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel,
                             submitter_subject="xerbs-ai-v2:principal:someone-else")
        assert e.value.code == "SUBMITTER_SUBJECT_MISMATCH"

    def test_15_last_editor_mismatch_rejected(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel,
                             last_material_editor_subject="xerbs-ai-v2:principal:nope")
        assert e.value.code == "EDITOR_SUBJECT_MISMATCH"

    def test_16_stale_attestation_after_content_change_rejected(self, client):
        """TOCTOU: core reviewed one thing, the object became another."""
        _pat, frm, _src, rel = _governed_pair(client)
        record = core_attestation_for("CLINICAL_RELATIONSHIP", rel)

        # Change the object after the attestation was minted.
        from app.db.models import ClinicalEntity
        from app.db.session import get_session_factory
        with get_session_factory().begin() as s:
            s.get(ClinicalEntity, frm).external_id = "xerbs:formula:moved-%s" % uuid.uuid4().hex[:6]

        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel, record=record)
        assert e.value.code in ("SUBJECT_MISMATCH", "CONTENT_HASH_MISMATCH")

    def test_17_stale_attestation_after_evidence_change_rejected(self, client):
        _pat, _frm, src, rel = _governed_pair(client)
        record = core_attestation_for("CLINICAL_RELATIONSHIP", rel)

        from app.db.models import GovernedObjectSource, SourceRegistry
        from app.db.session import get_session_factory
        extra = "SYNTH-extra-%s" % uuid.uuid4().hex[:8]
        with get_session_factory().begin() as s:
            s.add(SourceRegistry(source_id=extra, title="SYNTHETIC extra",
                                 source_type="GUIDELINE", review_status="REVIEWED"))
            s.flush()
            s.add(GovernedObjectSource(object_type="CLINICAL_RELATIONSHIP",
                                       object_id=rel, source_id=extra,
                                       source_version=1, locator=None))

        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel, record=record)
        assert e.value.code == "EVIDENCE_HASH_MISMATCH"

    def test_18_request_body_role_cannot_approve_entity(self, client):
        tag = uuid.uuid4().hex[:8]
        src = "SYNTH-role-src-%s" % tag
        eid = client.post("/api/v1/knowledge/clinical/ingest", json={
            "submitted_by": "synth-author", "source_label": "SYNTHETIC",
            "items": [{"entity_type": "formula",
                       "external_id": "xerbs:formula:role-%s" % tag,
                       "payload": {"name": "SYNTHETIC 方 %s" % tag},
                       "sources": [{"source_id": src, "title": "S",
                                    "source_type": "GUIDELINE"}]}]}).json()["created_entity_ids"][0]
        client.post("/api/v1/knowledge/clinical/entities/formula/%s/submit-review" % eid,
                    json={"submitted_by": "synth-author"})
        for role in ("CLINICAL_REVIEWER", "CLINICAL_ADMIN"):
            r = client.post("/api/v1/knowledge/clinical/entities/formula/%s/review" % eid,
                            json={"reviewer_id": "whoever", "reviewer_role": role,
                                  "decision": "APPROVE"})
            assert r.status_code >= 400, role
            assert "attestation" in r.text.lower()

    def test_19_request_body_role_cannot_approve_relationship(self, client):
        """There is no local approve at all -- from any state, any principal."""
        rel = self._rel(client)
        for state in lifecycle.GOVERNED_STATES:
            with pytest.raises(lifecycle.HumanAttestationRequired):
                lifecycle.next_state(state, "APPROVE")

    def test_20_request_body_role_cannot_approve_safety_rule(self, client):
        _pat, frm, src, _rel = _governed_pair(client)
        rule = client.post("/api/v1/safety/rules", json={
            "target_entity_id": frm, "rule_type": "PREGNANCY", "trigger_term": "孕",
            "severity": "CRITICAL", "action": "BLOCK", "message": "SYNTHETIC",
            "source_id": src, "actor_id": "synth-author",
            "actor_role": "CLINICAL_ADMIN"}).json()
        assert rule["review_status"] == "DRAFT"
        assert rule["clinical_ranking_eligible"] is False

    def test_21_machine_principal_cannot_approve(self, client):
        assert identity.PRINCIPALS_THAT_MAY_HUMAN_APPROVE == frozenset()
        for principal in identity.MACHINE_PRINCIPALS:
            assert ":admin:" not in principal

    def test_22_arbitrary_attestation_id_confers_no_eligibility(self, client):
        """§18: the column is not the proof; the event is."""
        _pat, frm, _src, rel = _governed_pair(client)
        from app.db.models import ClinicalRelationship
        from app.db.session import get_session_factory
        with get_session_factory().begin() as s:
            row = s.get(ClinicalRelationship, rel)
            row.review_status = "REVIEWED"
            row.governance_provenance = lifecycle.ATTESTED
            row.review_attestation_id = "att-I-JUST-MADE-THIS-UP"

        from app.services.knowledge.persistent_clinical import PersistentClinicalStore
        assert PersistentClinicalStore().eligible_formula_candidates_for_patterns(
            [_pat]) == []

    def test_23_replay_is_idempotent(self, client):
        rel = self._rel(client)
        first, record, _ = attested_approve("CLINICAL_RELATIONSHIP", rel)
        second, _r2, _ = attested_approve("CLINICAL_RELATIONSHIP", rel, record=record)
        assert first["idempotent_replay"] is False
        assert second["idempotent_replay"] is True
        assert second["review_attestation_id"] == first["review_attestation_id"]

        from sqlalchemy import select
        from app.db.models import GovernedObjectReviewEvent
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            events = s.scalars(select(GovernedObjectReviewEvent).where(
                GovernedObjectReviewEvent.object_id == rel,
                GovernedObjectReviewEvent.action == APPROVED_BY_ATTESTATION)).all()
        assert len(events) == 1, "replay must not duplicate the event"

    def test_24_second_conflicting_approval_rejected(self, client):
        rel = self._rel(client)
        attested_approve("CLINICAL_RELATIONSHIP", rel)
        other = core_attestation_for("CLINICAL_RELATIONSHIP", rel)
        other["attestation_id"] = "att-SYNTHETIC-TEST-other"
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel, record=other)
        assert e.value.code == "ALREADY_APPROVED_BY_DIFFERENT_ATTESTATION"

    def test_25_core_unavailable_fails_closed(self, client):
        rel = self._rel(client)
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel,
                             raises=AttestationLookupError(
                                 "core is down",
                                 code="ATTESTATION_VERIFICATION_UNAVAILABLE"))
        assert e.value.code == "ATTESTATION_VERIFICATION_UNAVAILABLE"
        from app.db.models import ClinicalRelationship
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            assert s.get(ClinicalRelationship, rel).review_status == "IN_REVIEW"

    def test_26_malformed_core_response_fails_closed(self, client):
        rel = self._rel(client)
        record = core_attestation_for("CLINICAL_RELATIONSHIP", rel)
        del record["is_active"]
        service = AttestedReviewService(
            client=StubCoreAttestationClient(record))
        # The stub returns it verbatim; the authoritative-state check refuses
        # because approval is never inferred from a row merely existing.
        with pytest.raises(AttestedReviewError) as e:
            service.approve(object_type="CLINICAL_RELATIONSHIP", object_id=rel,
                            attestation_id=record["attestation_id"])
        assert e.value.code == "ATTESTATION_NOT_ACTIVE"

    def test_26b_real_client_refuses_a_malformed_body(self):
        """The shape check lives in the client, before anything trusts it."""
        import httpx
        c = CoreAttestationClient(base_url="http://core.invalid",
                                  token="unused-in-this-test")

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"attestation_id": "att-1"}     # missing everything else

        original = httpx.get
        httpx.get = lambda *a, **k: _Resp()
        try:
            with pytest.raises(AttestationLookupError) as e:
                c.fetch("att-1")
        finally:
            httpx.get = original
        assert e.value.code == "ATTESTATION_RESPONSE_MALFORMED"

    def test_25b_unconfigured_lookup_fails_closed(self):
        c = CoreAttestationClient(base_url=None, token=None)
        assert c.configured() is False
        with pytest.raises(AttestationLookupError) as e:
            c.fetch("att-anything")
        assert e.value.code == "ATTESTATION_VERIFICATION_UNAVAILABLE"


# ----------------------------------------------------------------------
# 27 / 28: cross-repo contracts
# ----------------------------------------------------------------------

class TestCrossRepoContracts:
    """The subject grammar and hash spec are duplicated by hand in two repos.

    A pinned corpus is what keeps them from drifting apart silently. Both
    repositories run the same cases against their own implementation.
    """

    ACCEPTED = (
        "xerbs-core:admin:42",
        "xerbs-core:service:xerbs-core",
        "xerbs-ai-v2:principal:bootstrap",
        "xerbs-ai-v2:principal:corpus-authoring",
        "xerbs-ai-v2:principal:legacy-unknown",
        "xerbs-ai-v2:principal:migration",
        "xerbs-ai-v2:service:internal",
        "xerbs-core:admin:1",
    )
    REJECTED = (
        "42",                               # bare id -- the collision case
        "xerbs-core:42",                    # missing kind
        "unknown-system:principal:x",       # unknown namespace
        "xerbs-ai-v2:human:x",              # unknown kind
        "xerbs-ai-v2:principal:",           # empty id
        "xerbs-ai-v2:principal:-leading",   # must start alphanumeric
        "xerbs-ai-v2:principal:has space",  # whitespace
        "xerbs-ai-v2:principal:pipe|x",     # delimiter
        "xerbs-ai-v2:principal:tab\there",
        "xerbs-ai-v2:principal:ctrl\x01x",
        "xerbs-ai-v2:principal:证型",        # CJK not permitted in a subject
        "",
        "  xerbs-core:admin:42",            # leading whitespace
        "xerbs-core:admin:42 ",             # trailing whitespace
    )

    def test_27_accepted_subject_corpus(self):
        for subject in self.ACCEPTED:
            assert identity.is_valid_subject(subject) is True, subject

    def test_27b_rejected_subject_corpus(self):
        for subject in self.REJECTED:
            assert identity.is_valid_subject(subject) is False, repr(subject)

    def test_28_hash_spec_version_is_pinned(self):
        assert canonical.SPEC_VERSION == "xerbs-canonical-hash/1"

    def test_28b_vectors_file_is_the_contract(self):
        """The literal vectors must exist here for core to copy verbatim."""
        import json
        import pathlib
        p = pathlib.Path(__file__).parent / "vectors" / "canonical_hash_vectors.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert doc["spec_version"] == canonical.SPEC_VERSION
        assert len(doc["vectors"]) >= 13
        for v in doc["vectors"]:
            assert canonical.digest(v["subject"]) == v["sha256"], v["name"]


# ----------------------------------------------------------------------
# Object-type coverage
# ----------------------------------------------------------------------

class TestAllObjectTypes:

    def test_every_governed_type_has_an_adapter(self):
        from app.services.governance.attested_review import ADAPTERS, OBJECT_TYPES
        assert set(ADAPTERS) == set(OBJECT_TYPES)
        assert set(OBJECT_TYPES) == {"SOURCE", "CLINICAL_ENTITY",
                                     "CLINICAL_RELATIONSHIP", "SAFETY_RULE"}

    def test_source_binding_uses_its_real_integer_version(self, client):
        """§13: no faked version. SourceRegistry.version is genuine."""
        _pat, _frm, src, _rel = _governed_pair(client)
        binding = local_binding_for("SOURCE", src)
        assert binding.semantic_object_id == src
        assert isinstance(binding.object_version, int)
        assert binding.object_version >= 1
        # A Source is its own evidence: no separate evidence set to bind.
        assert binding.evidence_hash is None

    def test_safety_rule_can_be_attested(self, client):
        _pat, frm, src, _rel = _governed_pair(client)
        rule = client.post("/api/v1/safety/rules", json={
            "target_entity_id": frm, "rule_type": "PREGNANCY", "trigger_term": "孕",
            "severity": "CRITICAL", "action": "BLOCK", "message": "SYNTHETIC 孕期禁用",
            "source_id": src, "actor_id": "synth-author",
            "actor_role": "CLINICAL_REVIEWER"}).json()
        client.post("/api/v1/governance/SAFETY_RULE/%s/submit-review" % rule["id"],
                    json={"submitted_by": "synth-submitter"})
        result, _record, _ = attested_approve("SAFETY_RULE", rule["id"])
        assert result["review_status"] == "REVIEWED"

        screened = client.post("/api/v1/safety/screen", json={
            "formula_id": frm, "formula_name": "SYNTHETIC",
            "patient_context": {"pregnancy": True}})
        assert screened.json()["eligible_for_selection"] is False
