"""X1D-AIV2-GOVCLOSURE1: attested source review, revocation, reconciliation.

Three properties, all of which some earlier version of this system got wrong:

  * a Source becoming REVIEWED is what makes everything downstream
    ranking-eligible, so it is a human clinical decision like any other and
    can no longer be done by naming a role in a request body;
  * an approval xerbs-core has withdrawn must stop being an approval here --
    ATTEST1 left that gap unbounded and said so;
  * when the withdrawal notification is lost, reconciliation closes it, and
    the human decision is never unwound because a network call failed.

Everything is synthetic. Nothing here approves a real clinical object.
"""

import uuid

import pytest

from app.services.governance import identity, lifecycle
from app.services.governance.attestation_client import AttestationLookupError
from app.services.governance.attested_review import (APPROVED_BY_ATTESTATION,
                                                     ATTESTATION_REVOKED,
                                                     AttestedReviewError,
                                                     AttestedReviewService,
                                                     has_verified_attested_approval)
from tests.governed_fixtures import (StubCoreAttestationClient, attested_approve,
                                     core_attestation_for, drive_source_to_reviewed)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def _governed(client):
    """Two REVIEWED entities, a REVIEWED source, and an IN_REVIEW relationship."""
    tag = uuid.uuid4().hex[:8]
    src = "SYNTH-gc-src-%s" % tag
    body = {"submitted_by": "synth-author", "source_label": "SYNTHETIC GOVCLOSURE1",
            "items": [
                {"entity_type": "pattern", "external_id": "xerbs:pattern:gc-%s" % tag,
                 "payload": {"name": "SYNTHETIC 证 %s" % tag},
                 "sources": [{"source_id": src, "title": "SYNTHETIC",
                              "source_type": "GUIDELINE"}]},
                {"entity_type": "formula", "external_id": "xerbs:formula:gc-%s" % tag,
                 "payload": {"name": "SYNTHETIC 方 %s" % tag},
                 "sources": [{"source_id": src, "title": "SYNTHETIC",
                              "source_type": "GUIDELINE"}]},
            ]}
    ids = client.post("/api/v1/knowledge/clinical/ingest", json=body).json()["created_entity_ids"]
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


def _eligible(pattern_id):
    from app.services.knowledge.persistent_clinical import PersistentClinicalStore
    return PersistentClinicalStore().eligible_formula_candidates_for_patterns([pattern_id])


# ----------------------------------------------------------------------
# Source approval closure (27.15 - 27.18)
# ----------------------------------------------------------------------

class TestSourceApprovalClosure:

    def _draft_source(self, client):
        tag = uuid.uuid4().hex[:8]
        src = "SYNTH-close-%s" % tag
        client.post("/api/v1/knowledge/clinical/sources",
                    json={"source_id": src, "title": "SYNTHETIC source",
                          "source_type": "GUIDELINE", "actor_id": "synth-author"})
        cur = client.get("/api/v1/knowledge/clinical/sources/%s" % src).json()
        client.post("/api/v1/knowledge/clinical/sources/%s/submit-review" % src,
                    json={"submitted_by": "synth-author",
                          "expected_version": cur["version"]})
        return src

    @pytest.mark.parametrize("role", ["CLINICAL_REVIEWER", "CLINICAL_ADMIN"])
    def test_15_16_request_body_role_cannot_approve_a_source(self, client, role):
        src = self._draft_source(client)
        v = client.get("/api/v1/knowledge/clinical/sources/%s" % src).json()["version"]
        r = client.post("/api/v1/knowledge/clinical/sources/%s/review" % src,
                        json={"reviewer_id": "whoever", "reviewer_role": role,
                              "decision": "APPROVE", "expected_version": v})
        assert r.status_code >= 400, role
        assert "attestation" in r.text.lower()
        assert client.get("/api/v1/knowledge/clinical/sources/%s" % src
                          ).json()["review_status"] == "IN_REVIEW"

    def test_17_valid_attestation_approves_the_exact_source(self, client):
        src = self._draft_source(client)
        result, record, _ = attested_approve("SOURCE", src)
        assert result["review_status"] == "REVIEWED"
        got = client.get("/api/v1/knowledge/clinical/sources/%s" % src).json()
        assert got["review_status"] == "REVIEWED"
        assert got["reviewed_by"] == record["reviewer_subject"]
        assert got["reviewed_by"].startswith("xerbs-core:admin:")

    def test_17b_source_approval_keeps_its_own_governance_trail(self, client):
        src = self._draft_source(client)
        attested_approve("SOURCE", src)
        events = client.get("/api/v1/knowledge/clinical/sources/%s/reviews" % src).json()
        actions = [e["action"] for e in events["results"]]
        assert actions == ["CREATED", "SUBMITTED_FOR_REVIEW", "APPROVED"], actions

    def test_18_stale_source_attestation_is_rejected(self, client):
        """Content changed after the decision -- the digest no longer matches."""
        src = self._draft_source(client)
        record = core_attestation_for("SOURCE", src)
        from app.db.models import SourceRegistry
        from app.db.session import get_session_factory
        with get_session_factory().begin() as s:
            s.get(SourceRegistry, src).title = "SYNTHETIC retitled"
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("SOURCE", src, record=record)
        assert e.value.code == "CONTENT_HASH_MISMATCH"

    def test_18b_rejection_and_change_requests_remain_local(self, client):
        """Making it harder to WITHHOLD approval would be backwards."""
        for decision, expected in (("REJECT", "REJECTED"),
                                   ("REQUEST_CHANGES", "DRAFT")):
            src = self._draft_source(client)
            v = client.get("/api/v1/knowledge/clinical/sources/%s" % src).json()["version"]
            r = client.post("/api/v1/knowledge/clinical/sources/%s/review" % src,
                            json={"reviewer_id": "synth", "reviewer_role": "CLINICAL_REVIEWER",
                                  "decision": decision, "expected_version": v})
            assert r.status_code == 200, r.text
            assert r.json()["review_status"] == expected


# ----------------------------------------------------------------------
# The other three object types still work (27.19 - 27.21)
# ----------------------------------------------------------------------

class TestOtherObjectTypesUnaffected:

    def test_19_relationship_attestation_still_supported(self, client):
        _pat, _frm, _src, rel = _governed(client)
        result, _r, _ = attested_approve("CLINICAL_RELATIONSHIP", rel)
        assert result["review_status"] == "REVIEWED"

    def test_20_safety_rule_attestation_still_supported(self, client):
        _pat, frm, src, _rel = _governed(client)
        rule = client.post("/api/v1/safety/rules", json={
            "target_entity_id": frm, "rule_type": "PREGNANCY", "trigger_term": "孕",
            "severity": "CRITICAL", "action": "BLOCK", "message": "SYNTHETIC",
            "source_id": src, "actor_id": "synth-author",
            "actor_role": "CLINICAL_REVIEWER"}).json()
        client.post("/api/v1/governance/SAFETY_RULE/%s/submit-review" % rule["id"],
                    json={"submitted_by": "synth-submitter"})
        result, _r, _ = attested_approve("SAFETY_RULE", rule["id"])
        assert result["review_status"] == "REVIEWED"

    def test_21_entity_attestation_still_supported(self, client):
        pat, _frm, _src, _rel = _governed(client)
        from app.db.models import ClinicalEntity
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            assert s.get(ClinicalEntity, pat).review_status == "REVIEWED"


# ----------------------------------------------------------------------
# Revocation (27.22 - 27.28)
# ----------------------------------------------------------------------

def _revoked_record(record):
    r = dict(record)
    r.update(is_revoked=True, is_active=False,
             revoked_at="2026-01-02T00:00:00+00:00",
             revocation_reason="SYNTHETIC withdrawal")
    return r


class TestRevocation:

    def _approved(self, client):
        _pat, frm, _src, rel = _governed(client)
        _result, record, _ = attested_approve("CLINICAL_RELATIONSHIP", rel)
        return _pat, frm, rel, record

    def test_22_verified_revocation_removes_eligibility(self, client):
        pat, _frm, rel, record = self._approved(client)
        assert _eligible(pat), "precondition: approved and eligible"

        service = AttestedReviewService(
            client=StubCoreAttestationClient(_revoked_record(record)))
        out = service.revoke(attestation_id=record["attestation_id"])
        assert out["review_status"] == lifecycle.IN_REVIEW
        assert _eligible(pat) == []

    def test_22b_safety_rule_effectiveness_stops_too(self, client):
        _pat, frm, src, _rel = _governed(client)
        rule = client.post("/api/v1/safety/rules", json={
            "target_entity_id": frm, "rule_type": "PREGNANCY", "trigger_term": "孕",
            "severity": "CRITICAL", "action": "BLOCK", "message": "SYNTHETIC",
            "source_id": src, "actor_id": "synth-author",
            "actor_role": "CLINICAL_REVIEWER"}).json()
        client.post("/api/v1/governance/SAFETY_RULE/%s/submit-review" % rule["id"],
                    json={"submitted_by": "synth-submitter"})
        _res, record, _ = attested_approve("SAFETY_RULE", rule["id"])

        payload = {"formula_id": frm, "formula_name": "SYNTHETIC",
                   "patient_context": {"pregnancy": True}}
        assert client.post("/api/v1/safety/screen", json=payload
                           ).json()["eligible_for_selection"] is False

        AttestedReviewService(client=StubCoreAttestationClient(
            _revoked_record(record))).revoke(attestation_id=record["attestation_id"])
        assert client.post("/api/v1/safety/screen", json=payload
                           ).json()["eligible_for_selection"] is True

    def test_23_24_both_events_are_retained(self, client):
        """The approval is not deleted. Pretending it never happened would be
        its own falsification."""
        _pat, _frm, rel, record = self._approved(client)
        AttestedReviewService(client=StubCoreAttestationClient(
            _revoked_record(record))).revoke(attestation_id=record["attestation_id"])

        from sqlalchemy import select
        from app.db.models import GovernedObjectReviewEvent
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            events = s.scalars(select(GovernedObjectReviewEvent).where(
                GovernedObjectReviewEvent.object_id == rel)).all()
        actions = {e.action for e in events}
        assert APPROVED_BY_ATTESTATION in actions
        assert ATTESTATION_REVOKED in actions
        revoked = [e for e in events if e.action == ATTESTATION_REVOKED][0]
        assert revoked.attestation_id == record["attestation_id"]
        assert "revoked=True" in revoked.notes

    def test_25_revoked_attestation_cannot_re_approve(self, client):
        _pat, _frm, rel, record = self._approved(client)
        AttestedReviewService(client=StubCoreAttestationClient(
            _revoked_record(record))).revoke(attestation_id=record["attestation_id"])
        # The object is IN_REVIEW again; the old attestation is not usable.
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel,
                             record=_revoked_record(record))
        assert e.value.code in ("ATTESTATION_REVOKED", "ATTESTATION_NOT_ACTIVE")

    def test_25b_re_approving_the_same_version_is_deferred_not_guessed(self, client):
        """A revoked version cannot simply be approved again.

        Two APPROVED_BY_ATTESTATION rows at one version would break the
        per-action uniqueness, and relaxing that would make "which approval is
        live?" ambiguous precisely where ambiguity costs most. So the
        re-review workflow is deferred and this fails closed with a reason
        rather than inventing semantics.
        """
        pat, _frm, rel, record = self._approved(client)
        AttestedReviewService(client=StubCoreAttestationClient(
            _revoked_record(record))).revoke(attestation_id=record["attestation_id"])
        with pytest.raises(AttestedReviewError) as e:
            attested_approve("CLINICAL_RELATIONSHIP", rel)
        assert e.value.code == "RE_APPROVAL_AFTER_REVOCATION_UNSUPPORTED"
        assert "new version" in e.value.message
        assert _eligible(pat) == []

    def test_26_superseded_attestation_confers_no_eligibility(self, client):
        pat, _frm, rel, record = self._approved(client)
        superseded = dict(record, is_superseded=True, is_active=False,
                          superseded_by_attestation_id="att-SYNTHETIC-TEST-newer")
        AttestedReviewService(client=StubCoreAttestationClient(
            superseded)).revoke(attestation_id=record["attestation_id"])
        assert _eligible(pat) == []

    def test_26b_supersession_is_not_inherited(self, client):
        """B does not approve anything just because it supersedes A."""
        pat, _frm, rel, record = self._approved(client)
        superseded = dict(record, is_superseded=True, is_active=False,
                          superseded_by_attestation_id="att-SYNTHETIC-TEST-B")
        AttestedReviewService(client=StubCoreAttestationClient(
            superseded)).revoke(attestation_id=record["attestation_id"])
        assert _eligible(pat) == []
        from app.db.models import ClinicalRelationship
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            row = s.get(ClinicalRelationship, rel)
        assert row.review_attestation_id is None
        assert row.review_status == "IN_REVIEW"

    def test_27_forged_revoke_without_core_agreement_fails(self, client):
        """core still says active -> refuse. A service token is not authority."""
        pat, _frm, rel, record = self._approved(client)
        service = AttestedReviewService(client=StubCoreAttestationClient(record))
        with pytest.raises(AttestedReviewError) as e:
            service.revoke(attestation_id=record["attestation_id"])
        assert e.value.code == "ATTESTATION_STILL_ACTIVE"
        assert _eligible(pat), "a refused revocation must change nothing"

    def test_27b_revoke_endpoint_refuses_an_unknown_attestation(self, client):
        r = client.post("/api/v1/governance/attested-review/revoke",
                        json={"attestation_id": "att-never-existed"})
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "LOCAL_APPROVAL_NOT_FOUND"

    def test_28_core_unavailable_during_revoke_fails_closed(self, client):
        pat, _frm, _rel, record = self._approved(client)
        service = AttestedReviewService(client=StubCoreAttestationClient(
            None, raises=AttestationLookupError(
                "core down", code="ATTESTATION_VERIFICATION_UNAVAILABLE")))
        with pytest.raises(AttestedReviewError) as e:
            service.revoke(attestation_id=record["attestation_id"])
        assert e.value.code == "ATTESTATION_VERIFICATION_UNAVAILABLE"
        # Fails closed in the sense that matters here: no guess is recorded.
        assert _eligible(pat)

    def test_revocation_is_idempotent(self, client):
        _pat, _frm, rel, record = self._approved(client)
        service = AttestedReviewService(
            client=StubCoreAttestationClient(_revoked_record(record)))
        first = service.revoke(attestation_id=record["attestation_id"])
        second = service.revoke(attestation_id=record["attestation_id"])
        assert first["idempotent_replay"] is False
        assert second["idempotent_replay"] is True

        from sqlalchemy import select
        from app.db.models import GovernedObjectReviewEvent
        from app.db.session import get_session_factory
        with get_session_factory()() as s:
            events = s.scalars(select(GovernedObjectReviewEvent).where(
                GovernedObjectReviewEvent.object_id == rel,
                GovernedObjectReviewEvent.action == ATTESTATION_REVOKED)).all()
        assert len(events) == 1


# ----------------------------------------------------------------------
# Reconciliation (27.29 - 27.31)
# ----------------------------------------------------------------------

class TestReconciliation:

    def test_29_reconciliation_catches_a_missed_notification(self, client):
        """Core revoked, the notification never arrived, the job finds it."""
        pat, _frm, _rel, record = self._approved(client)
        assert _eligible(pat)

        # No revoke call was made -- this is the lost-notification case.
        service = AttestedReviewService(
            client=StubCoreAttestationClient(_revoked_record(record)))
        report = service.reconcile_attested_approvals()
        assert report["revoked"] >= 1
        assert any(d["outcome"] == "REVOKED_LOCALLY" for d in report["details"])
        assert _eligible(pat) == []

    def test_30_reconciliation_is_idempotent(self, client):
        pat, _frm, _rel, record = self._approved(client)
        service = AttestedReviewService(
            client=StubCoreAttestationClient(_revoked_record(record)))
        first = service.reconcile_attested_approvals()
        second = service.reconcile_attested_approvals()
        assert first["revoked"] >= 1
        # Already revoked locally, so it is no longer a live approval to check.
        assert second["revoked"] == 0
        assert _eligible(pat) == []

    def test_30b_still_active_approvals_are_left_alone(self, client):
        pat, _frm, _rel, record = self._approved(client)
        service = AttestedReviewService(client=StubCoreAttestationClient(record))
        report = service.reconcile_attested_approvals()
        assert report["revoked"] == 0
        assert any(d["outcome"] == "STILL_ACTIVE" for d in report["details"])
        assert _eligible(pat)

    def test_30c_unreachable_core_is_reported_not_assumed(self, client):
        pat, _frm, _rel, _record = self._approved(client)
        service = AttestedReviewService(client=StubCoreAttestationClient(
            None, raises=AttestationLookupError(
                "down", code="ATTESTATION_VERIFICATION_UNAVAILABLE")))
        report = service.reconcile_attested_approvals()
        assert report["unreachable"] >= 1
        assert report["revoked"] == 0
        assert any(d["outcome"] == "UNCHECKED" for d in report["details"])
        # Availability must not silently revoke, and must not silently pass.
        assert _eligible(pat)

    def test_31_normal_ranking_does_not_call_core(self, client):
        """Retrieval must not acquire a synchronous dependency on core."""
        pat, _frm, _rel, _record = self._approved(client)
        stub = StubCoreAttestationClient(None, raises=AttestationLookupError(
            "core must not be consulted here",
            code="ATTESTATION_VERIFICATION_UNAVAILABLE"))
        import app.services.governance.attested_review as ar
        original = ar.CoreAttestationClient
        ar.CoreAttestationClient = lambda *a, **k: stub
        try:
            assert _eligible(pat), "ranking used locally verified state"
            payload = {"formula_id": _frm, "formula_name": "SYNTHETIC",
                       "patient_context": {}}
            assert client.post("/api/v1/safety/screen", json=payload).status_code == 200
        finally:
            ar.CoreAttestationClient = original
        assert stub.calls == [], "ranking must not have contacted core"

    def test_reconciliation_is_a_job_entry_point_not_a_loop(self):
        """No polling and no schedule ships in this repository."""
        import inspect
        src = inspect.getsource(AttestedReviewService.reconcile_attested_approvals)
        # Code only: the docstring legitimately says the word "schedule"
        # while explaining that nothing here schedules anything.
        code = chr(10).join(l for l in src.splitlines()
                            if not l.strip().startswith("#"))
        code = code.split('"""')[0] + "".join(code.split('"""')[2:])
        for forbidden in ("while True", "sleep(", "cron", "Thread(",
                          "BackgroundTasks", "apscheduler"):
            assert forbidden not in code, forbidden

    _approved = TestRevocation._approved


# ----------------------------------------------------------------------
# Tampering, legacy, guard (27.32 - 27.34)
# ----------------------------------------------------------------------

class TestTamperingAndInvariants:

    def test_32_restoring_the_columns_by_hand_cannot_restore_eligibility(self, client):
        """§22: the revocation event wins over hand-set columns."""
        _pat, _frm, _src, rel = _governed(client)
        _result, record, _ = attested_approve("CLINICAL_RELATIONSHIP", rel)
        pat = _pat
        AttestedReviewService(client=StubCoreAttestationClient(
            _revoked_record(record))).revoke(attestation_id=record["attestation_id"])
        assert _eligible(pat) == []

        from app.db.models import ClinicalRelationship
        from app.db.session import get_session_factory
        with get_session_factory().begin() as s:
            row = s.get(ClinicalRelationship, rel)
            row.review_status = "REVIEWED"
            row.governance_provenance = lifecycle.ATTESTED
            row.review_attestation_id = record["attestation_id"]
        assert _eligible(pat) == [], (
            "hand-restored columns must not resurrect a withdrawn approval")

        with get_session_factory()() as s:
            row = s.get(ClinicalRelationship, rel)
            assert has_verified_attested_approval(
                s, "CLINICAL_RELATIONSHIP", rel, row.version,
                record["attestation_id"]) is False

    def test_32b_deleting_the_revocation_event_is_the_only_way_back(self, client):
        """Stated as a limit, not a defence: the event table is the authority.

        Anyone who can delete rows from governed_object_review_event can undo
        this, exactly as anyone who can edit the database can undo anything.
        What matters is that no application path does.
        """
        import pathlib
        app_dir = pathlib.Path(__file__).resolve().parents[1] / "app"
        offenders = []
        for f in app_dir.rglob("*.py"):
            code = "\n".join(l for l in f.read_text(encoding="utf-8").splitlines()
                             if not l.strip().startswith("#"))
            if "GovernedObjectReviewEvent" in code and "delete(" in code:
                offenders.append(f.name)
        assert offenders == [], offenders

    def test_33_legacy_grandfathering_is_unchanged(self):
        assert lifecycle.LEGACY_ELIGIBILITY_GRANDFATHERED is True
        assert lifecycle.LEGACY_PROVENANCE == frozenset({
            lifecycle.LEGACY_SELF_REVIEWED,
            lifecycle.LEGACY_INDEPENDENTLY_REVIEWED,
            lifecycle.LEGACY_UNREVIEWED})

    def test_34_governance_guard_remains_closed_by_default(self):
        from app.core.config import Settings
        assert Settings.model_fields[
            "allow_external_governance_mutation"].default is False

    def test_no_principal_may_approve_or_revoke(self):
        assert identity.PRINCIPALS_THAT_MAY_HUMAN_APPROVE == frozenset()
        for state in lifecycle.GOVERNED_STATES:
            with pytest.raises(lifecycle.HumanAttestationRequired):
                lifecycle.next_state(state, "APPROVE")
