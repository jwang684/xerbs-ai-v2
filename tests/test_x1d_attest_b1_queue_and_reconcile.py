"""X1D-CORE-ATTEST-B1：ai-v2 侧的待审队列、证据检视与对账入口。

本阶段在 ai-v2 只加了三样东西，而且只在核心确实读不到证据时才加：

  GET  /api/v1/governance/pending-review                    待审队列
  GET  /api/v1/governance/pending-review/{type}/{id}        证据与溯源
  POST /api/v1/governance/reconcile                         对账触发

前两个是只读的；第三个调用的是 X1D-AIV2-GOVCLOSURE1 已经实现、却一直没有
任何调用方的 ``reconcile_attested_approvals``——不是第二套对账算法。

这一套要钉住的是：队列不撒谎（不可审的对象带着原因出现，而不是被藏起来），
证据够人做判断（缺什么明说缺什么），以及对账既不批准任何东西，也不会在
core 不可达时乱猜。
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.services.governance import identity, lifecycle
from app.services.governance.attested_review import (AttestedReviewService,
                                                     AttestedReviewError)


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


#: 这套 fixture 写进去的适应证必须是任何真实检索都匹配不到的记号。
#:
#: 套件共用同一个 sqlite 文件，行会一直留在里面。本文件里有几个用例会把
#: 实体走完批准流程，于是它就成了一条"已审核 + 有已审核来源"的语料——
#: 只要适应证写的是"发热"，它就会真的出现在推荐候选里，把
#: test_phase2/test_phase3 那两条"候选必须为空"的既有断言打挂。
#: 那不是产品缺陷，是测试数据串味；用一个不可能被匹配到的记号即可。
SYNTHETIC_INDICATION = "B1-SYNTHETIC-INDICATION-DO-NOT-MATCH"


def _ingest_entity(client, name_suffix=""):
    """一个带证据、处于 DRAFT 的 formula。"""
    src = "src-b1-" + uuid.uuid4().hex[:8]
    body = {
        "submitted_by": "synth-author", "source_label": "B1 fixture",
        "items": [{
            "entity_type": "formula",
            "external_id": "xerbs-core:canonical-formula:B1%s-%s" % (
                name_suffix, uuid.uuid4().hex[:6]),
            "payload": {"name": "B1-SYNTHETIC-FORMULA%s-%s" % (
                            name_suffix, uuid.uuid4().hex[:6]),
                        "ingredients": ["B1-SYNTHETIC-HERB"],
                        "indications": [SYNTHETIC_INDICATION]},
            "sources": [{"source_id": src, "title": "B1 source",
                         "citation": "c", "url": "https://example.invalid/b1",
                         "source_type": "GUIDELINE"}],
        }],
    }
    r = client.post("/api/v1/knowledge/clinical/ingest", json=body)
    assert r.status_code in (200, 201), r.text
    return r.json()["created_entity_ids"][0], src


# ======================================================================
# 队列
# ======================================================================
class TestPendingQueue:

    def test_queue_lists_a_draft_entity_and_says_why_it_is_not_attestable(self, client):
        eid, _src = _ingest_entity(client)
        r = client.get("/api/v1/governance/pending-review")
        assert r.status_code == 200, r.text
        body = r.json()
        mine = [i for i in body["items"] if i["object_id"] == eid]
        assert len(mine) == 1
        item = mine[0]
        assert item["object_type"] == "CLINICAL_ENTITY"
        assert item["review_status"] == "DRAFT"
        # DRAFT 还没被提交给人审，所以此刻不可背书——但必须说清楚原因。
        assert item["attestable"] is False
        assert item["ineligible_code"] == "OBJECT_NOT_IN_REVIEW"

    def test_submitted_object_becomes_attestable(self, client):
        eid, _src = _ingest_entity(client)
        from app.schemas.clinical_workflow import ClinicalEntityType
        from app.services.knowledge.persistent_clinical import PersistentClinicalStore
        PersistentClinicalStore().submit_for_review(
            ClinicalEntityType("formula"), eid, "synth-author")
        item = [i for i in client.get(
            "/api/v1/governance/pending-review").json()["items"]
            if i["object_id"] == eid][0]
        assert item["review_status"] == "IN_REVIEW"
        assert item["attestable"] is True
        assert item["ineligible_code"] is None
        assert item["content_hash"] and len(item["content_hash"]) == 64
        assert item["semantic_object_id"]

    def test_queue_never_lists_a_reviewed_object(self, client):
        """已批准的对象不是"待审"。"""
        from tests.governed_fixtures import submit_and_attest_entity
        eid, _src = _ingest_entity(client)
        from tests.governed_fixtures import drive_source_to_reviewed
        drive_source_to_reviewed(client, _src)
        submit_and_attest_entity("formula", eid)
        ids = [i["object_id"] for i in client.get(
            "/api/v1/governance/pending-review").json()["items"]]
        assert eid not in ids

    def test_legacy_row_without_semantic_identity_is_shown_as_ineligible(self, client):
        """§21/§27：没有语义标识的历史行出现在队列里，但不可审。

        它必须出现——隐藏它会让队列看起来是完整工作清单；它又必须不可审——
        没有语义标识的对象无法成为跨服务背书的主体。
        """
        from app.db.models import ClinicalEntity
        from app.db.session import get_session_factory
        legacy_id = "legacy-b1-" + uuid.uuid4().hex[:8]
        with get_session_factory().begin() as s:
            s.add(ClinicalEntity(id=legacy_id, entity_type="formula",
                                 name="遗留方", review_status="IN_REVIEW",
                                 current_version=1,
                                 migration_origin="LEGACY_STATIC_DATA_FIXTURE"))
        item = [i for i in client.get(
            "/api/v1/governance/pending-review").json()["items"]
            if i["object_id"] == legacy_id][0]
        assert item["attestable"] is False
        assert item["ineligible_code"] == "SEMANTIC_IDENTITY_MISSING"

    def test_queue_can_be_filtered_by_type(self, client):
        _ingest_entity(client)
        body = client.get("/api/v1/governance/pending-review",
                          params={"object_type": "SAFETY_RULE"}).json()
        assert all(i["object_type"] == "SAFETY_RULE" for i in body["items"])

    def test_unknown_object_type_is_refused(self, client):
        r = client.get("/api/v1/governance/pending-review",
                       params={"object_type": "NOT_A_TYPE"})
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "UNKNOWN_OBJECT_TYPE"


# ======================================================================
# 证据检视
# ======================================================================
class TestBindingDetail:

    def test_detail_carries_everything_needed_to_decide(self, client):
        eid, src = _ingest_entity(client)
        r = client.get("/api/v1/governance/pending-review/CLINICAL_ENTITY/%s" % eid)
        assert r.status_code == 200, r.text
        d = r.json()
        # 内容、身份、版本、两个哈希
        assert d["content"]["name"].startswith("B1-SYNTHETIC-FORMULA")
        assert d["semantic_object_id"].startswith("xerbs-core:canonical-formula:")
        assert d["object_version"] >= 1
        assert len(d["content_hash"]) == 64
        assert len(d["evidence_hash"]) == 64
        # 证据，连同它自己的审核状态
        assert [e["source_id"] for e in d["evidence"]] == [src]
        assert d["evidence"][0]["review_status"] == "DRAFT"
        assert d["evidence"][0]["citation"] == "c"
        assert d["reviewed_evidence_count"] == 0
        # 三个主体，以及各自是人还是机器
        assert d["author_kind"] in ("MACHINE", "UNKNOWN")
        assert d["submitter_kind"] in ("MACHINE", "UNKNOWN")
        # 治理历史
        assert any(h["action"] == "INGESTED_AS_DRAFT" for h in d["history"])

    def test_detail_reports_missing_provenance_explicitly(self, client):
        """缺什么就说缺什么，不要用空串冒充"没有特别的人"。"""
        eid, _src = _ingest_entity(client)
        d = client.get(
            "/api/v1/governance/pending-review/CLINICAL_ENTITY/%s" % eid).json()
        assert isinstance(d["missing_provenance"], list)
        # 尚未提交，所以 submitter 必然缺位
        assert "submitter_subject" in d["missing_provenance"] or \
               d["submitter_kind"] == "UNKNOWN"

    def test_history_never_labels_an_automation_actor_as_human(self, client):
        """§22：自动化主体绝不能显示成人工审核。"""
        eid, _src = _ingest_entity(client)
        d = client.get(
            "/api/v1/governance/pending-review/CLINICAL_ENTITY/%s" % eid).json()
        for h in d["history"]:
            assert h["actor_kind"] != "HUMAN", h
            # 原始记录值保留下来，便于核对
            assert "actor_recorded_as" in h

    def test_detail_of_a_missing_object_is_404(self, client):
        r = client.get("/api/v1/governance/pending-review/CLINICAL_ENTITY/nope")
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "OBJECT_NOT_FOUND"

    def test_detail_is_read_only(self, client):
        """检视不得改变任何状态。"""
        eid, _src = _ingest_entity(client)
        path = "/api/v1/governance/pending-review/CLINICAL_ENTITY/%s" % eid
        before = client.get(path).json()
        for _ in range(3):
            client.get(path)
        after = client.get(path).json()
        assert before["object_version"] == after["object_version"]
        assert before["review_status"] == after["review_status"]
        assert before["content_hash"] == after["content_hash"]
        assert len(before["history"]) == len(after["history"])


# ======================================================================
# 主体分类
# ======================================================================
class TestSubjectClassification:

    def test_only_core_admin_subjects_are_human(self):
        assert identity.is_human_subject("xerbs-core:admin:7") is True
        assert identity.classify_subject("xerbs-core:admin:7") == "HUMAN"
        for machine in (identity.BOOTSTRAP, identity.CORPUS_AUTHORING,
                        identity.MIGRATION, identity.SERVICE_CORE):
            assert identity.is_human_subject(machine) is False
            assert identity.classify_subject(machine) == "MACHINE"

    def test_unknown_is_not_folded_into_machine(self):
        """"机器干的"和"没人记录是谁干的"是两件事。"""
        assert identity.classify_subject(None) == "UNKNOWN"
        assert identity.classify_subject(identity.LEGACY_UNKNOWN) == "UNKNOWN"

    def test_an_unenumerated_actor_is_never_human_by_default(self):
        """phase12c4b-automation 那类主体必须默认不是人。"""
        subject = identity.as_subject("phase12c4b-automation")
        assert identity.classify_subject(subject) == "MACHINE"
        assert identity.is_human_subject(subject) is False


# ======================================================================
# 对账入口
# ======================================================================
class TestReconciliationEntryPoint:

    def test_reconcile_on_an_empty_system_is_a_noop(self, client):
        r = client.post("/api/v1/governance/reconcile", json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["checked"] >= 0
        assert body["revoked"] == 0
        assert "details" in body

    def test_reconcile_echoes_the_correlation_id(self, client):
        r = client.post("/api/v1/governance/reconcile", json={},
                        headers={"X-Correlation-ID": "corr-b1-test"})
        assert r.json()["correlation_id"] == "corr-b1-test"

    def test_reconcile_detects_a_revocation_that_was_never_delivered(self, client):
        """§28.24：错过的吊销通知，由对账补上。"""
        from tests.governed_fixtures import (StubCoreAttestationClient,
                                             attested_approve,
                                             core_attestation_for,
                                             drive_source_to_reviewed)
        eid, src = _ingest_entity(client)
        drive_source_to_reviewed(client, src)
        from app.schemas.clinical_workflow import ClinicalEntityType
        from app.services.knowledge.persistent_clinical import PersistentClinicalStore
        PersistentClinicalStore().submit_for_review(
            ClinicalEntityType("formula"), eid, "synth-author")
        _result, record, _svc = attested_approve("CLINICAL_ENTITY", eid)

        # core 侧已吊销，但通知从未送达 ai-v2。
        revoked = dict(record, is_revoked=True, is_active=False,
                       revoked_at="2026-02-02T00:00:00+00:00",
                       revocation_reason="withdrawn upstream")
        service = AttestedReviewService(
            client=StubCoreAttestationClient(revoked))
        out = service.reconcile_attested_approvals()
        assert out["revoked"] >= 1
        assert any(d["outcome"] == "REVOKED_LOCALLY" for d in out["details"])

        # 资格必须随之消失。
        d = client.get(
            "/api/v1/governance/pending-review/CLINICAL_ENTITY/%s" % eid)
        if d.status_code == 200:
            assert d.json()["has_verified_attested_approval"] is False

    def test_reconcile_is_idempotent(self, client):
        """§28.25：跑第二遍和跑第一遍一样安全。"""
        from tests.governed_fixtures import (StubCoreAttestationClient,
                                             attested_approve,
                                             drive_source_to_reviewed)
        eid, src = _ingest_entity(client)
        drive_source_to_reviewed(client, src)
        from app.schemas.clinical_workflow import ClinicalEntityType
        from app.services.knowledge.persistent_clinical import PersistentClinicalStore
        PersistentClinicalStore().submit_for_review(
            ClinicalEntityType("formula"), eid, "synth-author")
        _r, record, _s = attested_approve("CLINICAL_ENTITY", eid)
        revoked = dict(record, is_revoked=True, is_active=False,
                       revoked_at="2026-02-02T00:00:00+00:00")
        service = AttestedReviewService(
            client=StubCoreAttestationClient(revoked))
        first = service.reconcile_attested_approvals()
        second = service.reconcile_attested_approvals()
        assert first["revoked"] >= 1
        # 第二遍不该再"吊销"一次同一条：它已经不是生效批准了。
        assert second["revoked"] == 0

    def test_reconcile_fails_closed_when_core_is_unreachable(self, client):
        """§19：core 不可达时报告 UNCHECKED，绝不猜。"""
        from app.services.governance.attestation_client import AttestationLookupError
        from tests.governed_fixtures import (StubCoreAttestationClient,
                                             attested_approve,
                                             drive_source_to_reviewed)
        eid, src = _ingest_entity(client)
        drive_source_to_reviewed(client, src)
        from app.schemas.clinical_workflow import ClinicalEntityType
        from app.services.knowledge.persistent_clinical import PersistentClinicalStore
        PersistentClinicalStore().submit_for_review(
            ClinicalEntityType("formula"), eid, "synth-author")
        attested_approve("CLINICAL_ENTITY", eid)

        service = AttestedReviewService(client=StubCoreAttestationClient(
            None, raises=AttestationLookupError(
                "core down", code="ATTESTATION_VERIFICATION_UNAVAILABLE")))
        out = service.reconcile_attested_approvals()
        assert out["unreachable"] >= 1
        assert out["revoked"] == 0
        assert all(d["outcome"] in ("UNCHECKED", "STILL_ACTIVE")
                   for d in out["details"])

    def test_reconcile_never_mints_an_approval(self, client):
        """§19：对账只会移除资格，永远不会授予资格。"""
        import inspect
        src = inspect.getsource(
            AttestedReviewService.reconcile_attested_approvals)
        assert "self.approve(" not in src
        assert "APPROVED_BY_ATTESTATION" not in src.replace(
            "GovernedObjectReviewEvent.action == APPROVED_BY_ATTESTATION", "")

    def test_reconcile_does_not_touch_grandfathering(self):
        """§19/§21：对账不得改动 legacy 豁免。"""
        import inspect
        src = inspect.getsource(
            AttestedReviewService.reconcile_attested_approvals)
        assert "LEGACY_ELIGIBILITY_GRANDFATHERED" not in src
        assert lifecycle.LEGACY_ELIGIBILITY_GRANDFATHERED is True


# ======================================================================
# 未被重新打开的东西
# ======================================================================
class TestNothingReopened:

    def test_the_new_routes_add_no_approval_path(self):
        """新加的三个路由里没有任何一个能批准对象。"""
        import pathlib
        src = pathlib.Path("app/api/governance.py").read_text(encoding="utf-8")
        head = src.split('@router.post("/{object_type}/{object_id}/submit-review")')[0]
        assert "attested_review_service.approve(" not in head

    def test_reconcile_route_creates_no_schedule(self):
        """§20：仓库里不得出现任何调度器。"""
        import pathlib
        import re
        pattern = re.compile(r"(?i)apscheduler|celery|crontab|BackgroundScheduler")
        for f in pathlib.Path("app").rglob("*.py"):
            assert not pattern.search(f.read_text(encoding="utf-8",
                                                  errors="replace")), f

    def test_governance_guard_still_closed_by_default(self):
        """断言的是**出厂默认值**，不是本套件里的实例。

        conftest 为了让 E2E1 之前的存量用例还能跑，把
        ALLOW_EXTERNAL_GOVERNANCE_MUTATION 打开了；实例化一个 Settings 会
        把那个测试环境读进来，于是这条断言会变成在检查 conftest，而不是在
        检查发布出去的默认值。
        """
        from app.core.config import Settings
        field = Settings.model_fields["allow_external_governance_mutation"]
        assert field.default is False

    def test_no_principal_may_human_approve(self):
        assert identity.PRINCIPALS_THAT_MAY_HUMAN_APPROVE == frozenset()

    def test_frozen_migrations_are_untouched(self):
        """§25：0008 / 0009 冻结。本阶段不新增迁移。"""
        import pathlib
        versions = pathlib.Path("migrations/versions")
        assert (versions / "0008_x1d_gov2c1_governed_objects.py").exists()
        assert (versions / "0009_x1d_attest1_entity_attestation.py").exists()
        assert not list(versions.glob("0010_*.py"))
