"""X1D-AUDITORDER1：治理事件按因果顺序返回，而不是按一个随机 uuid 排序。

三处历史查询都是这样写的：

    .order_by(X.created_at, X.event_id)

created_at 来自一个大约每 1-2 毫秒才前进一次的时钟，event_id 是
``aud-``/``srcevt-``/``evt-`` 加一段随机 uuid。于是同一个聚合的两次转换只要落在
同一个时钟刻度里，先后就由 uuid 的字典序决定——三个事件时，排对的概率是 1/6。
审计把时间戳钉死之后，12 次里有 11 次是错的。临床治理的审计轨迹可以把
SOURCE_APPROVED 排在 SOURCE_CREATED 前面，而且不留任何痕迹。

因果键本来就有。_transition 先查 VERSION_CONFLICT，再把 row.version 加一，然后
用这个新值给事件盖章，所以 version 就是这条生命线上的位置。review_event 与
source_review_event 早就存了它；audit_event 只把它放在 JSON payload 里。

本阶段做三件事：给 audit_event 加一个可空的 version 列并从 payload 精确回填；
把三处排序改成因果排序；给三个 (聚合, version) 域加唯一约束。

最后一件不是装饰。_transition 的乐观检查读的时候不加锁，审计证明了两个并发写入
方可以各自提交同一个 version。约束把"按约定唯一"变成"数据库拒绝违反"，这也是本
文件里那条并发回归要钉住的东西。

刻意没做：全局序列，或者 MAX(version)+1。前者给一个没人需要的全局单调性引入写
竞争点，后者和它要修的那个读-改-写是同一个形状。
"""
import os
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.db import models  # noqa: F401 - registers the tables
from app.db.base import Base
from app.db.models import AuditEvent, ReviewEvent, SourceReviewEvent

CAUSAL = ["SOURCE_CREATED", "SOURCE_SUBMITTED_FOR_REVIEW", "SOURCE_APPROVED"]
# One frozen instant: exactly the collision a 1-2ms clock produces on its own.
TS = datetime(2026, 9, 19, 12, 0, 0, 123456, tzinfo=timezone.utc)


@pytest.fixture
def factory():
    """A private database per test. Nothing global is touched."""
    tmp = tempfile.mkdtemp().replace("\\", "/")
    engine = create_engine("sqlite:///%s/ao.db" % tmp,
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    try:
        yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    finally:
        engine.dispose()


def aud(source_id, event_type, version, ts=TS):
    return AuditEvent(event_id="aud-%s" % uuid4().hex[:16], event_type=event_type,
                      source_id=source_id, actor_id="curator", payload={},
                      version=version, created_at=ts)


def read_audit(factory, source_id):
    """Exactly the production query, post-change."""
    with factory() as s:
        return [r.event_type for r in s.scalars(
            select(AuditEvent).where(AuditEvent.source_id == source_id)
            .order_by(AuditEvent.version)).all()]


# ======================================================================
# 一、确定性排序：时间戳全部相同也必须对
# ======================================================================

class TestCausalOrderIsDeterministic:

    @pytest.mark.parametrize("trial", range(25))
    def test_identical_timestamps_still_order_causally(self, factory, trial):
        """审计里这个场景 12 次错 11 次。现在必须 25 次全对。"""
        sid = "src-%s" % uuid4().hex[:8]
        with factory.begin() as s:
            for i, et in enumerate(CAUSAL, start=1):
                s.add(aud(sid, et, i))
        assert read_audit(factory, sid) == CAUSAL

    def test_insertion_order_does_not_matter(self, factory):
        """按因果顺序存的是 version，不是写入顺序。"""
        sid = "src-%s" % uuid4().hex[:8]
        with factory.begin() as s:
            for i, et in reversed(list(enumerate(CAUSAL, start=1))):
                s.add(aud(sid, et, i))
        assert read_audit(factory, sid) == CAUSAL

    def test_event_id_no_longer_influences_order(self, factory):
        """以前 'aud-zzz' 会排在 'aud-aaa' 后面，与因果无关。"""
        sid = "src-%s" % uuid4().hex[:8]
        with factory.begin() as s:
            for i, (et, eid) in enumerate(
                    zip(CAUSAL, ["aud-zzzz", "aud-mmmm", "aud-aaaa"]), start=1):
                s.add(AuditEvent(event_id=eid, event_type=et, source_id=sid,
                                 actor_id="curator", payload={}, version=i,
                                 created_at=TS))
        assert read_audit(factory, sid) == CAUSAL

    def test_created_at_no_longer_influences_order(self, factory):
        """时钟倒退或乱序也不再动摇因果顺序。"""
        sid = "src-%s" % uuid4().hex[:8]
        skewed = [TS.replace(microsecond=900000), TS.replace(microsecond=100000),
                  TS.replace(microsecond=500000)]
        with factory.begin() as s:
            for i, (et, ts) in enumerate(zip(CAUSAL, skewed), start=1):
                s.add(aud(sid, et, i, ts=ts))
        assert read_audit(factory, sid) == CAUSAL

    def test_a_long_lifecycle_stays_ordered(self, factory):
        sid = "src-%s" % uuid4().hex[:8]
        with factory.begin() as s:
            for i in range(1, 31):
                s.add(aud(sid, "STEP_%02d" % i, i))
        assert read_audit(factory, sid) == ["STEP_%02d" % i for i in range(1, 31)]

    def test_other_aggregates_do_not_interleave(self, factory):
        a, b = "src-aaa", "src-bbb"
        with factory.begin() as s:
            for i, et in enumerate(CAUSAL, start=1):
                s.add(aud(a, et, i))
                s.add(aud(b, et, i))
        assert read_audit(factory, a) == CAUSAL
        assert read_audit(factory, b) == CAUSAL


# ======================================================================
# 二、并发：两个写入方不能各自提交同一个 (聚合, version)
# ======================================================================

class TestConcurrentWritersCannotShareAVersion:
    """审计里这条在加约束前是会失败的——两边都提交了 v2。

    这不是排序的副作用，是本阶段自己的验收项：_transition 的乐观检查不加锁，
    所以"同一个 version 只能有一条事件"必须由数据库来守。
    """

    def _race(self, factory, table_add, read):
        s1, s2 = factory(), factory()
        results = []
        for sess, tag in ((s1, "A"), (s2, "B")):
            try:
                table_add(sess, tag)
                sess.commit()
                results.append("%s committed" % tag)
            except IntegrityError:
                sess.rollback()
                results.append("%s refused" % tag)
            finally:
                sess.close()
        return results, read()

    def test_two_audit_writers_cannot_both_persist_version_2(self, factory):
        sid = "src-%s" % uuid4().hex[:8]

        def add(sess, tag):
            sess.add(aud(sid, "SOURCE_APPROVED_%s" % tag, 2))

        def versions():
            with factory() as s:
                return [r.version for r in s.scalars(
                    select(AuditEvent).where(AuditEvent.source_id == sid)).all()]

        results, rows = self._race(factory, add, versions)
        assert results.count("A committed") == 1
        assert "B refused" in results, results
        assert rows == [2], rows       # 只有一条活下来

    def test_two_source_review_writers_cannot_share_a_version(self, factory):
        sid = "src-%s" % uuid4().hex[:8]

        def add(sess, tag):
            sess.add(SourceReviewEvent(
                event_id="srcevt-%s" % uuid4().hex[:16], source_id=sid,
                action="APPROVED", actor_id=tag, version=2, created_at=TS))

        results, _ = self._race(factory, add, lambda: None)
        assert results.count("A committed") == 1
        assert "B refused" in results, results

    def test_two_review_writers_cannot_share_a_version(self, factory):
        eid = "ent-%s" % uuid4().hex[:8]

        def add(sess, tag):
            sess.add(ReviewEvent(
                event_id="evt-%s" % uuid4().hex[:16], entity_id=eid,
                entity_type="formula", action="APPROVED", actor_id=tag,
                version=2, created_at=TS))

        results, _ = self._race(factory, add, lambda: None)
        assert results.count("A committed") == 1
        assert "B refused" in results, results

    def test_the_same_version_is_fine_for_different_aggregates(self, factory):
        with factory.begin() as s:
            s.add(aud("src-one", "SOURCE_CREATED", 1))
            s.add(aud("src-two", "SOURCE_CREATED", 1))
        assert read_audit(factory, "src-one") == ["SOURCE_CREATED"]
        assert read_audit(factory, "src-two") == ["SOURCE_CREATED"]


# ======================================================================
# 三、无聚合的事件：刻意不排序，刻意留 NULL
# ======================================================================

class TestUnscopedEventsStayNull:
    """遥测与 gap 事件不属于任何生命线，没有因果位置可言。"""

    def test_many_null_versions_coexist(self, factory):
        """NULL 在 SQLite 与 PostgreSQL 的唯一约束里都互不相等。"""
        with factory.begin() as s:
            for i in range(50):
                s.add(AuditEvent(event_id="aud-%s" % uuid4().hex[:16],
                                 event_type="LLM_PROVIDER_USAGE",
                                 actor_id="system", payload={"n": i},
                                 created_at=TS))
        with factory() as s:
            rows = s.scalars(select(AuditEvent).where(
                AuditEvent.version.is_(None))).all()
        assert len(rows) == 50
        assert all(r.source_id is None and r.entity_id is None for r in rows)

    def test_they_are_unreachable_from_the_ordered_query(self, factory):
        sid = "src-%s" % uuid4().hex[:8]
        with factory.begin() as s:
            s.add(aud(sid, "SOURCE_CREATED", 1))
            s.add(AuditEvent(event_id="aud-%s" % uuid4().hex[:16],
                             event_type="LLM_PROVIDER_USAGE",
                             actor_id="system", payload={}, created_at=TS))
        assert read_audit(factory, sid) == ["SOURCE_CREATED"]


# ======================================================================
# 四、约束与模型契约
# ======================================================================

class TestSchemaContract:

    def test_the_three_constraints_exist(self):
        names = lambda m: {c.name for c in m.__table__.constraints if c.name}
        assert "uq_audit_event_source_version" in names(AuditEvent)
        assert "uq_audit_event_entity_version" in names(AuditEvent)
        assert "uq_source_review_event_source_version" in names(SourceReviewEvent)
        assert "uq_review_event_entity_version" in names(ReviewEvent)

    def test_audit_event_version_is_nullable_and_integer(self):
        col = AuditEvent.__table__.c.version
        assert col.nullable is True
        assert isinstance(col.type, type(ReviewEvent.__table__.c.version.type))

    def test_the_internal_column_is_not_exposed_by_the_api_schema(self):
        from app.schemas.clinical_knowledge import SourceAuditEventRecord
        assert "version" not in SourceAuditEventRecord.model_fields

    def test_no_global_sequence_or_max_plus_one_was_introduced(self):
        import inspect

        from app.services.knowledge import persistent_clinical as pc

        src = inspect.getsource(pc)
        for forbidden in ("max(", "MAX(", "nextval", "autoincrement",
                          "AUTOINCREMENT", "Sequence("):
            assert forbidden not in src, forbidden


# ======================================================================
# 五、读路径仍然只读它该读的
# ======================================================================

class TestReadPathsUnchangedOtherwiseK:

    def test_source_audit_still_filters_by_source(self, factory):
        with factory.begin() as s:
            s.add(aud("src-x", "SOURCE_CREATED", 1))
            s.add(aud("src-y", "SOURCE_CREATED", 1))
        assert read_audit(factory, "src-x") == ["SOURCE_CREATED"]

    def test_review_history_orders_by_version(self, factory):
        sid = "src-%s" % uuid4().hex[:8]
        with factory.begin() as s:
            for i, action in enumerate(
                    ["CREATED", "SUBMITTED_FOR_REVIEW", "APPROVED"], start=1):
                s.add(SourceReviewEvent(
                    event_id="srcevt-%s" % uuid4().hex[:16], source_id=sid,
                    action=action, actor_id="curator", version=i, created_at=TS))
        with factory() as s:
            got = [r.action for r in s.scalars(
                select(SourceReviewEvent)
                .where(SourceReviewEvent.source_id == sid)
                .order_by(SourceReviewEvent.version)).all()]
        assert got == ["CREATED", "SUBMITTED_FOR_REVIEW", "APPROVED"]

    def test_entity_audit_groups_by_entity_then_version(self, factory):
        with factory.begin() as s:
            for eid in ("ent-a", "ent-b"):
                for i, action in enumerate(["CREATED", "APPROVED"], start=1):
                    s.add(ReviewEvent(
                        event_id="evt-%s" % uuid4().hex[:16], entity_id=eid,
                        entity_type="formula", action=action, actor_id="c",
                        version=i, created_at=TS))
        with factory() as s:
            rows = s.scalars(select(ReviewEvent).order_by(
                ReviewEvent.created_at, ReviewEvent.entity_id,
                ReviewEvent.version)).all()
        assert [(r.entity_id, r.action) for r in rows] == [
            ("ent-a", "CREATED"), ("ent-a", "APPROVED"),
            ("ent-b", "CREATED"), ("ent-b", "APPROVED")]
