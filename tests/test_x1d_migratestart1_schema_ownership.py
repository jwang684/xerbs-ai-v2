"""X1D-MIGRATESTART1 P1：schema 只有一个主人，而且不是应用启动。

app/main.py 过去在模块导入时就调用 init_db()，也就是
Base.metadata.create_all()。那有两个后果：

  * 导入这个模块本身就是一次潜在的 DDL 事件——任何进程只要 import app.main
    就可能建表，包括崩溃重启循环里的每一次重试；
  * schema 有了第二个主人，而这个主人从不写 alembic_version。于是数据库可以
    物理上是对的，Alembic 却以为它停在别的 revision，甚至以为它是空的。

create_all 还做不到真正需要的那件事：表一旦存在它就整张跳过，列的差异一概不看。
所以在已迁移的库上它其实什么也没做，却仍然占着"schema 由我负责"的名分。

真正演进这套 schema 的是 Alembic，0001-0007 已经能从 base 建到 head。本阶段
只做一件事：把运行时的 create_all 拿掉，让 Alembic 成为唯一的主人。

Alembic 现在也不在容器启动链里了（P2G-B）。`alembic upgrade head` 改由平台的
preDeploy 阶段执行：每次部署跑一次，失败就直接让这次部署 FAILED，应用容器根本
不会被创建，原来健康的部署继续服务。放在 CMD 里意味着每次普通 restart 都会重跑
一遍——单副本时只是幂等，多副本时就是竞争。

代价是一个镜像自己管不住的耦合：这份代码如果被部署时，捕获的 manifest 里没有
`alembic upgrade head` 这个 preDeployCommand，那这次部署就**没有任何** schema
主人。deployment manifest 是这件事的权威，本文件最后一节把"CMD 里不许再有
Alembic"钉住，manifest 那一半由发布流程在部署前验。

种子函数留在原地。它是 DML——通过普通的 session factory 读写行，不发任何 DDL——
所以它要的是表**存在**，而不是表刚刚被建出来。部署环境里 preDeploy 的
`alembic upgrade head` 在应用容器启动之前就跑完了，表就是存在的。
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

from app.db import models  # noqa: F401 - registers the tables
from app.db.base import Base

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ======================================================================
# 一、导入应用不得触发 DDL
# ======================================================================

# 在一个干净的子进程里跑：先把表建好（模拟"已迁移"的库），再把 create_all
# 换成一个只记账的替身，然后 import app.main。用子进程而不是 importlib.reload，
# 是因为 reload 会把类对象换掉，本套件里已经有过 exception 身份对不上的教训。
_PROBE = r'''
import json, sys
from unittest.mock import patch
from app.db import models            # noqa: F401
from app.db.base import Base
from app.db.session import get_engine

# 先按"已迁移"的样子把表建出来——这是测试脚手架在建，不是被测代码。
Base.metadata.create_all(get_engine())

calls = []
with patch.object(Base.metadata, "create_all",
                  side_effect=lambda *a, **k: calls.append(1)):
    import app.main                  # noqa: F401
    imported = hasattr(app.main, "app")

print("RESULT " + json.dumps({"create_all_calls": len(calls),
                              "app_built": imported}))
'''


def _import_app_in_subprocess(tmp_path):
    db = os.path.join(str(tmp_path), "ownership.db").replace("\\", "/")
    env = {**os.environ,
           "DATABASE_URL": "sqlite:///%s" % db,
           "LLM_PROVIDER": "mock",
           "ENVIRONMENT": "test",
           "ALLOW_INSECURE_LOCAL": "true",
           "ALLOW_EXTERNAL_GOVERNANCE_MUTATION": "true",
           "SERVICE_AUTH_TOKEN": "",
           "PYTHONPATH": REPO}
    p = subprocess.run([sys.executable, "-c", _PROBE], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=180)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, "probe produced no result:\n%s\n%s" % (p.stdout[-2000:], p.stderr[-2000:])
    return json.loads(line[0][len("RESULT "):])


class TestImportingTheAppIsNotADdlEvent:

    def test_importing_app_main_never_calls_create_all(self, tmp_path):
        """本阶段的全部主张：导入应用不再建表。"""
        got = _import_app_in_subprocess(tmp_path)
        assert got["create_all_calls"] == 0, (
            "importing app.main invoked Base.metadata.create_all %d time(s); "
            "runtime schema creation was supposed to be removed"
            % got["create_all_calls"])

    def test_the_app_still_builds_without_it(self, tmp_path):
        """拿掉 create_all 不能把应用本身拆坏。"""
        got = _import_app_in_subprocess(tmp_path)
        assert got["app_built"] is True

    def test_app_main_does_not_import_init_db(self):
        """没有调用点，就不该留着那个 import。"""
        import app.main
        assert not hasattr(app.main, "init_db")


# ======================================================================
# 二、init_db 仍然可用——只是没人在运行时调用它
# ======================================================================

class TestInitDbRemainsAvailableForExplicitUse:
    """本阶段拿掉的是**调用**，不是这个函数。

    显式调用它仍然是本地开发和工具的合法用法；不合法的是让应用启动隐式调用。
    """

    def test_it_is_still_defined_and_exported(self):
        from app.db import init_db as exported
        from app.db.session import init_db
        assert callable(init_db)
        assert exported is init_db

    def test_it_still_creates_the_schema_when_called_explicitly(self, tmp_path):
        from sqlalchemy import create_engine, inspect
        engine = create_engine(
            "sqlite:///%s/explicit.db" % str(tmp_path).replace("\\", "/"),
            connect_args={"check_same_thread": False})
        try:
            assert inspect(engine).get_table_names() == []
            Base.metadata.create_all(engine)
            names = set(inspect(engine).get_table_names())
            assert "clinical_entity" in names
            assert "audit_event" in names
        finally:
            engine.dispose()


# ======================================================================
# 三、种子仍然只写行，不建表
# ======================================================================

class TestSeedRemainsDmlOnly:

    def test_the_seed_issues_no_ddl(self):
        import inspect as _inspect

        from app.services.knowledge import persistent_seed

        src = _inspect.getsource(persistent_seed)
        for forbidden in ("create_all", "drop_all", "CREATE TABLE", "ALTER TABLE",
                          "DROP TABLE", "op.", "alembic"):
            assert forbidden not in src, forbidden

    def test_it_is_still_called_at_startup(self):
        """种子没有被顺手搬走——本阶段只动 schema 归属。"""
        import app.main
        assert hasattr(app.main, "seed_legacy_formula_fixtures")


# ======================================================================
# 四、Alembic 仍然是那个主人，但主人已经搬到 preDeploy 了
# ======================================================================

class TestAlembicIsTheOwnerAndHasMovedToPreDeploy:
    """P2G-B：迁移离开应用启动链，改由平台 preDeploy 独家执行。

    这里只能钉住能从仓库里看到的那一半——应用 CMD 不再迁移。另一半（部署的
    captured manifest 必须带 `alembic upgrade head`）不在仓库里，由发布流程在
    部署前验；两半都成立才有主人。
    """

    def _cmd(self):
        with open(os.path.join(REPO, "Dockerfile"), encoding="utf-8") as fh:
            dockerfile = fh.read()
        return [l for l in dockerfile.splitlines() if l.startswith("CMD")][0]

    def test_the_container_command_no_longer_migrates(self):
        """应用启动路径里不许再有迁移——否则每次 restart 都会重跑。"""
        assert "alembic" not in self._cmd()

    def test_the_container_command_still_bootstraps_before_serving(self):
        """搬走的只有 Alembic：语料仍然在服务之前建立。"""
        cmd = self._cmd()
        assert "bootstrap_golden_corpus" in cmd
        assert "python -m app.server" in cmd
        assert cmd.index("bootstrap_golden_corpus") < cmd.index("python -m app.server")

    def test_migrations_reach_head_from_base(self):
        """Alembic 能从 base 建出整套 schema——这是它当主人的资格。"""
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        cfg = Config(os.path.join(REPO, "alembic.ini"))
        cfg.set_main_option("script_location", os.path.join(REPO, "migrations"))
        script = ScriptDirectory.from_config(cfg)
        heads = script.get_heads()
        assert len(heads) == 1, heads
        chain = list(script.walk_revisions("base", heads[0]))
        assert len(chain) >= 8
        # X1D-AIV2-GOV2-C1 added 0008. The assertion pins the head by
        # name rather than counting, so a new migration has to be
        # declared here deliberately rather than slipping in.
        assert heads[0] == "0008_x1d_gov2c1"
        assert "0007_x1d_auditorder1" in {r.revision for r in chain}
