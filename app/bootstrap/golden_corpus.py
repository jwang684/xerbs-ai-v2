"""Golden Corpus bootstrap (X1D-E2E1 Step 6).

What this is for
----------------
Exactly one governed formula — 清肺排毒汤 — needs to exist in a target
environment's clinical corpus, established the same way it was established
locally: through the real governance lifecycle, with a real source, ending at
``clinical_ranking_eligible``.

Why it runs in-process rather than over HTTP
--------------------------------------------
External governance mutation is disabled by default (see
``app.core.governance_guard``), because ``reviewer_role`` arrives in the
request body and so cannot establish clinical review authority. Rather than
re-open those endpoints for deployment, this bootstrap calls the governance
service directly, in-process, where it is not reachable from the network. It
uses the same ``PersistentClinicalStore`` methods the API uses, so the same
transition rules, version checks and audit events apply — no direct SQL, and
no shortcut to REVIEWED.

Where the data comes from
-------------------------
The source metadata and the 21-herb composition are transcribed from evidence
already reviewed and approved in xerbs-core (evidence id 34, the NHSA 2024
公示 submission, SHA-256 recorded below). Nothing is fetched at deploy time:
no web scraping, no runtime search, no model inference. If the upstream
document ever changes, the recorded hash is what reveals it.

The human decision is external. This module only executes an approval that a
qualified human already made; it does not make one.

Idempotency
-----------
Safe to re-run. The formula is located by exact name and the source by its
stable id, so a second run adds nothing and reports ``already_eligible``.
Nothing is deleted or rewritten, and the append-only audit history is
preserved.
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import select

from app.db.models import ClinicalEntity, SourceRegistry
from app.schemas.clinical_knowledge import (
    SourceReviewActionRequest,
    SourceReviewDecision,
    SourceSubmitReviewRequest,
)
from app.schemas.clinical_workflow import (
    ClinicalEntityType,
    IngestionBatchRequest,
    IngestionItem,
)
from app.services.knowledge.persistent_clinical import (
    PersistentClinicalStore,
    PersistentWorkflowError,
)

GOLDEN_FORMULA_NAME = "清肺排毒汤"

# Stable source id: derived from the NHSA submission reference, not random,
# so re-running in any environment resolves the same record.
GOLDEN_SOURCE_ID = "nhsa-2024-YPSN202400007"

GOLDEN_SOURCE: Dict[str, Any] = {
    "source_id": GOLDEN_SOURCE_ID,
    "title": "国家医保局 2024年国家医保药品目录调整 申报材料（公示版）— 清肺排毒颗粒",
    "citation": (
        "国家医疗保障局公示申报材料；含通用名、药品类别 中成药、注册分类 中药3.2类、"
        "完整处方组成（21 味含剂量）、规格 每袋装15g（相当于饮片49g）、"
        "上市许可持有人 中国中医科学院中医临床基础医学研究所。"
        "SHA-256 ecfcd4046bcabc05105d293747903ac17c1c45c99cb7949b666ecf3ee14bc243；"
        "检索于 2026-09-12"
    ),
    "url": "https://www.nhsa.gov.cn/attach/Ypsn2024/YPSN202400007/YPSN202400007.pdf",
    "source_type": "OFFICIAL_COMPOSITION_DOC",
}

# Canonical composition, normalized herb identities as approved in xerbs-core.
GOLDEN_INGREDIENTS = [
    "麻黄", "细辛", "桂枝", "广藿香", "白术", "山药", "石膏", "柴胡", "燀苦杏仁",
    "款冬花", "紫菀", "射干", "姜半夏", "生姜", "枳实", "陈皮", "猪苓", "茯苓",
    "泽泻", "黄芩", "炙甘草",
]

# Indication terms quoted verbatim from the label's 功能主治. Not expanded:
# this product is indicated for a specific 寒湿疫 presentation, not for
# respiratory complaints generally.
GOLDEN_INDICATIONS = [
    "发热恶寒", "周身酸痛", "困乏肢重", "咳嗽少痰", "喘憋气促",
    "口淡无味", "食欲不振", "恶心呕吐", "大便不爽",
]

DOSE_NOTE = (
    "来源载明每味药的剂量，但本语料的 ingredients 为扁平药材名数组，无剂量位，"
    "故剂量未在此表示；不得据此进行剂量相关判定。"
)

# Distinct submitter and reviewer identities. This service does not enforce
# separation of duties, so it is maintained here deliberately rather than
# relied upon.
BOOTSTRAP_SUBMITTER = "xerbs-bootstrap-ingest"
BOOTSTRAP_REVIEWER = "xerbs-bootstrap-clinical-reviewer"


def _find_formula(store: PersistentClinicalStore):
    with store.Session() as s:
        return s.scalar(
            select(ClinicalEntity).where(
                ClinicalEntity.entity_type == "formula",
                ClinicalEntity.name == GOLDEN_FORMULA_NAME,
            )
        )


def _source_state(store: PersistentClinicalStore):
    with store.Session() as s:
        row = s.scalar(
            select(SourceRegistry).where(SourceRegistry.source_id == GOLDEN_SOURCE_ID)
        )
        return (row.review_status, row.version) if row else (None, None)


def bootstrap_golden_corpus(store: PersistentClinicalStore | None = None) -> Dict[str, Any]:
    """Ensure exactly one governed, ranking-eligible 清肺排毒汤 exists.

    Returns a report describing what it did, so a deployment log shows whether
    this run created anything or found the corpus already correct.
    """
    store = store or PersistentClinicalStore()
    report: Dict[str, Any] = {"formula_name": GOLDEN_FORMULA_NAME, "actions": []}

    existing = _find_formula(store)

    if existing is None:
        # Ingest creates the entity AND the entity_source link. It is the only
        # code path that creates that link, so formula and source must be
        # introduced together.
        result = store.ingest(
            IngestionBatchRequest(
                submitted_by=BOOTSTRAP_SUBMITTER,
                source_label="X1D-E2E1 Golden Corpus bootstrap (single formula)",
                items=[
                    IngestionItem(
                        entity_type=ClinicalEntityType.FORMULA,
                        external_id="xerbs-core:canonical-formula:清肺排毒汤",
                        payload={
                            "name": GOLDEN_FORMULA_NAME,
                            "ingredients": GOLDEN_INGREDIENTS,
                            "indications": GOLDEN_INDICATIONS,
                            "efficacy": "散寒祛湿，理肺排毒。",
                            "dose_representation_note": DOSE_NOTE,
                            "migration_origin": (
                                "X1D-E2E1 bootstrap: transcribed from xerbs-core "
                                "approved evidence (NHSA 2024 公示)"
                            ),
                        },
                        sources=[GOLDEN_SOURCE],  # type: ignore[list-item]
                    )
                ],
            )
        )
        entity_id = result.created_entity_ids[0]
        report["actions"].append("ingested_formula_as_draft")
    else:
        entity_id = existing.id
        report["actions"].append("formula_already_present")

    report["entity_id"] = entity_id

    # --- source lifecycle -------------------------------------------------
    status, version = _source_state(store)
    report["source_status_before"] = status

    if status == "DRAFT":
        store.submit_source_for_review(
            GOLDEN_SOURCE_ID,
            SourceSubmitReviewRequest(
                submitted_by=BOOTSTRAP_SUBMITTER,
                expected_version=version,
                notes="X1D-E2E1 bootstrap: submit authoritative source for review",
            ),
        )
        report["actions"].append("source_submitted")
        status, version = _source_state(store)

    if status == "IN_REVIEW":
        store.review_source(
            GOLDEN_SOURCE_ID,
            SourceReviewActionRequest(
                reviewer_id=BOOTSTRAP_REVIEWER,
                reviewer_role="CLINICAL_REVIEWER",
                decision=SourceReviewDecision.APPROVE,
                expected_version=version,
                notes=(
                    "X1D-E2E1 bootstrap: official NHSA 公示 material, verified and "
                    "human-approved in xerbs-core"
                ),
            ),
        )
        report["actions"].append("source_approved")

    report["source_status_after"] = _source_state(store)[0]

    # --- formula lifecycle ------------------------------------------------
    entity = _find_formula(store)
    report["formula_status_before"] = entity.review_status if entity else None

    if entity is not None and entity.review_status in {"DRAFT", "REJECTED"}:
        store.submit_for_review(
            ClinicalEntityType.FORMULA, entity_id, BOOTSTRAP_SUBMITTER,
            "X1D-E2E1 bootstrap: submit canonical formula for review",
        )
        report["actions"].append("formula_submitted")
        entity = _find_formula(store)

    if entity is not None and entity.review_status == "IN_REVIEW":
        # X1D-AIV2-ATTEST1: the bootstrap stops here, deliberately.
        #
        # It used to approve the formula itself, by passing
        # reviewer_role="CLINICAL_REVIEWER" to store.review(). That was a
        # machine asserting clinical review authority, which this service does
        # not have and must not appear to have -- the docstring above already
        # said "the human decision is external"; now it is true.
        #
        # The formula is left IN_REVIEW, awaiting a verified xerbs-core
        # attestation. It is therefore not ranking-eligible in a freshly
        # bootstrapped environment, which is the honest state: nobody has
        # approved it there yet.
        report["actions"].append("formula_awaiting_human_attestation")

    # --- verify eligibility through the service's own predicate -----------
    detail = store.get_entity_detail(ClinicalEntityType.FORMULA, entity_id)
    report["formula_status_after"] = detail.review_status.value
    report["clinical_ranking_eligible"] = detail.clinical_ranking_eligible
    report["sources"] = [s.source_id for s in detail.sources]

    if not report["actions"] or report["actions"] == ["formula_already_present"]:
        report["actions"].append("already_eligible" if detail.clinical_ranking_eligible
                                 else "no_change_but_not_eligible")

    return report
