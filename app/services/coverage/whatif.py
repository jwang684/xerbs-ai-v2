"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P13: the what-if corpus, in memory only.

Each pair is a pattern and its 基本方药 exactly as 《流行性感冒诊疗方案（2025年版）》
(nhc-natcm-flu-dx-tx-2025) states them, with the verbatim locator strings that
prove it. Building a what-if corpus returns a NEW ``CorpusSnapshot``; every
object it had to invent is ``simulated=True`` and SOURCE_VERIFIED only in
the sense "as if a human verified it" -- nothing is created, verified or
attested anywhere.

SEVERE pairs (重症) are kept apart: a severe-illness formula reached by a
consumer self-assessment is not consumer coverage, it is a safety signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.coverage.harness import (
    PATTERN_FORMULA, SOURCE_VERIFIED, CorpusSnapshot, EntitySnap, RelationshipSnap, _squash)

FLU_SOURCE_ID = "nhc-natcm-flu-dx-tx-2025"

MILD, RECOVERY, SEVERE = "MILD", "RECOVERY", "SEVERE"
_SCOPE = {
    MILD: "流行性感冒 轻症辨证治疗方案（2025年版）",
    SEVERE: "流行性感冒 重症辨证治疗方案（2025年版）",
    RECOVERY: "流行性感冒 恢复期辨证治疗方案（2025年版）",
}


@dataclass(frozen=True)
class WhatIfPair:
    key: str
    pattern_name: str
    formula_name: str
    tier: str
    pattern_locator: str      # verbatim heading in the source text
    formula_locator: str      # verbatim 基本方药 line in the source text
    source_id: str = FLU_SOURCE_ID

    @property
    def source_scope(self):
        return _SCOPE[self.tier]

    @property
    def consumer(self):
        return self.tier != SEVERE


PAIRS = (
    WhatIfPair("P1", "风寒束表", "麻黄汤加味", MILD, "（2）风寒束表", "基本方药：麻黄汤加味"),
    WhatIfPair("P2", "风热犯卫", "银翘散加减", MILD, "（1）风热犯卫", "基本方药：银翘散加减"),
    WhatIfPair("P3", "表寒里热", "大青龙汤加减", MILD, "（3）表寒里热", "基本方药：大青龙汤加减"),
    WhatIfPair("P4", "热毒袭肺", "麻杏石甘汤加减", MILD, "（4）热毒袭肺", "基本方药：麻杏石甘汤加减"),
    WhatIfPair("P5", "气阴两虚，正气未复", "沙参麦门冬汤加减", RECOVERY,
               "3.恢复期辨证治疗方案气阴两虚，正气未复", "基本方药：沙参麦门冬汤加减"),
    WhatIfPair("S1", "毒热壅盛", "宣白承气汤加减", SEVERE, "（1）毒热壅盛", "基本方药：宣白承气汤加减"),
    WhatIfPair("S2", "毒热内陷，内闭外脱", "参附汤加减", SEVERE, "（2）毒热内陷，内闭外脱", "基本方药：参附汤加减"),
)
CONSUMER_PAIRS = tuple(p for p in PAIRS if p.consumer)
SEVERE_PAIRS = tuple(p for p in PAIRS if not p.consumer)


def verify_pairs_against_source(pairs, source_text):
    """Every locator must occur verbatim (whitespace-insensitive, as PDF text is) in the source."""
    text = _squash(source_text)
    return {p.key: (_squash(p.pattern_locator) in text and _squash(p.formula_locator) in text) for p in pairs}


def _find(snap, entity_type, name):
    return next((e for e in snap.entities if e.entity_type == entity_type and e.name == name), None)


def build_whatif(snap: CorpusSnapshot, pairs, *, label, add_core_rows=False):
    """(new snapshot, provenance rows). Existing live objects are reused, not duplicated."""
    src = snap.source(FLU_SOURCE_ID)
    if src is None or src.review_status != SOURCE_VERIFIED or not src.live_verification:
        raise ValueError("what-if pairs cite %s, which is not a live SOURCE_VERIFIED source in this snapshot"
                         % FLU_SOURCE_ID)
    entities, rels, core_rows, provenance = [], [], [], []
    view = snap
    for p in pairs:
        row = {"pair": p.key, "tier": p.tier, "pattern": p.pattern_name, "formula": p.formula_name}
        pat = _find(view, "pattern", p.pattern_name)
        if pat is None:
            pat = EntitySnap("sim-pat-%s" % p.key.lower(), "pattern", p.pattern_name, SOURCE_VERIFIED, 3,
                             source_ids=(p.source_id,), live_verification=True, source_scope=p.source_scope,
                             tier=p.tier, simulated=True)
            entities.append(pat)
        row["pattern_object"] = "SIMULATED" if pat.simulated else "EXISTING:%s" % pat.entity_id
        frm = _find(view, "formula", p.formula_name)
        if frm is None:
            frm = EntitySnap("sim-frm-%s" % p.key.lower(), "formula", p.formula_name, SOURCE_VERIFIED, 3,
                             source_ids=(p.source_id,), live_verification=True, source_scope=p.source_scope,
                             tier=p.tier, simulated=True)
            entities.append(frm)
        row["formula_object"] = "SIMULATED" if frm.simulated else "EXISTING:%s" % frm.entity_id
        rel = next((r for r in view.relationships if r.pattern_id == pat.entity_id
                    and r.formula_id == frm.entity_id and r.relationship_type == PATTERN_FORMULA), None)
        if rel is None:
            rel = RelationshipSnap("sim-rel-%s" % p.key.lower(), pat.entity_id, frm.entity_id, SOURCE_VERIFIED, 3,
                                   evidence_source_ids=(p.source_id,), live_verification=True, simulated=True)
            rels.append(rel)
        row["relationship_object"] = "SIMULATED" if rel.simulated else "EXISTING:%s" % rel.relationship_id
        if add_core_rows and p.formula_name not in view.core_formula_names:
            core_rows.append(p.formula_name)
            row["core_row"] = "HYPOTHETICAL"
        else:
            row["core_row"] = "EXISTING" if p.formula_name in snap.core_formula_names else "ABSENT"
        provenance.append(row)
        view = snap.with_additions(entities=entities, relationships=rels)
    return snap.with_additions(entities=entities, relationships=rels, core_formula_names=core_rows,
                               label=label), provenance
