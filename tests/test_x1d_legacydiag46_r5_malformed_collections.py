"""X1D-LEGACYDIAG4.6-R5：六处列表读取，一处畸形只赔上它自己。

R4 修的是 also_resolved_by 一个字段。审计接着把同一个读法在本模块里数了一遍，
共七处：

    (x.get(field) or [])[:cap]

`or []` 只挡假值。真值里的非序列一路走到切片上——7 与 True 抛 TypeError，非空
dict 抛 KeyError（dict[slice]），set 抛 TypeError。validate_state 里没有任何
try，异常直接穿出去，被 assembler 的兜底接住，于是**整份**工作鉴别被丢弃，只
留下一个 DIFFERENTIAL_STATE_UNAVAILABLE。

R5 只收其中六处——它们今天会抛异常，加固不改变任何现有的成功路径：

    1. _resolve_all 内部的切片
    2. raw["hypotheses"]
    3. item["supporting_evidence"]（两处读法：_resolve_all 与 len()）
    4. item["contradicting_evidence"]（同样两处）
    5. item["unresolved_discriminators"]
    6. raw["evidence_gaps"]

第七处 gap["separates"] 被**刻意留下**，交给 R6。它今天不抛异常：裸字符串
"AB" 会被逐字符拆成 ['A','B']，凭空造出一个模型从未声明的 gap。那是语义纠正
而不是崩溃加固，改动方向相反，需要自己的证据与回滚边界。本文件最后一节把这个
已知缺陷钉成断言，既证明 R5 没碰它，也给 R6 留下基线。

EVIDENCE_REF_UNRESOLVED
-----------------------
畸形的证据集合不等于干净的空证据。模型确实写了东西，只是一条都没能落地——这正
是这个 note 本来要说的话。所以降级时它照旧发出。发出的形状集合与今天完全一致：
非空字符串今天就靠 dropped>0 发出，真值 dict/int/True/set 今天是抛异常，假值形
状今天不发、之后也不发。
"""
import pytest

from app.schemas.reasoning import ResolvableEvidence
from app.services.clarification import coverage as cov
from app.services.clarification.coverage import (
    KNOWN, MIN_ADAPTIVE_SCORE, assess_coverage, assess_sufficiency,
    governed_question_candidates, preference_rank, select_questions,
)
from app.services.interview.differential import (
    MAX_DISCRIMINATORS, MAX_EVIDENCE_PER_SIDE, MAX_GAPS, MAX_HYPOTHESES,
    _as_sequence, _collection_discarded, _resolve_all,
    differential_required_domains, signals_from_state, validate_state,
)

TEXT = "咳嗽三天，有点发热，身上酸痛。"
MISSING = object()

EV = ResolvableEvidence(
    has_complaint=True,
    observations=["sweating"],
    answers=[{"turn_id": 1, "question_field": "thirst",
              "domain": "thirst", "answer": "yes"}])

SUPPORT = [{"origin": "COMPLAINT"}]
CONTRA = [{"origin": "OBSERVATION", "field": "sweating"}]
DISC = {"domain": "thirst", "separates": ["A", "B"],
        "if_present_supports": ["A"], "if_absent_supports": ["B"],
        "rationale": "r"}
GAP = {"domain": "sweat", "separates": ["A", "B"]}


def raw_state(field=None, value=MISSING):
    """一份处处合法的 raw，只在 `field` 上放 `value`。"""
    hyp_a = {"pattern_name": "A", "standing": "PRIMARY_WORKING",
             "supporting_evidence": list(SUPPORT),
             "contradicting_evidence": list(CONTRA),
             "unresolved_discriminators": [dict(DISC)]}
    hyp_b = {"pattern_name": "B", "standing": "PLAUSIBLE",
             "supporting_evidence": list(SUPPORT),
             "unresolved_discriminators": []}
    raw = {"hypotheses": [hyp_a, hyp_b], "evidence_gaps": [dict(GAP)]}
    if field is None:
        return raw
    target = raw if field in ("hypotheses", "evidence_gaps") else hyp_a
    if value is MISSING:
        target.pop(field, None)
    else:
        target[field] = value
    return raw


def run(field=None, value=MISSING):
    return validate_state(raw_state(field, value), EV)


def hyp(state, name="A"):
    return next((h for h in state.hypotheses if h.pattern_name == name), None)


# ======================================================================
# 形状矩阵
# ======================================================================
#
# 每处站点都按同一张表过一遍。三组的意义不同，断言也不同：
#   BENIGN   —— 今天就安全，R5 之后必须一模一样
#   CRASHING —— 今天抛异常并带走整份状态，R5 之后必须降级
#   VALID    —— 合法输入，R5 之后必须一字不改

BENIGN = [("missing", MISSING), ("None", None), ("[]", []),
          ('""', ""), ("{}", {}), ("0", 0), ("False", False)]

CRASHING = [("int 7", 7), ("True", True), ("truthy dict", {"a": 1}),
            ("set", {"x", "y"})]

# 裸字符串两边都不属于：它今天不抛异常（字符串可切片），但也从来没被当成数组用
# 过——每个字符都过不了下游的类型判断。R5 之后它继续得到空集合。
BARE_STRING = "nose"

EVIDENCE_FIELDS = ["supporting_evidence", "contradicting_evidence"]
ALL_SITES = ["hypotheses", "supporting_evidence", "contradicting_evidence",
             "unresolved_discriminators", "evidence_gaps"]


class TestNothingRaisesAnyMore:
    """R5 的全部目的：这一类不再抛异常。"""

    @pytest.mark.parametrize("field", ALL_SITES)
    @pytest.mark.parametrize("label,value", CRASHING, ids=[c[0] for c in CRASHING])
    def test_a_crashing_shape_no_longer_raises(self, field, label, value):
        run(field, value)   # 抛出即失败

    @pytest.mark.parametrize("field", ALL_SITES)
    @pytest.mark.parametrize("label,value", BENIGN, ids=[b[0] for b in BENIGN])
    def test_a_benign_shape_still_does_not_raise(self, field, label, value):
        run(field, value)

    @pytest.mark.parametrize("field", ALL_SITES)
    def test_a_bare_string_does_not_raise(self, field):
        run(field, BARE_STRING)


class TestTheDifferentialSurvives:
    """畸形字段不再带走整份鉴别——hypotheses 除外，见下一节。"""

    SURVIVING = ["supporting_evidence", "contradicting_evidence",
                 "unresolved_discriminators", "evidence_gaps"]

    @pytest.mark.parametrize("field", SURVIVING)
    @pytest.mark.parametrize("label,value", CRASHING, ids=[c[0] for c in CRASHING])
    def test_both_hypotheses_survive(self, field, label, value):
        state, _ = run(field, value)
        assert state is not None, (field, label)
        assert [h.pattern_name for h in state.hypotheses] == ["A", "B"]

    @pytest.mark.parametrize("field", SURVIVING)
    @pytest.mark.parametrize("label,value", CRASHING, ids=[c[0] for c in CRASHING])
    def test_the_other_fields_are_untouched(self, field, label, value):
        """畸形只赔上它自己那一格。"""
        state, _ = run(field, value)
        a = hyp(state)
        if field != "supporting_evidence":
            assert len(a.supporting_evidence) == 1, field
        if field != "contradicting_evidence":
            assert len(a.contradicting_evidence) == 1, field
        if field != "unresolved_discriminators":
            assert len(a.unresolved_discriminators) == 1, field
        if field != "evidence_gaps":
            assert state.evidence_gaps == [
                {"domain": "sweat", "separates": ["A", "B"]}], field


# ======================================================================
# 站点 2：hypotheses —— 审计定下的语义就是 state=None
# ======================================================================

class TestSite2Hypotheses:
    """结构性根节点。降级成空，正好走到本来就在那儿的空分支。"""

    @pytest.mark.parametrize("label,value", CRASHING, ids=[c[0] for c in CRASHING])
    def test_malformed_yields_state_none_not_an_exception(self, label, value):
        state, notes = run("hypotheses", value)
        assert state is None, label

    @pytest.mark.parametrize("label,value", BENIGN, ids=[b[0] for b in BENIGN])
    def test_benign_shapes_keep_their_existing_meaning(self, label, value):
        """missing / None / [] / "" / {} / 0 / False 今天就是 state=None。"""
        state, _ = run("hypotheses", value)
        assert state is None, label

    def test_a_bare_string_is_not_a_list_of_hypotheses(self):
        state, _ = run("hypotheses", BARE_STRING)
        assert state is None

    def test_this_is_the_same_outcome_the_empty_list_already_gave(self):
        """降级路径与既有空路径不可区分——R5 只是不再经由异常到达它。"""
        assert run("hypotheses", [])[0] is None
        assert run("hypotheses", 7)[0] is None

    def test_a_valid_list_is_unchanged(self):
        state, _ = run()
        assert [h.pattern_name for h in state.hypotheses] == ["A", "B"]

    def test_a_tuple_is_accepted(self):
        raw = raw_state()
        raw["hypotheses"] = tuple(raw["hypotheses"])
        state, _ = validate_state(raw, EV)
        assert [h.pattern_name for h in state.hypotheses] == ["A", "B"]

    def test_a_mixed_list_keeps_the_real_hypotheses(self):
        raw = raw_state()
        raw["hypotheses"] = [raw["hypotheses"][0], 7, None, "junk", {},
                             raw["hypotheses"][1]]
        state, _ = validate_state(raw, EV)
        assert [h.pattern_name for h in state.hypotheses] == ["A", "B"]

    def test_the_cap_is_preserved(self):
        raw = raw_state()
        base = raw["hypotheses"][0]
        raw["hypotheses"] = [dict(base, pattern_name="H%02d" % i)
                             for i in range(12)]
        state, _ = validate_state(raw, EV)
        assert MAX_HYPOTHESES == 5
        assert len(state.hypotheses) == MAX_HYPOTHESES
        assert [h.pattern_name for h in state.hypotheses] == [
            "H00", "H01", "H02", "H03", "H04"]


# ======================================================================
# 站点 3 / 4：证据两侧 —— 只赔那一侧，并且照旧报告
# ======================================================================

class TestSites34Evidence:

    @pytest.mark.parametrize("field", EVIDENCE_FIELDS)
    @pytest.mark.parametrize("label,value", CRASHING, ids=[c[0] for c in CRASHING])
    def test_only_that_side_empties(self, field, label, value):
        state, _ = run(field, value)
        a = hyp(state)
        assert a is not None, (field, label)
        if field == "supporting_evidence":
            assert a.supporting_evidence == []
            assert len(a.contradicting_evidence) == 1
        else:
            assert len(a.supporting_evidence) == 1
            assert a.contradicting_evidence == []

    @pytest.mark.parametrize("field", EVIDENCE_FIELDS)
    @pytest.mark.parametrize("label,value", CRASHING, ids=[c[0] for c in CRASHING])
    def test_the_hypothesis_itself_survives(self, field, label, value):
        state, _ = run(field, value)
        assert hyp(state) is not None, (field, label)
        assert hyp(state).standing == "PRIMARY_WORKING"

    @pytest.mark.parametrize("field", EVIDENCE_FIELDS)
    @pytest.mark.parametrize("label,value", CRASHING, ids=[c[0] for c in CRASHING])
    def test_the_note_is_emitted(self, field, label, value):
        """畸形证据集合不得静悄悄地变成“干净地没有证据”。"""
        _, notes = run(field, value)
        assert "EVIDENCE_REF_UNRESOLVED" in notes, (field, label)

    @pytest.mark.parametrize("field", EVIDENCE_FIELDS)
    @pytest.mark.parametrize("label,value", BENIGN, ids=[b[0] for b in BENIGN])
    def test_absent_evidence_stays_silent(self, field, label, value):
        """假值形状今天不发这个 note，之后也不发——那是干净的缺席。"""
        _, notes = run(field, value)
        assert "EVIDENCE_REF_UNRESOLVED" not in notes, (field, label)

    @pytest.mark.parametrize("field", EVIDENCE_FIELDS)
    def test_a_bare_string_keeps_the_note_it_already_had(self, field):
        """非空字符串今天就靠 dropped>0 发出这个 note；发出的集合没有变。"""
        state, notes = run(field, BARE_STRING)
        assert "EVIDENCE_REF_UNRESOLVED" in notes
        assert hyp(state) is not None

    def test_a_clean_state_emits_nothing(self):
        _, notes = run()
        assert "EVIDENCE_REF_UNRESOLVED" not in notes

    def test_an_unresolvable_citation_still_emits_it(self):
        """既有触发路径原样保留。"""
        _, notes = run("supporting_evidence",
                       [{"origin": "OBSERVATION", "field": "tongue_coating"}])
        assert "EVIDENCE_REF_UNRESOLVED" in notes

    @pytest.mark.parametrize("field", EVIDENCE_FIELDS)
    def test_a_valid_list_resolves_unchanged(self, field):
        state, _ = run(field, list(SUPPORT if field == "supporting_evidence"
                                   else CONTRA))
        a = hyp(state)
        assert len(getattr(a, field)) == 1

    @pytest.mark.parametrize("field", EVIDENCE_FIELDS)
    def test_a_tuple_resolves_unchanged(self, field):
        state, _ = run(field, tuple(SUPPORT if field == "supporting_evidence"
                                    else CONTRA))
        assert len(getattr(hyp(state), field)) == 1

    def test_a_mixed_list_keeps_only_what_resolves(self):
        state, notes = run("supporting_evidence",
                           [{"origin": "COMPLAINT"}, 7, None, "junk",
                            {"origin": "INVENTED"}])
        assert len(hyp(state).supporting_evidence) == 1
        assert "EVIDENCE_REF_UNRESOLVED" in notes

    def test_the_cap_is_preserved(self):
        assert MAX_EVIDENCE_PER_SIDE == 8
        refs = [{"origin": "OBSERVATION", "field": "sweating"}] * 30
        evidence = ResolvableEvidence(has_complaint=True,
                                      observations=["sweating"])
        assert len(_resolve_all(refs, evidence)) == 1      # 去重后只剩一条
        many = [{"origin": "ANSWER", "question_field": "d%d" % i, "turn_id": 1}
                for i in range(30)]
        answers = [{"turn_id": 1, "question_field": "d%d" % i} for i in range(30)]
        resolved = _resolve_all(many, ResolvableEvidence(answers=answers))
        assert len(resolved) == MAX_EVIDENCE_PER_SIDE


# ======================================================================
# 站点 1：_resolve_all 本身
# ======================================================================

class TestSite1ResolveAll:

    @pytest.mark.parametrize("label,value", CRASHING + BENIGN,
                             ids=[c[0] for c in CRASHING + BENIGN])
    def test_it_never_raises(self, label, value):
        if value is MISSING:
            return
        assert _resolve_all(value, EV) == [], label

    def test_a_bare_string_resolves_to_nothing(self):
        assert _resolve_all("COMPLAINT", EV) == []

    def test_valid_input_is_unchanged(self):
        assert len(_resolve_all(SUPPORT, EV)) == 1
        assert len(_resolve_all(tuple(SUPPORT), EV)) == 1


# ======================================================================
# 站点 5：unresolved_discriminators
# ======================================================================

class TestSite5Discriminators:

    @pytest.mark.parametrize("label,value", CRASHING, ids=[c[0] for c in CRASHING])
    def test_the_hypothesis_survives_without_discriminators(self, label, value):
        """审计定下的语义：零个鉴别项是完全合法的假设形状。"""
        state, _ = run("unresolved_discriminators", value)
        a = hyp(state)
        assert a is not None, label
        assert a.unresolved_discriminators == [], label
        assert len(a.supporting_evidence) == 1, label

    @pytest.mark.parametrize("label,value", CRASHING, ids=[c[0] for c in CRASHING])
    def test_no_eligible_domain_is_invented(self, label, value):
        """鉴别项贡献的 thirst 随字段一起消失；gap 贡献的 sweat 与它无关。"""
        state, _ = run("unresolved_discriminators", value)
        assert differential_required_domains(state) == {"sweat"}, label

    def test_a_bare_string_yields_no_discriminators(self):
        state, _ = run("unresolved_discriminators", BARE_STRING)
        assert hyp(state).unresolved_discriminators == []

    def test_a_valid_list_is_unchanged(self):
        state, _ = run()
        assert len(hyp(state).unresolved_discriminators) == 1
        # thirst 来自鉴别项，sweat 来自 evidence_gaps —— 两条既有路径都在。
        assert differential_required_domains(state) == {"thirst", "sweat"}

    def test_a_tuple_is_accepted(self):
        state, _ = run("unresolved_discriminators", (dict(DISC),))
        assert len(hyp(state).unresolved_discriminators) == 1

    def test_a_mixed_list_keeps_the_real_entries(self):
        state, _ = run("unresolved_discriminators",
                       [7, dict(DISC), None, "junk"])
        assert len(hyp(state).unresolved_discriminators) == 1

    def test_the_cap_is_preserved(self):
        assert MAX_DISCRIMINATORS == 4
        entries = [dict(DISC, domain=d) for d in
                   ("thirst", "sweat", "nose", "throat", "sputum", "diet")]
        state, _ = run("unresolved_discriminators", entries)
        kept = [e["domain"] for e in hyp(state).unresolved_discriminators]
        assert kept == ["thirst", "sweat", "nose", "throat"]


# ======================================================================
# 站点 6：evidence_gaps
# ======================================================================

class TestSite6EvidenceGaps:

    @pytest.mark.parametrize("label,value", CRASHING, ids=[c[0] for c in CRASHING])
    def test_gaps_degrade_and_hypotheses_survive(self, label, value):
        state, _ = run("evidence_gaps", value)
        assert state is not None, label
        assert state.evidence_gaps == [], label
        assert len(state.hypotheses) == 2, label

    def test_a_bare_string_yields_no_gaps(self):
        state, _ = run("evidence_gaps", BARE_STRING)
        assert state.evidence_gaps == []

    def test_a_valid_list_is_unchanged(self):
        state, _ = run()
        assert state.evidence_gaps == [{"domain": "sweat",
                                        "separates": ["A", "B"]}]

    def test_a_tuple_is_accepted(self):
        state, _ = run("evidence_gaps", (dict(GAP),))
        assert state.evidence_gaps == [{"domain": "sweat",
                                        "separates": ["A", "B"]}]

    def test_a_mixed_list_keeps_the_real_gaps(self):
        state, _ = run("evidence_gaps", [7, dict(GAP), None, "junk", {}])
        assert state.evidence_gaps == [{"domain": "sweat",
                                        "separates": ["A", "B"]}]

    def test_the_cap_is_preserved(self):
        assert MAX_GAPS == 6
        gaps = [dict(GAP, domain="d%d" % i) for i in range(10)]
        state, _ = run("evidence_gaps", gaps)
        assert len(state.evidence_gaps) == MAX_GAPS
        assert [g["domain"] for g in state.evidence_gaps] == [
            "d%d" % i for i in range(6)]


# ======================================================================
# 助手契约
# ======================================================================

class TestHelperContract:

    def test_only_list_and_tuple_are_collections(self):
        assert _as_sequence([1, 2]) == [1, 2]
        assert _as_sequence((1, 2)) == (1, 2)
        for value in (None, "", "nose", 7, 0, True, False, {}, {"a": 1},
                      {"x", "y"}, 1.5, object()):
            assert _as_sequence(value) == (), repr(value)

    def test_it_does_not_copy_a_valid_collection(self):
        """合法输入原样透传，切片、顺序、去重的语义都不受影响。"""
        original = [1, 2, 3]
        assert _as_sequence(original) is original

    def test_a_set_is_not_accepted_here(self):
        """competition_key 接受 set，本助手刻意不接受。

        那是给鉴别项身份用的、顺序无关的比较；这里读的字段契约写的是数组，
        而 set 无序——接受它会让切片上限取到哪几个元素变成不确定。
        """
        assert _as_sequence({"a", "b"}) == ()

    def test_a_bare_string_is_never_an_array(self):
        assert _as_sequence("nose") == ()
        assert _as_sequence("AB") == ()

    def test_discarded_reports_content_not_absence(self):
        for value in (7, True, {"a": 1}, {"x", "y"}, "nose", 1.5):
            assert _collection_discarded(value) is True, repr(value)
        for value in (None, "", 0, False, {}, [], (), [1], (1,)):
            assert _collection_discarded(value) is False, repr(value)


# ======================================================================
# 站点 7：刻意未修改，留给 R6
# ======================================================================

class TestSite7CorrectedByR6:
    """站点 7 的历史：R5 刻意留下，R6 纠正。

    R5 只改今天会抛异常的路径，所以这里原本钉的是**当前错误行为**——"AB" 被逐
    字符拆成 ['A','B']，凭空造出一个模型从未声明的 gap。那是语义纠正而不是崩溃
    加固，按约定单独成phase。

    R6 做了那次纠正，于是本节的断言按审计列出的方式逐条反转。保留这一节而不是
    删掉它，是因为它记录的是一个真实发生过的缺陷：读到这里的人应该看见它曾经
    怎样表现、为什么当时没有顺手改掉、以及现在被改成了什么。

    畸形形状的完整矩阵在 test_x1d_legacydiag46_r6_gap_separates.py。
    """

    def gaps_for(self, separates):
        raw = raw_state()
        raw["evidence_gaps"][0]["separates"] = separates
        state, _ = validate_state(raw, EV)
        return state.evidence_gaps

    def test_a_string_is_no_longer_split_into_characters(self):
        """R5 之前：'AB' -> [{'separates': ['A','B']}]。R6 之后：没有 gap。"""
        assert self.gaps_for("AB") == []
        assert self.gaps_for("BA") == []
        assert self.gaps_for("A") == []
        assert self.gaps_for("AXB") == []

    def test_a_set_is_no_longer_accepted_here(self):
        """曾经被接受，且 [:2] 让保留哪两个名字变得不确定。"""
        assert self.gaps_for({"A", "B"}) == []

    def test_a_truthy_mapping_no_longer_fabricates(self):
        """dict 迭代的是键。{'A': 1} 曾经造出 separates=['A']。

        R5 那版断言用的是 {'a': 1}——键对不上，改前改后都得空，所以它其实从来
        没有钉住这个缺陷。这里换成会匹配的键。
        """
        assert self.gaps_for({"A": 1}) == []
        assert self.gaps_for({"a": 1}) == []

    @pytest.mark.parametrize("value", [7, True], ids=["int", "bool"])
    def test_it_no_longer_raises(self, value):
        """站点 7 的崩溃面曾经开着；R5 明确划出去，R6 关上。"""
        raw = raw_state()
        raw["evidence_gaps"][0]["separates"] = value
        state, _ = validate_state(raw, EV)      # 抛出即失败
        assert state is not None
        assert state.evidence_gaps == []
        assert len(state.hypotheses) == 2


# ======================================================================
# R3 / R4 平价
# ======================================================================

class TestR3R4ParityIntact:

    def alts_state(self, alternatives):
        raw = raw_state()
        raw["hypotheses"][0]["unresolved_discriminators"] = [
            dict(DISC, also_resolved_by=alternatives)]
        return validate_state(raw, EV)[0]

    def test_r4_malformed_alternatives_still_degrade_locally(self):
        for value in (7, True, {"a": "nose"}, {"nose", "throat"}, "nose"):
            state = self.alts_state(value)
            disc = hyp(state).unresolved_discriminators[0]
            assert disc["also_resolved_by"] == [], repr(value)
            assert disc["separates"] == ["A", "B"]

    def test_r3_valid_alternatives_still_reach_gap_domains(self):
        state = self.alts_state(["nose", "throat"])
        assert {"thirst", "nose", "throat"} <= set(
            signals_from_state(state).gap_domains)

    def test_r3_required_domains_still_widen(self):
        state = self.alts_state(["nose", "throat"])
        assert {"thirst", "nose", "throat"} <= differential_required_domains(state)

    def test_r3_scoring_weight_is_unchanged(self):
        state = self.alts_state(["throat"])
        required = differential_required_domains(state)
        coverage = assess_coverage(TEXT, required)
        signals = signals_from_state(state)
        throat = [c for c in governed_question_candidates(required, coverage, [])
                  if c.domain == "throat"][0]
        assert cov.score_candidate(
            throat, coverage, signals).score >= MIN_ADAPTIVE_SCORE


# ======================================================================
# 周边不变量
# ======================================================================

class TestSurroundingInvariantsUnchanged:

    def test_the_floor_is_unchanged(self):
        assert MIN_ADAPTIVE_SCORE == 40

    def test_the_budget_is_unchanged_for_degraded_input(self):
        valid, _ = run()
        degraded, _ = run("unresolved_discriminators", 7)
        coverage = assess_coverage(TEXT, set(cov.DOMAINS_BY_KEY))
        assert (assess_sufficiency(coverage, signals_from_state(valid), False)[1]
                == assess_sufficiency(
                    coverage, signals_from_state(degraded), False)[1])

    def test_preference_rank_is_untouched(self):
        coverage = assess_coverage(TEXT, set(cov.DOMAINS_BY_KEY))
        assert preference_rank("cold_heat", coverage)[0] == 1   # PARTIAL
        assert preference_rank("throat", coverage)[0] == 0      # UNKNOWN

    def test_known_domains_remain_excluded(self):
        raw = raw_state()
        raw["hypotheses"][0]["unresolved_discriminators"] = [
            dict(DISC, domain="sweat")]
        state, _ = validate_state(raw, EV)
        required = differential_required_domains(state)
        coverage = assess_coverage("发热无汗身痛", required)
        assert coverage.states.get("sweat") == KNOWN
        produced = [c.domain for c in
                    governed_question_candidates(required, coverage, [])]
        assert "sweat" not in produced

    def test_generic_coverage_still_fills_capacity(self):
        state, _ = run("supporting_evidence", 7)
        required = differential_required_domains(state)
        coverage = assess_coverage("咳嗽三天", required)
        selection = select_questions(
            deterministic=[], adaptive=[], coverage=coverage,
            signals=signals_from_state(state), differential_domains=required)
        assert selection.sufficiency == cov.NEEDS_MORE_INFORMATION
        assert selection.fallback_used > 0

    def test_nothing_consumer_visible_or_authoritative_was_added(self):
        import inspect

        from app.services.interview import differential as diff

        for fn in (diff._as_sequence, diff._collection_discarded,
                   diff.validate_state, diff._resolve_all):
            source = inspect.getsource(fn)
            for forbidden in ("clinical_ranking_eligible", "VERIFIED",
                              "REVIEWED", "ready_for_formula_retrieval",
                              "consumer_purchasable", "PATTERN_FORMULA",
                              "FORMULA_HERB", "commit(", "insert("):
                assert forbidden not in source, (fn.__name__, forbidden)

    def test_the_validated_shapes_stay_constant(self):
        """下游读的是这些形状，畸形输入不得改变类型。"""
        for field in ["supporting_evidence", "contradicting_evidence",
                      "unresolved_discriminators"]:
            for value in (MISSING, None, [], 7, True, {"a": 1}, {"x"}, "nose"):
                state, _ = run(field, value)
                assert isinstance(getattr(hyp(state), field), list)
        for value in (MISSING, None, [], 7, True, {"a": 1}, {"x"}, "nose"):
            state, _ = run("evidence_gaps", value)
            assert isinstance(state.evidence_gaps, list)
