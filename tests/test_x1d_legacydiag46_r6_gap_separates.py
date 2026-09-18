"""X1D-LEGACYDIAG4.6-R6：evidence_gaps 的 separates，最后一处，也是唯一一处错的。

R4 修了一个字段，R5 修了六处会抛异常的读法。剩下的第七处不抛异常——它静悄悄
地产出错误答案，所以被单独留成一个 phase：

    separates = [str(n).strip() for n in (entry.get("separates") or [])
                 if str(n).strip()]
    separates = [n for n in separates if n in named][:2]

`or []` 只挡假值，而字符串是可迭代的。于是 "AB" 被逐字符拆开，如果 A 和 B 恰好
是台面上的两个读法，就凭空造出一个模型从未声明的 gap——没有异常，没有 note，
没有 reason code。下游根本分不出真假：它照样进 gap_domains，照样拿
WEIGHT_ENVELOPE_GAP，照样在 differential_required_domains 里变成合格域，照样
去抢那几个稀缺的追问名额。审计量到的数字是：伪造的 gap 与真实的 gap 都让 sweat
得 84 分，门槛是 40。

同一个洞还有三种形状。dict 迭代的是键，所以 {'A': 1} 同样伪造；set 被接受，而
[:2] 让保留哪两个名字随迭代顺序变化；7 与 True 则直接抛异常，代价和 R5 那六处
一样是整份鉴别。

还有一处跟畸形输入无关、在**合法输入**上就已经错了的地方：没有去重。
['A','A','B'] 会用两个 A 填满两个名额，把真正的第二个分隔者 B 挤掉。

R6 的取舍
---------
_as_sequence 原样复用，一个字未改：list 与 tuple，别的都不是数组。审计里试过
改用 _names，被否决了——它只预扫 limit*2 个元素，['X','Y','Z','W','A'] 会静悄
悄地丢掉第五个位置上合法的 A，那是拿一个缺陷换另一个。

刻意没做：不要求至少两个分隔者。契约里写的是 "the two readings it would
separate"，但今天单个分隔者是被接受的、并且会产生一个追问，砍掉它属于更宽的语
义政策变更，不在本次测到的缺陷范围内。
"""
import pytest

from app.schemas.reasoning import ResolvableEvidence
from app.services.clarification import coverage as cov
from app.services.clarification.coverage import (
    KNOWN, MIN_ADAPTIVE_SCORE, WEIGHT_ENVELOPE_GAP, assess_coverage,
    assess_sufficiency, governed_question_candidates, preference_rank,
    select_questions,
)
from app.services.interview.differential import (
    MAX_GAPS, _as_sequence, differential_required_domains, signals_from_state,
    validate_state,
)

TEXT = "咳嗽三天，有点发热，身上酸痛。"
MISSING = object()
EV = ResolvableEvidence(has_complaint=True)

DISC = {"domain": "thirst", "separates": ["A", "B"],
        "if_present_supports": ["A"], "if_absent_supports": ["B"],
        "rationale": "r"}


def raw_state(separates=MISSING, domain="sweat", standing_b="PLAUSIBLE",
              gaps=None):
    """A、B 两个假设都成立；只有 gap 的 separates 是变量。"""
    raw = {"hypotheses": [
        {"pattern_name": "A", "standing": "PRIMARY_WORKING",
         "supporting_evidence": [{"origin": "COMPLAINT"}],
         "unresolved_discriminators": []},
        {"pattern_name": "B", "standing": standing_b,
         "supporting_evidence": [{"origin": "COMPLAINT"}],
         "unresolved_discriminators": []}]}
    if gaps is not None:
        raw["evidence_gaps"] = gaps
    else:
        entry = {"domain": domain}
        if separates is not MISSING:
            entry["separates"] = separates
        raw["evidence_gaps"] = [entry]
    return raw


def gaps_for(separates, **kw):
    state, _ = validate_state(raw_state(separates, **kw), EV)
    return state.evidence_gaps


def seps(separates, **kw):
    g = gaps_for(separates, **kw)
    return g[0]["separates"] if g else None


# ======================================================================
# 一、伪造路径关上了
# ======================================================================

FABRICATING = [
    ("bare string AB", "AB"),
    ("bare string BA", "BA"),
    ("bare string A", "A"),
    ("bare string AXB", "AXB"),
    ("mapping with matching key", {"A": 1}),
    ("mapping with two keys", {"A": 1, "B": 2}),
    ("set", {"A", "B"}),
]


class TestFabricationIsClosed:

    @pytest.mark.parametrize("label,value", FABRICATING,
                             ids=[f[0] for f in FABRICATING])
    def test_no_gap_is_produced(self, label, value):
        assert gaps_for(value) == [], label

    @pytest.mark.parametrize("label,value", FABRICATING,
                             ids=[f[0] for f in FABRICATING])
    def test_the_domain_does_not_become_eligible(self, label, value):
        """伪造的 gap 曾经让 sweat 直接变成合格域。"""
        state, _ = validate_state(raw_state(value), EV)
        assert "sweat" not in differential_required_domains(state), label

    @pytest.mark.parametrize("label,value", FABRICATING,
                             ids=[f[0] for f in FABRICATING])
    def test_the_domain_does_not_reach_gap_domains(self, label, value):
        state, _ = validate_state(raw_state(value), EV)
        assert "sweat" not in signals_from_state(state).gap_domains, label

    @pytest.mark.parametrize("label,value", FABRICATING,
                             ids=[f[0] for f in FABRICATING])
    def test_it_buys_no_question_slot(self, label, value):
        """审计量到的就是这一条：伪造与真实都得 84 分，门槛 40。"""
        state, _ = validate_state(raw_state(value), EV)
        required = differential_required_domains(state)
        coverage = assess_coverage(TEXT, required | {"sweat"})
        produced = [c.domain for c in
                    governed_question_candidates(required, coverage, [])]
        assert "sweat" not in produced, label

    def test_a_string_of_non_matching_characters_was_always_harmless(self):
        """'xyz' 改前改后都得空——这条不是 R6 修的，是用来划边界的。"""
        assert gaps_for("xyz") == []

    def test_the_whole_differential_survives_a_fabricating_shape(self):
        for _, value in FABRICATING:
            state, _ = validate_state(raw_state(value), EV)
            assert state is not None
            assert [h.pattern_name for h in state.hypotheses] == ["A", "B"]


# ======================================================================
# 二、崩溃面关上了
# ======================================================================

class TestCrashingShapes:

    @pytest.mark.parametrize("value", [7, True, 1.5],
                             ids=["int", "bool", "float"])
    def test_it_does_not_raise(self, value):
        state, _ = validate_state(raw_state(value), EV)
        assert state is not None
        assert state.evidence_gaps == []

    @pytest.mark.parametrize("value", [7, True],
                             ids=["int", "bool"])
    def test_the_differential_is_not_discarded(self, value):
        """以前这里抛 TypeError，assembler 再丢掉整份状态。"""
        state, _ = validate_state(raw_state(value), EV)
        assert len(state.hypotheses) == 2
        assert len(state.hypotheses[0].supporting_evidence) == 1


# ======================================================================
# 三、去重：合法输入上的那个缺陷
# ======================================================================

class TestDeduplication:

    def test_a_duplicate_no_longer_crowds_out_a_real_separator(self):
        """本阶段的核心：['A','A','B'] 曾经变成 ['A','A']，B 被挤掉。"""
        assert seps(["A", "A", "B"]) == ["A", "B"]

    def test_duplicates_collapse(self):
        assert seps(["A", "A"]) == ["A"]
        assert seps(["B", "B", "B"]) == ["B"]

    def test_first_seen_order_is_preserved(self):
        assert seps(["B", "A"]) == ["B", "A"]
        assert seps(["B", "B", "A"]) == ["B", "A"]
        assert seps(["A", "B"]) == ["A", "B"]

    def test_a_single_separator_is_still_accepted(self):
        """刻意不加 >=2 的要求：今天单个分隔者会产生一个追问。"""
        assert seps(["A"]) == ["A"]
        assert gaps_for(["A"]) == [{"domain": "sweat", "separates": ["A"]}]

    def test_a_gap_naming_one_reading_twice_is_not_two_readings(self):
        assert seps(["A", "A"]) == ["A"]
        assert len(seps(["A", "A"])) == 1


# ======================================================================
# 四、合法输入平价
# ======================================================================

class TestValidInputParity:

    @pytest.mark.parametrize("label,value", [
        ("missing", MISSING), ("None", None), ("[]", []),
        ('""', ""), ("{}", {}), ("0", 0), ("False", False)],
        ids=["missing", "None", "empty", "empty_str", "empty_dict", "zero",
             "false"])
    def test_absent_shapes_yield_no_gap(self, label, value):
        assert gaps_for(value) == [], label

    def test_a_valid_list_is_unchanged(self):
        assert gaps_for(["A", "B"]) == [{"domain": "sweat",
                                         "separates": ["A", "B"]}]

    def test_a_tuple_is_accepted(self):
        assert gaps_for(("A", "B")) == [{"domain": "sweat",
                                         "separates": ["A", "B"]}]

    def test_junk_inside_a_list_is_dropped_as_before(self):
        assert seps(["A", 7, None, "B"]) == ["A", "B"]
        assert seps([7, None, {}]) is None

    def test_names_nobody_holds_are_dropped_as_before(self):
        assert gaps_for(["X", "Y"]) == []

    def test_whitespace_is_stripped_as_before(self):
        assert seps(["  A  ", "B"]) == ["A", "B"]

    def test_a_later_valid_name_is_not_lost(self):
        """审计因此否决了 _names：它只预扫 limit*2 个元素。"""
        assert seps(["X", "Y", "Z", "W", "A"]) == ["A"]

    def test_the_two_name_cap_is_preserved(self):
        raw = raw_state(gaps=[{"domain": "sweat",
                               "separates": ["A", "B", "C"]}])
        raw["hypotheses"].append(
            {"pattern_name": "C", "standing": "PLAUSIBLE",
             "supporting_evidence": [{"origin": "COMPLAINT"}],
             "unresolved_discriminators": []})
        state, _ = validate_state(raw, EV)
        assert state.evidence_gaps[0]["separates"] == ["A", "B"]

    def test_max_gaps_is_unchanged(self):
        assert MAX_GAPS == 6
        gaps = [{"domain": "d%d" % i, "separates": ["A", "B"]}
                for i in range(10)]
        state, _ = validate_state(raw_state(gaps=gaps), EV)
        assert len(state.evidence_gaps) == MAX_GAPS

    def test_the_named_filter_still_uses_every_hypothesis_not_only_live(self):
        """live 过滤属于 differential_required_domains，不属于这里。"""
        for standing in ("PLAUSIBLE", "WEAKENED", "RULED_OUT_FOR_NOW"):
            assert seps(["A", "B"], standing_b=standing) == ["A", "B"], standing

    def test_the_validated_shape_is_always_a_list(self):
        for value in (MISSING, None, [], 7, True, {"A": 1}, {"A"}, "AB",
                      ["A", "B"], ("A", "B")):
            state, _ = validate_state(raw_state(value), EV)
            assert isinstance(state.evidence_gaps, list)
            for gap in state.evidence_gaps:
                assert isinstance(gap["separates"], list)


# ======================================================================
# 五、真实 gap 一切照旧
# ======================================================================

class TestGenuineGapUnaffected:

    def test_it_still_reaches_gap_domains_and_scores(self):
        state, _ = validate_state(raw_state(["A", "B"]), EV)
        assert "sweat" in signals_from_state(state).gap_domains
        assert "sweat" in differential_required_domains(state)

    def test_the_weight_is_unchanged(self):
        assert WEIGHT_ENVELOPE_GAP == 20
        state, _ = validate_state(raw_state(["A", "B"]), EV)
        required = differential_required_domains(state)
        coverage = assess_coverage(TEXT, required)
        signals = signals_from_state(state)
        sweat = [c for c in governed_question_candidates(required, coverage, [])
                 if c.domain == "sweat"][0]
        assert cov.score_candidate(
            sweat, coverage, signals).score >= MIN_ADAPTIVE_SCORE

    def test_a_gap_whose_names_are_all_dead_still_yields_no_eligibility(self):
        raw = raw_state(["A", "B"])
        for h in raw["hypotheses"]:
            h["standing"] = "RULED_OUT_FOR_NOW"
        state, _ = validate_state(raw, EV)
        assert differential_required_domains(state) == set()


# ======================================================================
# 六、R3 / R4 / R5 不变量
# ======================================================================

class TestEarlierPhasesIntact:

    def alts_state(self, alternatives):
        raw = raw_state(["A", "B"])
        raw["hypotheses"][0]["unresolved_discriminators"] = [
            dict(DISC, also_resolved_by=alternatives)]
        return validate_state(raw, EV)[0]

    def test_r3_alternatives_still_reach_gap_domains(self):
        state = self.alts_state(["nose", "throat"])
        assert {"thirst", "nose", "throat"} <= set(
            signals_from_state(state).gap_domains)

    def test_r4_malformed_alternatives_still_degrade_locally(self):
        for value in (7, True, {"a": "nose"}, {"nose"}, "nose"):
            state = self.alts_state(value)
            disc = state.hypotheses[0].unresolved_discriminators[0]
            assert disc["also_resolved_by"] == [], repr(value)
            assert disc["separates"] == ["A", "B"]

    def test_r5_sites_still_degrade_without_raising(self):
        for field in ("hypotheses", "evidence_gaps"):
            raw = raw_state(["A", "B"])
            raw[field] = 7
            state, _ = validate_state(raw, EV)
            if field == "hypotheses":
                assert state is None
            else:
                assert state.evidence_gaps == []
        for field in ("supporting_evidence", "contradicting_evidence",
                      "unresolved_discriminators"):
            raw = raw_state(["A", "B"])
            raw["hypotheses"][0][field] = 7
            state, _ = validate_state(raw, EV)
            assert state is not None

    def test_r5_helper_is_untouched(self):
        assert _as_sequence([1, 2]) == [1, 2]
        assert _as_sequence((1, 2)) == (1, 2)
        for value in (None, "", "nose", 7, 0, True, False, {}, {"a": 1},
                      {"x", "y"}, 1.5):
            assert _as_sequence(value) == (), repr(value)
        original = [1, 2, 3]
        assert _as_sequence(original) is original


# ======================================================================
# 七、周边不变量
# ======================================================================

class TestSurroundingInvariantsUnchanged:

    def test_the_floor_is_unchanged(self):
        assert MIN_ADAPTIVE_SCORE == 40

    def test_the_budget_is_unchanged(self):
        genuine, _ = validate_state(raw_state(["A", "B"]), EV)
        fabricated, _ = validate_state(raw_state("AB"), EV)
        coverage = assess_coverage(TEXT, set(cov.DOMAINS_BY_KEY))
        assert assess_sufficiency(
            coverage, signals_from_state(genuine), False)[1] == 3
        assert assess_sufficiency(
            coverage, signals_from_state(fabricated), False)[1] == 3

    def test_preference_rank_is_untouched(self):
        coverage = assess_coverage(TEXT, set(cov.DOMAINS_BY_KEY))
        assert preference_rank("cold_heat", coverage)[0] == 1   # PARTIAL
        assert preference_rank("throat", coverage)[0] == 0      # UNKNOWN

    def test_known_domains_remain_excluded(self):
        state, _ = validate_state(raw_state(["A", "B"]), EV)
        required = differential_required_domains(state)
        coverage = assess_coverage("发热无汗身痛", required)
        assert coverage.states.get("sweat") == KNOWN
        produced = [c.domain for c in
                    governed_question_candidates(required, coverage, [])]
        assert "sweat" not in produced

    def test_generic_coverage_still_fills_capacity(self):
        state, _ = validate_state(raw_state("AB"), EV)
        required = differential_required_domains(state)
        coverage = assess_coverage("咳嗽三天", required)
        selection = select_questions(
            deterministic=[], adaptive=[], coverage=coverage,
            signals=signals_from_state(state), differential_domains=required)
        assert selection.sufficiency == cov.NEEDS_MORE_INFORMATION
        assert selection.fallback_used > 0

    def test_no_uncertainty_or_reason_code_policy_changed(self):
        """R6 不新增 note。伪造的 gap 现在是"没有 gap"，不是"有问题的 gap"。"""
        _, notes = validate_state(raw_state("AB"), EV)
        assert notes == []
        _, notes = validate_state(raw_state(["A", "B"]), EV)
        assert notes == []

    def test_nothing_consumer_visible_or_authoritative_was_added(self):
        import inspect

        from app.services.interview import differential as diff

        source = inspect.getsource(diff.validate_state)
        for forbidden in ("clinical_ranking_eligible", "VERIFIED", "REVIEWED",
                          "ready_for_formula_retrieval", "consumer_purchasable",
                          "PATTERN_FORMULA", "FORMULA_HERB", "commit(",
                          "insert("):
            assert forbidden not in source, forbidden
