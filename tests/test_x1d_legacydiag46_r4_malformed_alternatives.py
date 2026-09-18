"""X1D-LEGACYDIAG4.6-R4：一个畸形字段只赔上它自己。

_validate_discriminator 过去这样读 also_resolved_by：

    (entry.get("also_resolved_by") or [])[:MAX_DISCRIMINATORS * 2]

`or []` 只挡得住假值。真值里的非序列会一路走到切片上：7 与 True 抛
TypeError，非空 dict 抛 KeyError（dict[slice]），set 抛 TypeError。这里没有
任何 try，异常一路穿出 validate_state，被 assembler 的兜底接住，于是**整份**
工作鉴别被丢弃，只留下一个 DIFFERENTIAL_STATE_UNAVAILABLE。模型答对的每一
条假设、每一处引用、每一个鉴别项，都被一个装饰性字段带走了。

R4 补上的正是 _names 早就在 separates / if_present_supports /
if_absent_supports 上用的那道类型判断——also_resolved_by 是本函数里唯一一个
列表形状却没有这道判断的字段。降级成 [] 之后鉴别项本身留下来，按剩下的内容
受审，这正是 validate_state 给自己定的规矩：fail toward less，而不是 toward
nothing。

一条刻意不做的事：裸字符串 "nose" 仍然得到 []，不会被拆成 ["nose"]。契约写
的是数组，接受字符串是新增行为而不是加固，_names 对同样的形状也是这么判的。
"""
import pytest

from app.schemas.reasoning import ResolvableEvidence
from app.services.clarification import coverage as cov
from app.services.clarification.coverage import (
    KNOWN, MIN_ADAPTIVE_SCORE, assess_coverage, assess_sufficiency,
    governed_question_candidates, preference_rank, select_questions,
)
from app.services.interview.differential import (
    MAX_DISCRIMINATORS, differential_required_domains, signals_from_state,
    validate_state,
)

TEXT = "咳嗽三天，有点发热，身上酸痛。"
MISSING = object()


def state_for(alternatives=MISSING, domain="thirst"):
    entry = {"domain": domain, "separates": ["A", "B"],
             "if_present_supports": ["A"], "if_absent_supports": ["B"],
             "rationale": "r"}
    if alternatives is not MISSING:
        entry["also_resolved_by"] = alternatives
    raw = {"hypotheses": [
        {"pattern_name": "A", "standing": "PRIMARY_WORKING",
         "supporting_evidence": [{"origin": "COMPLAINT"}],
         "unresolved_discriminators": [entry]},
        {"pattern_name": "B", "standing": "PLAUSIBLE",
         "supporting_evidence": [{"origin": "COMPLAINT"}],
         "unresolved_discriminators": []}]}
    return validate_state(raw, ResolvableEvidence())


def alternatives_of(state):
    return state.hypotheses[0].unresolved_discriminators[0]["also_resolved_by"]


# ======================================================================
# 畸形输入一律降级，且不带走鉴别项
# ======================================================================

MALFORMED = [
    ("int", 7),
    ("bool True", True),
    ("truthy dict", {"a": "nose"}),
    ("set", {"nose", "throat"}),
    ("bare string", "nose"),
    ("empty string", ""),
    ("empty dict", {}),
    ("falsy int", 0),
    ("float", 1.5),
    ("object", object()),
]


class TestMalformedDegradesToEmpty:

    @pytest.mark.parametrize("label,value", MALFORMED,
                             ids=[m[0] for m in MALFORMED])
    def test_it_degrades_to_an_empty_list(self, label, value):
        state, _ = state_for(value)
        assert state is not None, label
        assert alternatives_of(state) == [], label

    @pytest.mark.parametrize("label,value", MALFORMED,
                             ids=[m[0] for m in MALFORMED])
    def test_the_discriminator_itself_survives(self, label, value):
        """只赔上替代域，不赔上鉴别项。"""
        state, _ = state_for(value)
        kept = state.hypotheses[0].unresolved_discriminators
        assert len(kept) == 1, label
        assert kept[0]["domain"] == "thirst", label
        assert kept[0]["separates"] == ["A", "B"], label
        assert kept[0]["discriminating"] is True, label

    @pytest.mark.parametrize("label,value", MALFORMED,
                             ids=[m[0] for m in MALFORMED])
    def test_the_whole_differential_is_not_lost(self, label, value):
        """以前这里会抛异常，assembler 再把整份状态丢掉。"""
        state, notes = state_for(value)
        assert state is not None, label
        assert len(state.hypotheses) == 2, label
        assert differential_required_domains(state) == {"thirst"}, label

    @pytest.mark.parametrize("label,value", MALFORMED,
                             ids=[m[0] for m in MALFORMED])
    def test_validate_state_does_not_raise(self, label, value):
        """整个类别都不再抛异常——这是本阶段的全部目的。"""
        state_for(value)   # 抛出即失败

    def test_a_bare_string_is_not_coerced(self):
        """契约写的是数组。接受字符串属于新增行为，本阶段刻意不做。"""
        state, _ = state_for("nose")
        assert alternatives_of(state) == []
        assert "nose" not in signals_from_state(state).gap_domains


# ======================================================================
# 合法输入一字未改
# ======================================================================

class TestValidInputUnchanged:

    def test_missing_none_and_empty_remain_empty(self):
        for value in (MISSING, None, []):
            state, _ = state_for(value)
            assert alternatives_of(state) == []

    def test_a_valid_list_is_unchanged(self):
        state, _ = state_for(["nose"])
        assert alternatives_of(state) == ["nose"]
        state, _ = state_for(["nose", "throat"])
        assert alternatives_of(state) == ["nose", "throat"]

    def test_a_tuple_is_still_accepted(self):
        state, _ = state_for(("nose", "throat"))
        assert alternatives_of(state) == ["nose", "throat"]

    def test_junk_inside_a_list_normalises_away(self):
        state, _ = state_for([7, {}])
        assert alternatives_of(state) == []

    def test_a_mixed_list_keeps_only_canonical_domains(self):
        state, _ = state_for(["nose", 7, "not_a_domain", "throat", None])
        assert alternatives_of(state) == ["nose", "throat"]

    def test_order_and_dedup_are_preserved(self):
        state, _ = state_for(["throat", "nose", "throat"])
        assert alternatives_of(state) == ["throat", "nose"]

    def test_the_primary_is_still_excluded_from_its_own_alternatives(self):
        state, _ = state_for(["thirst", "nose"])
        assert alternatives_of(state) == ["nose"]

    def test_the_slice_cap_is_preserved(self):
        """切片仍在归一化之前，上限仍是 MAX_DISCRIMINATORS * 2。"""
        assert MAX_DISCRIMINATORS == 4
        oversized = (["nose", "throat", "sputum", "sweat", "thirst",
                      "cold_heat", "diet", "sleep"]          # 前 8 个
                     + ["ear_eye", "excretion", "head_body"])  # 被切掉
        state, _ = state_for(oversized)
        kept = alternatives_of(state)
        assert "ear_eye" not in kept          # 超出切片
        assert "excretion" not in kept
        assert "thirst" not in kept           # 主鉴别域被排除
        assert kept == ["nose", "throat", "sputum", "sweat",
                        "cold_heat", "diet", "sleep"]


# ======================================================================
# R3 的 gap 平价未受影响
# ======================================================================

class TestR3ParityIntact:

    def test_valid_alternatives_still_enter_gap_domains(self):
        state, _ = state_for(["nose", "throat"])
        gaps = set(signals_from_state(state).gap_domains)
        assert {"thirst", "nose", "throat"} <= gaps

    def test_degraded_alternatives_contribute_no_gap(self):
        state, _ = state_for(7)
        assert set(signals_from_state(state).gap_domains) == {"thirst"}

    def test_required_domains_follow_the_same_rule(self):
        valid, _ = state_for(["nose", "throat"])
        assert differential_required_domains(valid) == {
            "thirst", "nose", "throat"}
        degraded, _ = state_for({"a": "nose"})
        assert differential_required_domains(degraded) == {"thirst"}

    def test_scoring_of_a_valid_alternative_is_unchanged(self):
        """R3 的 +20 gap 权重照旧。"""
        state, _ = state_for(["throat"])
        required = differential_required_domains(state)
        coverage = assess_coverage(TEXT, required)
        signals = signals_from_state(state)
        throat = [c for c in governed_question_candidates(required, coverage, [])
                  if c.domain == "throat"][0]
        assert cov.score_candidate(throat, coverage, signals).score >= MIN_ADAPTIVE_SCORE


# ======================================================================
# 周边不变量
# ======================================================================

class TestSurroundingInvariantsUnchanged:

    def test_the_floor_is_unchanged(self):
        assert MIN_ADAPTIVE_SCORE == 40

    def test_the_budget_is_unchanged_for_degraded_input(self):
        valid, _ = state_for(["nose", "throat"])
        degraded, _ = state_for(7)
        coverage = assess_coverage(TEXT, set(cov.DOMAINS_BY_KEY))
        assert (assess_sufficiency(coverage, signals_from_state(valid), False)[1]
                == assess_sufficiency(coverage,
                                      signals_from_state(degraded), False)[1])

    def test_known_domains_remain_excluded(self):
        state, _ = state_for(7, domain="sweat")
        required = differential_required_domains(state)
        coverage = assess_coverage("发热无汗身痛", required)
        assert coverage.states.get("sweat") == KNOWN
        produced = [c.domain for c in
                    governed_question_candidates(required, coverage, [])]
        assert "sweat" not in produced

    def test_preference_rank_is_untouched(self):
        coverage = assess_coverage(TEXT, set(cov.DOMAINS_BY_KEY))
        assert preference_rank("cold_heat", coverage)[0] == 1   # PARTIAL
        assert preference_rank("throat", coverage)[0] == 0      # UNKNOWN

    def test_generic_coverage_still_fills_capacity(self):
        state, _ = state_for(7)
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

        source = inspect.getsource(diff._validate_discriminator)
        for forbidden in ("clinical_ranking_eligible", "VERIFIED", "REVIEWED",
                          "ready_for_formula_retrieval", "consumer_purchasable",
                          "PATTERN_FORMULA", "FORMULA_HERB", "commit", "insert"):
            assert forbidden not in source

    def test_the_validated_representation_is_always_a_list(self):
        """下游三处读的都是这个字段，形状必须恒定。"""
        for value in (MISSING, None, [], 7, True, {"a": 1}, {"nose"},
                      "nose", ["nose"], ("nose",)):
            state, _ = state_for(value)
            assert isinstance(alternatives_of(state), list)
