"""X1D-LEGACYDIAG4.6-R3：被提名的替代域也是一个 gap。

4.6 让 also_resolved_by 拓宽了**资格**——一个替代域会进 required_domains，
也会拿到一条受治理的默认问题——却没有把这件事告诉排序层。于是它带着
differential=0 到达打分环节，就像鉴别里从没提过它一样。

自适应分数下限正好卡在"资格"与"排序"之间。staging 两次实测量到同一个后果：
咽喉得 36 分，下限是 40，于是它在进入分层与偏好比较之前就被丢掉；其中一次
剩下的候选恰好等于预算，4.6 的偏好层压根没有东西可选。

修法是最小的那一种：模型说"这个域同样能分开这一对"，这正是 gap_domains
本来就在表达的主张，所以复用同一个类别与同一份权重，不新增第二套信号。

下面钉住的不只是"替代域现在有信号了"，更是"别的东西一律没动"——主鉴别域、
无关域、下限、预算、KNOWN 抑制，以及 4.6 自己的偏好仲裁。
"""
import pytest

from app.schemas.reasoning import ResolvableEvidence
from app.services.clarification import coverage as cov
from app.services.clarification.coverage import (
    KNOWN,
    MIN_ADAPTIVE_SCORE,
    NOT_RELEVANT,
    Candidate,
    DifferentialSignals,
    assess_coverage,
    assess_sufficiency,
    governed_question_candidates,
    preference_rank,
    resolve_domain,
    select_questions,
)
from app.services.interview.differential import (
    differential_required_domains,
    signals_from_state,
    validate_state,
)

TEXT = "咳嗽三天，有点发热，身上酸痛。"


def hyp(name, standing="PLAUSIBLE", discs=()):
    return {"pattern_name": name, "standing": standing,
            "supporting_evidence": [{"origin": "COMPLAINT"}],
            "unresolved_discriminators": list(discs)}


def disc(domain, sep=("A", "B"), alts=(), present=None, absent=None):
    entry = {"domain": domain, "separates": list(sep),
             "if_present_supports": list(present or [sep[0]]),
             "if_absent_supports": list(absent or [sep[1]]),
             "rationale": "r"}
    if alts:
        entry["also_resolved_by"] = list(alts)
    return entry


def build(*hypotheses):
    return validate_state({"hypotheses": list(hypotheses)}, ResolvableEvidence())


def scored(candidate, coverage, signals):
    result = cov.score_candidate(candidate, coverage, signals)
    return result.score if result else None


def candidate_for(domain):
    resolved, certain = resolve_domain(domain, "q")
    return Candidate(field=domain, question="q", domain=resolved,
                     kind="adaptive", domain_certain=certain,
                     payload={"field": domain})


# ======================================================================
# 替代域进入信号
# ======================================================================

class TestAlternativesBecomeGaps:

    def test_a_validated_alternative_enters_gap_domains(self):
        state, _ = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("thirst", alts=["nose", "sputum", "throat"])]),
            hyp("B"))
        gaps = set(signals_from_state(state).gap_domains)
        assert {"nose", "sputum", "throat"} <= gaps
        assert "thirst" in gaps          # 主鉴别域仍在

    def test_no_eligible_domain_is_left_without_a_signal(self):
        """资格与信号不再脱节——这正是本阶段要消除的那道缝。"""
        state, _ = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("thirst", alts=["nose", "sputum", "throat"])]),
            hyp("B"))
        required = differential_required_domains(state)
        signals = signals_from_state(state)
        unsignalled = (required - set(signals.gap_domains)
                       - set(signals.discriminating_domains)
                       - set(signals.contradiction_domains))
        assert unsignalled == set()

    def test_the_alternative_now_clears_the_floor(self):
        """staging 实测的那一格：咽喉 36 -> 56。"""
        state, _ = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("thirst", alts=["nose", "sputum", "throat"])]),
            hyp("B"))
        required = differential_required_domains(state)
        coverage = assess_coverage(TEXT, required)
        signals = signals_from_state(state)
        throat = [c for c in governed_question_candidates(required, coverage, [])
                  if c.domain == "throat"][0]
        assert scored(throat, coverage, signals) >= MIN_ADAPTIVE_SCORE

    def test_it_reuses_the_existing_gap_weight(self):
        """没有第二套信号，也没有第二份权重。"""
        assert cov.WEIGHT_ENVELOPE_GAP == 20
        assert not hasattr(cov.DifferentialSignals, "alternative_domains")
        fields = cov.DifferentialSignals.__dataclass_fields__
        assert set(fields) == {"gap_domains", "contradiction_domains",
                               "discriminating_domains"}


# ======================================================================
# 别的东西一律没动
# ======================================================================

class TestNothingElseMoved:

    def _state(self):
        return build(hyp("A", "PRIMARY_WORKING",
                         [disc("thirst", alts=["nose", "sputum", "throat"])]),
                     hyp("B"))

    def test_the_primary_discriminator_scores_the_same(self):
        """主鉴别域本来就在 gaps 里，集合多了别的成员不影响它。"""
        state, _ = self._state()
        coverage = assess_coverage(TEXT, differential_required_domains(state))
        narrow = DifferentialSignals(gap_domains=frozenset({"thirst"}))
        wide = signals_from_state(state)
        primary = candidate_for("thirst")
        assert scored(primary, coverage, narrow) == scored(primary, coverage, wide)

    def test_an_unrelated_domain_scores_the_same(self):
        """既不是主鉴别域也不是替代域的域，分数不得变化。"""
        state, _ = self._state()
        coverage = assess_coverage(TEXT, differential_required_domains(state)
                                   | {"cold_heat"})
        narrow = DifferentialSignals(gap_domains=frozenset({"thirst"}))
        wide = signals_from_state(state)
        other = candidate_for("cold_heat")
        assert scored(other, coverage, narrow) == scored(other, coverage, wide)

    def test_the_adaptive_floor_is_unchanged(self):
        assert MIN_ADAPTIVE_SCORE == 40

    def test_a_weak_candidate_still_falls_below_the_floor(self):
        """没有任何信号支撑的域，依旧上不了台面。"""
        coverage = assess_coverage(TEXT, {"throat"})
        bare = DifferentialSignals()
        throat = [c for c in governed_question_candidates({"throat"}, coverage, [])
                  if c.domain == "throat"][0]
        assert scored(throat, coverage, bare) < MIN_ADAPTIVE_SCORE

    def test_the_budget_is_unchanged(self):
        state, _ = self._state()
        coverage = assess_coverage(TEXT, differential_required_domains(state))
        narrow = DifferentialSignals(gap_domains=frozenset({"thirst"}))
        wide = signals_from_state(state)
        assert (assess_sufficiency(coverage, narrow, False)
                == assess_sufficiency(coverage, wide, False))

    def test_the_narrow_empty_gap_case(self):
        """H3 的那一格：gaps 本来为空，替代域把它填成非空。

        只有在覆盖已经完备时 assess_sufficiency 才会读到 gap_domains，而且
        只当成布尔量。这里把两种输入都跑一遍，确保预算要么不变，要么只在
        "确实还有东西要narrow"这个正确方向上变化。
        """
        coverage = assess_coverage("发热无汗身痛口渴头痛大便干小便黄"
                                   "睡眠差食欲差胸闷耳鸣咽痛鼻塞黄痰",
                                   set(cov.DOMAINS_BY_KEY))
        empty = DifferentialSignals()
        filled = DifferentialSignals(gap_domains=frozenset({"throat"}))
        suff_empty, budget_empty = assess_sufficiency(coverage, empty, False)
        suff_filled, budget_filled = assess_sufficiency(coverage, filled, False)
        # 非空 gaps 至多把结论推向"还需要narrow"，绝不会放宽预算
        assert budget_filled <= cov.MAX_ADAPTIVE_WHEN_OPEN
        assert budget_empty <= cov.MAX_ADAPTIVE_WHEN_OPEN
        if suff_empty == cov.SUFFICIENT_FOR_REASONING:
            assert budget_empty == 0

    def test_known_and_not_relevant_suppression_is_unchanged(self):
        """已经答过的域不会因为多了一个信号就被重新问。"""
        state, _ = build(
            hyp("A", "PRIMARY_WORKING", [disc("sweat", alts=["nose"])]),
            hyp("B"))
        required = differential_required_domains(state)
        coverage = assess_coverage("发热无汗身痛", required)
        assert coverage.states.get("sweat") == KNOWN
        produced = [c.domain for c in
                    governed_question_candidates(required, coverage, [])]
        assert "sweat" not in produced
        sweat = candidate_for("sweat")
        assert scored(sweat, coverage, signals_from_state(state)) is None


# ======================================================================
# 无效的替代域什么也拿不到
# ======================================================================

class TestInvalidAlternativesGainNothing:

    def test_an_unnormalisable_alternative_contributes_no_signal(self):
        state, _ = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("thirst", alts=["not_a_domain", "完全不是一个域"])]),
            hyp("B"))
        gaps = set(signals_from_state(state).gap_domains)
        assert gaps == {"thirst"}
        assert all(g in cov.DOMAINS_BY_KEY for g in gaps)

    def test_a_discriminator_with_no_live_competition_contributes_nothing(self):
        """验证阶段就丢掉的鉴别项，连同它的替代域一起消失。"""
        state, notes = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("thirst", ("A", "GHOST"), alts=["nose", "throat"],
                      present=["GHOST"], absent=["A"])]),
            hyp("B"))
        assert "DISCRIMINATOR_NO_LIVE_COMPETITION" in notes
        assert set(signals_from_state(state).gap_domains) == set()

    def test_a_state_with_no_hypotheses_yields_no_signal(self):
        assert signals_from_state(None) == DifferentialSignals()

    def test_malformed_alternatives_are_survivable(self):
        """畸形的 also_resolved_by 原样送进生产代码，不经测试辅助函数整形。

        只覆盖能走完校验的形状。非序列（int / bool）在 4.6 的
        _validate_discriminator 里就会抛 TypeError，那是本阶段之前就存在的
        缺陷，见下面那条单独钉住它的用例。
        """
        for junk in (None, "", {}, ["  "], [None], [7, {}], "nose"):
            entry = {"domain": "thirst", "separates": ["A", "B"],
                     "if_present_supports": ["A"], "if_absent_supports": ["B"],
                     "rationale": "r", "also_resolved_by": junk}
            state, _ = build(hyp("A", "PRIMARY_WORKING", [entry]), hyp("B"))
            gaps = set(signals_from_state(state).gap_domains)
            assert gaps <= set(cov.DOMAINS_BY_KEY), (junk, gaps)
            assert None not in gaps

    @pytest.mark.parametrize("junk", [7, True])
    def test_a_non_sequence_alternative_still_raises_today(self, junk):
        """既有缺陷，本阶段刻意不修，只是钉住它别被悄悄改掉。

        _validate_discriminator 对 also_resolved_by 直接切片：

            (entry.get("also_resolved_by") or [])[:MAX_DISCRIMINATORS * 2]

        送进一个 int 或 bool 就会抛 TypeError。assembler 外层有兜底，所以
        一轮问诊不会失败，但**整份**工作鉴别会被丢掉，而不只是这一条畸形的
        鉴别项。修它属于另一个阶段的范围。
        """
        entry = {"domain": "thirst", "separates": ["A", "B"],
                 "if_present_supports": ["A"], "if_absent_supports": ["B"],
                 "rationale": "r", "also_resolved_by": junk}
        with pytest.raises(TypeError):
            build(hyp("A", "PRIMARY_WORKING", [entry]), hyp("B"))


# ======================================================================
# 4.6 的偏好仲裁仍然成立
# ======================================================================

class TestPreferenceStillArbitrates:

    def _pipeline(self, text=TEXT):
        state, _ = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("thirst", alts=["nose", "sputum", "throat"])]),
            hyp("B"))
        required = differential_required_domains(state)
        signals = signals_from_state(state)
        coverage = assess_coverage(text, required)
        cands = governed_question_candidates(required, coverage, [])
        selection = select_questions(deterministic=[], adaptive=cands,
                                     coverage=coverage, signals=signals,
                                     differential_domains=required)
        return required, coverage, signals, cands, selection

    def test_a_genuine_post_floor_surplus_now_exists(self):
        """这正是 R2 想证明却证不出来的那一步。"""
        required, coverage, signals, cands, selection = self._pipeline()
        above = [c for c in cands
                 if (scored(c, coverage, signals) or 0) >= MIN_ADAPTIVE_SCORE]
        assert len(above) > selection.adaptive_budget

    def test_the_primary_is_not_displaced_by_its_own_alternatives(self):
        required, coverage, signals, cands, selection = self._pipeline()
        chosen = {s.candidate.domain for s in selection.scores
                  if s.candidate.kind == "adaptive"}
        assert "thirst" in chosen

    def test_selection_is_deterministic(self):
        first = self._pipeline()[4]
        second = self._pipeline()[4]
        assert ([s.candidate.domain for s in first.scores]
                == [s.candidate.domain for s in second.scores])

    def test_preference_still_orders_within_the_tier(self):
        """偏好键没有被改动：PARTIAL 仍然排在 UNKNOWN 之后。"""
        coverage = assess_coverage(TEXT, set(cov.DOMAINS_BY_KEY))
        assert coverage.states.get("cold_heat") == cov.PARTIAL
        assert preference_rank("cold_heat", coverage)[0] == 1
        assert preference_rank("throat", coverage)[0] == 0

    def test_generic_coverage_still_fills_spare_capacity(self):
        """不得形成只问鉴别项的隧道。"""
        state, _ = build(hyp("A", "PRIMARY_WORKING"), hyp("B"))
        required = differential_required_domains(state)
        assert required == set()
        coverage = assess_coverage("咳嗽三天", required)
        selection = select_questions(
            deterministic=[], adaptive=[], coverage=coverage,
            signals=signals_from_state(state), differential_domains=required)
        assert selection.sufficiency == cov.NEEDS_MORE_INFORMATION
        assert selection.fallback_used > 0


# ======================================================================
# 治理边界未被触碰
# ======================================================================

class TestGovernanceUnchanged:

    def test_no_authority_surface_was_added(self):
        import inspect

        from app.services.interview import differential as diff

        source = inspect.getsource(diff.signals_from_state)
        for forbidden in ("clinical_ranking_eligible", "VERIFIED", "REVIEWED",
                          "ready_for_formula_retrieval", "consumer_purchasable",
                          "PATTERN_FORMULA", "FORMULA_HERB", "commit", "insert"):
            assert forbidden not in source

    def test_required_domains_logic_was_not_touched(self):
        """资格判定不属于本阶段的改动范围。"""
        state, _ = build(
            hyp("A", "PRIMARY_WORKING",
                [disc("thirst", alts=["nose", "sputum", "throat"])]),
            hyp("B"))
        assert differential_required_domains(state) == {
            "thirst", "nose", "sputum", "throat"}
