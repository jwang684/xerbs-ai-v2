"""X1D-LEGACYDIAG3.2.1: the depth escape, made reachable and unambiguous.

What was wrong
--------------
LEGACYDIAG3.2 shipped the escape and said so: the router checked a turn number
the intake contract did not carry, so it was implemented, tested and dead. The
only live exit from interview mode was COVERAGE_SUFFICIENT, and a case whose
coverage never converges could interview forever.

The unit, stated once
---------------------
``interview_depth`` counts the governed turns this case has ALREADY COMPLETED,
excluding the submission being routed.

    turn 1  depth 0  ->  INTERVIEW
    turn 2  depth 1  ->  INTERVIEW
    turn 3  depth 2  ->  INTERVIEW
    turn 4  depth 3  ->  FULL_REASONING / INTERVIEW_DEPTH_REACHED

So ``depth >= MAX_INTERVIEW_TURNS`` reads as "three interview turns have
happened; that is enough", and there is no off-by-one to argue about. The
tests below walk that table explicitly rather than trusting the arithmetic.

Who is allowed to say
---------------------
core, from its own trace rows. Not the browser, which also sends a turn_count
on the form -- handing the depth limit to the party being limited is not a
limit. Not the model, whose information_sufficient stays advisory. Several
tests exist only to pin that.

What the escape does not mean
-----------------------------
"Enough information exists for a safe recommendation." It means "stop asking
and reason with what we have". The full-reasoning attempt that follows may
legitimately end at NEEDS_MORE_INFORMATION, NO_CANDIDATE, SAFETY_BLOCKED or
anything else the governed pipeline decides -- it takes no shortcut because
the interview budget ran out.
"""

import ast
import inspect
import textwrap

import pytest

from app.schemas.intake import RecommendationRequest
from app.services.interview import mode as interview_mode
from app.services.interview.mode import (
    FULL_REASONING,
    INTERVIEW,
    MAX_INTERVIEW_TURNS,
    REASON_COVERAGE_SUFFICIENT,
    REASON_DEPTH_REACHED,
    REASON_MATERIALLY_MISSING,
    decide_mode,
    describe_policy,
)

SPARSE = "咳嗽发热3天。"
RICH = ("发热3天，怕冷无汗，咳嗽有白痰，口不渴，食欲正常，大便正常，"
        "睡眠可，头身酸痛，胸不闷，受凉后起病。")


def route(depth, text=SPARSE):
    """Route one turn at a given governed depth, with coverage still open."""
    return decide_mode(accumulated_text=text, missing_information=[],
                       interview_depth=depth)


def _executable_code(obj):
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
    return ast.unparse(tree)


# ======================================================================
# A-E: the turn-by-turn table
# ======================================================================

class TestTheDepthTable:
    def test_a_a_new_case_starts_at_depth_zero(self):
        """Turn 1 has no completed turns behind it."""
        assert RecommendationRequest(text_input=SPARSE).interview_depth == 0

    def test_b_turn_one_interviews(self):
        assert route(0) == (INTERVIEW, REASON_MATERIALLY_MISSING)

    def test_c_turn_two_interviews(self):
        assert route(1) == (INTERVIEW, REASON_MATERIALLY_MISSING)

    def test_d_turn_three_interviews(self):
        assert route(2) == (INTERVIEW, REASON_MATERIALLY_MISSING)

    def test_e_turn_four_takes_the_escape(self):
        assert route(3) == (FULL_REASONING, REASON_DEPTH_REACHED)

    def test_e2_and_every_turn_after_it(self):
        for depth in range(MAX_INTERVIEW_TURNS, MAX_INTERVIEW_TURNS + 5):
            assert route(depth) == (FULL_REASONING, REASON_DEPTH_REACHED)

    def test_exactly_three_interview_turns_are_allowed(self):
        """The whole point, counted rather than reasoned about."""
        interviews = [d for d in range(0, 10) if route(d)[0] == INTERVIEW]
        assert interviews == [0, 1, 2]
        assert len(interviews) == MAX_INTERVIEW_TURNS

    def test_the_escape_is_actually_reachable_from_the_contract(self):
        """The 3.2 defect: the router read a field the contract lacked."""
        assert "interview_depth" in RecommendationRequest.model_fields
        signature = inspect.signature(decide_mode)
        assert "interview_depth" in signature.parameters

    def test_the_assembler_reads_the_contract_field(self):
        from app.services.recommendation import assembler as module

        code = _executable_code(module.RecommendationAssembler.generate)
        assert 'getattr(request, \'interview_depth\', 0)' in code or \
               'getattr(request, "interview_depth", 0)' in code
        assert "turn_count" not in code

    def test_the_unit_is_documented_where_the_limit_lives(self):
        assert "already completed" in describe_policy()["depth_unit"]


# ======================================================================
# F: coverage can still exit earlier
# ======================================================================

class TestScenarioF_EarlyExit:
    def test_sufficient_coverage_exits_before_the_limit(self):
        assert decide_mode(accumulated_text=RICH, missing_information=[],
                           interview_depth=0) == (FULL_REASONING,
                                                  REASON_COVERAGE_SUFFICIENT)

    def test_the_early_exit_is_not_the_depth_reason(self):
        _mode, reason = decide_mode(accumulated_text=RICH,
                                    missing_information=[],
                                    interview_depth=1)
        assert reason == REASON_COVERAGE_SUFFICIENT

    def test_depth_is_checked_before_coverage(self):
        """At the limit the answer is the same whatever coverage says."""
        for text in (SPARSE, RICH):
            assert route(MAX_INTERVIEW_TURNS, text) == (FULL_REASONING,
                                                        REASON_DEPTH_REACHED)


# ======================================================================
# J-K: nobody outside core may set the depth
# ======================================================================

class TestScenarioJK_NotClientOrModelAuthoritative:
    def test_j_the_router_does_not_read_a_client_turn_count(self):
        code = _executable_code(interview_mode)
        assert "turn_count" not in code

    def test_j2_the_assembler_does_not_read_a_client_turn_count(self):
        from app.services.recommendation import assembler as module

        assert "turn_count" not in _executable_code(module)

    def test_j3_the_contract_carries_no_client_turn_field(self):
        """turn_count stays a core-side form value; it never reaches ai-v2."""
        assert "turn_count" not in RecommendationRequest.model_fields

    def test_j4_the_depth_is_bounded_by_the_type(self):
        """A caller cannot buy unlimited interviewing with a huge number."""
        with pytest.raises(Exception):
            RecommendationRequest(text_input="x", interview_depth=-1)
        with pytest.raises(Exception):
            RecommendationRequest(text_input="x", interview_depth=10_000)

    def test_j5_a_large_depth_only_ever_ends_the_interview(self):
        """The direction of the bound is safe: more depth never buys more."""
        assert route(50)[0] == FULL_REASONING

    def test_k_the_model_cannot_alter_the_depth(self):
        code = _executable_code(interview_mode)
        for forbidden in ("information_sufficient", "ProviderResult",
                          "clarification_proposals", "await "):
            assert forbidden not in code

    def test_k2_routing_happens_before_any_call(self):
        from app.services.recommendation import assembler as module

        code = _executable_code(module.RecommendationAssembler.generate)
        assert code.index("decide_mode(") < code.index("generate_interview(")
        assert code.index("decide_mode(") < code.index("generate_recommendation(")

    def test_k3_the_advisory_flag_is_still_only_advisory(self):
        from app.schemas.reasoning import InterviewReasoning

        assert "information_sufficient" in InterviewReasoning.model_fields
        assert "information_sufficient" not in _executable_code(interview_mode)


# ======================================================================
# L-M: the escape grants nothing
# ======================================================================

class TestScenarioLM_NoAuthority:
    def test_l_full_reasoning_after_the_escape_is_the_ordinary_path(self):
        """The escape picks a prompt. It does not pick an outcome."""
        mode, _ = route(MAX_INTERVIEW_TURNS)
        assert mode == FULL_REASONING
        from app.services.llm import openai_compatible as oc

        code = _executable_code(oc.OpenAICompatibleProvider.generate_recommendation)
        assert "SYSTEM_PROMPT" in code

    def test_l2_the_envelope_and_corpus_gate_still_run(self):
        from app.services.recommendation import assembler as module

        code = _executable_code(module.RecommendationAssembler.generate)
        assert "ClinicalReasoningEnvelope(" in code
        assert "eligible_formula_candidates" in code
        assert "MODEL_FORMULA_CANDIDATES_SUPPRESSED_UNVERIFIED_CORPUS" in code

    @pytest.mark.parametrize("forbidden", [
        "consumer_purchasable", "clinical_ranking_eligible", "safety_verdict",
        "REVIEWED", "review_status", "SafetyEngine", "formula_id",
        "eligible_formula_candidates", "corpus_match"])
    def test_m_the_depth_escape_names_no_authority(self, forbidden):
        assert forbidden not in _executable_code(interview_mode)

    def test_m2_the_escape_cannot_shortcut_readiness(self):
        code = _executable_code(interview_mode)
        assert "ready_for_formula_retrieval" not in code

    def test_m3_the_reason_is_a_routing_label_not_a_clinical_claim(self):
        assert REASON_DEPTH_REACHED == "INTERVIEW_DEPTH_REACHED"
        for claim in ("SUFFICIENT", "READY", "ELIGIBLE", "SAFE", "VERIFIED"):
            assert claim not in REASON_DEPTH_REACHED


# ======================================================================
# N-O: 3.2 behaviour below the threshold is untouched
# ======================================================================

class TestScenarioNO_Unchanged:
    def test_n_interview_routing_below_the_threshold_is_unchanged(self):
        for depth in range(MAX_INTERVIEW_TURNS):
            assert route(depth) == (INTERVIEW, REASON_MATERIALLY_MISSING)

    def test_n2_the_interview_prompt_is_unchanged(self):
        from app.services.llm import openai_compatible as oc

        assert "at most 3" in oc.INTERVIEW_SYSTEM_PROMPT
        assert "Two good questions beat four" in oc.INTERVIEW_SYSTEM_PROMPT
        assert "interview_depth" not in oc.INTERVIEW_SYSTEM_PROMPT

    def test_o_the_clarify_pipeline_is_unchanged(self):
        from app.services.clarification import coverage as cov
        from app.services.clarification.validator import (
            FIELD_PATTERN, MAX_PROPOSALS_PER_TURN, REJECTION_REASONS)

        assert cov.MAX_VISIBLE_QUESTIONS_PER_TURN == 4
        assert cov.MAX_ADAPTIVE_WHEN_OPEN == 3
        assert cov.MIN_QUESTIONS_WHEN_INSUFFICIENT == 2
        assert MAX_PROPOSALS_PER_TURN == 3
        assert len(REJECTION_REASONS) == 10
        assert FIELD_PATTERN.pattern == r"^[a-z][a-z0-9_]{1,38}[a-z0-9]$"

    def test_o2_the_router_still_reports_every_reason(self):
        policy = describe_policy()
        assert set(policy["reasons"]) == {
            "PROVIDER_DOES_NOT_SUPPORT_INTERVIEW", REASON_DEPTH_REACHED,
            REASON_MATERIALLY_MISSING, REASON_COVERAGE_SUFFICIENT}

    def test_o3_routing_stays_reproducible(self):
        for depth in range(5):
            first = route(depth)
            for _ in range(5):
                assert route(depth) == first
