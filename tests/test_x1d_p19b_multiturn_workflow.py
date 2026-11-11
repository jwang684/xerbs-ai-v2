"""X1D-P19B-FIX: the multi-turn patient workflow, replayed with synthetic data.

Staging records #313-#315 (a real browser test) never reached a syndrome or a
formula. Two confirmed ai-v2 causes:

  * turn 3 was still routed to INTERVIEW (MAX_INTERVIEW_TURNS was 3) although
    core shows adaptive questions on turns 1-2 only, so it produced questions
    nobody would see and no hypotheses;
  * earlier rounds' answers were invisible to routing, coverage and the
    full-reasoning call, because the turn text holds only the latest round.

This suite replays that sequence with an INTERVIEW-CAPABLE stub (the P19R
suite's stub cannot interview, so it never exercised this path). Every
clinical object is a SYNTHETIC in-memory fixture; nothing is a patient record.
The stub's hypotheses are scripted per case: the point is what the pipeline
does with a given model output, never which syndrome the model "should" pick.
"""

import asyncio

import pytest

from app.schemas.intake import RecommendationRequest
from app.services.clarification.coverage import KNOWN, UNKNOWN, assess_coverage
from app.services.governance import applicability as A
from app.services.interview import answers as carried
from app.services.interview.mode import (FULL_REASONING, INTERVIEW, MAX_INTERVIEW_TURNS,
                                         REASON_DEPTH_REACHED, decide_mode)
from app.services.knowledge.resolver import KnowledgeResolver
from app.services.llm.provider import INTERVIEW as P_INTERVIEW, LLMProvider, ProviderResult
from app.services.reasoning import primary as P
from app.services.reasoning.engine import DiagnosticReasoningEngine
from app.services.recommendation.assembler import RecommendationAssembler
from tests.test_x1d_p19r_approved_cough_contract import cough_slice
from tests.test_x1d_p7_source_verification import env, store  # noqa: F401

COMPLAINT = "合成病例：咳嗽，发热，怕冷，胸闷，痰多4天"
R1 = [("sweat", "你这几天有出汗吗？", "否"), ("nose", "有没有鼻塞或流鼻涕？", "流清涕"),
      ("throat", "你有咽喉疼痛吗？", "是")]
R2 = [("thirst", "是否口渴？", "口干但不想喝"), ("head_body", "是否有头痛、头晕或身体酸痛？", "头痛")]
TURN1_PROPOSALS = [
    {"field": "sweat", "question": "你这几天有出汗吗？", "answer_type": "yes_no", "priority": "medium"},
    {"field": "nose", "question": "有没有鼻塞或流鼻涕？鼻涕是清的还是黄的？", "answer_type": "single_choice",
     "choices": ["鼻塞为主", "流清涕", "流黄涕", "打喷嚏", "都没有"], "priority": "medium"},
    {"field": "throat", "question": "你有咽喉疼痛吗？", "answer_type": "yes_no", "priority": "high"},
]
YES_NONE = {"cough_primary": "YES", "cough_exclusion_features": "NONE"}


class InterviewStub(LLMProvider):
    """Interview-capable and recording. Scripted output; no inference."""

    def __init__(self, full_hyps=(), proposals=()):
        self.full_hyps = list(full_hyps)
        self.proposals = list(proposals)
        self.calls = []

    async def generate_interview(self, *, text_input, symptoms, language, on_display_text=None,
                                 carry_state=None, known_domains=None):
        self.calls.append(("INTERVIEW", {"text_input": text_input, "known_domains": known_domains,
                                         "carry_state": carry_state}))
        return ProviderResult(summary="synthetic interview", clarification_proposals=list(self.proposals),
                              provider="stub", model="p19b-interview", inference_purpose=P_INTERVIEW)

    async def generate_recommendation(self, *, text_input, symptoms, goals, constraints, image_data,
                                      language, on_display_text=None, patient_answers=None):
        self.calls.append(("FULL_REASONING", {"text_input": text_input, "image_data": image_data,
                                              "patient_answers": patient_answers}))
        return ProviderResult(summary="synthetic full reasoning", pattern_hypotheses=list(self.full_hyps),
                              model_confidence=max([h["confidence"] for h in self.full_hyps] or [0.0]),
                              provider="stub", model="p19b-full")


def hyp(name, confidence):
    return {"name": name, "confidence": confidence, "reasoning": "synthetic"}


def text_for(round_answers):
    """What core sends today: the complaint plus THIS round's answers only."""
    if not round_answers:
        return COMPLAINT
    return COMPLAINT + "。补充信息：" + "；".join("%s %s" % (q, a) for _, q, a in round_answers)


def carry(*rounds, extra=()):
    """core's interview_state: structured answers from governed history."""
    answers = [{"turn_id": i + 1, "question_field": f, "answer": a}
               for i, r in enumerate(rounds) for f, _, a in r]
    answers += list(extra)
    return {"prior": None, "resolvable_evidence": {"has_complaint": True, "observations": [], "answers": answers}}


def run(store, provider, depth, round_answers=(), state=None, context=None):
    asm = RecommendationAssembler(provider)
    asm.corpus = store
    asm.safety.Session = store.Session
    asm.reasoning = DiagnosticReasoningEngine(store, KnowledgeResolver(store, record_gaps=False))
    req = RecommendationRequest(text_input=text_for(round_answers), interview_depth=depth,
                                interview_state=state, condition_context=context)
    return asyncio.run(asm.generate(req))


def asked(resp):
    return [q.field for q in resp.reasoning.clarification_questions]


# ----------------------------------------------------------------------
# The budget: two interview turns, then full reasoning
# ----------------------------------------------------------------------
def test_the_interview_budget_matches_core_two_rounds_of_questions():
    assert MAX_INTERVIEW_TURNS == 2
    routes = [decide_mode(accumulated_text=COMPLAINT, interview_depth=d)[0] for d in range(5)]
    assert routes == [INTERVIEW, INTERVIEW, FULL_REASONING, FULL_REASONING, FULL_REASONING]
    assert decide_mode(accumulated_text=COMPLAINT, interview_depth=2) == (FULL_REASONING, REASON_DEPTH_REACHED)


# 1-4: the #313-#315 sequence
def test_1_turn_one_interviews_and_asks_a_validated_first_round(store):
    stub = InterviewStub(proposals=TURN1_PROPOSALS)
    resp = run(store, stub, 0)
    assert [c[0] for c in stub.calls] == ["INTERVIEW"]
    assert resp.pattern_hypotheses == [] and resp.formula_candidates == []
    # ranking among the three is the existing selector's job (unchanged here);
    # core's replay pins the exact recorded 3 -> 2 question rounds
    assert len(asked(resp)) >= 2 and set(asked(resp)) <= {"sweat", "nose", "throat"}


def test_2_turn_two_interviews_without_re_asking_round_one(store):
    stub = InterviewStub(proposals=[TURN1_PROPOSALS[0]])           # the model re-proposes sweat
    resp = run(store, stub, 1, R1, state=carry(R1))
    assert [c[0] for c in stub.calls] == ["INTERVIEW"]
    assert not {"sweat", "nose", "throat"} & set(asked(resp))
    assert {"sweat", "nose", "throat"} <= set(stub.calls[0][1]["known_domains"])


def test_3_turn_three_runs_full_reasoning_with_every_earlier_answer(store):
    stub = InterviewStub(full_hyps=[hyp("风寒束肺", 0.7)])
    run(store, stub, 2, R2, state=carry(R1, R2))
    assert [c[0] for c in stub.calls] == ["FULL_REASONING"]
    got = {a["question_field"]: a["answer"] for a in stub.calls[0][1]["patient_answers"]}
    assert got == {"sweat": "否", "nose": "流清涕", "throat": "是", "thirst": "口干但不想喝", "head_body": "头痛"}
    assert stub.calls[0][1]["image_data"] is None                  # 15: no image inference


def test_4_turn_three_never_returns_an_unanswerable_interview(store):
    resp = run(store, InterviewStub(full_hyps=[hyp("风寒束肺", 0.7)]), 2, R2, state=carry(R1, R2))
    assert "INFERENCE_MODE_INTERVIEW" not in resp.uncertainty_flags
    assert resp.pattern_hypotheses                                 # the model's output is carried, unforced


# 5-6: answers carried and not repeated
def test_5_earlier_answers_reach_routing_and_coverage():
    text = text_for(R2)
    bare = assess_coverage(text)
    assert {bare.states[d] for d in ("sweat", "nose", "throat")} == {UNKNOWN}
    req = RecommendationRequest(text_input=text, interview_state=carry(R1, R2))
    with_history = assess_coverage(text, answered_domains=carried.answered_domains(req))
    assert all(with_history.states[d] == KNOWN for d in ("sweat", "nose", "throat", "thirst", "head_body"))


def test_6_carried_answers_dedupe_ignore_blanks_and_never_invent_negatives():
    req = RecommendationRequest(text_input="x", interview_state=carry(extra=[
        {"turn_id": 1, "question_field": "sweat", "answer": "否"},
        {"turn_id": 2, "question_field": "sweat", "answer": "容易出汗"},      # later answer wins
        {"turn_id": 2, "question_field": "thirst", "answer": "  "},          # blank: not an answer
        {"turn_id": 2, "question_field": "", "answer": "是"},                # no field: dropped
        {"turn_id": 2, "question_field": "cough_primary", "answer": "是"},   # governance input, excluded
        {"turn_id": 2, "question_field": "nose", "answer": "不清楚"},          # kept as the patient said it
    ]))
    got = carried.patient_answers(req)
    assert [(a["question_field"], a["answer"]) for a in got] == [("sweat", "容易出汗"), ("nose", "不清楚")]
    assert "thirst" not in carried.answered_domains(req)


def test_6b_no_history_means_no_carried_answers():
    assert carried.patient_answers(RecommendationRequest(text_input="x")) == []
    assert carried.answered_domains(RecommendationRequest(text_input="x")) == set()


# ----------------------------------------------------------------------
# 7-12: what turn 3 does with the model's output (primary-anchored, unforced)
# ----------------------------------------------------------------------
def test_7_9_no_hypothesis_or_ambiguous_primary_fails_closed(store):
    cough_slice(store)
    for hyps in ([], [hyp("风热犯肺", 0.5), hyp("风寒束表", 0.5)]):
        resp = run(store, InterviewStub(full_hyps=hyps), 2, R2, state=carry(R1, R2), context=YES_NONE)
        assert resp.formula_candidates == [] and resp.reasoning.condition_scope_required == []


def test_10_an_unmatched_wind_cold_primary_never_receives_formula_8(store):
    cough_slice(store)
    resp = run(store, InterviewStub(full_hyps=[hyp("风寒束肺", 0.7), hyp("风热犯肺", 0.3)]), 2, R2,
               state=carry(R1, R2), context=YES_NONE)
    assert resp.formula_candidates == []
    assert P.PRIMARY_PATTERN_NOT_IN_GOVERNED_CORPUS in resp.uncertainty_flags
    assert resp.reasoning.condition_scope_required == []           # no P19C question for a non-primary


def test_11_a_wind_heat_primary_triggers_the_approved_first_question(store):
    cough_slice(store)
    resp = run(store, InterviewStub(full_hyps=[hyp("风热犯肺", 0.8)]), 2, R2, state=carry(R1, R2))
    assert resp.reasoning.condition_scope_required == ["cough_primary"]
    assert resp.formula_candidates == []


@pytest.mark.parametrize("ctx, expected", [
    ({"cough_primary": "YES"}, ["cough_exclusion_features"]),
    (YES_NONE, None),
])
def test_11b_answers_advance_the_gate(store, ctx, expected):
    cough_slice(store)
    resp = run(store, InterviewStub(full_hyps=[hyp("风热犯肺", 0.8)]), 3, (), state=carry(R1, R2), context=ctx)
    if expected:
        assert resp.reasoning.condition_scope_required == expected and resp.formula_candidates == []
    else:
        assert [c.name for c in resp.formula_candidates] == ["桑菊饮加减"]
        assert resp.formula_candidates[0].governance["applicability_contract"] == "X1D-P19C-V1"


@pytest.mark.parametrize("ctx", [{"cough_primary": "NO"}, {"cough_primary": "UNKNOWN"},
                                 {"cough_primary": "YES", "cough_exclusion_features": "PRESENT"},
                                 {"cough_primary": "YES", "cough_exclusion_features": "UNKNOWN"}])
def test_12_negative_or_unknown_applicability_answers_fail_closed(store, ctx):
    cough_slice(store)
    resp = run(store, InterviewStub(full_hyps=[hyp("风热犯肺", 0.8)]), 3, (), state=carry(R1, R2), context=ctx)
    assert resp.formula_candidates == [] and resp.reasoning.condition_scope_required == []
    assert A.FLAG_NOT_SATISFIED in resp.uncertainty_flags


def test_condition_scope_answers_never_reach_the_model(store):
    cough_slice(store)
    stub = InterviewStub(full_hyps=[hyp("风热犯肺", 0.8)])
    run(store, stub, 3, (), context=YES_NONE, state=carry(R1, R2, extra=[
        {"turn_id": 3, "question_field": "cough_primary", "answer": "是"},
        {"turn_id": 4, "question_field": "cough_exclusion_features", "answer": "都没有"}]))
    fields = {a["question_field"] for a in stub.calls[0][1]["patient_answers"]}
    assert not fields & {"cough_primary", "cough_exclusion_features"}


def test_a_provider_without_the_new_argument_still_works(store):
    """Providers that predate patient_answers are called exactly as before."""
    class Legacy(LLMProvider):
        async def generate_recommendation(self, *, text_input, symptoms, goals, constraints, image_data, language,
                                          on_display_text=None):
            return ProviderResult(summary="legacy", pattern_hypotheses=[hyp("风寒束肺", 0.7)])
    resp = run(store, Legacy(), 2, R2, state=carry(R1, R2))
    assert resp.pattern_hypotheses
