from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import select
from app.db.models import DiagnosticInterview, DiagnosticInterviewTurn
from app.db.session import get_session_factory
from app.schemas.intake import RecommendationRequest
from app.schemas.interview import InterviewState, InterviewQuestion, InterviewAnswerRequest
from app.schemas.reasoning import ReasoningResponse
from app.services.reasoning.engine import DiagnosticReasoningEngine
from app.services.reasoning.convergence import PatternConvergenceEngine
from app.services.llm.provider import LLMProvider

QUESTION_BANK = {
    "duration": "症状持续多久？何时开始？",
    "temperature": "目前怕冷、怕热或有发热吗？",
    "appetite": "近期食欲和进食情况如何？",
    "stool": "大便情况如何？",
    "sleep": "近期睡眠情况如何？",
}
PRIORITY_SCORE = {"HIGH": 1.0, "MEDIUM": 0.72, "LOW": 0.45}
CONVERGENCE_READY_THRESHOLD = 0.72

class InterviewNotFoundError(ValueError):
    pass

class AdaptiveInterviewEngine:
    def __init__(self, provider: LLMProvider, reasoning: DiagnosticReasoningEngine | None = None):
        self.Session = get_session_factory()
        self.provider = provider
        self.reasoning = reasoning or DiagnosticReasoningEngine()
        self.convergence = PatternConvergenceEngine()

    async def _reason(self, intake: RecommendationRequest):
        result=await self.provider.generate_recommendation(
            text_input=intake.text_input,symptoms=intake.symptoms,goals=intake.goals,constraints=intake.constraints,
            image_data=intake.image_data,language=intake.language
        )
        return self.reasoning.analyze(intake,result.pattern_hypotheses)

    async def start(self, intake: RecommendationRequest, max_questions_per_round: int = 3) -> InterviewState:
        reasoning = await self._reason(intake)
        convergence = self.convergence.evaluate(reasoning, None)
        reasoning.convergence = convergence
        questions = self._select_questions(reasoning, set(), max_questions_per_round)
        now = datetime.now(timezone.utc)
        interview_id = f"int-{uuid4().hex[:16]}"
        status = self._status(reasoning, questions, convergence.score)
        with self.Session.begin() as s:
            row = DiagnosticInterview(
                id=interview_id,
                status=status,
                intake_snapshot=intake.model_dump(),
                reasoning_snapshot=reasoning.model_dump(),
                current_turn=0,
                convergence_score=convergence.score,
                convergence_snapshot=convergence.model_dump(),
                created_at=now,
                updated_at=now,
            )
            s.add(row)
            s.add(DiagnosticInterviewTurn(
                id=f"turn-{uuid4().hex[:16]}", interview_id=interview_id, turn_number=0,
                answers=[], asked_questions=[q.model_dump() for q in questions],
                reasoning_snapshot=reasoning.model_dump(), created_at=now,
            ))
        return self.get(interview_id)

    async def answer(self, interview_id: str, request: InterviewAnswerRequest) -> InterviewState:
        with self.Session.begin() as s:
            row = s.get(DiagnosticInterview, interview_id)
            if row is None:
                raise InterviewNotFoundError("Interview not found")
            if row.status == "COMPLETED":
                raise ValueError("Interview already completed")
            turns = list(s.scalars(select(DiagnosticInterviewTurn).where(DiagnosticInterviewTurn.interview_id==interview_id).order_by(DiagnosticInterviewTurn.turn_number)).all())
            asked = {q.get("question_id") for t in turns for q in (t.asked_questions or [])}
            allowed = {q.get("question_id"): q for t in turns for q in (t.asked_questions or [])}
            for a in request.answers:
                if a.question_id not in allowed:
                    raise ValueError(f"Unknown question_id: {a.question_id}")
            previous_reasoning=ReasoningResponse(**row.reasoning_snapshot)
            intake = RecommendationRequest(**row.intake_snapshot)
            for a in request.answers:
                q=allowed[a.question_id]
                if a.answer.strip():
                    intake.text_input = (intake.text_input + "\n" + f"{q.get('field')}：{a.answer.strip()}").strip()
            reasoning = await self._reason(intake)
            answered_ids={a.question_id for a in request.answers}
            answered_fields={allowed[a.question_id].get("field") for a in request.answers}
            reasoning.missing_information=[m for m in reasoning.missing_information if m.field not in answered_fields]
            reasoning.followup_questions=[QUESTION_BANK.get(m.field,m.reason) for m in reasoning.missing_information]
            if reasoning.missing_information:
                if "MISSING_CLINICAL_INFORMATION" not in reasoning.uncertainty_flags:
                    reasoning.uncertainty_flags.append("MISSING_CLINICAL_INFORMATION")
            else:
                reasoning.uncertainty_flags=[x for x in reasoning.uncertainty_flags if x!="MISSING_CLINICAL_INFORMATION"]
            reasoning.ready_for_formula_retrieval=bool(reasoning.pattern_assessments) and any(x.corpus_match for x in reasoning.pattern_assessments) and not any(x.priority=="HIGH" for x in reasoning.missing_information) and not any(x.contradictions for x in reasoning.pattern_assessments)
            convergence=self.convergence.evaluate(reasoning, previous_reasoning)
            reasoning.convergence=convergence
            if convergence.pattern_stability < .5 and previous_reasoning.pattern_assessments:
                if "PATTERN_HYPOTHESIS_UNSTABLE" not in reasoning.uncertainty_flags:
                    reasoning.uncertainty_flags.append("PATTERN_HYPOTHESIS_UNSTABLE")
            next_questions=self._select_questions(reasoning, asked | answered_ids, request.max_questions_per_round)
            row.current_turn += 1
            row.intake_snapshot=intake.model_dump()
            row.reasoning_snapshot=reasoning.model_dump()
            row.convergence_score=convergence.score
            row.convergence_snapshot=convergence.model_dump()
            row.status=self._status(reasoning,next_questions,convergence.score)
            row.updated_at=datetime.now(timezone.utc)
            s.add(DiagnosticInterviewTurn(
                id=f"turn-{uuid4().hex[:16]}", interview_id=interview_id, turn_number=row.current_turn,
                answers=[a.model_dump() for a in request.answers], asked_questions=[q.model_dump() for q in next_questions],
                reasoning_snapshot=reasoning.model_dump(), created_at=row.updated_at,
            ))
        return self.get(interview_id)

    def complete(self, interview_id: str) -> InterviewState:
        with self.Session.begin() as s:
            row=s.get(DiagnosticInterview,interview_id)
            if row is None: raise InterviewNotFoundError("Interview not found")
            row.status="COMPLETED"; row.updated_at=datetime.now(timezone.utc)
        return self.get(interview_id)

    def get(self, interview_id: str) -> InterviewState:
        with self.Session() as s:
            row=s.get(DiagnosticInterview,interview_id)
            if row is None: raise InterviewNotFoundError("Interview not found")
            latest=s.scalar(select(DiagnosticInterviewTurn).where(DiagnosticInterviewTurn.interview_id==interview_id).order_by(DiagnosticInterviewTurn.turn_number.desc()).limit(1))
            reasoning=ReasoningResponse(**row.reasoning_snapshot)
            return InterviewState(
                interview_id=row.id,status=row.status,intake=RecommendationRequest(**row.intake_snapshot),
                reasoning=reasoning,next_questions=[InterviewQuestion(**x) for x in (latest.asked_questions if latest else [])],
                turn_count=row.current_turn,convergence_score=row.convergence_score,convergence=reasoning.convergence,
                created_at=row.created_at,updated_at=row.updated_at,
            )

    def _select_questions(self, reasoning, already_asked:set[str], limit:int):
        ranked=[]
        for m in reasoning.missing_information:
            qid=f"field:{m.field}"
            if qid in already_asked: continue
            base=PRIORITY_SCORE.get(m.priority,0.5)
            bonus={"duration":.05,"temperature":.04,"appetite":.03,"stool":.02,"sleep":.01}.get(m.field,0)
            ranked.append((min(1.0,base+bonus),m.field,qid,QUESTION_BANK.get(m.field,m.reason),m.priority))
        ranked.sort(key=lambda x:(-x[0],x[1]))
        return [InterviewQuestion(question_id=qid,field=field,question=q,priority=priority,information_gain_score=score) for score,field,qid,q,priority in ranked[:limit]]

    def _status(self, reasoning, questions, convergence_score:float):
        if reasoning.ready_for_formula_retrieval and convergence_score>=CONVERGENCE_READY_THRESHOLD:
            return "READY_FOR_RECOMMENDATION"
        if not questions and not reasoning.missing_information and not any(x.contradictions for x in reasoning.pattern_assessments):
            return "READY_FOR_RECOMMENDATION"
        return "OPEN"
