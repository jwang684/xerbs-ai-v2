from app.schemas.intake import RecommendationRequest
from app.schemas.recommendation import FormulaCandidate, ModelProvenance, PatternHypothesis, RecommendationResponse
from app.services.knowledge.persistent_clinical import PersistentClinicalStore
from app.services.knowledge.tse_repository import TSEKnowledgeRepository
from app.services.llm.provider import LLMProvider
from app.services.safety.engine import SafetyEngine
from app.schemas.safety import SafetyScreenRequest
from app.services.reasoning.engine import DiagnosticReasoningEngine

PROMPT_VERSION = "xerbs-v2-recommendation-0.10-convergence-base44-contract"

class RecommendationAssembler:
    def __init__(self, provider: LLMProvider):
        self.provider=provider; self.corpus=PersistentClinicalStore(); self.tse=TSEKnowledgeRepository(); self.safety=SafetyEngine(); self.reasoning=DiagnosticReasoningEngine(self.corpus)

    async def generate(self, request: RecommendationRequest) -> RecommendationResponse:
        result=await self.provider.generate_recommendation(text_input=request.text_input,symptoms=request.symptoms,goals=request.goals,constraints=request.constraints,image_data=request.image_data,language=request.language)
        reasoning=self.reasoning.analyze(request,result.pattern_hypotheses)
        uncertainty_flags=list(dict.fromkeys([*result.uncertainty_flags,*reasoning.uncertainty_flags]))
        candidates=[]
        # Model candidates are preserved as hypotheses, but deterministic corpus retrieval is preferred when a reviewed pattern is resolved.
        # Backward-compatible reviewed indication retrieval remains available even
        # when pattern verification is incomplete. It is explicitly flagged and
        # never promoted to a verified pattern conclusion.
        verified_pattern_ids=[x.pattern_id for x in reasoning.pattern_assessments if x.corpus_match and x.pattern_id]
        relationship_matches=self.corpus.eligible_formula_candidates_for_patterns(verified_pattern_ids) if reasoning.ready_for_formula_retrieval else []
        reviewed_matches=[] if relationship_matches else self.corpus.eligible_formula_candidates(request.symptoms,request.text_input)
        if relationship_matches:
            candidates=relationship_matches
            uncertainty_flags.extend(["REVIEWED_PATTERN_FORMULA_RELATIONSHIP_RETRIEVAL","REVIEWED_CLINICAL_CORPUS","AI_FORMULA_RANKING_NOT_USED"])
        elif reviewed_matches:
            candidates=reviewed_matches
            if not reasoning.ready_for_formula_retrieval:
                uncertainty_flags.append("REVIEWED_INDICATION_RETRIEVAL_WITH_INCOMPLETE_REASONING")
            else:
                uncertainty_flags.append("NO_REVIEWED_PATTERN_FORMULA_RELATIONSHIP_MATCH")
            uncertainty_flags.extend(["REVIEWED_CLINICAL_CORPUS","AI_FORMULA_RANKING_NOT_USED"])
        elif result.formula_candidates:
            candidates=result.formula_candidates; uncertainty_flags.append("MODEL_FORMULA_CANDIDATES_REQUIRE_VERIFICATION")
        else:
            uncertainty_flags.append("NO_FORMULA_CANDIDATE")
        enriched=[]
        for candidate in candidates:
            item=dict(candidate); item["catalog_matches"]=self.tse.map_formula_candidate(item.get("name",""),item.get("ingredients",[]))
            assessment=self.safety.screen(SafetyScreenRequest(formula_id=item.get("formula_id"),formula_name=item.get("name",""),ingredients=item.get("ingredients",[]),constraints=request.constraints,patient_context=request.patient_context))
            item["safety_assessment"]=assessment.model_dump()
            if assessment.findings: item["safety_flags"]=list(dict.fromkeys([*item.get("safety_flags",[]),*[f.message for f in assessment.findings]]))
            if not assessment.eligible_for_selection: uncertainty_flags.append("SAFETY_BLOCKING_FINDINGS")
            enriched.append(item)
        summary=result.summary
        if reasoning.followup_questions and not reasoning.ready_for_formula_retrieval:
            summary=(summary+" Missing information should be clarified before deterministic corpus formula retrieval.").strip()
        return RecommendationResponse(request_id=request.request_id,summary=summary,pattern_hypotheses=[PatternHypothesis(**x) for x in result.pattern_hypotheses],formula_candidates=[FormulaCandidate(**x) for x in enriched],uncertainty_flags=list(dict.fromkeys(uncertainty_flags)),model_confidence=result.model_confidence,provenance=ModelProvenance(provider=result.provider,model=result.model,prompt_version=PROMPT_VERSION),requires_practitioner_review=True,reasoning=reasoning)
