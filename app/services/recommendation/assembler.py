from app.schemas.intake import RecommendationRequest
from app.schemas.recommendation import (
    FormulaCandidate,
    ModelProvenance,
    PatternHypothesis,
    RecommendationResponse,
)
from app.schemas.safety import SafetyScreenRequest
from app.services.knowledge.persistent_clinical import PersistentClinicalStore
from app.services.knowledge.tse_repository import TSEKnowledgeRepository
from app.services.llm.provider import LLMProvider
from app.services.reasoning.engine import DiagnosticReasoningEngine
from app.services.safety.engine import SafetyEngine


PROMPT_VERSION = "xerbs-v2-recommendation-0.10-convergence-base44-contract"


class RecommendationAssembler:
    """
    Assemble the final Xerbs recommendation response.

    Safety / governance boundary:

    1. The LLM may generate diagnostic pattern hypotheses.
    2. The LLM may internally suggest formula hypotheses.
    3. A formula may enter the public `formula_candidates` collection only when
       it is retrieved from the reviewed clinical corpus.
    4. Model-generated formulas that are not backed by reviewed corpus records
       are suppressed from selectable candidates.
    5. TSE catalog data is metadata only and never makes a formula clinically
       ranking eligible.
    6. Safety screening is performed only after a formula has passed the
       reviewed-corpus retrieval gate.
    7. TrustScore is not calculated or owned by this service.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.corpus = PersistentClinicalStore()
        self.tse = TSEKnowledgeRepository()
        self.safety = SafetyEngine()
        self.reasoning = DiagnosticReasoningEngine(self.corpus)

    async def generate(
        self,
        request: RecommendationRequest,
    ) -> RecommendationResponse:
        # -------------------------------------------------------------
        # 1. Generate model hypotheses.
        #
        # Model output is hypothesis-level intelligence only.
        # It is not a clinically verified recommendation.
        # -------------------------------------------------------------
        result = await self.provider.generate_recommendation(
            text_input=request.text_input,
            symptoms=request.symptoms,
            goals=request.goals,
            constraints=request.constraints,
            image_data=request.image_data,
            language=request.language,
        )

        # -------------------------------------------------------------
        # 2. Run deterministic diagnostic reasoning against the
        #    governed clinical corpus.
        # -------------------------------------------------------------
        reasoning = self.reasoning.analyze(
            request,
            result.pattern_hypotheses,
        )

        uncertainty_flags = list(
            dict.fromkeys(
                [
                    *result.uncertainty_flags,
                    *reasoning.uncertainty_flags,
                ]
            )
        )

        # -------------------------------------------------------------
        # 3. Determine whether reviewed corpus formulas are available.
        #
        # Preferred path:
        #
        # verified pattern
        #   -> reviewed PATTERN_FORMULA relationship
        #
        # Compatibility path:
        #
        # reviewed indication retrieval
        #
        # Raw model formulas are NOT allowed to become selectable
        # formula candidates.
        # -------------------------------------------------------------
        candidates = []

        verified_pattern_ids = [
            assessment.pattern_id
            for assessment in reasoning.pattern_assessments
            if assessment.corpus_match and assessment.pattern_id
        ]

        relationship_matches = []

        if reasoning.ready_for_formula_retrieval and verified_pattern_ids:
            relationship_matches = (
                self.corpus.eligible_formula_candidates_for_patterns(
                    verified_pattern_ids
                )
            )

        # Backward-compatible reviewed indication retrieval.
        #
        # This retrieval is still governed by corpus eligibility.
        # It is used only when reviewed pattern relationships did not
        # produce a result.
        reviewed_matches = []

        if not relationship_matches:
            reviewed_matches = self.corpus.eligible_formula_candidates(
                request.symptoms,
                request.text_input,
            )

        # -------------------------------------------------------------
        # 4. Formula-selection governance gate.
        # -------------------------------------------------------------
        if relationship_matches:
            candidates = relationship_matches

            uncertainty_flags.extend(
                [
                    "REVIEWED_PATTERN_FORMULA_RELATIONSHIP_RETRIEVAL",
                    "REVIEWED_CLINICAL_CORPUS",
                    "AI_FORMULA_RANKING_NOT_USED",
                ]
            )

        elif reviewed_matches:
            candidates = reviewed_matches

            if not reasoning.ready_for_formula_retrieval:
                uncertainty_flags.append(
                    "REVIEWED_INDICATION_RETRIEVAL_WITH_INCOMPLETE_REASONING"
                )
            else:
                uncertainty_flags.append(
                    "NO_REVIEWED_PATTERN_FORMULA_RELATIONSHIP_MATCH"
                )

            uncertainty_flags.extend(
                [
                    "REVIEWED_CLINICAL_CORPUS",
                    "AI_FORMULA_RANKING_NOT_USED",
                ]
            )

        elif result.formula_candidates:
            # ---------------------------------------------------------
            # CRITICAL SAFETY / GOVERNANCE GATE
            #
            # The model proposed one or more formulas, but no eligible
            # reviewed clinical corpus record supports their promotion
            # into selectable candidates.
            #
            # Therefore:
            # - do NOT expose them in formula_candidates
            # - do NOT run them through deterministic safety screening
            # - do NOT mark them eligible_for_selection
            #
            # The fact that the model proposed formulas is represented
            # only through uncertainty / provenance metadata.
            # ---------------------------------------------------------
            candidates = []

            uncertainty_flags.extend(
                [
                    "MODEL_FORMULA_CANDIDATES_SUPPRESSED_UNVERIFIED_CORPUS",
                    "REVIEWED_CLINICAL_CORPUS_REQUIRED_FOR_FORMULA_SELECTION",
                    "NO_VERIFIED_FORMULA_CANDIDATE",
                ]
            )

        else:
            candidates = []

            uncertainty_flags.append(
                "NO_FORMULA_CANDIDATE"
            )

        # Deduplicate flags after retrieval decision.
        uncertainty_flags = list(
            dict.fromkeys(uncertainty_flags)
        )

        # -------------------------------------------------------------
        # 5. Enrich ONLY corpus-governed candidates.
        #
        # TSE is metadata only.
        # Safety Engine operates only after corpus eligibility.
        # -------------------------------------------------------------
        enriched_candidates = []

        for candidate in candidates:
            # Corpus repositories may return dict-like or Pydantic data.
            if hasattr(candidate, "model_dump"):
                item = candidate.model_dump()
            else:
                item = dict(candidate)

            formula_name = item.get("name", "")
            ingredients = item.get("ingredients", [])
            formula_id = item.get("formula_id")

            # ---------------------------------------------------------
            # TSE metadata mapping.
            #
            # IMPORTANT:
            # clinical_ranking_eligible=false catalog rows must remain
            # metadata only and cannot increase confidence or eligibility.
            # ---------------------------------------------------------
            item["catalog_matches"] = self.tse.map_formula_candidate(
                formula_name,
                ingredients,
            )

            # ---------------------------------------------------------
            # Deterministic safety screening.
            #
            # At this point the formula has already passed the reviewed
            # clinical corpus gate.
            # ---------------------------------------------------------
            assessment = self.safety.screen(
                SafetyScreenRequest(
                    formula_id=formula_id,
                    formula_name=formula_name,
                    ingredients=ingredients,
                    constraints=request.constraints,
                    patient_context=request.patient_context,
                )
            )

            item["safety_assessment"] = assessment.model_dump()

            if assessment.findings:
                existing_safety_flags = item.get(
                    "safety_flags",
                    [],
                )

                finding_messages = [
                    finding.message
                    for finding in assessment.findings
                ]

                item["safety_flags"] = list(
                    dict.fromkeys(
                        [
                            *existing_safety_flags,
                            *finding_messages,
                        ]
                    )
                )

            if not assessment.eligible_for_selection:
                uncertainty_flags.append(
                    "SAFETY_BLOCKING_FINDINGS"
                )

            enriched_candidates.append(
                FormulaCandidate(**item)
            )

        uncertainty_flags = list(
            dict.fromkeys(uncertainty_flags)
        )

        # -------------------------------------------------------------
        # 6. Build summary.
        # -------------------------------------------------------------
        summary = result.summary.strip()

        if (
            reasoning.followup_questions
            and not reasoning.ready_for_formula_retrieval
        ):
            clarification_message = (
                "Missing information should be clarified before "
                "deterministic corpus formula retrieval."
            )

            if clarification_message not in summary:
                summary = (
                    f"{summary} {clarification_message}"
                ).strip()

        # -------------------------------------------------------------
        # 7. Convert model pattern hypotheses to response schema.
        # -------------------------------------------------------------
        pattern_hypotheses = []

        for hypothesis in result.pattern_hypotheses:
            if hasattr(hypothesis, "model_dump"):
                hypothesis_data = hypothesis.model_dump()
            else:
                hypothesis_data = dict(hypothesis)

            pattern_hypotheses.append(
                PatternHypothesis(**hypothesis_data)
            )

        # -------------------------------------------------------------
        # 8. Return recommendation contract.
        #
        # AI output always remains a draft and always requires
        # practitioner review.
        #
        # TrustScore remains Base44-owned.
        # -------------------------------------------------------------
        return RecommendationResponse(
            request_id=request.request_id,
            status="DRAFT_AI_RECOMMENDATION",
            summary=summary,
            pattern_hypotheses=pattern_hypotheses,
            formula_candidates=enriched_candidates,
            uncertainty_flags=uncertainty_flags,
            model_confidence=result.model_confidence,
            provenance=ModelProvenance(
                provider=result.provider,
                model=result.model,
                prompt_version=PROMPT_VERSION,
            ),
            requires_practitioner_review=True,
            reasoning=reasoning,
        )
