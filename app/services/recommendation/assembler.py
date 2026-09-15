from app.schemas.intake import RecommendationRequest
from app.schemas.recommendation import (
    FormulaCandidate,
    ModelProvenance,
    PatternHypothesis,
    RecommendationResponse,
)
from app.schemas.reasoning import (
    ClarificationQuestion,
    ClinicalReasoningEnvelope,
    InterviewReasoning,
)
from app.schemas.safety import SafetyScreenRequest
from app.services.knowledge.persistent_clinical import PersistentClinicalStore
from app.services.knowledge.tse_repository import TSEKnowledgeRepository
from app.services.llm.provider import LLMProvider
from app.services.reasoning.engine import DiagnosticReasoningEngine
from app.services.safety.engine import SafetyEngine
from app.services.clarification.validator import validate_proposals_detailed
from app.services.interview.mode import (
    INTERVIEW,
    decide_mode,
    provider_supports_interview,
)
from app.services.telemetry.clarification_rejection import outcome_flags
from app.services.clarification.coverage import (
    Candidate,
    assess_coverage,
    extract_differential_signals,
    resolve_domain,
    select_questions,
)


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
        on_provider_result=None,
        on_clarification_outcomes=None,
    ) -> RecommendationResponse:
        """on_provider_result is an X1D-TELEMETRY1 observer.

        The raw ProviderResult carries token counts and call latency that the
        clinical response deliberately does not, so persistence needs a way to
        see it without those numbers entering the contract. An optional
        callback keeps every existing caller working unchanged and keeps the
        assembler's clinical behaviour identical: it is invoked for its side
        effect and its return value is ignored.
        """
        # -------------------------------------------------------------
        # 1. Generate model hypotheses.
        #
        # Model output is hypothesis-level intelligence only.
        # It is not a clinically verified recommendation.
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # X1D-LEGACYDIAG3.2: decide which contract this turn needs, before
        # calling anything.
        #
        # Both inputs are deterministic and available pre-call: the engine's
        # own marker-matched missing_information, and CLARIFY2 coverage over
        # the accumulated text. The model is not consulted about whether the
        # interview should continue -- see interview/mode.py for why.
        # -------------------------------------------------------------
        accumulated_text = " ".join([request.text_input or "", *request.symptoms])
        mode, mode_reason = decide_mode(
            accumulated_text=accumulated_text,
            missing_information=self.reasoning.analyze(
                request, []).missing_information,
            turn_count=getattr(request, "turn_count", 1),
            supports_interview=provider_supports_interview(self.provider),
        )
        interview_mode = mode == INTERVIEW

        if interview_mode:
            result = await self.provider.generate_interview(
                text_input=request.text_input,
                symptoms=request.symptoms,
                language=request.language,
            )
        else:
            result = await self.provider.generate_recommendation(
                text_input=request.text_input,
                symptoms=request.symptoms,
                goals=request.goals,
                constraints=request.constraints,
                image_data=request.image_data,
                language=request.language,
            )

        if on_provider_result is not None:
            # Observational only; never allowed to affect what follows.
            on_provider_result(result)

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
        # X1D-CLARIFY1: the model proposes, deterministic code decides.
        #
        # Proposals are derived from patient-supplied text and are therefore
        # data to be checked, never instructions to follow. Fields the
        # deterministic engine already asks about are passed in as known, so
        # adaptive questions supplement the checklist instead of repeating it.
        #
        # Validation failure is inert: rejected proposals simply do not appear
        # and the governed result is untouched.
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # X1D-LEGACYDIAG2: retain the model's clinical reasoning.
        #
        # The legacy system produced this reasoning and it is the main reason
        # that system felt clinically useful. Its mistake was letting the same
        # prose become a purchasable formula by way of a regex. Here the
        # reasoning is kept as typed MODEL_HYPOTHESIS objects and nothing more.
        #
        # This deliberately does NOT touch the block below, which decides
        # formula_candidates. formula_hypotheses and formula_candidates are
        # separate channels: one records what the model thought, the other is
        # governed by the reviewed corpus and is unchanged by this phase.
        #
        # Validation failure is contained: an unusable envelope becomes an
        # empty one, never a failed recommendation.
        # -------------------------------------------------------------
        # X1D-LEGACYDIAG3.2: an interview turn produces no envelope at all.
        # The small contract never asks for one, so there is nothing to parse
        # and nothing to attach -- the absence is structural, not filtered.
        # A compact InterviewReasoning is built instead, used only to give
        # CLARIFY2 ranking its differential signals; it is never attached to
        # the response and never persisted as clinical reasoning.
        signal_source = None
        if interview_mode:
            envelope = None
            try:
                signal_source = InterviewReasoning(**(result.interview or {}))
            except Exception:  # noqa: BLE001 - ranking must not break clinical work
                signal_source = None
                uncertainty_flags.append("INTERVIEW_REASONING_UNPARSEABLE")
        else:
            try:
                envelope = ClinicalReasoningEnvelope(**(result.clinical_reasoning or {}))
            except Exception:  # noqa: BLE001 - reasoning must not break clinical work
                envelope = ClinicalReasoningEnvelope()
                uncertainty_flags.append("MODEL_REASONING_ENVELOPE_UNPARSEABLE")
            reasoning.clinical_reasoning = envelope
            signal_source = envelope

        uncertainty_flags.append("INFERENCE_MODE_%s" % mode)
        uncertainty_flags.append("INFERENCE_ROUTE_%s" % mode_reason)

        if envelope is not None and envelope.formula_hypotheses:
            # Stated explicitly so the distinction is legible in the output
            # itself, not only in the type system.
            uncertainty_flags.append(
                "MODEL_FORMULA_HYPOTHESES_RETAINED_NOT_ELIGIBLE")

        # X1D-CLARIFY3: keep the decisions, not just the survivors.
        #
        # validate_proposals_detailed makes exactly the same accept/reject
        # calls as before and additionally reports why, so a turn that shows
        # no adaptive questions can be explained from the record instead of
        # guessed at. The accepted questions are taken from the same outcomes,
        # so there is one decision procedure rather than two.
        deterministic_fields = [m.field for m in reasoning.missing_information]
        proposal_outcomes = []
        try:
            proposal_outcomes = validate_proposals_detailed(
                result.clarification_proposals,
                known_fields=deterministic_fields,
            )
            reasoning.clarification_questions = [
                ClarificationQuestion(**o.question.as_dict())
                for o in proposal_outcomes if o.accepted
            ]
        except Exception:  # noqa: BLE001 - clarification must not break clinical work
            proposal_outcomes = []
            reasoning.clarification_questions = []

        # -------------------------------------------------------------
        # X1D-CLARIFY2: ask the fewest questions that most reduce
        # clinically relevant uncertainty.
        #
        # The validator above decided what MAY be asked. This decides what is
        # WORTH asking, by scoring the deterministic checklist and the
        # validated adaptive proposals against one another -- the checklist is
        # not privileged, because giving a generic category question automatic
        # precedence is what crowded complaint-specific questions out of the
        # visible budget in the first place.
        #
        # Three deliberate boundaries:
        #
        #   * missing_information is NOT touched. It remains the complete
        #     record of what is unknown and the sole input to the HIGH-priority
        #     component of ready_for_formula_retrieval. Pruning the question
        #     list never converts an unknown into a known.
        #
        #   * the envelope is advisory input only. Its missing_information and
        #     contradicting_findings can raise a question the validator already
        #     approved; they cannot create, edit or answer one.
        #
        #   * the sufficiency verdict governs question emission and nothing
        #     else. It is not readiness, not safety, and not eligibility.
        #
        # Failure is inert: any problem here leaves the CLARIFY1 result in
        # place, because a ranking fault must never cost a valid diagnosis.
        # -------------------------------------------------------------
        try:
            accumulated_text = " ".join(
                [request.text_input or "", *request.symptoms])
            coverage = assess_coverage(accumulated_text)
            signals = extract_differential_signals(signal_source)

            # The reasoning engine appends to missing_information and
            # followup_questions in lockstep, so index i pairs. If that ever
            # stops holding, pair nothing and leave the list alone rather than
            # guessing which question belongs to which field.
            paired = (len(reasoning.followup_questions)
                      == len(reasoning.missing_information))

            deterministic_candidates = []
            if paired:
                for item, text in zip(reasoning.missing_information,
                                      reasoning.followup_questions):
                    domain, certain = resolve_domain(item.field, text)
                    deterministic_candidates.append(Candidate(
                        field=item.field,
                        question=text,
                        domain=domain,
                        kind="deterministic",
                        high_priority_missing=(
                            str(item.priority or "").upper() == "HIGH"),
                        domain_certain=certain,
                        payload=text,
                    ))

            adaptive_candidates = []
            for question in reasoning.clarification_questions:
                domain, certain = resolve_domain(question.field,
                                                 question.question)
                adaptive_candidates.append(Candidate(
                    field=question.field,
                    question=question.question,
                    domain=domain,
                    kind="adaptive",
                    model_priority=question.priority,
                    domain_certain=certain,
                    payload=question,
                ))

            selection = select_questions(
                deterministic=deterministic_candidates,
                adaptive=adaptive_candidates,
                coverage=coverage,
                signals=signals,
            )

            if paired:
                reasoning.followup_questions = list(selection.deterministic)
            # The coverage floor rides the typed channel alongside the model's
            # own questions: it carries choice controls the plain-text followup
            # list cannot express, and core's state gate then governs both.
            reasoning.clarification_questions = (
                list(selection.adaptive)
                + [ClarificationQuestion(**q) for q in selection.fallback])
            reasoning.clarification_sufficiency = selection.sufficiency
            reasoning.clinical_coverage = coverage.as_dict()

            uncertainty_flags.append(
                "CLARIFICATION_%s" % selection.sufficiency)
            if selection.suppressed_count:
                uncertainty_flags.append(
                    "CLARIFICATION_QUESTIONS_SUPPRESSED_NOT_MATERIAL")
            # Why a turn asked what it asked. Without these, a turn that asks
            # nothing is indistinguishable from a turn whose proposals were all
            # rejected, which is exactly the ambiguity that made the first
            # staging run hard to read.
            if not result.clarification_proposals:
                uncertainty_flags.append("CLARIFICATION_NO_MODEL_PROPOSALS")
            elif not adaptive_candidates:
                uncertainty_flags.append(
                    "CLARIFICATION_PROPOSALS_REJECTED_BY_VALIDATOR")
            if selection.fallback_used:
                uncertainty_flags.append("CLARIFICATION_COVERAGE_FALLBACK_USED")
            if selection.same_domain_suppressed:
                # X1D-CLARIFY3.1. Not rejection telemetry: the proposal was
                # valid and accepted, and was then not shown because another
                # selected question already covered its domain this turn.
                uncertainty_flags.append(
                    "CLARIFICATION_SAME_DOMAIN_DUPLICATE_SUPPRESSED_%d"
                    % selection.same_domain_suppressed)
            # X1D-CLARIFY3: and now, which rule fired. Fail-open on its own,
            # so an observability fault costs a log line rather than a
            # governed clinical result.
            try:
                uncertainty_flags.extend(outcome_flags(
                    proposal_outcomes,
                    malformed_count=result.clarification_proposals_discarded))
            except Exception:  # noqa: BLE001 - observability is never fatal
                uncertainty_flags.append("CLARIFICATION_REASONS_UNAVAILABLE")
        except Exception:  # noqa: BLE001 - ranking must not break clinical work
            uncertainty_flags.append("CLARIFICATION_COVERAGE_UNAVAILABLE")

        if on_clarification_outcomes is not None:
            # Observational only, exactly like on_provider_result: invoked for
            # its side effect, return value ignored, and its own failure
            # contained so observability can never cost a diagnosis.
            try:
                on_clarification_outcomes(
                    proposal_outcomes,
                    result.clarification_proposals_discarded)
            except Exception:  # noqa: BLE001 - observability is never fatal
                pass

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

        # X1D-LEGACYDIAG3.2 deliberately does NOT gate this on interview mode.
        #
        # The first cut skipped retrieval on interview turns, reasoning that
        # mid-interview nobody should see a formula. Four existing tests caught
        # it: a REVIEWED, sourced formula reaching the assembler is a corpus
        # governance guarantee, and suppressing it was scope creep dressed up
        # as caution.
        #
        # The interview path changes which prompt runs. It does not change what
        # the deterministic corpus does, and it contributes nothing to formula
        # selection -- generate_interview returns formula_candidates=[] always,
        # and the interview contract has no formula field to populate. Any
        # candidate here came from the reviewed corpus, exactly as before.
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

        # Keyed off missing_information rather than followup_questions:
        # X1D-CLARIFY2 prunes the question list down to what is worth asking
        # now, while missing_information stays the complete record of what is
        # unknown. The summary should reflect the record, not the shortlist.
        if (
            reasoning.missing_information
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
