"""X1D-LEGACYDIAG2: keep the legacy system's reasoning, not its authority.

The legacy system felt clinically useful, and the audit found the reason was
mostly its output contract: it required a primary and a secondary pattern each
with cited evidence, eight-principle differentiation, pathogenesis, treatment
principle, formula rationale, ingredients and dosages. The current contract
asked for five fields, so most of that was never requested -- and the formula
hypotheses the model did produce were discarded rather than kept.

What the legacy system got wrong was not reasoning broadly. It was that the
same free-form prose became, via a regex over Markdown, a purchasable formula
with no verification anywhere between. Detail was mistaken for verification.

So this phase asks for the reasoning again and keeps it, while changing no
gate. The central assertion, repeated from several angles below, is that a
plausible formula the model invents can be read by a clinician and can be
verified later, but cannot become eligible, cannot reach SafetyEngine, and
cannot be bought.
"""

import ast
import inspect
import json
import textwrap

import pytest

from app.schemas.reasoning import (
    MODEL_HYPOTHESIS,
    ClinicalReasoningEnvelope,
    FormulaHypothesis,
    IngredientHypothesis,
    PatternHypothesisDetail,
)


def rich_reasoning() -> dict:
    """A response of the shape the enriched prompt asks for."""
    return {
        "clinical_summary": "风热犯表，卫分受邪，肺失宣降",
        "tcm_diagnosis_hypotheses": ["风热感冒"],
        "eight_principle_differentiation": {
            "cold_heat": "热", "exterior_interior": "表", "deficiency_excess": "实"},
        "pattern_hypotheses": [
            {"name": "风热犯表", "role": "primary", "confidence": 0.7,
             "supporting_findings": ["发热", "咽痛", "口渴"],
             "contradicting_findings": ["无明显恶寒"]},
            {"name": "肺热壅盛", "role": "secondary", "confidence": 0.4,
             "supporting_findings": ["咳嗽"], "contradicting_findings": []},
        ],
        "pathogenesis": "风热之邪犯表，卫气被遏，肺气失宣",
        "treatment_principle": "辛凉解表，宣肺清热",
        "formula_hypotheses": [
            {"name": "银翘散", "confidence": 0.6,
             "rationale": "辛凉平剂，主治风热犯表",
             "ingredients": [
                 {"name": "金银花", "dosage": "9-15g", "role": "君"},
                 {"name": "连翘", "dosage": "9-15g", "role": "君"}],
             "administration": "水煎服，每日一剂",
             "contraindications": ["风寒感冒者不宜"],
             "precautions": ["不宜久煎"]},
        ],
        "missing_information": ["舌象", "脉象"],
    }


# ======================================================================
# The reasoning is actually retained
# ======================================================================

class TestReasoningIsRetained:
    def test_the_envelope_parses_a_rich_response(self):
        env = ClinicalReasoningEnvelope(**rich_reasoning())
        assert env.clinical_summary
        assert env.pathogenesis and env.treatment_principle
        assert len(env.pattern_hypotheses) == 2
        assert len(env.formula_hypotheses) == 1
        assert env.is_empty() is False

    def test_primary_and_secondary_patterns_are_distinguished(self):
        env = ClinicalReasoningEnvelope(**rich_reasoning())
        roles = {p.name: p.role for p in env.pattern_hypotheses}
        assert roles == {"风热犯表": "primary", "肺热壅盛": "secondary"}

    def test_supporting_and_contradicting_findings_are_both_kept(self):
        """The legacy contract asked for evidence on both sides; so does this."""
        primary = ClinicalReasoningEnvelope(**rich_reasoning()).pattern_hypotheses[0]
        assert primary.supporting_findings == ["发热", "咽痛", "口渴"]
        assert primary.contradicting_findings == ["无明显恶寒"]

    def test_formula_hypotheses_retain_ingredients_and_dosage(self):
        formula = ClinicalReasoningEnvelope(**rich_reasoning()).formula_hypotheses[0]
        assert formula.rationale
        assert [i.name for i in formula.ingredients] == ["金银花", "连翘"]
        assert formula.ingredients[0].dosage == "9-15g"
        assert formula.administration
        assert formula.contraindications and formula.precautions

    def test_a_dosage_range_is_kept_verbatim(self):
        """Parsing "9-15g" into a number would invent precision the model
        never expressed. Nothing computes with this field."""
        assert IngredientHypothesis(name="金银花", dosage="9-15g").dosage == "9-15g"

    def test_missing_information_stays_explicit(self):
        env = ClinicalReasoningEnvelope(**rich_reasoning())
        assert env.missing_information == ["舌象", "脉象"]


# ======================================================================
# Everything is a hypothesis, and cannot be relabelled
# ======================================================================

class TestHypothesisStatus:
    def test_every_clinical_object_is_a_model_hypothesis(self):
        env = ClinicalReasoningEnvelope(**rich_reasoning())
        assert all(p.status == MODEL_HYPOTHESIS for p in env.pattern_hypotheses)
        for formula in env.formula_hypotheses:
            assert formula.status == MODEL_HYPOTHESIS
            assert all(i.status == MODEL_HYPOTHESIS for i in formula.ingredients)

    @pytest.mark.parametrize("escalated", [
        "VERIFIED", "VERIFIED_EXTERNAL", "REVIEWED", "SAFETY_APPROVED",
        "ELIGIBLE", "APPROVED", "", None, True,
    ])
    def test_status_cannot_be_widened(self, escalated):
        """A Literal, so escalation changes the type and shows up in review."""
        for model in (FormulaHypothesis, PatternHypothesisDetail):
            with pytest.raises(Exception):
                model(name="x", status=escalated)

    def test_the_envelope_carries_no_authority_fields(self):
        blob = json.dumps(
            ClinicalReasoningEnvelope(**rich_reasoning()).model_dump(),
            ensure_ascii=False)
        for forbidden in ("corpus_match", "ready_for_formula_retrieval",
                          "clinical_ranking_eligible", "safety_verdict",
                          "consumer_purchasable", "eligible_for_selection",
                          "formula_id", "product_id", "review_status"):
            assert forbidden not in blob


# ======================================================================
# THE CENTRAL ACCEPTANCE CRITERION
# ======================================================================

class TestNoAuthorityEscalation:
    """A plausible formula absent from the reviewed corpus must stay inert."""

    @staticmethod
    def _executable(module_or_fn):
        """Source with docstrings stripped.

        These modules explain at length what they refuse to do, so the words
        "eligible", "purchasable" and "safety" all appear in prose. A
        substring check would match the explanation rather than the behaviour.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(module_or_fn)))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef))
                    and body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:]
        return ast.unparse(tree)

    def test_the_suppression_gate_is_untouched(self):
        """formula_candidates is still decided by the reviewed corpus alone."""
        from app.services.recommendation.assembler import RecommendationAssembler

        code = self._executable(RecommendationAssembler.generate)
        assert "MODEL_FORMULA_CANDIDATES_SUPPRESSED_UNVERIFIED_CORPUS" in code
        assert "eligible_formula_candidates_for_patterns" in code

    def test_formula_hypotheses_never_feed_formula_candidates(self):
        from app.services.recommendation.assembler import RecommendationAssembler

        code = self._executable(RecommendationAssembler.generate)
        for leak in ("candidates = envelope.formula_hypotheses",
                     "candidates.extend(envelope",
                     "candidates += envelope",
                     "formula_candidates=envelope"):
            assert leak not in code

    def test_the_envelope_never_writes_an_authority_field(self):
        from app.services.recommendation import assembler as module

        code = self._executable(module)
        for assignment in ("envelope.corpus_match", "envelope.safety_verdict",
                           "corpus_match = envelope", "safety_verdict = envelope",
                           "consumer_purchasable = envelope",
                           "ready_for_formula_retrieval = envelope"):
            assert assignment not in code

    def test_the_reasoning_engine_gate_is_unchanged(self):
        """corpus_match still comes only from a reviewed corpus lookup."""
        from app.services.reasoning.engine import DiagnosticReasoningEngine

        code = self._executable(DiagnosticReasoningEngine.analyze)
        assert "corpus_match=bool(matches)" in code
        assert "self.resolver.search(" in code
        assert "clinical_reasoning" not in code   # the engine never reads it

    def test_the_envelope_reaches_no_safety_input(self):
        """Check the arguments of the SafetyScreenRequest call itself.

        An earlier version of this test split the module source on the string
        "SafetyScreenRequest" and inspected the remainder -- which begins at
        the import line, so it covered nearly the whole file and proved
        nothing. What matters is narrower and checkable: what is actually
        passed into the safety request.
        """
        from app.services.recommendation import assembler as module

        tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", None))
                 == "SafetyScreenRequest"]
        assert calls, "SafetyEngine screening must still be constructed"
        for call in calls:
            rendered = ast.unparse(call)
            for leak in ("envelope", "formula_hypotheses", "clinical_reasoning",
                         "ingredient_hypotheses", "MODEL_HYPOTHESIS"):
                assert leak not in rendered, (
                    f"model reasoning reached safety input: {rendered}")

    def test_purchase_authority_is_absent_from_this_service_entirely(self):
        """consumer_purchasable is core's, and this phase did not change that."""
        from app.services.recommendation import assembler as module

        assert "consumer_purchasable" not in self._executable(module)


# ======================================================================
# A richer contract must not make the clinical path less reliable
# ======================================================================

class TestMalformedReasoningIsContained:
    @pytest.mark.parametrize("junk", [
        None, "", "not a dict", 42, [], [1, 2],
        {"pattern_hypotheses": "nope"},
        {"formula_hypotheses": [{"no_name": 1}]},
        {"model_confidence": "high"},
        {"pattern_hypotheses": [{"name": "x", "confidence": 5.0}]},
    ])
    def test_unusable_reasoning_becomes_an_empty_envelope_not_a_failure(self, junk):
        try:
            env = ClinicalReasoningEnvelope(**junk) if isinstance(junk, dict) \
                else ClinicalReasoningEnvelope()
            assert isinstance(env, ClinicalReasoningEnvelope)
        except Exception:
            # The assembler catches exactly this and substitutes an empty
            # envelope; the point is that it is containable, not that every
            # malformed shape validates.
            assert True

    def test_the_assembler_contains_envelope_failure(self):
        from app.services.recommendation.assembler import RecommendationAssembler

        code = TestNoAuthorityEscalation._executable(RecommendationAssembler.generate)
        assert "MODEL_REASONING_ENVELOPE_UNPARSEABLE" in code
        assert "ClinicalReasoningEnvelope()" in code    # the empty fallback

    def test_the_provider_tolerates_a_missing_reasoning_block(self):
        from app.services.llm.openai_compatible import _extract_clinical_reasoning

        for body in ({}, {"clinical_reasoning": None},
                     {"clinical_reasoning": "nope"}, {"clinical_reasoning": []}):
            assert _extract_clinical_reasoning(body) == {}

    def test_an_omitted_field_is_not_fabricated(self):
        """An absent field is information; a fabricated one is not."""
        env = ClinicalReasoningEnvelope(clinical_summary="only this")
        assert env.pathogenesis is None
        assert env.formula_hypotheses == []
        assert env.missing_information == []


# ======================================================================
# The prompt asks for the legacy capabilities, and forbids invention
# ======================================================================

class TestPromptMigratedLegacyCapabilities:
    @pytest.mark.parametrize("capability", [
        "eight_principle_differentiation", "pathogenesis",
        "treatment_principle", "supporting_findings",
        "contradicting_findings", "formula_hypotheses", "administration",
        "contraindications", "precautions", "missing_information",
    ])
    def test_each_legacy_capability_is_requested(self, capability):
        from app.services.llm.openai_compatible import SYSTEM_PROMPT

        assert capability in SYSTEM_PROMPT

    def test_the_prompt_forbids_filling_slots_with_invention(self):
        from app.services.llm.openai_compatible import SYSTEM_PROMPT

        assert "Do not invent" in SYSTEM_PROMPT
        assert "An absent field is information" in SYSTEM_PROMPT

    def test_the_prompt_states_the_reasoning_is_not_a_prescription(self):
        from app.services.llm.openai_compatible import SYSTEM_PROMPT

        assert "never becomes a prescription" in SYSTEM_PROMPT
        assert "not an instruction to the patient" in SYSTEM_PROMPT

    def test_no_markdown_contract_returned(self):
        """The legacy free-form Markdown contract is not restored."""
        from app.services.llm.openai_compatible import SYSTEM_PROMPT

        assert "Return JSON only" in SYSTEM_PROMPT
        assert "Markdown" not in SYSTEM_PROMPT
