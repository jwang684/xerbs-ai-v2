"""X1D-LEGACYDIAG4.2: show the reasoning, never the prescription.

The finding this answers
------------------------
LEGACYDIAG4 established something awkward: Xerbs asks the model for a full TCM
reasoning envelope, keeps it, ships it across the service boundary -- and shows
the patient a one-line summary. The legacy system displayed far more and felt
far better for it, while being far less safe.

So a patient may now read how the model reasoned: the eight principles, the
competing patterns with the findings for and against each, the pathogenesis,
the treatment principle, what is still missing.

What a patient may still never read is the part the legacy system got wrong --
a formula, a herb, a dose, a decoction instruction, a contraindication framed
as advice. Those live in the same envelope, one attribute away.

Why the tests look the way they do
----------------------------------
Mostly they assert absence, and they assert it structurally. It is easy to
write a projection that happens not to include formula_hypotheses today; the
useful question is whether it *could* start including them tomorrow by someone
adding a field. So the projection is built as an allowlist and the tests check
the allowlist itself -- an unlisted field cannot arrive by being forgotten.

The treatment_principle guard has a section of its own. It is the one approved
field a prescription could hide inside, and it is the only heuristic here, so
it is tested for what it drops as well as what it keeps.
"""

import ast
import inspect
import json
import textwrap

import pytest

from app.schemas.reasoning import (
    ClinicalReasoningEnvelope,
    FormulaHypothesis,
    IngredientHypothesis,
    PatternHypothesisDetail,
    ReasoningResponse,
)
from app.services.recommendation import consumer_projection as cp
from app.services.recommendation.consumer_projection import (
    build_consumer_reasoning,
    mentions_treatment_product,
    project_eight_principle,
    project_patterns,
    project_uncertainty,
)

APPROVED_KEYS = {"summary", "eight_principle", "pattern_hypotheses",
                 "pathogenesis", "treatment_principle", "missing_information",
                 "uncertainty"}


def rich_envelope(**over) -> ClinicalReasoningEnvelope:
    """A full envelope, including the parts a patient must never see."""
    data = dict(
        clinical_summary="外感风寒袭表，兼肺失宣降。",
        tcm_diagnosis_hypotheses=["风寒束表"],
        eight_principle_differentiation={
            "exterior_interior": "表证为主",
            "cold_heat": "寒证",
            "deficiency_excess": "实证",
            "yin_yang": "阳证",
            "made_up_dimension": "模型自创的维度",
        },
        pattern_hypotheses=[
            PatternHypothesisDetail(name="风热犯表", role="secondary",
                                    confidence=0.3,
                                    supporting_findings=["发热3天"],
                                    contradicting_findings=["明显怕冷", "无汗"]),
            PatternHypothesisDetail(name="风寒束表", role="primary",
                                    confidence=0.7,
                                    supporting_findings=["恶寒无汗", "受凉起病"],
                                    contradicting_findings=[]),
        ],
        pathogenesis="风寒外束，卫阳被遏，肺气失宣。",
        treatment_principle="辛温解表，宣肺止咳。",
        formula_hypotheses=[FormulaHypothesis(
            name="麻黄汤", confidence=0.6, rationale="风寒表实证的代表方",
            ingredients=[IngredientHypothesis(name="麻黄", dosage="6-9g",
                                              role="君")],
            administration="水煎温服，取微汗",
            contraindications=["表虚自汗者忌用"],
            precautions=["得汗即止"])],
        missing_information=["缺少咽痛与鼻涕性状"],
        uncertainty_flags=["症状仅3天，证候可能变化",
                           "PATTERN_NOT_VERIFIED_IN_REVIEWED_CORPUS"],
        model_confidence=0.55,
    )
    data.update(over)
    return ClinicalReasoningEnvelope(**data)


def code(obj):
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
# A-J: the approved content reaches the patient
# ======================================================================

class TestApprovedContent:
    def setup_method(self):
        self.p = build_consumer_reasoning(rich_envelope())

    def test_a_summary_is_projected(self):
        assert self.p["summary"] == "外感风寒袭表，兼肺失宣降。"

    def test_a2_summary_falls_back_when_the_envelope_has_none(self):
        p = build_consumer_reasoning(rich_envelope(clinical_summary=None),
                                     fallback_summary="回退摘要")
        assert p["summary"] == "回退摘要"

    def test_b_eight_principle_is_projected(self):
        dims = {d["dimension"]: d["finding"] for d in self.p["eight_principle"]}
        assert dims == {"表里": "表证为主", "寒热": "寒证",
                        "虚实": "实证", "阴阳": "阳证"}

    def test_b2_an_invented_dimension_is_not_projected(self):
        """The model does not get to add a ninth principle."""
        blob = json.dumps(self.p, ensure_ascii=False)
        assert "模型自创的维度" not in blob

    def test_b3_absent_dimensions_are_simply_absent(self):
        p = build_consumer_reasoning(rich_envelope(
            eight_principle_differentiation={"cold_heat": "寒证"}))
        assert p["eight_principle"] == [{"dimension": "寒热", "finding": "寒证"}]

    def test_c_the_primary_pattern_is_projected_first(self):
        first = self.p["pattern_hypotheses"][0]
        assert first["pattern_name"] == "风寒束表"
        assert first["role"] == "primary"

    def test_d_the_secondary_pattern_is_projected(self):
        names = [h["pattern_name"] for h in self.p["pattern_hypotheses"]]
        assert "风热犯表" in names

    def test_e_supporting_findings_are_projected(self):
        primary = self.p["pattern_hypotheses"][0]
        assert primary["supporting_findings"] == ["恶寒无汗", "受凉起病"]

    def test_f_contradicting_findings_are_projected(self):
        """Never suppressed to make an answer look tidier."""
        secondary = next(h for h in self.p["pattern_hypotheses"]
                         if h["role"] == "secondary")
        assert secondary["contradicting_findings"] == ["明显怕冷", "无汗"]

    def test_g_pathogenesis_is_projected(self):
        assert self.p["pathogenesis"] == "风寒外束，卫阳被遏，肺气失宣。"

    def test_h_treatment_principle_is_projected(self):
        assert self.p["treatment_principle"] == "辛温解表，宣肺止咳。"

    def test_i_missing_information_is_projected(self):
        assert self.p["missing_information"] == ["缺少咽痛与鼻涕性状"]

    def test_j_uncertainty_is_projected(self):
        assert "症状仅3天，证候可能变化" in self.p["uncertainty"]

    def test_j2_internal_governance_tokens_are_not_shown_as_advice(self):
        assert "PATTERN_NOT_VERIFIED_IN_REVIEWED_CORPUS" not in \
            self.p["uncertainty"]

    def test_k_empty_fields_produce_no_section(self):
        p = build_consumer_reasoning(ClinicalReasoningEnvelope())
        assert p == {}

    def test_k2_a_partially_filled_envelope_yields_only_what_exists(self):
        p = build_consumer_reasoning(ClinicalReasoningEnvelope(
            clinical_summary="只有摘要。"))
        assert set(p) == {"summary"}

    def test_k3_an_absent_envelope_is_not_an_error(self):
        assert build_consumer_reasoning(None) == {}
        assert build_consumer_reasoning(None, "x") == {"summary": "x"}


# ======================================================================
# L-T: the prohibited content does not
# ======================================================================

class TestProhibitedContent:
    def setup_method(self):
        self.p = build_consumer_reasoning(rich_envelope())
        self.blob = json.dumps(self.p, ensure_ascii=False)

    def test_the_projection_is_an_allowlist_not_a_filter(self):
        """An unlisted field cannot arrive by being forgotten."""
        assert set(self.p) <= APPROVED_KEYS

    @pytest.mark.parametrize("forbidden", [
        "麻黄汤",          # L/M: formula hypothesis and its name
        "麻黄",            # N: herb
        "6-9g",            # O: dosage
        "水煎温服",         # P: administration
        "表虚自汗者忌用",    # Q: contraindication as advice
        "得汗即止",         # precaution
        "风寒表实证的代表方",  # formula rationale
    ])
    def test_l_q_no_treatment_content_reaches_the_patient(self, forbidden):
        assert forbidden not in self.blob

    def test_l2_the_formula_key_itself_is_absent(self):
        for key in ("formula_hypotheses", "formula_candidates", "formula",
                    "ingredients", "dosage", "administration",
                    "contraindications", "precautions", "rationale"):
            assert key not in self.p

    def test_r_s_products_and_purchase_are_absent(self):
        for key in ("product", "product_id", "price", "purchase",
                    "consumer_purchasable", "eligibility", "listed"):
            assert key not in self.p
            assert key not in self.blob

    def test_t_confidence_and_authority_metadata_are_absent(self):
        """A number beside a pattern reads as certainty however it is labelled."""
        for key in ("confidence", "model_confidence", "status",
                    "corpus_match", "clinical_ranking_eligible",
                    "review_status", "REVIEWED", "VERIFIED"):
            assert key not in self.blob
        for pattern in self.p["pattern_hypotheses"]:
            assert set(pattern) == {"pattern_name", "role",
                                    "supporting_findings",
                                    "contradicting_findings"}

    def test_t2_tcm_diagnosis_hypotheses_is_not_projected(self):
        """Available, but not on the approved list and duplicative of patterns."""
        assert "tcm_diagnosis_hypotheses" not in self.p

    def test_the_builder_never_reads_a_forbidden_attribute(self):
        source = code(cp.build_consumer_reasoning)
        for forbidden in ("ingredients", "dosage", "administration",
                          "precautions", "rationale", "model_confidence"):
            assert forbidden not in source

    def test_formula_names_are_read_only_to_suppress(self):
        """The one place formula_hypotheses is touched, and only as a blocklist."""
        source = code(cp.build_consumer_reasoning)
        assert "formula_hypotheses" in source
        assert "mentions_treatment_product" in source
        # it feeds the guard, never the output
        assert '"formula' not in source.replace("formula_hypotheses", "")


# ======================================================================
# The treatment-principle guard
# ======================================================================

class TestTreatmentPrincipleGuard:
    @pytest.mark.parametrize("text", [
        "桂枝汤加减", "予麻黄汤", "银翘散合方", "六味地黄丸",
        "麻黄9g，桂枝6g", "水煎温服，每日2次", "每次 1 剂",
        "服用川贝枇杷膏",
    ])
    def test_a_prescription_shaped_principle_is_dropped(self, text):
        assert mentions_treatment_product(text) is True
        p = build_consumer_reasoning(rich_envelope(treatment_principle=text))
        assert "treatment_principle" not in p

    @pytest.mark.parametrize("text", [
        "辛温解表，宣肺止咳。", "清热解毒，化痰止咳。",
        "健脾益气，和胃降逆。", "疏肝解郁，理气和中。",
    ])
    def test_a_genuine_principle_survives(self, text):
        assert mentions_treatment_product(text) is False
        p = build_consumer_reasoning(rich_envelope(treatment_principle=text))
        assert p["treatment_principle"] == text

    def test_a_principle_naming_the_models_own_formula_is_dropped(self):
        p = build_consumer_reasoning(rich_envelope(
            treatment_principle="以麻黄汤法辛温发汗"))
        assert "treatment_principle" not in p

    def test_the_guard_only_ever_suppresses(self):
        """It can drop a field; it can never add or rewrite one."""
        source = code(cp.mentions_treatment_product)
        assert "return True" in source and "return False" in source
        assert "replace(" not in source and "+" not in source


# ======================================================================
# U-AA: nothing is upgraded, nothing is mutated
# ======================================================================

class TestNoAuthorityChange:
    def test_u_the_envelope_is_not_mutated(self):
        envelope = rich_envelope()
        before = envelope.model_dump_json()
        build_consumer_reasoning(envelope)
        assert envelope.model_dump_json() == before

    def test_u2_the_builder_assigns_nothing_to_the_envelope(self):
        """Reading envelope.pattern_hypotheses is the point; writing is not.

        Checked over the AST rather than by substring, because every legitimate
        read looks like a write to a substring search.
        """
        from app.services.recommendation import consumer_projection as module

        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute):
                    raise AssertionError(
                        "projection must not assign to any attribute: %s"
                        % ast.unparse(target))
        source = code(module)
        assert "setattr(" not in source
        assert "__dict__" not in source

    @pytest.mark.parametrize("forbidden", [
        "corpus", "relationship", "REVIEWED", "VERIFIED", "verification",
        "safety", "SafetyEngine", "consumer_purchasable", "product_id",
        "resolver", "session", "commit", "insert", "update(",
    ])
    def test_v_z_the_module_has_no_write_path_anywhere(self, forbidden):
        """"product" itself is excluded: mentions_treatment_product is the
        guard that keeps products out, so its own name is not evidence of one
        getting in. product_id -- the thing that would actually be a leak --
        is checked instead."""
        from app.services.recommendation import consumer_projection as module

        assert forbidden not in code(module)

    def test_aa_model_generated_stays_model_generated(self):
        """The projection carries no status that could read as verified."""
        p = build_consumer_reasoning(rich_envelope())
        blob = json.dumps(p, ensure_ascii=False)
        for upgrade in ("VERIFIED", "REVIEWED", "确诊", "最终诊断",
                        "已验证", "医生诊断"):
            assert upgrade not in blob

    def test_the_projection_field_is_additive_and_defaulted(self):
        assert "consumer_reasoning" in ReasoningResponse.model_fields
        assert ReasoningResponse().consumer_reasoning == {}

    def test_the_assembler_builds_it_only_on_the_full_reasoning_branch(self):
        from app.services.recommendation import assembler as module

        source = code(module.RecommendationAssembler.generate)
        assert "build_consumer_reasoning(" in source
        # it sits with the envelope, after it is typed
        assert source.index("ClinicalReasoningEnvelope(") < \
            source.index("build_consumer_reasoning(")

    def test_a_projection_failure_does_not_break_the_diagnosis(self):
        from app.services.recommendation import assembler as module

        source = code(module.RecommendationAssembler.generate)
        assert "CONSUMER_REASONING_UNAVAILABLE" in source


# ======================================================================
# AB-AH: everything from earlier phases is untouched
# ======================================================================

class TestNoRegression:
    def test_ab_streaming_remains_display_only(self):
        from app.services.llm.streaming import STREAMABLE_KEYS

        assert STREAMABLE_KEYS == ("summary", "interview_summary")

    def test_ac_the_scanner_still_emits_no_raw_json(self):
        from app.services.llm.streaming import SummaryStreamScanner

        body = json.dumps({"summary": "外感风热。", "x": 1}, ensure_ascii=False)
        scanner = SummaryStreamScanner()
        out = "".join(scanner.feed(body[:i]) for i in range(1, len(body) + 1))
        assert out == "外感风热。"
        for syntax in ("{", "}", '"summary"'):
            assert syntax not in out

    def test_ad_ae_replay_and_idempotency_are_untouched(self):
        from app.services.integration import base44

        source = code(base44.Base44GenerationService.generate)
        assert "return self._response(existing)" in source
        head = source[:source.index("return self._response(existing)")]
        assert "_execute" not in head

    def test_af_interview_behaviour_is_unchanged(self):
        from app.services.interview.mode import (MAX_INTERVIEW_TURNS,
                                                 REASON_DEPTH_REACHED,
                                                 decide_mode)

        assert MAX_INTERVIEW_TURNS == 3
        assert decide_mode(accumulated_text="咳嗽3天。", missing_information=[],
                           interview_depth=3) == ("FULL_REASONING",
                                                  REASON_DEPTH_REACHED)

    def test_ag_the_question_budget_is_unchanged(self):
        from app.services.clarification import coverage as cov
        from app.services.clarification.validator import (
            FIELD_PATTERN, MAX_PROPOSALS_PER_TURN, REJECTION_REASONS)

        assert cov.MAX_VISIBLE_QUESTIONS_PER_TURN == 4
        assert cov.MAX_ADAPTIVE_WHEN_OPEN == 3
        assert MAX_PROPOSALS_PER_TURN == 3
        assert len(REJECTION_REASONS) == 10
        assert FIELD_PATTERN.pattern == r"^[a-z][a-z0-9_]{1,38}[a-z0-9]$"

    def test_ah_the_corpus_gate_is_unchanged(self):
        from app.services.recommendation import assembler as module

        source = code(module.RecommendationAssembler.generate)
        assert "eligible_formula_candidates" in source
        assert "MODEL_FORMULA_CANDIDATES_SUPPRESSED_UNVERIFIED_CORPUS" in source
        assert "MODEL_FORMULA_HYPOTHESES_RETAINED_NOT_ELIGIBLE" in source

    def test_the_envelope_still_retains_everything_internally(self):
        """4.2 shows less than is kept. It must not start keeping less."""
        envelope = rich_envelope()
        assert envelope.formula_hypotheses
        assert envelope.formula_hypotheses[0].ingredients[0].dosage == "6-9g"
