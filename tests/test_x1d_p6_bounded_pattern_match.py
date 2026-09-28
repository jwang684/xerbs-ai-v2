"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P6: a model pattern name finds the reviewed pattern it names.

The model writes "风寒束表，肺气失宣"; the source-backed reviewed pattern is "风寒束表".
The old lookup asked whether the model's whole string occurs inside a corpus
field, so the precise model name never matched while the fragment "风寒" did.
The engine now matches by bounded component equality -- never by fragment,
character overlap, or alias the reviewed record does not carry.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.schemas.clinical_knowledge import SourceRef
from app.schemas.clinical_workflow import ClinicalEntityType, IngestionBatchRequest, IngestionItem
from app.schemas.intake import RecommendationRequest
from app.services.knowledge.pattern_match import matches_reviewed_name
from app.services.knowledge.persistent_clinical import PersistentClinicalStore
from app.services.knowledge.resolver import KnowledgeResolver
from app.services.reasoning.engine import DiagnosticReasoningEngine
from tests.governed_fixtures import approve_entity_for_test, approve_source_for_test

CANON = "风寒束表"


# ----------------------------------------------------------------------
# The rule itself
# ----------------------------------------------------------------------

def test_exact_canonical_name_matches():
    assert matches_reviewed_name("风寒束表", [CANON])


def test_canonical_name_as_a_bounded_component_matches():
    for model in ("风寒束表，肺气失宣", "风寒束表,肺气失宣", "风寒束表、肺气失宣",
                  "肺气失宣；风寒束表", "风寒束表（太阳伤寒）", " 风寒束表 "):
        assert matches_reviewed_name(model, [CANON]), model


def test_the_pattern_suffix_is_not_a_different_pattern():
    assert matches_reviewed_name("风寒束表证", [CANON])
    assert matches_reviewed_name("风寒束表证，肺气失宣", [CANON])
    assert matches_reviewed_name("风寒束表", ["风寒束表证"])


def test_a_generic_fragment_does_not_match():
    for model in ("风寒", "束表", "表", "寒", "风寒证", "证"):
        assert not matches_reviewed_name(model, [CANON]), model


def test_overlapping_characters_in_an_unrelated_pattern_do_not_match():
    for model in ("风热束表", "风寒束肺", "风寒袭表", "外感风寒束表", "风寒束表兼湿阻",
                  "寒湿束表", "风寒束表肺气失宣"):
        assert not matches_reviewed_name(model, [CANON]), model


def test_a_short_reviewed_name_is_not_found_inside_a_longer_model_component():
    assert not matches_reviewed_name("风寒束表，肺气失宣", ["风寒"])
    assert not matches_reviewed_name("风寒束表", ["表"])


def test_only_aliases_the_reviewed_record_carries_are_used():
    assert not matches_reviewed_name("风寒表证", [CANON])
    assert matches_reviewed_name("风寒表证", [CANON, "风寒表证"])


def test_empty_names_never_match():
    assert not matches_reviewed_name("", [CANON])
    assert not matches_reviewed_name("，、", [CANON])
    assert not matches_reviewed_name("风寒束表", ["", "证"])


# ----------------------------------------------------------------------
# Against a real store: reviewed-only and source rules are unchanged
# ----------------------------------------------------------------------

def _store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return PersistentClinicalStore(sessionmaker(bind=engine, expire_on_commit=False))


def _pattern(store, name, source_id, reviewed, **payload):
    req = IngestionBatchRequest(submitted_by="p6", source_label="p6", items=[IngestionItem(
        entity_type=ClinicalEntityType.PATTERN, payload={"name": name, **payload},
        sources=[SourceRef(source_id=source_id, title="p6 source", source_type="TEST")])])
    eid = store.ingest(req).created_entity_ids[0]
    if reviewed:
        store.submit_for_review(ClinicalEntityType.PATTERN, eid, "p6")
        approve_entity_for_test("pattern", eid, session_factory=store.Session)
    return eid


def test_store_matches_the_reviewed_pattern_from_a_compound_model_name():
    store = _store()
    pid = _pattern(store, CANON, "p6-src", reviewed=True, indications=["恶寒", "无汗"])
    found = store.match_reviewed_patterns("风寒束表，肺气失宣")
    assert [x["pattern_id"] for x in found] == [pid]


def test_store_never_matches_a_draft_pattern():
    store = _store()
    _pattern(store, CANON, "p6-src", reviewed=False)
    assert store.match_reviewed_patterns("风寒束表") == []
    assert store.match_reviewed_patterns("风寒束表，肺气失宣") == []


def test_store_does_not_match_a_pattern_name_against_indications():
    store = _store()
    _pattern(store, CANON, "p6-src", reviewed=True, indications=["恶寒", "无汗"])
    assert store.match_reviewed_patterns("恶寒") == []


def test_ranking_eligibility_still_requires_a_reviewed_source():
    store = _store()
    _pattern(store, CANON, "p6-src", reviewed=True)
    found = store.match_reviewed_patterns(CANON)
    assert found and found[0]["clinical_ranking_eligible"] is False
    cur = store.get_source("p6-src")
    from app.schemas.clinical_knowledge import SourceSubmitReviewRequest
    store.submit_source_for_review("p6-src", SourceSubmitReviewRequest(submitted_by="curator", expected_version=cur.version))
    approve_source_for_test("p6-src", session_factory=store.Session)
    assert store.match_reviewed_patterns(CANON)[0]["clinical_ranking_eligible"] is True


def test_the_general_search_api_is_unchanged():
    store = _store()
    _pattern(store, CANON, "p6-src", reviewed=True)
    assert store.search("风寒", ["pattern"], reviewed_only=True)
    assert store.search("风寒束表，肺气失宣", ["pattern"], reviewed_only=True) == []


# ----------------------------------------------------------------------
# Through the reasoning engine
# ----------------------------------------------------------------------

def test_engine_marks_the_compound_model_pattern_as_a_corpus_match():
    store = _store()
    pid = _pattern(store, CANON, "p6-src", reviewed=True)
    engine = DiagnosticReasoningEngine(store, KnowledgeResolver(store, record_gaps=False))
    r = engine.analyze(RecommendationRequest(text_input="恶寒发热两天，无汗，身痛头痛，鼻流清涕。"),
                       [{"name": "风寒束表，肺气失宣", "confidence": 0.8, "reasoning": "r"},
                        {"name": "风寒", "confidence": 0.3, "reasoning": "r"}])
    a, b = r.pattern_assessments
    assert a.corpus_match is True and a.pattern_id == pid
    assert b.corpus_match is False and b.pattern_id is None
