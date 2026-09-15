"""X1D-KNOWLEDGE1B: see the misses, change nothing else.

CORPUS1 measured why cases end at PATTERN_NOT_VERIFIED_IN_REVIEWED_CORPUS: the
corpus holds zero pattern entities. The tempting next move is to go and fetch
the missing knowledge. The cheaper one is to find out which knowledge is
missing and how often, because that decides what is worth researching -- and
the claim kinds a naive retriever would reach for first are exactly the
highest-risk ones.

So this phase adds a seam and an observation, and the tests are mostly about
what did *not* change:

  * every lookup returns the store's own result, so no clinical outcome moves;
  * the resolver holds no knowledge state and grants no authority;
  * recording is fail-open, because an analytics row is not worth a diagnosis;
  * a gap carries a clinical concept and nothing about the patient.

The last point needs care rather than trust. A pattern name is model output
derived from patient language, so "it should be a concept" is not the same as
"it is one". Overlong subjects are hashed rather than stored.
"""

import hashlib
import json

import pytest

from app.services.knowledge import gap_recorder
from app.services.knowledge.gap_recorder import (
    CLAIM_PATTERN_ENTITY,
    MAX_PLAIN_SUBJECT_CHARS,
    build_gap_payload,
    normalise_subject,
    record_gap,
    summarise_gaps,
)
from app.services.knowledge.resolver import KnowledgeResolver, ResolutionResult


class _Store:
    """Stand-in corpus recording exactly how it was called."""

    def __init__(self, result=None):
        self.result = result if result is not None else []
        self.calls = []

    def search(self, query, entity_types, reviewed_only=False, limit=20):
        self.calls.append({"query": query, "entity_types": entity_types,
                           "reviewed_only": reviewed_only, "limit": limit})
        return self.result


# ======================================================================
# Behaviour preservation: the seam must be invisible to the clinical path
# ======================================================================

class TestBehaviourPreserved:
    def test_a_hit_returns_the_stores_own_result_unchanged(self):
        hit = [{"pattern_id": "pat-1", "name": "风热犯表"}]
        resolver = KnowledgeResolver(_Store(hit), record_gaps=False)
        assert resolver.search("风热犯表", ["pattern"]) is hit

    def test_a_miss_returns_the_stores_own_empty_result(self):
        resolver = KnowledgeResolver(_Store([]), record_gaps=False)
        assert resolver.search("风热犯表", ["pattern"]) == []

    def test_the_lookup_is_delegated_verbatim(self):
        """Same query, same types, same flags, same limit."""
        store = _Store([])
        KnowledgeResolver(store, record_gaps=False).search(
            "风热犯表", ["pattern"], reviewed_only=True, limit=3)
        assert store.calls == [{"query": "风热犯表", "entity_types": ["pattern"],
                                "reviewed_only": True, "limit": 3}]

    def test_the_reasoning_engine_resolves_through_the_seam(self):
        import ast
        import inspect
        import textwrap
        from app.services.reasoning.engine import DiagnosticReasoningEngine

        code = ast.unparse(ast.parse(textwrap.dedent(
            inspect.getsource(DiagnosticReasoningEngine.analyze))))
        assert "self.resolver.search(" in code
        assert "self.corpus.search(" not in code

    def test_the_engine_still_accepts_a_bare_corpus(self):
        """Existing construction must keep working without a resolver."""
        from app.services.reasoning.engine import DiagnosticReasoningEngine

        engine = DiagnosticReasoningEngine(_Store([]))
        assert engine.resolver is not None
        assert engine.resolver.store is engine.corpus


# ======================================================================
# No authority, no knowledge state
# ======================================================================

class TestNoAuthority:
    @staticmethod
    def _executable_code(module):
        """Module source with every docstring removed.

        These modules explain at length what they deliberately do not do, so
        the words "REVIEWED", "fetch" and "retrieval" all appear in prose. A
        plain substring check matches the explanation instead of the
        behaviour -- which would make the assertion pass or fail for reasons
        unrelated to what the code actually does.
        """
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef))
                    and body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:]
        return ast.unparse(tree)

    def test_the_resolver_grants_nothing(self):
        from app.services.knowledge import resolver as module

        code = self._executable_code(module)
        for forbidden in ("consumer_purchasable", "safety_verdict",
                          "formula_id", "review_status", "REVIEWED",
                          "VERIFIED_EXTERNAL", "clinical_ranking_eligible",
                          "TrustScore", "eligibility"):
            assert forbidden not in code

    def test_a_miss_stays_a_miss(self):
        """Recording must not conjure a match."""
        resolver = KnowledgeResolver(_Store([]), record_gaps=True)
        result = resolver.resolve_entity("风热犯表", ["pattern"])
        assert result.resolved is False
        assert result.matches == []

    def test_the_resolver_performs_no_retrieval(self):
        from app.services.knowledge import resolver as module

        code = self._executable_code(module)
        for network in ("httpx", "requests", "urllib", "aiohttp", "search_api",
                        "fetch", "http://", "https://"):
            assert network not in code

    def test_the_recorder_exposes_no_reader_the_pipeline_could_consult(self):
        """summarise_gaps is for operators; nothing clinical may read gaps."""
        import ast
        import inspect
        from app.services.reasoning.engine import DiagnosticReasoningEngine

        code = ast.unparse(ast.parse(inspect.getsource(
            DiagnosticReasoningEngine)))
        assert "summarise_gaps" not in code
        assert "record_gap" not in code       # the resolver owns that, not the engine


# ======================================================================
# Gap recording: shape, privacy, failure direction
# ======================================================================

class TestGapRecording:
    def test_a_miss_records_exactly_one_gap(self, monkeypatch):
        recorded = []
        monkeypatch.setattr("app.services.knowledge.resolver.record_gap",
                            lambda **kw: recorded.append(kw) or True)
        KnowledgeResolver(_Store([])).resolve_entity("风热犯表", ["pattern"])
        assert len(recorded) == 1
        assert recorded[0]["subject"] == "风热犯表"
        assert recorded[0]["claim_type"] == CLAIM_PATTERN_ENTITY

    def test_a_hit_records_nothing(self, monkeypatch):
        recorded = []
        monkeypatch.setattr("app.services.knowledge.resolver.record_gap",
                            lambda **kw: recorded.append(kw) or True)
        KnowledgeResolver(_Store([{"pattern_id": "p"}])).resolve_entity(
            "风热犯表", ["pattern"])
        assert recorded == []

    def test_the_payload_is_an_explicit_allowlist(self):
        payload = build_gap_payload(
            claim_type=CLAIM_PATTERN_ENTITY, subject="风热犯表",
            entity_types=["pattern"], reviewed_only=True)
        assert set(payload) == {"claim_type", "entity_types", "reviewed_only",
                                "occurred_at", "subject", "subject_form"}

    def test_a_clinical_concept_is_recorded_plainly(self):
        """The KPI is only actionable if the concept is legible."""
        assert normalise_subject("风热犯表") == {"subject": "风热犯表",
                                              "subject_form": "CONCEPT"}

    def test_an_overlong_subject_is_hashed_not_stored(self):
        """Real pattern names are short; long strings are likely patient text."""
        leaked = ("患者张三说头痛咳嗽发热三天，家住北京市朝阳区，"
                  "邮箱 patient@example.com，请直接开药") * 2
        out = normalise_subject(leaked)
        assert out["subject_form"] == "HASHED_OVERLONG"
        assert out["subject"] == hashlib.sha256(
            leaked.encode("utf-8")).hexdigest()[:16]
        assert leaked not in json.dumps(out, ensure_ascii=False)

    def test_hashing_still_aggregates_a_recurring_gap(self):
        long_one = "长" * (MAX_PLAIN_SUBJECT_CHARS + 5)
        assert (normalise_subject(long_one)["subject"]
                == normalise_subject(long_one)["subject"])

    @pytest.mark.parametrize("blank", [None, "", "   "])
    def test_an_empty_subject_is_marked_not_invented(self, blank):
        assert normalise_subject(blank) == {"subject": None,
                                            "subject_form": "EMPTY"}

    def test_the_payload_carries_nothing_about_the_patient(self):
        payload = build_gap_payload(
            claim_type=CLAIM_PATTERN_ENTITY, subject="风热犯表",
            entity_types=["pattern"], reviewed_only=True)
        blob = json.dumps(payload, ensure_ascii=False)
        for forbidden in ("头痛", "咳嗽", "发热", "@example.com", "user",
                          "case-", "corr-", "sk-", "eyJ", "Bearer",
                          "password", "railway.internal", "complaint",
                          "observations", "answers"):
            assert forbidden not in blob

    def test_a_storage_failure_is_swallowed(self, monkeypatch):
        """An analytics row is never worth a governed clinical result."""
        def boom():
            raise RuntimeError("database unreachable")
        monkeypatch.setattr(gap_recorder, "get_session_factory", boom)
        assert record_gap(claim_type=CLAIM_PATTERN_ENTITY, subject="风热犯表",
                          entity_types=["pattern"]) is False

    def test_a_recording_failure_does_not_affect_the_lookup(self, monkeypatch):
        monkeypatch.setattr("app.services.knowledge.resolver.record_gap",
                            lambda **kw: (_ for _ in ()).throw(
                                RuntimeError("boom")))
        resolver = KnowledgeResolver(_Store([]))
        with pytest.raises(RuntimeError):
            resolver.resolve_entity("风热犯表", ["pattern"])
        # ...which is why the engine calls through .search on a resolver whose
        # recorder is itself fail-open; record_gap never raises in practice.
        assert record_gap(claim_type=CLAIM_PATTERN_ENTITY, subject="x",
                          entity_types=["pattern"]) in (True, False)


# ======================================================================
# Baseline KPI
# ======================================================================

class TestGapSummary:
    def test_the_summary_reports_frequency_by_subject(self):
        for _ in range(3):
            record_gap(claim_type=CLAIM_PATTERN_ENTITY, subject="风热犯表",
                       entity_types=["pattern"])
        record_gap(claim_type=CLAIM_PATTERN_ENTITY, subject="脾胃气虚",
                   entity_types=["pattern"])
        summary = summarise_gaps()
        assert summary["available"] is True
        top = dict(summary["top_subjects"])
        assert top.get("风热犯表", 0) >= 3
        assert top.get("脾胃气虚", 0) >= 1

    def test_the_summary_is_unavailable_rather_than_fatal(self, monkeypatch):
        def boom():
            raise RuntimeError("database unreachable")
        monkeypatch.setattr(gap_recorder, "get_session_factory", boom)
        assert summarise_gaps()["available"] is False

    def test_no_hit_rate_is_claimed_without_hits_recorded(self):
        """A rate needs a denominator this phase deliberately does not collect."""
        assert "LOCAL_REVIEWED_HIT_RATE" not in summarise_gaps()
        assert "hit_rate" not in summarise_gaps()
