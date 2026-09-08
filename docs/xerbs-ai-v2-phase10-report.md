# Xerbs AI v2 — Phase 10 Report

**Pattern Convergence + Contradiction Engine + Base44 Integration Contract**

Status: **implemented and verified.** Service version `0.10.0`, Alembic head `0004_phase10`.

> **Note on provenance of this document.** Phase 10 code was already present in the
> repository when this report was written; the report documents the implementation as
> it actually exists rather than describing newly written code. No Phase 10 code was
> modified to produce it. The repository has since advanced through Phases 12B-1,
> 12B-2 and 12B-3 (clinical entity detail, relationships read API, safety rules read
> API), all of which are additive and leave the Phase 10 surface unchanged.

---

## 1. Files

Git history does not isolate a Phase 10 diff: the initial import (`022fa16 Add files
via upload`) landed the tree as a single bulk commit, so the files below are listed by
role rather than as a commit range.

| File | Role |
|---|---|
| `app/services/reasoning/convergence.py` | `PatternConvergenceEngine` — deterministic multi-turn convergence |
| `app/services/reasoning/contradictions.py` | `ContradictionEngine` — explicit-negation annotator |
| `app/services/reasoning/engine.py` | structured reasoning over intake |
| `app/services/interview/engine.py` | adaptive interview, question selection, stopping |
| `app/services/integration/base44.py` | `Base44GenerationService` — idempotency, retry, persisted status |
| `app/api/base44.py` | Base44 HTTP contract |
| `app/schemas/reasoning.py` | `ConvergenceMetrics`, `ReasoningResponse` |
| `app/schemas/interview.py` | interview state / question schemas |
| `app/schemas/integration.py` | `Base44GenerateRequest`, `Base44GenerationResponse` |
| `app/services/recommendation/assembler.py` | retrieval precedence + governance gate |
| `app/db/models.py` | `DiagnosticInterview`, `DiagnosticInterviewTurn`, `GenerationRequest` |
| `migrations/versions/0004_phase10_convergence_base44.py` | Phase 10 migration |
| `tests/test_phase10.py` | Phase 10 test suite (8 tests) |

## 2. Architecture

```
Base44 /ask
   │  POST /api/v1/integrations/base44/generate   (Idempotency-Key, X-Correlation-ID)
   ▼
Base44GenerationService ──► GenerationRequest row (PENDING → SUCCEEDED | FAILED)
   │
   ▼
RecommendationAssembler
   ├─ ReasoningEngine        → structured symptoms, pattern hypotheses, missing info
   ├─ ContradictionEngine    → explicit contradictions, PATTERN_CONTRADICTIONS_PRESENT
   ├─ PatternConvergenceEngine → ConvergenceMetrics (deterministic)
   ├─ PersistentClinicalStore → reviewed-corpus retrieval (governance gate)
   └─ SafetyEngine           → deterministic source-backed screening
   ▼
Draft recommendation → Base44 Recommendation Domain
```

The convergence and contradiction engines are **pure functions over structured
reasoning snapshots**. They never call the model, so a provider outage degrades
candidate retrieval but cannot corrupt convergence arithmetic.

## 3. API contract

| Method | Path |
|---|---|
| POST | `/api/v1/integrations/base44/generate` |
| GET | `/api/v1/integrations/base44/generations/{generation_id}` |
| POST | `/api/v1/integrations/base44/generations/{generation_id}/retry` |
| POST | `/api/v1/interviews/start` |
| GET | `/api/v1/interviews/{interview_id}` |
| POST | `/api/v1/interviews/{interview_id}/answers` |
| POST | `/api/v1/interviews/{interview_id}/recommendation` |
| POST | `/api/v1/interviews/{interview_id}/complete` |

`POST /generate` **requires** the `Idempotency-Key` header (400 without it) and accepts
optional `X-Correlation-ID`; when absent a `corr-<hex>` id is generated. Response
carries `generation_id`, `correlation_id`, `request_id`, `organization_id`, `status`,
`retry_count`, `recommendation`, `error_code`, `error_message`.

No Base44 entity IDs are invented. `request_id` and `organization_id` are echoed from
the caller; `generation_id` is Xerbs-owned and namespaced `gen-`.

## 4. Convergence algorithm

`PatternConvergenceEngine.evaluate(current, previous)` — deterministic, no model call:

```
evidence_sufficiency = max(0, 1 − min(1, 0.22·HIGH + 0.11·MEDIUM + 0.05·LOW))
pattern_stability    = |current ∩ previous| / |current ∪ previous|      (0.5 on first turn)
verified_pattern_strength = max(model_confidence where corpus_match)     (0.0 if none)
contradiction_penalty = min(1, 0.18 · contradiction_count)

score = 0.34·evidence_sufficiency
      + 0.30·pattern_stability
      + 0.36·verified_pattern_strength
      − 0.25·contradiction_penalty          → clamped to [0,1], rounded to 3dp
```

`ConvergenceMetrics` exposes every component plus `stable_pattern_names`,
`changed_pattern_names`, `contradiction_count`, and a human-readable `rationale` list —
so convergence is auditable, not a black box.

**Convergence is not TrustScore.** It is a Xerbs-internal interview-progress signal.
`GET /api/v1/knowledge/clinical/governance` reports
`trust_score_owned_by_ai_service: false`.

**LLM confidence alone cannot establish convergence.** `verified_pattern_strength`
counts model confidence *only* for hypotheses with `corpus_match` true, and carries
weight 0.36 — below the 0.72 readiness threshold. A confident but unverified model
hypothesis cannot reach readiness on its own.

## 5. Contradiction behavior

`ContradictionEngine.annotate` is deliberately conservative: it fires **only on explicit
patient text**. For each clinical term the model cites as supporting evidence, it checks
the patient's own words for an explicit negated form (e.g. `发热` vs `无发热 / 没有发热 /
否认发热 / 不发热`).

On a hit it appends an `EvidenceItem` to `assessment.contradictions` sourced
`deterministic_contradiction_check`, adds the `PATTERN_CONTRADICTIONS_PRESENT`
uncertainty flag, and sets `ready_for_formula_retrieval = False`.

Missing information is **never** converted into a contradiction — absence of a statement
is not a denial. Contradictions are never silently resolved: they are surfaced in
structured output and mechanically suppress automatic formula readiness.

## 6. Stopping rules

Question selection (`_select_questions`) ranks unresolved slots by
`PRIORITY_SCORE {HIGH 1.0, MEDIUM 0.72, LOW 0.45}` plus a small deterministic
field-specific tiebreak, excludes already-asked ids, and returns the top *N*.

Status (`_status`) — the interview reaches `READY_FOR_RECOMMENDATION` when **either**:

1. `ready_for_formula_retrieval` **and** `convergence_score ≥ 0.72`; **or**
2. no remaining questions **and** no missing information **and** no contradictions.

Otherwise `OPEN`. Convergence is never forced: an interview that cannot resolve returns
an explicit uncertainty state rather than a manufactured verdict, and any contradiction
blocks path 1 via `ready_for_formula_retrieval = False`.

## 7. Reviewed knowledge retrieval precedence

Enforced in `RecommendationAssembler`, in strict order:

1. **Reviewed `PATTERN_FORMULA` relationship** → flags
   `REVIEWED_PATTERN_FORMULA_RELATIONSHIP_RETRIEVAL`, `REVIEWED_CLINICAL_CORPUS`,
   `AI_FORMULA_RANKING_NOT_USED`.
2. **Reviewed indication compatibility** → flagged explicitly, either
   `NO_REVIEWED_PATTERN_FORMULA_RELATIONSHIP_MATCH` or
   `REVIEWED_INDICATION_RETRIEVAL_WITH_INCOMPLETE_REASONING`.
3. **Model-proposed formulas** → **suppressed entirely**. `candidates = []`; they are not
   exposed, not safety-screened, and not selectable. Represented only as
   `MODEL_FORMULA_CANDIDATES_SUPPRESSED_UNVERIFIED_CORPUS`,
   `REVIEWED_CLINICAL_CORPUS_REQUIRED_FOR_FORMULA_SELECTION`,
   `NO_VERIFIED_FORMULA_CANDIDATE`.
4. **Nothing** → `NO_FORMULA_CANDIDATE`.

Eligibility requires `review_status == REVIEWED` **and** at least one persisted
`entity_source` row. DRAFT / IN_REVIEW / REJECTED / RETIRED records and
`clinical_ranking_eligible = false` records can never rank. No legacy fallback formula is
ever substituted.

## 8. Base44 ownership boundary

| Xerbs AI v2 owns | Base44 owns |
|---|---|
| intake interpretation | Request |
| diagnostic interview | Recommendation aggregate |
| structured reasoning | Evidence |
| pattern hypotheses | Verification |
| convergence | RecommendationVersion |
| formula candidate retrieval | **TrustScore** |
| AI/model confidence | OutcomeLink |
| uncertainty | practitioner approval |
| safety screening | patient outcome workflow |
| provenance / model metadata | analytics |

Xerbs AI v2 does not calculate TrustScore, and contains no commerce, payments, auth UI,
clinic management, outcome, or analytics code.

## 9. Idempotency design

Key: the `Idempotency-Key` header, stored on `generation_request` alongside a
`request_hash` — SHA-256 over the canonically serialized payload (`sort_keys`,
`ensure_ascii=False`, tight separators).

- **Same key + same payload** → returns the existing persisted generation. No duplicate
  interview, no duplicate generation event.
- **Same key + different payload** → `IdempotencyConflictError` → **HTTP 409**.
- **Missing key** → **HTTP 400**.
- **Retry** → only `FAILED` generations may retry; anything else → **HTTP 422**. Retry
  increments `retry_count`, clears the error fields, and re-executes against the stored
  `request_snapshot`.

Status is persisted (`PENDING → SUCCEEDED | FAILED`) so a caller that loses its
connection can recover state via `GET /generations/{id}`.

## 10. Test results

All commands run against Python 3.12.10 in `.venv`.

| Command | Result |
|---|---|
| `pytest` | **80 passed**, 2 warnings |
| `python -m compileall -q app` | **OK** (exit 0) |
| `alembic upgrade head` (clean DB) | **OK** → head `0004_phase10`, 14 tables |
| FastAPI/OpenAPI smoke | `/health` 200, `/openapi.json` 200 (30 paths, 50 schemas), `/docs` 200 |

The two warnings are pre-existing third-party deprecations (Starlette `TestClient`
httpx notice; `anyio.abc.BlockingPortal` alias) — not project code.

`tests/test_phase10.py` (8 tests): convergence rewards stability + verified pattern;
convergence penalizes missing information and instability; explicit patient negation
becomes an auditable contradiction; interview exposes the convergence breakdown; Base44
contract requires an idempotency key; contract is idempotent and correlated; key reuse
with a different payload conflicts; a succeeded generation cannot be retried.

## 11. Remaining production blockers

1. **Empty production corpus.** Production holds 4 formulas — 3 DRAFT legacy fixtures
   plus 1 RETIRED test artifact — and **0 patterns, 0 herbs, 0 relationships, 0 safety
   rules**. Every retrieval path therefore returns `NO_FORMULA_CANDIDATE`. The pipeline
   is correct but has nothing eligible to serve.
2. **Source architecture gaps** (see the Phase 12C-1 audit). Most serious: version
   snapshots embed *submitted* source text while `source_registry` keeps the
   *first-written* row, so `GET /entities/…` and `GET …/history` can return contradictory
   provenance for the same entity. There is also no Source API, sources are not
   review-gated, and no reverse lookup exists.
3. **Error contract is coarse.** Failures collapse to a single `GENERATION_FAILED` code.
   The distinct machine-readable codes (provider unavailable, provider auth failure,
   invalid structured output, insufficient information, corpus unavailable, no eligible
   candidate, safety-blocked) are expressed as *uncertainty flags* on the recommendation
   rather than as top-level error codes.
4. **`app_version` is pinned at `0.10.0`.** `/health` cannot distinguish deployments;
   production served Phases 12B-1..3 while still reporting `0.10.0`.
5. **No auth on read endpoints.** Acceptable for server-to-server behind a private
   network; needs a decision before public exposure.
