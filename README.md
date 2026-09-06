# Xerbs AI v2

Clinical-intelligence service extracted from the legacy AI Herbal Formula System.

## Current version: 0.10.0

Phase 10 adds deterministic multi-turn pattern convergence, explicit contradiction handling, and a stable Base44 integration contract with idempotency, correlation IDs, persisted generation status, and retry semantics.

### Core APIs

- `GET /health`
- `POST /api/v1/reasoning/analyze`
- `POST /api/v1/recommendations/generate`
- `POST /api/v1/followups/generate`
- `POST /api/v1/interviews/start`
- `GET /api/v1/interviews/{interview_id}`
- `POST /api/v1/interviews/{interview_id}/answers`
- `POST /api/v1/interviews/{interview_id}/recommendation`
- `POST /api/v1/interviews/{interview_id}/complete`
- `POST /api/v1/integrations/base44/generate`
- `GET /api/v1/integrations/base44/generations/{generation_id}`
- `POST /api/v1/integrations/base44/generations/{generation_id}/retry`

### Pattern convergence

Each interview reasoning snapshot may include:

- evidence sufficiency
- pattern stability across turns
- reviewed-pattern strength
- contradiction penalty
- stable / changed pattern names
- convergence rationale

The convergence score is deterministic and is not Xerbs TrustScore.

### Contradiction policy

Only explicit patient statements are used to create contradiction findings. Missing information is never silently converted into a contradiction. A detected contradiction prevents automatic readiness for deterministic formula retrieval and is surfaced via `PATTERN_CONTRADICTIONS_PRESENT`.

### Base44 integration contract

`POST /api/v1/integrations/base44/generate` requires the `Idempotency-Key` header and accepts optional `X-Correlation-ID`.

Same idempotency key + same payload returns the same persisted generation. Same key + different payload returns HTTP 409. Generation records persist PENDING / SUCCEEDED / FAILED state plus retry count and error information.

The service explicitly reports `trust_score_owned_by_ai_service=false`. Xerbs TrustScore remains owned by the Base44 Recommendation Domain.

### Formula retrieval precedence

`Reviewed Pattern → reviewed PATTERN_FORMULA relationship → Reviewed Formula`

The reviewed indication-overlap path remains a compatibility path and is explicitly flagged when pattern verification is incomplete.

### Existing governance and safety guarantees

- TSE catalog data is isolated from clinical ranking.
- Pattern / Formula / Herb records are review-gated and source-gated.
- Persistent SQLAlchemy corpus with immutable versions and append-only review/audit events.
- Reviewer-role enforcement, optimistic concurrency, retire/supersede lifecycle.
- Reviewed Pattern→Formula and Formula→Herb relationships.
- Deterministic source-backed contraindication / interaction safety rules.
- AI confidence, convergence, safety state, catalog matching, and Xerbs TrustScore remain separate concepts.
- AI output always requires practitioner review.

## Database

Local:

```bash
DATABASE_URL=sqlite:///./xerbs_ai_v2.db
```

Production:

```bash
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME
```

Apply migrations:

```bash
alembic upgrade head
```

Phase 10 migration: `0004_phase10_convergence_base44.py`.

## Railway-ready container

The Docker image now runs migrations before startup and honors Railway's `PORT` environment variable:

```text
alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
```

No credentials are embedded in the image or source tree.
