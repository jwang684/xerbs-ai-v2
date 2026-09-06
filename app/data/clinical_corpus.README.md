# Clinical Corpus Data Policy

This directory is the future home of reviewed, versioned clinical knowledge records.

Phase 4 intentionally does **not** manufacture or infer production clinical facts from the legacy TSE catalog. The three legacy formula fixtures are loaded in memory as `DRAFT` migration records only and are not clinical-ranking eligible.

A record becomes ranking eligible only when:

1. `review_status == REVIEWED`, and
2. at least one explicit source reference is attached.

Future imports should preserve source provenance, reviewer identity outside this stateless AI package, and immutable version history in the system of record.
