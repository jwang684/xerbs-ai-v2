# Governed clinical objects (X1D-AIV2-GOV2-C1)

One page on what changed and why. Implementation lives in
`app/services/governance/` and migration `0008_x1d_gov2c1`.

## The defect this closes

`SafetyEngine.create_relationship` wrote `review_status='REVIEWED'` as a
literal, and `create_rule` did the same. The ORM default was `DRAFT`; the code
overrode it. So a relationship's authority came from a constant in a source
file rather than from any review, and the ranking layer trusted it. Production
holds one `PATTERN_FORMULA` edge of exactly that shape: no reviewer, no review
event, no version, status `REVIEWED`.

Safety rules had the identical defect, and it mattered more there: a rule can
carry `action=BLOCK`, so an unreviewed rule could decide what a patient may buy.

## Who owns what

| ai-v2 owns | xerbs-core owns |
| --- | --- |
| object content, versions, semantic identity | authenticated **human** reviewer identity |
| evidence provenance, lifecycle state | reviewer authorization |
| ranking eligibility, local audit history | human separation of duties |
| validating a core attestation binds to the right object | the immutable attestation itself |

**This service has no end-user identity and must not acquire one.** Everything
that acts here is a machine and is named as one. There is deliberately no
`CLINICAL_REVIEWER` principal: creating one to stand in for a person is the
failure this programme exists to undo.

## Lifecycle

```
create ──> DRAFT ──SUBMIT──> IN_REVIEW ──REQUEST_CHANGES──> DRAFT
                                       ──REJECT──────────> REJECTED
                             IN_REVIEW ──APPROVE─────────> ✗ refused
DRAFT / REJECTED / REVIEWED ──RETIRE──> RETIRED
```

`IN_REVIEW -> REVIEWED` is **absent**, not restricted. `lifecycle.next_state`
raises `HumanAttestationRequired` for every approve attempt, from every state,
for every principal. There is no configurable bypass — a switch that can be
turned on is a switch that will be turned on. AI-GOV2 replaces the raise with a
verified-attestation path; until then this is the entire implementation of
"a machine may not approve".

## Semantic identity

| object | identity | how |
| --- | --- | --- |
| source | `source_registry.source_id` | already semantic; reused, no new field |
| entity | `external_id` | **supplied** by authoring tooling |
| relationship | `rel:<src>\|<TYPE>\|<tgt>` | **derived** from endpoints |
| safety rule | `rule:<target>\|<TYPE>\|<trigger>` | **derived** |

Entity identity is never derived from `name`: a display name is mutable, and
deriving identity from it would fork the identity on rename. CJK is allowed —
the established convention is `xerbs-core:canonical-formula:清肺排毒汤`, and an
ASCII-only grammar would have rejected the one semantic identity this corpus
already uses. Whitespace, control characters and `|` are refused.

Legacy rows keep `external_id IS NULL`. Nothing invents one; a fabricated
identity would be indistinguishable from a real one later.

## Canonical hashing

`app/services/governance/canonical.py`, spec `xerbs-canonical-hash/1`. UTF-8,
never escaped; every string NFC-normalised; keys sorted; no insignificant
whitespace; set-valued arrays sorted and ordered arrays left alone; `None`
omitted rather than serialised as `null`; floats refused; timestamps, primary
keys, review status, reviewer identity and the version counter excluded.

Literal vectors live in `tests/vectors/canonical_hash_vectors.json` for
xerbs-core to consume verbatim. Recomputing an expected digest with the same
implementation inside an assertion proves nothing, so the expected values are
constants in the artifact.

## Governance provenance

How an object came to hold its status. `LEGACY_` here means *pre-attestation*,
not old.

| value | meaning |
| --- | --- |
| `UNREVIEWED` | authored under GOV2, not yet reviewed |
| `LEGACY_UNREVIEWED` | holds REVIEWED with **no review record at all** |
| `LEGACY_SELF_REVIEWED` | a lifecycle ran, but submitter == approver |
| `LEGACY_INDEPENDENTLY_REVIEWED` | two distinct identities, still no attestation |
| `ATTESTED` | approved against a verified core attestation — nothing has this yet |

## Ranking eligibility

One predicate, `lifecycle.is_governed_object_ranking_eligible`, used by ranking
and by safety screening so they cannot drift into two answers. A GOV2-era
object needs `review_attestation_id` — which only a verified core decision can
produce — plus at least one REVIEWED evidence source. A lifecycle string is
necessary and never sufficient.

`LEGACY_ELIGIBILITY_GRANDFATHERED` is `True`: legacy rows keep the eligibility
they already had, so this phase changes no live retrieval behaviour. **Flipping
it to False is AI-GOV2-D, and it makes every pre-attestation REVIEWED
relationship non-ranking-eligible until a human re-reviews it.** For Production
that means 风热犯卫 → 银翘散 stops producing a formula candidate. That belongs
to a phase that schedules it.

## Breaking change

`POST /api/v1/safety/relationships` and `POST /api/v1/safety/rules` now return
`review_status: "DRAFT"` and `clinical_ranking_eligible: false`, plus
`external_id`, `version`, `content_hash`, `evidence_hash` and
`governance_provenance`. Existing keys are unchanged; the old REVIEWED response
is gone, deliberately. Backward compatibility here would have meant preserving
the defect.

`actor_role` in the request body still validates as a declared field and now
**authorizes nothing** — it cannot, because creation confers nothing.

## Testing

Tests that are really about retrieval or screening use
`tests/governed_fixtures.simulate_core_attestation_for_test`, which writes rows
directly to stand in for the future AI-GOV2 integration. It is a test fixture,
not a bypass: no application code path reaches it, and the attestation ids it
writes carry an unmistakable `att-SYNTHETIC-TEST-` prefix.
