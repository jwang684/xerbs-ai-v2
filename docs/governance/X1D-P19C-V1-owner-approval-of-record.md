# Approval of Record: Applicability Contract X1D-P19C-V1

| Field | Entry |
|---|---|
| Contract | `X1D-P19C-V1` |
| Approval basis | `OWNER_INSTRUCTION` |
| Approved by | The project owner, by explicit instruction. Not a named clinician. |
| Owner instruction (verbatim) | 「原稿无修改通过，P19C 第3节问题、第5节决策表及 A–H 项均已明确批准。」 |
| Designation as approval of record | The project owner explicitly designated the instruction above as the owner approval of record for `X1D-P19C-V1`. |
| Approval date | 2026-10-11 |
| Approval scope | P19C §3 (patient clarification wording and choices), §5 (decision table) and decisions A–H, all **unchanged** |
| Reviewed document | `docs/governance/X1D-P19C-cough-consensus-2021-applicability-review.md`, SHA-256 `82db5e2a57a47cf2aac0e42d5d2dffe623afd846859cbd820049a2ccd27a1b5e` (byte-identical to the reviewed packet) |
| Clinical review status | **NOT INDEPENDENTLY CLINICALLY ATTESTED.** No clinician identity, qualification, signature or clinical attestation is recorded. No `clinical:review` authority is granted. |
| Source-verification status | **FOUR OBJECTS SOURCE_VERIFIED** by Human B (admin 4), each verified at v2 and now at v3 |
| Environment | **STAGING ONLY.** This record does not authorise any Production use. |

## Source and governed objects

**Source:** 《咳嗽中医诊疗专家共识意见（2021）》, 中华中医药学会肺系病分会, 世界中医药学会联合会肺系病专业委员会. 中医杂志 2021;62(16):1465–1472, DOI 10.13288/j.11-2166/r.2021.16.018.

**Inspected copy:** 333,069 bytes, SHA-256 `8a116119722fc54a271a350e0ce338325042347481c20851ed3e831f4f10de3e`.

| Object | ID | Content hash | Source verification (attestation id, verified at object v2, not revoked) |
|---|---|---|---|
| Source | `cacm-pulm-cough-consensus-2021` | `87ede41f34f8add8eac401679b30884687180b5101b19127edba8e58b5aae125` | `1930237a-2bbb-4f0b-b589-579573c57d4c` (2026-10-11 22:12:07 UTC) |
| Pattern 风热犯肺证 | `pat-1db7a0fbc59f4e28` | `16e01708aa8bdf1b1c402c60ab6ced75d9ec72c8eff57c9f67f6423a173ac215` | `9bdefef1-74eb-4e52-ae56-ebbd8079c51a` (22:12:15 UTC) |
| Formula 桑菊饮加减 | `frm-cf8d442637a04257` | `d35988422e4c8c3db25f916c2dc3ea6eb54da06d2dd16628b9a59fba8c9f9539` | `8daf2ad7-eecf-49a7-a279-8124b8094fef` (22:12:31 UTC) |
| Relationship 风热犯肺证 → 桑菊饮加减 | `rel-84ce66fe24114d9b` | `e95527ec58f5150042238390d713eabb232e5c3bb1c22656071988746f54ccfa` (evidence `e56d814d9162c7d743c5340c164807af82203936f67ca662ce6cc499d335590d`) | `1c7f3398-d63f-4081-8148-a31a539f8999` (22:39:21 UTC) |

**Core canonical formula:** 桑菊饮加减, id 8. It has no product mapping, no product and no purchase eligibility.

## What the approved contract requires

Answers count only when given to the fixed questions, asked in this order and at most once per case.

1. 「咳嗽是您这次最主要或唯一的不适吗？」 (是 / 否 / 不清楚). Only 是 continues.
2. 「除了咳嗽，您现在是否有下面任何一种情况？」 followed by the five P19C §3 items (都没有 / 有 / 不清楚). Only 都没有 satisfies the condition.

Every other answer, missing answer, malformed answer or unsupported condition fails closed.

- 有 shows only a bounded practitioner-review notice.
- 都没有 is a patient report, not a determination that disease is absent.
- The independent Core safety and profile checks still apply.
- Nothing is purchasable.

## Limits of this record

This is a project-owner approval of an applicability rule for Staging engineering use. It is not an independent clinical review of the formula, the syndrome or the safety of any recommendation, and it must never be presented as one. Changing the conditions, the question wording or the reviewed document requires a new contract version and new approval.
