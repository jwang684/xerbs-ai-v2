# Clinical Applicability Review: 2021 Cough Consensus

**Review ID:** X1D-P19C
**Status:** AWAITING CLINICAL DECISION. Nothing in this document is an approved clinical rule.
**Purpose:** decide what patient context must be established before an AI system may suggest one source-bounded formula.

> **What has and has not been checked.** The source text quoted here was checked against the published article; this is *source verification*. It is **not** an independent clinical review of the formula, the syndrome, or the safety of any recommendation. This review asks you for exactly that clinical judgement, limited to the questions in §4.

---

## 1. Source and scope

**Source:** 《咳嗽中医诊疗专家共识意见（2021）》, 中华中医药学会肺系病分会, 世界中医药学会联合会肺系病专业委员会. 中医杂志 2021年8月 第62卷第16期 1465–1472. DOI 10.13288/j.11-2166/r.2021.16.018.

**Copy inspected:** a CNKI-generated PDF of the published article, 8 pages, 333,069 bytes, SHA-256 `8a116119722fc54a271a350e0ce338325042347481c20851ed3e831f4f10de3e`. Please check against your own copy.

**§1.1 applicability definition** (p. 1465, right column), verbatim:

> 本共识定义的咳嗽是特指以咳嗽为主要或唯一症状者，作为其他疾病伴随症状的咳嗽，如哮病、喘证、肺胀、肺痿、肺痈等兼有的咳嗽症状不纳入本次共识的论述范围。

**Related scope statements** (verbatim):

- **§1.3** (p. 1465), on what is included:
  > 亚急性咳嗽主要包括感染后咳嗽、咳嗽变异性哮喘和上气道咳嗽综合征；慢性咳嗽主要包括咳嗽变异性哮喘、上气道咳嗽综合征、嗜酸粒细胞性支气管炎、胃食管反流性咳嗽、变应性咳嗽以及咳嗽高敏综合征。上述咳嗽可参考本专家共识治疗。
- **§5.1.1** (p. 1467) places 风热 among acute exogenous cough (急性咳嗽多属外感咳嗽 … 以风寒、风热、风燥为主).

**The relationship under review** (§5.3.2, pp. 1467–1468):

| Item | Source text |
|---|---|
| 证型 | 风热犯肺证 |
| 证候 | 咳嗽频剧，气粗或咳声嘶哑，咽喉燥痛，咯痰不爽，痰黏黄稠，鼻流黄涕，口渴，头痛，恶风，身热，舌质红、苔薄黄，脉浮数或浮滑。 |
| 病机 | 风热犯表，卫表不和，肺失清肃，肺热伤津。 |
| 治法 | 疏风清热，宣肺止咳。 |
| 方药 | 桑菊饮（《温病条辨》）加减 |
| 处方 | 桑叶 9g，菊花 9g，苦杏仁 9g，连翘 9g，薄荷 6g（后下），桔梗 9g，芦根 15g，甘草 6g |
| 加减 | 肺热甚 +黄芩、鱼腥草; 咳甚 +百部、枇杷叶; 咽痛 +射干; 内夹湿邪 +薏苡仁、佩兰; 热伤肺津 +南沙参、天花粉; 痰中带血 +白茅根、藕节; 夏令夹暑湿 +六一散 |

**Why this review is needed.** The system already asks one fixed question: 「咳嗽是您这次最主要或唯一的不适吗？」 (是 / 否 / 不清楚). That covers the *main or only symptom* part of §1.1. It does **not** cover the second part: the exclusion of cough that accompanies other diseases (哮病、喘证、肺胀、肺痿、肺痈 **等**). Note that the list is open-ended.

---

## 2. Source-derived exclusions (§4.2.3 咳嗽与其他疾病的鉴别)

All of §4.2.3 is on p. 1467: items 1–3 are in the left column, items 4–6 in the right column.

The first two columns are **source statements**. The last column is **only a proposed interpretation** for you to confirm or correct.

| Source term | Distinguishing features stated by the source (verbatim) | Named in §1.1? | Proposed interpretation: possible urgent concern? |
|---|---|---|---|
| 哮病 | 哮以声响言，以发作时喉中哮鸣有声为主要临床特征，呼吸气粗困难，甚则喘促不能平卧，发作与缓解均迅速 … 哮病为一种反复发作的独立疾病 | Yes | Possibly: breathlessness, cannot lie flat |
| 喘证 | 以气息言，呼吸困难，甚则张口抬肩，鼻翼煽动，不能平卧 … 喘证可并发于多种急、慢性疾病过程中，哮必兼喘，而喘未必兼哮 | Yes | Possibly: severe dyspnoea |
| 肺胀 | 是多种慢性肺系疾病反复、迁延日久所致 … 常患咳嗽、喘证、哮病等病史日久，以胸部膨满、喘咳上气、烦躁心悸等症为主要表现，甚则肢体浮肿、面色晦暗 | Yes | Possibly: oedema, palpitations, chronic lung disease |
| 肺痈 | 以咯吐大量腥臭脓血痰为特征，多伴有咳嗽、发热、胸痛等症 | Yes | Possibly: purulent or bloody sputum, chest pain, fever |
| 肺痿 | 以咯吐浊唾涎沫为主症，为肺脏慢性虚损性疾病，伴有咳嗽气短、动则气喘 … 肺功能及胸部影像学检查有助于鉴别诊断 | Yes | Ordinary practitioner review? |
| 肺痨 | 咳嗽为其四大主症之一，以干咳、痰中带血或咯吐血痰为主要特征，常伴有潮热盗汗、形体瘦削 … 胸部影像学检查有助于明确病灶或诊断 | **No** (§4.2.3 only) | Possibly: haemoptysis, weight loss |
| 肺癌 | 表现为刺激性呛咳，咳嗽顽固，持续日久不愈，反复咯血痰，出现不明原因的低热、胸痛、胸闷短气，伴消瘦、乏力等症 … 中老年男性吸烟患者好发 | **No** (§4.2.3 only) | Possibly: haemoptysis, weight loss, persistent cough |

**Notes on the table:**

- §1.1 names five conditions. 肺痨 and 肺癌 appear only in the §4.2.3 differential. Whether 「等」 covers them is your decision (§4, item B); no additional exclusions have been invented.
- 咳嗽变异性哮喘 is explicitly **included** (§1.3), while 哮病 is **excluded** (§1.1). A lay question such as "do you have asthma?" would therefore misclassify patients (§4, item E).

---

## 3. Proposed patient clarification (DRAFT, NOT APPROVED)

It would be shown **only** after the patient answers 是 to 「咳嗽是您这次最主要或唯一的不适吗？」, and only when this specific relationship is otherwise about to be used.

> **除了咳嗽，您现在是否有下面任何一种情况？**
> - 喘息时喉咙里有哮鸣音（像吹哨的声音）
> - 呼吸困难，或者躺平时喘不过气
> - 咳出大量有腥臭味的脓痰
> - 痰中带血或咯血
> - 已被医生诊断为慢性肺部疾病（例如慢阻肺、肺气肿）
>
> **都没有** ／ **有** ／ **不清楚**

**Limits of the draft:**

- An answer of 都没有 does **not** rule out any disease. At most it means the patient reports none of these features.
- The item list, the wording, and the omission of weight loss, night sweats and chest pain are all open for your revision.

---

## 4. Clinician decisions required

Please answer each item (you may also write "not acceptable").

- **A. Adequacy.** Is a single patient-answered question like the draft clinically adequate as the gate for *this narrow source-applicability purpose*?
- **B. Features.** Which exact features must the question include? Should 肺痨 / 肺癌 features (消瘦, 潮热盗汗, 持续日久不愈) or 胸痛 / 发热 be included?
- **C. Stopping rule.** Which answers or features must stop automatic formula suggestion?
- **D. Urgency.** Which features require advising **urgent** medical evaluation, and which only ordinary practitioner review? Please give exact wording if a message should be shown.
- **E. Cough-variant asthma.** How should 咳嗽变异性哮喘 (included) be distinguished from 哮病 (excluded) in patient-facing terms, if at all?
- **F. Sufficiency.** Can an explicit negative patient answer (都没有) establish sufficient eligibility for a *source-bounded AI formula suggestion*? Or must a practitioner confirm before any formula is shown? (All results are already labelled 「需执业医师审核」 and are never purchasable.)
- **G. 不清楚.** What should happen when the patient answers 不清楚?
- **H. Other context.** Is any additional patient context mandatory? Examples: cough duration (acute vs chronic), age, pregnancy, or the 证候 features in §5.3.2.

---

## 5. Proposed decision table (every row: PENDING CLINICAL APPROVAL)

The system applies these rules deterministically, from the patient's explicit answers only; no rule depends on model wording. Each question is asked at most once per case.

| # | 咳嗽为主要或唯一 (P18) | Exclusion question (§3) | Proposed handling | Formula from this relationship | Status |
|---|---|---|---|---|---|
| 1 | not yet asked | — | Ask the P18 question | No | PENDING |
| 2 | 否 | — | Do not ask further; this relationship does not apply | No | PENDING |
| 3 | 不清楚 | — | Do not ask again | No | PENDING |
| 4 | 是 | not yet asked | Ask the §3 question | No | PENDING |
| 5 | 是 | 都没有 | Relationship eligible; result shows source-verification and 需执业医师审核 labels (subject to decision F) | Yes, if F permits | PENDING |
| 6 | 是 | 有 | This relationship does not apply; show the review/referral message per decision D | No | PENDING |
| 7 | 是 | 不清楚 | Per decision G (proposed: not eligible, do not ask again) | No (proposed) | PENDING |
| 8 | any | malformed / unsupported answer | Treated as not established | No | PENDING |
| 9 | any | (free-text mentions of these features) | Not used to establish or exclude eligibility; only explicit answers count | Unchanged by free text | PENDING |

In every row:

- no product and no purchase eligibility is created;
- the formula is shown only as a source-verified suggestion requiring practitioner review;
- the other governed path (风寒束表 → 麻黄汤加味, from a different source) is unaffected.

---

## 6. Approval record

The reviewer fills this in; it is intentionally left blank.

| Field | Entry |
|---|---|
| Reviewer name | |
| Professional qualification / licence | |
| Review date | |
| Source version reviewed (title, DOI, copy checksum) | |
| Approved question wording (verbatim) | |
| Approved answer choices | |
| Approved handling for 都没有 / 有 / 不清楚 | |
| Urgent-evaluation wording (if any) | |
| Decisions A–H (summary) | |
| Required revisions | |
| Decision (one only) | ☐ Approved　☐ Approved with changes (listed above)　☐ Rejected　☐ More information requested |
| Signature | |

---

## 7. Engineering follow-up (only after approval; not started)

1. **Contract.** Add one applicability condition beside the existing `COUGH_PRIMARY`, with a matching patient-context field. Exact name and wording come from the approval.
   - The new governed pattern must carry **both** conditions in its hash-bound content.
   - Any change of wording or conditions later produces a new version that needs re-verification.
2. **Question.** One fixed question, the approved wording verbatim:
   - shown only after 咳嗽为主要或唯一 = 是, and only for this relationship's primary diagnosis;
   - asked once; unknown or malformed answers fail closed.
3. **Unchanged.** Primary-only retrieval (`PRIMARY_ANCHORED_FAIL_CLOSED_V1`), the P18 cough-primary question, and the existing 风寒束表 → 麻黄汤加味 behaviour.
4. **Then P19 resumes:**
   - an exact Core canonical record for 桑菊饮加减 (composition above; no product or mapping);
   - ingest of 1 source, 1 pattern, 1 formula and 1 relationship;
   - four source verifications by the authorized human verifier;
   - Staging deployment, deterministic acceptance, then one owner-operated browser test.
