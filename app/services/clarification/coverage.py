"""X1D-CLARIFY2: which few questions are actually worth a patient's time.

The problem this solves
-----------------------
CLARIFY1 gave the model a way to propose complaint-specific questions, and gave
deterministic code the final say over which of them a patient sees. That part
works. What it did not fix is *which* questions win the four visible slots.

Today the deterministic engine emits up to five fixed questions -- duration,
temperature, appetite, stool, sleep -- whenever their keyword markers are
absent, and core shows deterministic questions first. So a patient reporting
"咳嗽发热3天" is asked about appetite, stool and sleep, and at most one
complaint-specific question survives. The five-item checklist is a small
questionnaire, and it crowds out the questions that would actually move the
differential.

The legacy system had the opposite failure at ten times the scale: it asked all
ten 十问歌 domains every time, in a fixed order, with a hardcoded fallback list
whenever the model failed, and it mapped answers back onto categories by list
position. Broad coverage, zero adaptivity.

What is kept from the legacy system is the *coverage map*: the ten domains are
a genuinely good account of what a TCM intake should eventually know. What is
discarded is the questionnaire -- the idea that every domain must be asked, and
that asking in order is the same thing as knowing something.

So this module turns 十问歌 into a coverage assessment rather than a form:

  * read the accumulated patient facts and mark each domain KNOWN / PARTIAL /
    UNKNOWN, purely from explicit markers -- never inferred, never defaulted;
  * identify the complaint focus, and treat only the domains material to that
    focus as worth a slot (NOT_RELEVANT for the rest, meaning "not material to
    *this* differential", not "never matters");
  * score every candidate question -- deterministic and model-proposed alike --
    and keep the few highest.

What it deliberately does not do
--------------------------------
It has no authority and cannot acquire any. Coverage only *removes* and
*reorders* questions; it never adds one, edits one, or answers one. An unasked
domain stays unknown in missing_information -- the system stops asking, it does
not start assuming.

It never touches ready_for_formula_retrieval, corpus matching, safety screening
or purchasability. In particular the sufficiency verdict below decides one
thing only: whether to ask another question. It is not a clinical readiness
signal and nothing downstream reads it as one.

Ordering relative to the validator
----------------------------------
This runs strictly *after* the CLARIFY1 validator. The validator remains the
authority over what may reach a patient; coverage only narrows what the
validator already approved. A question ranked first here is still a question
that passed every CLARIFY1 check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

# The per-turn visible cap. core owns the authoritative bound
# (clarification_policy.MAX_VISIBLE_QUESTIONS); this mirrors it so the limit
# holds on both sides of the service boundary rather than in one place only.
MAX_VISIBLE_QUESTIONS_PER_TURN = 4

# Adaptive questions allowed while the differential is genuinely open. Matches
# the CLARIFY1 validator's own per-turn cap.
MAX_ADAPTIVE_WHEN_OPEN = 3

# Adaptive questions allowed once coverage is complete but something still
# needs narrowing: one focused question, not another round of three.
MAX_ADAPTIVE_WHEN_NARROWING = 1

CoverageState = Literal["KNOWN", "PARTIAL", "UNKNOWN", "NOT_RELEVANT"]

KNOWN: CoverageState = "KNOWN"
PARTIAL: CoverageState = "PARTIAL"
UNKNOWN: CoverageState = "UNKNOWN"
NOT_RELEVANT: CoverageState = "NOT_RELEVANT"

NEEDS_MORE_INFORMATION = "NEEDS_MORE_INFORMATION"
SUFFICIENT_FOR_REASONING = "SUFFICIENT_FOR_REASONING"


@dataclass(frozen=True)
class Domain:
    """One clinical coverage domain.

    ``strong`` markers mean the patient actually said something about this
    domain. ``weak`` markers mean the text brushes past it -- enough to call it
    PARTIAL, not enough to call it answered.
    """

    key: str
    label: str
    strong: Tuple[str, ...]
    weak: Tuple[str, ...] = ()
    # Field-name fragments, for mapping a model-proposed field such as
    # "aversion_to_cold" onto a domain. The model names its own keys in ASCII.
    aliases: Tuple[str, ...] = ()
    # Marker groups that must ALL be present before the domain counts as
    # answered. 寒热 is the case that needs it: a patient who reports 发热 has
    # said nothing about whether they also feel cold, and it is precisely that
    # relation which separates 风寒 from 风热. Treating 发热 alone as "寒热
    # known" would suppress the single most valuable follow-up question in an
    # exterior pattern -- the LEGACYDIAG2 staging run asked for exactly this
    # relation in its own missing_information.
    pairs: Tuple[Tuple[str, ...], ...] = ()
    # The legacy system's own default question for this domain, adapted. Used
    # only as the floor described in coverage_fallback -- never as a form.
    default_question: str = ""
    default_choices: Tuple[str, ...] = ()


# The legacy ten, in their classical order, plus two the current deterministic
# engine already asks about. 睡眠 and 病程 are not part of 十问歌; they are here
# because CORE_FIELDS asks them, and a coverage map that could not represent an
# existing question would silently exempt that question from ranking.
DOMAINS: Tuple[Domain, ...] = (
    Domain("cold_heat", "寒热",
           ("发热", "怕冷", "恶寒", "畏寒", "怕热", "恶风", "潮热", "低热",
            "高热", "寒热往来", "体温", "发烧"),
           ("热", "寒", "冷"),
           ("cold", "heat", "fever", "chill", "aversion", "temperature",
            "febrile"),
           pairs=(("发热", "怕热", "潮热", "低热", "高热", "体温", "发烧",
                   "无发热", "不发热"),
                  ("怕冷", "恶寒", "畏寒", "恶风", "不怕冷", "无恶寒")),
           default_question="您目前是怕冷、发热，还是两者都有？",
           default_choices=("怕冷为主", "发热为主", "又冷又热", "寒热交替", "都不明显",)),
    Domain("sweat", "汗",
           ("无汗", "出汗", "盗汗", "自汗", "多汗", "少汗", "大汗", "汗出"),
           ("汗",),
           ("sweat", "perspir", "diaphor"),
           default_question="近来出汗情况如何？",
           default_choices=("正常", "容易出汗", "几乎不出汗", "夜间盗汗", "出汗后怕风",)),
    Domain("head_body", "头身",
           ("头痛", "头晕", "头胀", "头重", "身痛", "身重", "腰痛", "关节痛",
            "肢体酸痛", "全身酸痛", "颈痛", "肩痛"),
           ("酸痛", "乏力"),
           ("head", "body_ache", "myalgia", "dizz", "limb"),
           default_question="是否有头痛、头晕或身体酸痛？",
           default_choices=("头痛", "头晕", "身体酸痛", "都有", "都没有",)),
    Domain("excretion", "二便",
           ("大便", "小便", "便秘", "腹泻", "泄泻", "便溏", "夜尿", "尿"),
           ("便",),
           ("stool", "bowel", "urin", "defec", "diarr", "constip"),
           default_question="大便和小便情况如何？",
           default_choices=("都正常", "便秘", "大便稀溏", "小便偏黄", "小便频繁",)),
    Domain("diet", "饮食",
           ("食欲", "胃口", "纳差", "进食", "厌食", "食少", "不思饮食"),
           ("饮食", "吃"),
           ("appetite", "eat", "diet", "food", "intake"),
           default_question="近期食欲和进食情况如何？",
           default_choices=("正常", "食欲减退", "不想吃", "吃一点就饱", "食量增加",)),
    Domain("chest_abdomen", "胸腹",
           ("胸闷", "胸痛", "心悸", "气短", "腹胀", "腹痛", "脘痞", "胁痛",
            "胃胀", "胃痛"),
           ("胸", "腹"),
           ("chest", "abdomen", "epigast", "palpitat", "breath"),
           default_question="是否有胸闷、心悸或腹部胀痛？",
           default_choices=("胸闷", "心悸", "腹胀", "腹痛", "都没有",)),
    Domain("ear_eye", "耳目",
           ("耳鸣", "听力", "耳聋", "耳痛", "目涩", "视物", "眼干", "眼花"),
           ("耳", "目"),
           ("ear", "eye", "hearing", "vision", "tinnitus"),
           default_question="是否有耳鸣、听力下降或眼睛干涩？",
           default_choices=("耳鸣", "听力下降", "眼睛干涩", "都有", "都没有",)),
    Domain("thirst", "渴饮",
           ("口渴", "口干", "不渴", "渴喜", "渴不欲饮", "饮水", "喜冷饮",
            "喜热饮"),
           ("渴",),
           ("thirst", "drink", "fluid"),
           default_question="是否口渴？想喝热的还是凉的？",
           default_choices=("不渴", "渴想喝凉的", "渴想喝热的", "口干但不想喝", "总是渴",)),
    Domain("past_illness", "旧病",
           ("既往", "病史", "慢性", "高血压", "糖尿病", "曾患", "旧疾"),
           (),
           ("past", "history", "chronic", "previous", "prior"),
           default_question="既往是否有慢性病或类似的发作？",
           default_choices=("没有", "有慢性病", "以前发作过类似情况", "不确定",)),
    Domain("cause", "病因",
           ("受凉", "着凉", "劳累", "熬夜", "情绪", "生气", "饮食不当",
            "淋雨", "诱因"),
           (),
           ("cause", "trigger", "onset_reason", "exposure", "precipit"),
           default_question="发病前有没有受凉、劳累、情绪波动或饮食不当？",
           default_choices=("受凉", "劳累", "情绪波动", "饮食不当", "想不起诱因",)),
    Domain("sleep", "睡眠",
           ("失眠", "入睡", "早醒", "多梦", "睡眠", "嗜睡"),
           ("睡",),
           ("sleep", "insomnia", "dream"),
           default_question="近期睡眠情况如何？",
           default_choices=("正常", "入睡困难", "容易醒", "多梦", "睡不够",)),
    Domain("onset_duration", "病程",
           ("天", "周", "月", "年", "小时", "持续", "开始", "反复"),
           ("久", "长期"),
           ("duration", "onset", "since", "how_long"),
           default_question="症状持续多久？从什么时候开始的？"),
)

DOMAINS_BY_KEY: Dict[str, Domain] = {d.key: d for d in DOMAINS}

# The deterministic engine's five fixed fields, mapped onto the coverage map so
# they are ranked by the same rule as everything else instead of being exempt.
CORE_FIELD_DOMAIN: Dict[str, str] = {
    "duration": "onset_duration",
    "temperature": "cold_heat",
    "appetite": "diet",
    "stool": "excretion",
    "sleep": "sleep",
}


@dataclass(frozen=True)
class FocusProfile:
    """A complaint shape, and the domains material to it.

    ``material`` is ordered most- to least-valuable. Position drives the
    ranking weight, so this ordering is the clinical content of the table and
    is meant to be read and argued with rather than buried.
    """

    key: str
    markers: Tuple[str, ...]
    material: Tuple[str, ...]


FOCUS_PROFILES: Tuple[FocusProfile, ...] = (
    FocusProfile(
        "respiratory",
        ("咳", "痰", "喘", "鼻塞", "流涕", "咽痛", "咽痒", "气促", "哮",
         "喷嚏", "声嘶", "感冒"),
        ("cold_heat", "sweat", "thirst", "head_body", "onset_duration",
         "chest_abdomen", "cause"),
    ),
    FocusProfile(
        "digestive",
        ("腹泻", "便秘", "腹胀", "腹痛", "胃痛", "胃胀", "恶心", "呕吐",
         "反酸", "嗳气", "纳差", "泄泻", "便溏"),
        ("excretion", "diet", "chest_abdomen", "onset_duration", "thirst",
         "cold_heat", "cause"),
    ),
    FocusProfile(
        "sleep_fatigue",
        ("失眠", "入睡", "早醒", "多梦", "嗜睡", "乏力", "疲劳", "倦怠",
         "疲乏", "精神不振", "健忘", "心烦"),
        ("sleep", "diet", "chest_abdomen", "onset_duration", "cold_heat",
         "sweat", "cause", "past_illness"),
    ),
    FocusProfile(
        "pain",
        ("头痛", "腰痛", "关节痛", "肩痛", "颈痛", "酸痛", "疼痛", "刺痛",
         "胀痛", "隐痛"),
        ("head_body", "onset_duration", "cold_heat", "cause", "sleep",
         "excretion", "past_illness"),
    ),
)

# No profile matched. Every domain stays material, so an unrecognised complaint
# behaves exactly as it did before this module existed. Failing open toward the
# previous behaviour is the only safe direction here: the cost of a profile
# miss is then an extra question, never a suppressed one.
GENERAL_FOCUS = "general"


@dataclass
class CoverageAssessment:
    """What the accumulated patient facts do and do not establish."""

    focus: str
    states: Dict[str, CoverageState] = field(default_factory=dict)
    material: Tuple[str, ...] = ()

    def state(self, domain_key: Optional[str]) -> Optional[CoverageState]:
        if domain_key is None:
            return None
        return self.states.get(domain_key)

    def unknown_material(self) -> List[str]:
        return [k for k in self.material if self.states.get(k) == UNKNOWN]

    def as_dict(self) -> Dict[str, Any]:
        """Compact operational view. Carries no patient text."""
        return {
            "focus": self.focus,
            "material_domains": list(self.material),
            "states": {k: self.states[k] for k in sorted(self.states)},
            "unknown_material": self.unknown_material(),
        }


def _hits(text: str, markers: Iterable[str]) -> bool:
    return any(m in text for m in markers)


def identify_focus(text: str) -> str:
    """Which complaint shape this is, by marker count. Ties keep table order."""
    best, best_score = GENERAL_FOCUS, 0
    for profile in FOCUS_PROFILES:
        score = sum(1 for marker in profile.markers if marker in (text or ""))
        if score > best_score:
            best, best_score = profile.key, score
    return best


def _material_for(focus: str) -> Tuple[str, ...]:
    for profile in FOCUS_PROFILES:
        if profile.key == focus:
            return profile.material
    return tuple(d.key for d in DOMAINS)


def assess_coverage(text: str) -> CoverageAssessment:
    """Mark every domain from explicit patient facts only.

    Nothing is inferred and nothing is defaulted: a domain the patient did not
    mention is UNKNOWN, which is a statement about the record rather than a
    claim about the patient.
    """
    text = text or ""
    focus = identify_focus(text)
    material = _material_for(focus)

    states: Dict[str, CoverageState] = {}
    for domain in DOMAINS:
        if domain.pairs:
            hit = sum(1 for group in domain.pairs if _hits(text, group))
            if hit == len(domain.pairs):
                states[domain.key] = KNOWN
            elif hit:
                # One side of the relation only. The domain has been touched,
                # not answered, so it stays askable at PARTIAL weight.
                states[domain.key] = PARTIAL
            elif domain.key not in material:
                states[domain.key] = NOT_RELEVANT
            else:
                states[domain.key] = UNKNOWN
        elif _hits(text, domain.strong):
            states[domain.key] = KNOWN
        elif _hits(text, domain.weak):
            states[domain.key] = PARTIAL
        elif domain.key not in material:
            # Not material to this differential. Still unknown in the record --
            # this says only that it is not worth a slot right now.
            states[domain.key] = NOT_RELEVANT
        else:
            states[domain.key] = UNKNOWN

    return CoverageAssessment(focus=focus, states=states, material=material)


# ======================================================================
# Mapping a question onto a domain
# ======================================================================

def domain_for_field(field_name: Any) -> Optional[str]:
    """Map a field identifier onto a domain, or None."""
    name = str(field_name or "").strip().lower()
    if not name:
        return None
    if name in CORE_FIELD_DOMAIN:
        return CORE_FIELD_DOMAIN[name]
    for domain in DOMAINS:
        if any(alias in name for alias in domain.aliases):
            return domain.key
    return None


def domain_for_text(text: Any) -> Optional[str]:
    """Map free text onto a domain by marker, strong matches first."""
    body = str(text or "")
    if not body:
        return None
    for domain in DOMAINS:
        if _hits(body, domain.strong):
            return domain.key
    for domain in DOMAINS:
        if _hits(body, domain.weak):
            return domain.key
    return None


def resolve_domain(field_name: Any,
                   question: Any) -> Tuple[Optional[str], bool]:
    """Map a question onto a domain, and say how confident that mapping is.

    Returns (domain, certain). ``certain`` is True only when the domain came
    from the model's own field identifier, which is a deliberate claim that the
    question IS that domain. A domain matched in the question *text* means only
    that the question mentions it -- "腰痛是酸沉还是刺痛？" mentions 头身 but
    asks about the quality of a symptom already reported, not whether it exists.

    The distinction matters because exclusion is destructive: an uncertain
    mapping may rank a question, but it may never silence one.
    """
    domain = domain_for_field(field_name)
    if domain is not None:
        return domain, True
    return domain_for_text(question), False


def domain_for_question(field_name: Any, question: Any) -> Optional[str]:
    """The field is the model's own key, so it is the more reliable signal."""
    return resolve_domain(field_name, question)[0]


# ======================================================================
# Ranking
# ======================================================================
#
# Integer weights, summed, sorted descending, ties broken by input order. No
# randomness and no second model call: the same inputs always produce the same
# question list, which is what makes the selection auditable.

# A deterministic question whose missing_information priority is HIGH is never
# ranked away. Tying that exemption to the existing priority signal, rather
# than to a new list, means pruning can never remove a question the reasoning
# engine already treats as blocking.
PRIORITY_FLOOR_SCORE = 1000

FOCUS_WEIGHT_TOP = 40
FOCUS_WEIGHT_STEP = 4
FOCUS_WEIGHT_FLOOR = 8
FOCUS_WEIGHT_GENERAL = 20
# A question that maps to no domain is complaint-specific by construction --
# which is exactly the value CLARIFY1 added. Weighted below the leading
# material domains and above their tail.
FOCUS_WEIGHT_UNMAPPED = 18
# Unmapped AND sharing no marker with an identified complaint: neither a
# material domain nor about what the patient came in for.
FOCUS_WEIGHT_UNMAPPED_OFF_TOPIC = 4

# An adaptive question must be worth a slot, not merely fit in one. Without
# this floor a leftover proposal is asked whenever the budget has room, which
# is the questionnaire habit in miniature.
MIN_ADAPTIVE_SCORE = 40

# When a turn has declared information insufficient it must not then ask the
# patient nothing. This is a floor on that case only -- never a target, and
# never used to fill the visible budget.
MIN_QUESTIONS_WHEN_INSUFFICIENT = 2

COVERAGE_WEIGHT = {UNKNOWN: 25, PARTIAL: 12}

# Advisory signals drawn from the LEGACYDIAG2 envelope. They can raise a
# question the validator already approved; they can never create one.
WEIGHT_ENVELOPE_GAP = 20
WEIGHT_CONTRADICTION = 12
WEIGHT_DISCRIMINATING = 8
MAX_DIFFERENTIAL_WEIGHT = 24

WEIGHT_COMPLAINT_LINK = 10

MODEL_PRIORITY_WEIGHT = {"high": 6, "medium": 3, "low": 0}


@dataclass(frozen=True)
class Candidate:
    """One question competing for a visible slot."""

    field: Optional[str]
    question: str
    domain: Optional[str]
    kind: str                      # "deterministic" | "adaptive"
    model_priority: str = "medium"
    high_priority_missing: bool = False
    # Whether ``domain`` came from the field identifier rather than the
    # question text. Only a certain mapping may exclude a question.
    domain_certain: bool = False
    payload: Any = None


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    score: int
    components: Dict[str, int]


@dataclass(frozen=True)
class DifferentialSignals:
    """Advisory input extracted from the reasoning envelope.

    Every entry is a domain key resolved through the fixed domain table. Model
    text that maps to no domain contributes nothing, which is what keeps
    patient-derived prose from steering question selection.
    """

    gap_domains: frozenset = frozenset()
    contradiction_domains: frozenset = frozenset()
    discriminating_domains: frozenset = frozenset()

    @property
    def has_contradiction(self) -> bool:
        return bool(self.contradiction_domains)


def extract_differential_signals(envelope: Any) -> DifferentialSignals:
    """Read the envelope for hints about what is still in question.

    Defensive throughout: the envelope is model output and may be absent,
    empty, or shaped unexpectedly. Reading nothing yields no signals, which
    degrades ranking to coverage and complaint focus alone.
    """
    if envelope is None:
        return DifferentialSignals()

    gaps: set = set()
    contradictions: set = set()
    discriminating: set = set()

    for item in getattr(envelope, "missing_information", None) or []:
        domain = domain_for_text(item)
        if domain:
            gaps.add(domain)

    hypotheses = list(getattr(envelope, "pattern_hypotheses", None) or [])
    support_counts: Dict[str, int] = {}
    for hypothesis in hypotheses:
        for finding in getattr(hypothesis, "contradicting_findings", None) or []:
            domain = domain_for_text(finding)
            if domain:
                contradictions.add(domain)
        seen_here: set = set()
        for finding in getattr(hypothesis, "supporting_findings", None) or []:
            domain = domain_for_text(finding)
            if domain:
                seen_here.add(domain)
        for domain in seen_here:
            support_counts[domain] = support_counts.get(domain, 0) + 1

    # A domain cited by exactly one of the competing hypotheses is what
    # separates them. With fewer than two hypotheses there is nothing to
    # separate, so no domain is discriminating.
    if len(hypotheses) >= 2:
        discriminating = {d for d, n in support_counts.items() if n == 1}

    return DifferentialSignals(
        gap_domains=frozenset(gaps),
        contradiction_domains=frozenset(contradictions),
        discriminating_domains=frozenset(discriminating),
    )


def _focus_weight(coverage: CoverageAssessment, domain: Optional[str],
                  on_topic: bool) -> int:
    if domain is None:
        if coverage.focus == GENERAL_FOCUS or on_topic:
            return FOCUS_WEIGHT_UNMAPPED
        return FOCUS_WEIGHT_UNMAPPED_OFF_TOPIC
    if coverage.focus == GENERAL_FOCUS:
        return FOCUS_WEIGHT_GENERAL
    if domain not in coverage.material:
        return 0
    index = coverage.material.index(domain)
    return max(FOCUS_WEIGHT_FLOOR, FOCUS_WEIGHT_TOP - FOCUS_WEIGHT_STEP * index)


def _shares_complaint_focus(candidate: "Candidate",
                            coverage: CoverageAssessment) -> bool:
    """Whether the question is about the presenting complaint itself."""
    if coverage.focus == GENERAL_FOCUS:
        return False
    profile = next((p for p in FOCUS_PROFILES if p.key == coverage.focus), None)
    if profile is None:
        return False
    body = "%s %s" % (candidate.question or "", candidate.field or "")
    return _hits(body, profile.markers)


def score_candidate(
    candidate: Candidate,
    coverage: CoverageAssessment,
    signals: DifferentialSignals,
) -> Optional[ScoredCandidate]:
    """Score one question, or return None if it should not be asked.

    Exclusion happens for exactly two reasons: the patient already told us
    (KNOWN), or the domain is not material to this differential. Both are
    statements about the record, never assumptions about the patient.
    """
    if candidate.high_priority_missing:
        return ScoredCandidate(candidate, PRIORITY_FLOOR_SCORE,
                               {"high_priority_missing": PRIORITY_FLOOR_SCORE})

    state = coverage.state(candidate.domain)
    if state in (KNOWN, NOT_RELEVANT) and candidate.domain_certain:
        return None

    on_topic = _shares_complaint_focus(candidate, coverage)

    components: Dict[str, int] = {}
    components["focus"] = _focus_weight(coverage, candidate.domain, on_topic)
    # An unmapped question is complaint-specific; the model asked it precisely
    # because the answer is not on record, so it is scored as uncovered.
    components["coverage"] = (COVERAGE_WEIGHT.get(state, 0)
                              if candidate.domain is not None
                              else COVERAGE_WEIGHT[UNKNOWN])

    differential = 0
    if candidate.domain in signals.gap_domains:
        differential += WEIGHT_ENVELOPE_GAP
    if candidate.domain in signals.contradiction_domains:
        differential += WEIGHT_CONTRADICTION
    if candidate.domain in signals.discriminating_domains:
        differential += WEIGHT_DISCRIMINATING
    components["differential"] = min(differential, MAX_DIFFERENTIAL_WEIGHT)

    components["complaint_link"] = WEIGHT_COMPLAINT_LINK if on_topic else 0

    components["model_priority"] = MODEL_PRIORITY_WEIGHT.get(
        str(candidate.model_priority or "").lower(), 0)

    return ScoredCandidate(candidate, sum(components.values()), components)


# ======================================================================
# Selection and the stop condition
# ======================================================================

@dataclass
class SelectionResult:
    sufficiency: str
    coverage: CoverageAssessment
    deterministic: List[Any] = field(default_factory=list)
    adaptive: List[Any] = field(default_factory=list)
    suppressed_count: int = 0
    adaptive_budget: int = 0
    scores: List[ScoredCandidate] = field(default_factory=list)
    # X1D-CLARIFY3.1: questions dropped for asking about a domain another
    # selected question already occupies. Counted, never silent.
    same_domain_suppressed: int = 0
    # Coverage-table questions, carried separately from the model's own. They
    # are typed patient questions, so the assembler routes them through the
    # clarification channel rather than the plain-text followup list.
    fallback: List[Any] = field(default_factory=list)

    @property
    def fallback_used(self) -> int:
        return len(self.fallback)


def coverage_fallback(coverage: CoverageAssessment,
                      already_covered: Sequence[Optional[str]]) -> List[Candidate]:
    """Legacy 十问歌 default questions for material domains still unspoken.

    Ordered by the profile's own materiality ordering, so the first question
    offered is the one the complaint most needs. Domains a selected question
    already covers are skipped, and domains the patient has spoken to are not
    candidates at all.
    """
    covered = {d for d in already_covered if d}
    out: List[Candidate] = []
    for key in coverage.material:
        if key in covered:
            continue
        if coverage.states.get(key) not in (UNKNOWN, PARTIAL):
            continue
        domain = DOMAINS_BY_KEY.get(key)
        if domain is None or not domain.default_question:
            continue
        out.append(Candidate(
            field=key,
            question=domain.default_question,
            domain=key,
            kind="coverage",
            domain_certain=True,
            payload={
                "field": key,
                "question": domain.default_question,
                "answer_type": ("single_choice" if domain.default_choices
                                else "short_text"),
                "choices": list(domain.default_choices),
                "priority": "medium",
                "source": "xerbs-ai-v2-coverage",
            },
        ))
    return out


def assess_sufficiency(
    coverage: CoverageAssessment,
    signals: DifferentialSignals,
    has_high_priority_missing: bool,
) -> Tuple[str, int]:
    """Decide whether to keep asking, and how much room the next turn gets.

    Deliberately not "are all ten domains KNOWN". The question is whether what
    remains unknown is material to the differential actually in play.

    This verdict governs question emission and nothing else. It is not a
    clinical readiness signal: ready_for_formula_retrieval, corpus matching,
    safety screening and purchasability are computed elsewhere and are not
    reachable from here.
    """
    if has_high_priority_missing or coverage.unknown_material():
        return NEEDS_MORE_INFORMATION, MAX_ADAPTIVE_WHEN_OPEN

    if signals.has_contradiction or signals.gap_domains:
        # Coverage is complete but something still needs narrowing: one
        # focused question, not another round of three.
        return NEEDS_MORE_INFORMATION, MAX_ADAPTIVE_WHEN_NARROWING

    return SUFFICIENT_FOR_REASONING, 0


def select_questions(
    *,
    deterministic: Sequence[Candidate],
    adaptive: Sequence[Candidate],
    coverage: CoverageAssessment,
    signals: DifferentialSignals,
    limit: int = MAX_VISIBLE_QUESTIONS_PER_TURN,
) -> SelectionResult:
    """Rank both kinds of question together and keep the highest few.

    Ranking the two lists jointly is the point. Scoring them separately would
    preserve exactly the behaviour this phase exists to fix, where a generic
    checklist question outranks a complaint-specific one purely by category.
    """
    has_high = any(c.high_priority_missing for c in deterministic)
    sufficiency, adaptive_budget = assess_sufficiency(
        coverage, signals, has_high)

    scored: List[ScoredCandidate] = []
    considered = list(deterministic) + list(adaptive)
    for candidate in considered:
        result = score_candidate(candidate, coverage, signals)
        if result is not None:
            scored.append(result)

    adaptive_scored = [s for s in scored
                       if s.candidate.kind == "adaptive"
                       and s.score >= MIN_ADAPTIVE_SCORE]
    kept_adaptive = sorted(
        enumerate(adaptive_scored),
        key=lambda pair: (-pair[1].score, pair[0]))[:max(0, adaptive_budget)]
    allowed_adaptive = {id(s.candidate) for _, s in kept_adaptive}

    eligible = [s for s in scored
                if s.candidate.kind != "adaptive"
                or id(s.candidate) in allowed_adaptive]

    ordered = [s for _, s in sorted(enumerate(eligible),
                                    key=lambda pair: (-pair[1].score, pair[0]))]

    # X1D-CLARIFY3.1: one clinical domain, one visible slot per turn.
    #
    # Walked in rank order, so the question that occupies a domain is the
    # highest-scoring one that wants it and the duplicate is the one dropped.
    # Only a certain (field-derived) mapping can occupy a domain or be
    # suppressed by one -- a text match says the question mentions the domain,
    # not that it is about it, and that is too weak to silence a question on.
    #
    # Dropping a duplicate frees its slot for the next candidate rather than
    # shortening the turn, which is why this filters during the walk instead
    # of after the cut.
    selected: List[ScoredCandidate] = []
    occupied: set = set()
    same_domain_suppressed = 0
    for scored_candidate in ordered:
        if len(selected) >= max(0, limit):
            break
        domain = scored_candidate.candidate.domain
        certain = (domain is not None
                   and scored_candidate.candidate.domain_certain)
        if certain and domain in occupied:
            same_domain_suppressed += 1
            continue
        if certain:
            occupied.add(domain)
        selected.append(scored_candidate)

    # The floor. Only when this turn has said the information is insufficient
    # and the ranking still produced almost nothing -- which is what happens
    # when the model proposes no questions and the checklist was immaterial.
    fallback: List[Candidate] = []
    if sufficiency == NEEDS_MORE_INFORMATION:
        shortfall = min(MIN_QUESTIONS_WHEN_INSUFFICIENT, limit) - len(selected)
        if shortfall > 0:
            fallback = coverage_fallback(
                coverage, [s.candidate.domain for s in selected])[:shortfall]

    return SelectionResult(
        sufficiency=sufficiency,
        coverage=coverage,
        deterministic=[s.candidate.payload for s in selected
                       if s.candidate.kind == "deterministic"],
        adaptive=[s.candidate.payload for s in selected
                  if s.candidate.kind == "adaptive"],
        suppressed_count=len(considered) - len(selected),
        adaptive_budget=adaptive_budget,
        scores=selected,
        same_domain_suppressed=same_domain_suppressed,
        fallback=[c.payload for c in fallback],
    )


def describe_policy() -> Dict[str, Any]:
    """Operational description of the model, for logs and tests."""
    return {
        "domains": [{"key": d.key, "label": d.label} for d in DOMAINS],
        "focus_profiles": [p.key for p in FOCUS_PROFILES] + [GENERAL_FOCUS],
        "states": [KNOWN, PARTIAL, UNKNOWN, NOT_RELEVANT],
        "max_visible_questions_per_turn": MAX_VISIBLE_QUESTIONS_PER_TURN,
        "max_adaptive_when_open": MAX_ADAPTIVE_WHEN_OPEN,
        "max_adaptive_when_narrowing": MAX_ADAPTIVE_WHEN_NARROWING,
        "min_adaptive_score": MIN_ADAPTIVE_SCORE,
        "min_questions_when_insufficient": MIN_QUESTIONS_WHEN_INSUFFICIENT,
    }
