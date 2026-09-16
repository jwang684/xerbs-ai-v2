"""X1D-LEGACYDIAG4.4B: carry a working differential, and refuse to let it harden.

What this adds
--------------
InterviewReasoning has produced working hypotheses since LEGACYDIAG3.2, and
they have been thrown away at the end of every turn. So each turn re-derives
its reading from accumulated text and cannot ask the one question that would
separate the two readings it just formed. This carries that state forward.

The whole risk lives in one sentence
------------------------------------
Feeding a model its own previous conclusion is how you build a machine that
agrees with itself. Three turns of "风热犯卫, still 风热犯卫, definitely 风热犯卫"
would look exactly like conviction and contain no information at all.

So the state carried here is not a conclusion. It is a set of candidates with
the patient evidence for and against each, and two deterministic rules stop it
hardening:

  * a hypothesis may only cite patient evidence that actually resolves to
    something the patient supplied. A model that writes "yellow tongue coating"
    for a patient who never mentioned their tongue loses that item -- it does
    not get to become next turn's premise;

  * a standing may not improve unless the model cites at least one piece of
    evidence it was not already citing. Repetition is not evidence, and
    without this rule repetition is indistinguishable from evidence.

Both are checked after the model answers, not asked for in the prompt. A prompt
instruction is a request; this is a boundary.

What it cannot do
-----------------
Anything clinical. The validated state reaches exactly one consumer --
extract_differential_signals, which already ranks clarification questions by
gap, contradiction and discrimination. It never touches the corpus, safety,
eligibility or purchase, and it is never shown to a patient: during the
interview the patient sees questions, not a running theory about themselves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.schemas.reasoning import (
    EvidenceRef,
    ResolvableEvidence,
    WorkingDifferentialState,
    WorkingHypothesis,
)
from app.services.clarification.coverage import (
    DifferentialSignals,
    normalize_clinical_domain,
    signals_from_domains,
)

# Ordered weakest to strongest. Position is the whole point: the anti-escalation
# rule is an index comparison, so "may not strengthen" needs no special cases.
STANDING_ORDER = ("RULED_OUT_FOR_NOW", "WEAKENED", "PLAUSIBLE",
                  "PRIMARY_WORKING")

# Origins a model may cite. Everything else -- its own prose, a summary, a
# previous hypothesis, an image -- is not patient evidence and never becomes it.
ALLOWED_ORIGINS = ("COMPLAINT", "OBSERVATION", "ANSWER")

MAX_HYPOTHESES = 5
MAX_SEPARATES = 4
MAX_RATIONALE_CHARS = 200
MAX_EVIDENCE_PER_SIDE = 8
MAX_DISCRIMINATORS = 4
MAX_GAPS = 6


def _rank(standing: Optional[str]) -> int:
    try:
        return STANDING_ORDER.index(str(standing))
    except ValueError:
        # An unknown standing is treated as the weakest thing it could be, so a
        # typo can never buy a promotion.
        return 0


def ref_key(ref: EvidenceRef) -> str:
    """Identity of one citation, for set comparison across turns."""
    if ref.origin == "COMPLAINT":
        return "COMPLAINT"
    if ref.origin == "OBSERVATION":
        return "OBSERVATION:%s" % (ref.field or "")
    return "ANSWER:%s:%s" % (ref.turn_id if ref.turn_id is not None else "",
                             ref.question_field or "")


def _answer_identities(entry: Any) -> set:
    """Every name one recorded answer can legitimately be cited by.

    X1D-LEGACYDIAG4.4B-R1. The model cites the domain it was reasoning about
    ("cold_heat"); the record holds the field the question was filed under
    ("temperature"). Before R1 those two never met and the citation was
    dropped as unresolvable -- a live answer treated as a fabrication. Both the
    raw field and its canonical domain are accepted, and nothing else.
    """
    if not isinstance(entry, dict):
        return set()
    names = set()
    for key in ("question_field", "domain"):
        value = str(entry.get(key) or "").strip()
        if value:
            names.add(value.lower())
            canonical = normalize_clinical_domain(value)
            if canonical:
                names.add(canonical)
    return names


def resolve_ref(ref: Any, evidence: ResolvableEvidence) -> Optional[EvidenceRef]:
    """Return the ref if it points at real patient evidence, else None.

    Deterministic and total. The model supplies a claim about where a finding
    came from; this checks the claim against what the patient actually gave,
    and the only outcomes are "keep it" and "drop it". There is no branch that
    turns unresolved model prose into evidence.
    """
    try:
        candidate = ref if isinstance(ref, EvidenceRef) else EvidenceRef(**ref)
    except Exception:  # noqa: BLE001 - a malformed citation is simply not one
        return None

    if candidate.origin not in ALLOWED_ORIGINS:
        return None

    if candidate.origin == "COMPLAINT":
        return candidate if evidence.has_complaint else None

    if candidate.origin == "OBSERVATION":
        field = (candidate.field or "").strip()
        return candidate if field and field in evidence.observations else None

    # ANSWER: the patient must actually have answered this, in this case.
    #
    # R1 changed what is being checked here. It used to be "was this question
    # asked"; it is now "did the patient answer something filed under this
    # name", matched on the field or its canonical domain. An asked but
    # unanswered question is not evidence, and a question is not evidence
    # either -- only the answer is.
    field = (candidate.question_field or "").strip().lower()
    if not field:
        return None
    canonical = normalize_clinical_domain(field)
    wanted = {field} | ({canonical} if canonical else set())
    for answered in evidence.answers:
        if not wanted & _answer_identities(answered):
            continue
        if candidate.turn_id is not None and \
                answered.get("turn_id") != candidate.turn_id:
            continue
        return candidate
    return None


def _resolve_all(refs: Any, evidence: ResolvableEvidence) -> List[EvidenceRef]:
    out: List[EvidenceRef] = []
    seen: set = set()
    for ref in (refs or [])[:MAX_EVIDENCE_PER_SIDE * 2]:
        resolved = resolve_ref(ref, evidence)
        if resolved is None:
            continue
        key = ref_key(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
        if len(out) >= MAX_EVIDENCE_PER_SIDE:
            break
    return out


def _prior_support_keys(prior: Optional[WorkingDifferentialState],
                        name: str) -> set:
    if prior is None:
        return set()
    for hypothesis in prior.hypotheses:
        if hypothesis.pattern_name == name:
            return {ref_key(r) for r in hypothesis.supporting_evidence}
    return set()


def _prior_standing(prior: Optional[WorkingDifferentialState],
                    name: str) -> Optional[str]:
    if prior is None:
        return None
    for hypothesis in prior.hypotheses:
        if hypothesis.pattern_name == name:
            return hypothesis.standing
    return None


def _names(raw: Any, live: set, limit: int = MAX_SEPARATES) -> List[str]:
    """Hypothesis names from model output, kept only if they are live here.

    X1D-LEGACYDIAG4.5. A model that writes "sweat distinguishes X from Y" when
    Y is not one of its own current readings has described a competition that
    is not happening, and must not gain differential priority for it. Order is
    preserved and duplicates collapse, so a list naming one hypothesis twice
    cannot pass as naming two.
    """
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[str] = []
    for item in raw[:limit * 2]:
        name = str(item or "").strip()
        if name and name in live and name not in out:
            out.append(name)
        if len(out) >= limit:
            break
    return out


def _validate_discriminator(entry: Any, live: set) -> Optional[Dict[str, Any]]:
    """One unresolved discriminator, or None.

    Two questions are asked of it, and they are different questions:

      * does it name a real competition? At least two distinct live
        hypotheses, or there is nothing here to separate;
      * does it demonstrate that the answer would MOVE that competition?
        A finding whose presence and absence support the same reading
        discriminates nothing, however plausible the prose sounds.

    Only the second earns ``discriminating``, and only a discriminating entry
    can claim one of the scarce question slots. An entry that names a real
    competition without the demonstration is kept -- it is honest observable
    reasoning -- but it does not get priority for it.
    """
    if not isinstance(entry, dict):
        return None
    domain = normalize_clinical_domain(entry.get("domain"))
    if not domain:
        return None

    separates = _names(entry.get("separates"), live)
    if len(separates) < 2:
        return None

    present = _names(entry.get("if_present_supports"), live)
    absent = _names(entry.get("if_absent_supports"), live)
    # The counterfactual has to point somewhere, and somewhere DIFFERENT.
    demonstrated = bool(present and absent and set(present) != set(absent))

    # X1D-LEGACYDIAG4.6: the other domains that would settle the same pair.
    # Named by the model, canonicalized here, and never extended by us -- the
    # deterministic layer orders what it is given and invents nothing.
    alternatives = []
    for extra in (entry.get("also_resolved_by") or [])[:MAX_DISCRIMINATORS * 2]:
        candidate = normalize_clinical_domain(extra)
        if candidate and candidate != domain and candidate not in alternatives:
            alternatives.append(candidate)

    return {
        "domain": domain,
        "separates": separates,
        "also_resolved_by": alternatives,
        "if_present_supports": present,
        "if_absent_supports": absent,
        "discriminating": demonstrated,
        # Model reasoning, kept for readability of the record. It is never
        # patient evidence, never cited, and never leaves the server.
        "rationale": str(entry.get("rationale")
                         or entry.get("question_needed")
                         or "").strip()[:MAX_RATIONALE_CHARS],
    }


def validate_state(
    raw: Any,
    evidence: ResolvableEvidence,
    prior: Optional[WorkingDifferentialState] = None,
) -> Tuple[Optional[WorkingDifferentialState], List[str]]:
    """Turn raw model state into state that may be carried. Never raises.

    Returns (state, notes). Notes are operational breadcrumbs -- which rules
    fired -- not clinical content.

    Every rule here fails toward LESS: an unresolvable citation is dropped
    rather than the hypothesis, a hypothesis with no name is dropped rather
    than the state, and a standing that cannot justify itself is lowered rather
    than the turn being failed. Losing a hypothesis costs a question; keeping a
    fabricated one costs the premise of every turn that follows.
    """
    notes: List[str] = []
    if not isinstance(raw, dict) or not raw:
        return None, notes

    pending: List[Dict[str, Any]] = []
    for item in (raw.get("hypotheses") or [])[:MAX_HYPOTHESES * 2]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("pattern_name") or "").strip()
        if not name:
            continue

        supporting = _resolve_all(item.get("supporting_evidence"), evidence)
        contradicting = _resolve_all(item.get("contradicting_evidence"),
                                     evidence)
        dropped = (len(item.get("supporting_evidence") or [])
                   + len(item.get("contradicting_evidence") or [])
                   - len(supporting) - len(contradicting))
        if dropped > 0:
            notes.append("EVIDENCE_REF_UNRESOLVED")

        standing = str(item.get("standing") or "PLAUSIBLE")
        if standing not in STANDING_ORDER:
            standing = "PLAUSIBLE"

        # Anti-escalation. A standing may rise only if this turn cites
        # something the previous turn did not. Repetition is not evidence.
        previous = _prior_standing(prior, name)
        if previous is not None and _rank(standing) > _rank(previous):
            fresh = {ref_key(r) for r in supporting} - _prior_support_keys(
                prior, name)
            if not fresh:
                standing = previous
                notes.append("STANDING_ESCALATION_WITHOUT_NEW_EVIDENCE")

        # Discriminators are validated in a second pass: whether one names a
        # real competition can only be judged once every hypothesis in this
        # state is known.
        raw_discriminators = [
            e for e in (item.get("unresolved_discriminators") or [])
            [:MAX_DISCRIMINATORS] if isinstance(e, dict)]

        pending.append({
            "pattern_name": name, "standing": standing,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "raw_discriminators": raw_discriminators,
        })
        if len(pending) >= MAX_HYPOTHESES:
            break
    # Second pass. Which competitions are real can only be judged once every
    # reading in this state is known, so the hypotheses are constructed here,
    # complete, rather than built empty and filled in afterwards.
    live_names = {p["pattern_name"] for p in pending
                  if p["standing"] in LIVE_STANDINGS}
    hypotheses: List[WorkingHypothesis] = []
    for item_data in pending:
        kept = []
        for entry in item_data["raw_discriminators"]:
            validated = _validate_discriminator(entry, live_names)
            if validated is None:
                notes.append("DISCRIMINATOR_NO_LIVE_COMPETITION")
                continue
            if not validated["discriminating"]:
                notes.append("DISCRIMINATOR_WITHOUT_COUNTERFACTUAL")
            kept.append(validated)
        hypotheses.append(WorkingHypothesis(
            pattern_name=item_data["pattern_name"],
            standing=item_data["standing"],
            supporting_evidence=item_data["supporting_evidence"],
            contradicting_evidence=item_data["contradicting_evidence"],
            unresolved_discriminators=kept,
        ))

    gaps = []
    named = {h.pattern_name for h in hypotheses}
    for entry in (raw.get("evidence_gaps") or [])[:MAX_GAPS]:
        if not isinstance(entry, dict):
            continue
        domain = str(entry.get("domain") or "").strip()
        separates = [str(n).strip() for n in (entry.get("separates") or [])
                     if str(n).strip()]
        # A gap that separates patterns nobody is holding is not a gap.
        separates = [n for n in separates if n in named][:2]
        if domain and separates:
            gaps.append({"domain": domain, "separates": separates})

    if not hypotheses:
        return None, notes

    return WorkingDifferentialState(
        turn_id=raw.get("turn_id") if isinstance(raw.get("turn_id"), int) else None,
        hypotheses=hypotheses, evidence_gaps=gaps), notes


# ======================================================================
# Reaching question selection
# ======================================================================
#
# The state has to arrive as domains, because domains are what the CLARIFY2
# ranker weighs. It is worth being precise about where those domains come
# from: an EvidenceRef carries the model's own ASCII field identifier, which
# is a deliberate claim that the finding IS that domain, so domain_for_field
# resolves it exactly. Passing "aversion_to_cold" through the free-text
# matcher instead would look for Chinese markers, find none, and silently drop
# a signal we already had with certainty.


def domain_for_ref(ref: EvidenceRef) -> Optional[str]:
    """The domain a citation belongs to, or None for the complaint itself.

    A COMPLAINT reference deliberately has no domain: it points at the whole
    presenting text, so counting it toward any one domain would claim the
    patient addressed something they may never have mentioned.
    """
    if ref.origin == "COMPLAINT":
        return None
    return normalize_clinical_domain(ref.field or ref.question_field)


def answered_domains(evidence: ResolvableEvidence) -> set:
    """Canonical domains the patient has actually answered in this case.

    Used to state, as a testable fact, that a domain already answered under
    one name is not outstanding merely because the model spells it another.
    """
    found = set()
    for entry in evidence.answers or []:
        if not isinstance(entry, dict):
            continue
        for name in _answer_identities(entry):
            canonical = normalize_clinical_domain(name)
            if canonical:
                found.add(canonical)
    for field in evidence.observations or []:
        canonical = normalize_clinical_domain(field)
        if canonical:
            found.add(canonical)
    return found


def signals_from_state(
    state: Optional[WorkingDifferentialState],
) -> DifferentialSignals:
    """Render a validated state as CLARIFY2 ranking signals.

    No new weights and no second ranker: this resolves domains and hands them
    to the same signals_from_domains the envelope path uses, so a gap found
    here counts for exactly what a gap found there counts for.
    """
    if state is None or not state.hypotheses:
        return DifferentialSignals()

    gaps: set = set()
    for gap in state.evidence_gaps:
        if isinstance(gap, dict):
            gaps.add(normalize_clinical_domain(gap.get("domain")))
    for hypothesis in state.hypotheses:
        for entry in hypothesis.unresolved_discriminators:
            if isinstance(entry, dict):
                gaps.add(normalize_clinical_domain(entry.get("domain")))

    pairs = [({domain_for_ref(r) for r in h.supporting_evidence},
              {domain_for_ref(r) for r in h.contradicting_evidence})
             for h in state.hypotheses]

    return signals_from_domains(gaps, pairs)


# X1D-LEGACYDIAG4.4B-R2: which standings still compete for a question.
#
# PRIMARY_WORKING and PLAUSIBLE only. WEAKENED is deliberately excluded: it is
# by definition the reading the evidence currently argues against, and letting
# it demand one of three scarce slots is how an interview stays open on a
# fading alternative. RULED_OUT_FOR_NOW is excluded for the stronger version of
# the same reason -- it can come back, but only by the model raising its
# standing on new evidence, which the 4.4B anti-escalation rule already
# governs. Neither is silenced: both keep their place in the carried state and
# both are re-evaluated every turn.
LIVE_STANDINGS = ("PRIMARY_WORKING", "PLAUSIBLE")


def live_hypotheses(
    state: Optional[WorkingDifferentialState],
) -> List[WorkingHypothesis]:
    """The hypotheses still genuinely in contention."""
    if state is None:
        return []
    return [h for h in state.hypotheses if h.standing in LIVE_STANDINGS]


def differential_required_domains(
    state: Optional[WorkingDifferentialState],
) -> set:
    """Canonical domains the live hypotheses still need in order to separate.

    Derived from the VALIDATED state only, so every citation behind it has
    already survived resolution and every standing has already survived the
    anti-escalation check. A domain enters only if the model explicitly named
    it -- as an unresolved discriminator of a live hypothesis, or as an
    evidence gap between two of them. Nothing is inferred, and a name that does
    not normalize is dropped rather than guessed at.

    This set does not rank anything and grants no score. It answers one
    question: would an answer here help tell the live readings apart?

    X1D-LEGACYDIAG4.6. Eligibility is the whole of what this decides. Which of
    these domains is actually put to the patient is settled downstream, and it
    is worth naming the real path because an earlier version of this docstring
    named a function that production never calls:

      1. governed_question_candidates() drops the domains that are no longer
         askable -- anything whose coverage state is not UNKNOWN or PARTIAL --
         and attaches the governed default question to those that remain;
      2. select_questions() scores every candidate on the existing CLARIFY
         weights and splits them into the two R2 tiers;
      3. order_within_tier() orders the differential tier by
      4. preference_rank(), which is deterministic and mentions no pattern, no
         complaint and no domain by name.

    Final selection stays bounded by the existing adaptive budget and the
    existing scoring path. Nothing here raises a score or buys a slot.
    """
    live = live_hypotheses(state)
    if not live:
        return set()
    live_names = {h.pattern_name for h in live}

    required: set = set()
    for hypothesis in live:
        for entry in hypothesis.unresolved_discriminators:
            if not isinstance(entry, dict):
                continue
            # X1D-LEGACYDIAG4.5: the scarce slots go to discriminators that
            # showed their work. One that names a competition but cannot say
            # how the two answers would move it stays in the record and waits.
            if not entry.get("discriminating"):
                continue
            domain = normalize_clinical_domain(entry.get("domain"))
            if domain:
                required.add(domain)
            # 4.6: the alternatives are equally able to settle the same
            # competition, so they are equally eligible. Which one is actually
            # asked is decided downstream by preference_rank, reached through
            # governed_question_candidates and select_questions -- not here.
            for extra in entry.get("also_resolved_by") or []:
                candidate = normalize_clinical_domain(extra)
                if candidate:
                    required.add(candidate)

    for gap in (state.evidence_gaps if state else []):
        if not isinstance(gap, dict):
            continue
        separates = [str(n).strip() for n in (gap.get("separates") or [])]
        # A gap is only a gap if what it separates is still in contention.
        if not (set(separates) & live_names):
            continue
        domain = normalize_clinical_domain(gap.get("domain"))
        if domain:
            required.add(domain)
    return required


# ======================================================================
# Which competition, and what could settle it
# ======================================================================
#
# X1D-LEGACYDIAG4.6. The 4.5 acceptance measured a clean split: which
# COMPETITION each branch was in separated perfectly (within 0.65, between
# 0.00), while which DOMAIN it picked to settle that competition did not
# reproduce (within 0.333, between 0.340). Several findings separate the same
# two readings about equally well, and nothing preferred one consistently.
#
# So the two decisions are pulled apart. The model says what competition it is
# in and which domains could resolve it -- both clinical judgements. Which of
# those gets one of three slots is decided deterministically, by rules that
# mention no pattern, no complaint and no domain by name.
#
# NON-PRODUCTION SECTION. Both functions below are diagnostic and test helpers
# with no caller in app/. Production reads the flat eligible set from
# differential_required_domains and never groups by competition: a domain is
# eligible because some live pair needs it, and which eligible domain is asked
# is decided by preference_rank. The grouping here exists so a competition can
# be INSPECTED -- which readings are actually in contention this turn, and what
# would settle each -- and so a future phase that wants to spread scarce slots
# ACROSS competitions has the shape it would need. Nothing today reads it.
#
# Keeping them costs nothing and deleting them would discard the only
# expression of competition identity in the codebase. Wiring them in is a
# design change and belongs to whichever phase decides to make it, not here.


def competition_key(names: Any) -> Tuple[str, ...]:
    """Identity of one competition: its participants, order-independent.

    A-vs-B and B-vs-A are the same question about the same patient, so they
    must hash the same. Internal reasoning identity only -- it confers no
    clinical authority and never reaches a patient.
    """
    if not isinstance(names, (list, tuple, set)):
        return ()
    cleaned = {str(n).strip() for n in names if str(n or "").strip()}
    return tuple(sorted(cleaned))


def competitions(state: Optional[WorkingDifferentialState]) -> Dict[Tuple[str, ...], set]:
    """Every live competition in this state, and the domains that could settle it.

    Only validated discriminators contribute, so every entry has already
    survived the 4.5 rules: a real live pair, and a counterfactual showing the
    two answers would move things differently.
    """
    live = {h.pattern_name for h in live_hypotheses(state)}
    found: Dict[Tuple[str, ...], set] = {}
    if not live:
        return found
    for hypothesis in live_hypotheses(state):
        for entry in hypothesis.unresolved_discriminators:
            if not isinstance(entry, dict) or not entry.get("discriminating"):
                continue
            key = competition_key(entry.get("separates"))
            if len(key) < 2 or not set(key) <= live:
                continue
            domain = normalize_clinical_domain(entry.get("domain"))
            if domain:
                found.setdefault(key, set()).add(domain)
            for extra in (entry.get("also_resolved_by") or []):
                candidate = normalize_clinical_domain(extra)
                if candidate:
                    found.setdefault(key, set()).add(candidate)
    return found


def describe_policy() -> Dict[str, Any]:
    """Operational description. No clinical content."""
    return {
        "standings": list(STANDING_ORDER),
        "live_standings": list(LIVE_STANDINGS),
        "discriminator_requires_live_pair": True,
        "discriminator_requires_counterfactual": True,
        "allowed_evidence_origins": list(ALLOWED_ORIGINS),
        "max_hypotheses": MAX_HYPOTHESES,
        "escalation_requires_new_evidence": True,
        "ruled_out_is_reversible": True,
    }
