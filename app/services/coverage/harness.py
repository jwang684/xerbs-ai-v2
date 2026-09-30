"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P13: the deterministic governed coverage harness.

Question it answers: for the pattern names the model actually emits, how far
down the governed chain does each turn get --

    model pattern name -> governed pattern -> governed PATTERN_FORMULA link
    -> governed formula -> Core canonical formula row

-- today, and under a stated, in-memory what-if corpus?

How it stays honest
-------------------
* Pattern names are compared with the production matcher itself
  (``pattern_match.name_components`` / ``matches_reviewed_name``); nothing here
  re-implements string comparison for the current-reality mode.
* The eligibility gates mirror, line for line, ``PersistentClinicalStore.
  match_reviewed_patterns`` and ``eligible_formula_candidates_for_patterns``
  (REVIEWED first, then SOURCE_VERIFIED where
  ``lifecycle.source_bounded_retrieval_enabled`` says so). A parity test runs
  both against the same governed state.
* It works on an immutable ``CorpusSnapshot``. It never opens a database
  session, never calls a model, and a what-if corpus is a new snapshot object,
  not a write. Simulated objects carry ``simulated=True`` and say so in every
  report line derived from them.
* Anything that would widen matching beyond the production rule (aliases,
  extra normalization) is a separately named strategy, never the baseline.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

from app.services.governance import lifecycle
from app.services.knowledge.pattern_match import _BOUNDARY, _normalize, name_components

REVIEWED = "REVIEWED"
SOURCE_VERIFIED = lifecycle.SOURCE_VERIFIED
PATTERN_FORMULA = "PATTERN_FORMULA"

# Full-chain outcome of one turn, from furthest to nearest failure.
FULL_PATH = "FULL_PATH"                                  # governed formula AND exactly one Core canonical row
CORE_CANONICAL_MISSING = "CORE_CANONICAL_MISSING"        # governed formula, no single Core canonical row
FORMULA_INELIGIBLE = "FORMULA_INELIGIBLE"
RELATIONSHIP_INELIGIBLE = "RELATIONSHIP_INELIGIBLE"
NO_RELATIONSHIP = "NO_RELATIONSHIP"
PATTERN_INELIGIBLE = "PATTERN_INELIGIBLE"
NO_PATTERN_MATCH = "NO_PATTERN_MATCH"
OUTCOMES = (FULL_PATH, CORE_CANONICAL_MISSING, FORMULA_INELIGIBLE, RELATIONSHIP_INELIGIBLE,
            NO_RELATIONSHIP, PATTERN_INELIGIBLE, NO_PATTERN_MATCH)
GOVERNED_OUTCOMES = frozenset({FULL_PATH, CORE_CANONICAL_MISSING})

# How a model name met a governed pattern.
EXACT_CANONICAL = "EXACT_CANONICAL"
NORMALIZED_CANONICAL = "NORMALIZED_CANONICAL"
COMPOUND_PIECE = "COMPOUND_PIECE"
RECORD_ALIAS = "RECORD_ALIAS"                            # alias carried by the governed record itself
NEUTRAL_NORMALIZATION = "NEUTRAL_NORMALIZATION"          # strategy 4 only
SOURCE_BACKED_ALIAS = "SOURCE_BACKED_ALIAS"              # strategy 3 only, simulation
NO_MATCH = "NO_MATCH"

CLINICAL_EQUIVALENCE_EVIDENCE_REQUIRED = "CLINICAL_EQUIVALENCE_EVIDENCE_REQUIRED"
SOURCE_BACKED = "SOURCE_BACKED"


# ----------------------------------------------------------------------
# Immutable corpus snapshot
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SourceSnap:
    source_id: str
    review_status: str
    version: int
    # A live SOURCE_VERIFIED_BY_ATTESTATION at the current version (not revoked).
    live_verification: bool = False
    title: str | None = None
    simulated: bool = False


@dataclass(frozen=True)
class EntitySnap:
    entity_id: str
    entity_type: str                 # "pattern" | "formula" | ...
    name: str
    review_status: str
    version: int
    aliases: tuple = ()
    source_ids: tuple = ()
    live_verification: bool = False
    clinical_ranking_eligible: bool = False
    retired: bool = False
    external_id: str | None = None
    source_scope: str | None = None
    tier: str | None = None          # what-if only: MILD | SEVERE | RECOVERY
    simulated: bool = False


@dataclass(frozen=True)
class RelationshipSnap:
    relationship_id: str
    pattern_id: str
    formula_id: str
    review_status: str
    version: int
    evidence_source_ids: tuple = ()
    live_verification: bool = False
    relationship_type: str = PATTERN_FORMULA
    # REVIEWED relationships need a verified clinical attestation; ai-v2 does
    # not expose that over HTTP, so a loader that cannot prove it leaves False.
    clinical_ranking_eligible: bool = False
    simulated: bool = False


@dataclass(frozen=True)
class CorpusSnapshot:
    environment: str
    sources: tuple = ()
    entities: tuple = ()
    relationships: tuple = ()
    # One entry per Core ``herbal_formulas`` row (duplicates kept: the resolver
    # refuses a name that is not unique).
    core_formula_names: tuple = ()
    # Core formula names that have exactly one APPROVED product mapping (informational).
    core_mapped_formula_names: tuple = ()
    label: str = "CURRENT_REALITY"

    def source(self, source_id):
        return next((s for s in self.sources if s.source_id == source_id), None)

    def entity(self, entity_id):
        return next((e for e in self.entities if e.entity_id == entity_id), None)

    @property
    def patterns(self):
        return tuple(sorted((e for e in self.entities if e.entity_type == "pattern"), key=lambda e: e.entity_id))

    def with_additions(self, *, sources=(), entities=(), relationships=(), core_formula_names=(), label=None):
        """A NEW snapshot with these objects added. The receiver is unchanged."""
        return replace(self, sources=self.sources + tuple(sources), entities=self.entities + tuple(entities),
                       relationships=self.relationships + tuple(relationships),
                       core_formula_names=self.core_formula_names + tuple(core_formula_names),
                       label=label or self.label)

    @property
    def simulated_object_count(self):
        return sum(1 for x in (*self.sources, *self.entities, *self.relationships) if x.simulated)


# ----------------------------------------------------------------------
# Matching (production rule + opt-in strategies)
# ----------------------------------------------------------------------
# Strategy 4, neutral normalization: typography only. No character is treated
# as a clinical separator here ("夹", "兼", "与" are clinical words, not punctuation).
_NEUTRAL_EXTRA_BOUNDARY = re.compile(r"[—–‒‐\-~～·•・“”\"'‘’《》〈〉<>「」『』+＋&＆]+")
_ZERO_WIDTH = re.compile(r"[​-‍﻿]")
_WS = re.compile(r"\s+")


def neutral_components(model_name: str) -> frozenset:
    """Production components PLUS those visible after typography-only cleanup."""
    base = set(name_components(model_name))
    text = _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", str(model_name or ""))).strip()
    pieces = [text, _WS.sub("", text)]
    for chunk in _BOUNDARY.split(text):
        pieces.append(chunk)
        pieces.extend(_NEUTRAL_EXTRA_BOUNDARY.split(chunk))
    for chunk in list(pieces):
        pieces.extend(_NEUTRAL_EXTRA_BOUNDARY.split(chunk))
    base.update(p for p in (_normalize(x) for x in pieces) if p)
    return frozenset(base)


@dataclass(frozen=True)
class Matcher:
    """How names are compared. The default IS the production rule."""
    name: str = "production"
    neutral_normalization: bool = False
    # pattern_id -> alias terms, ONLY from accepted source-backed alias proposals.
    source_backed_aliases: Mapping[str, tuple] = field(default_factory=dict)

    def components(self, model_name):
        return neutral_components(model_name) if self.neutral_normalization else name_components(model_name)

    def explain(self, model_name: str, e: EntitySnap) -> str:
        """Why this model name names this pattern, or NO_MATCH. Ordered by strength."""
        whole = str(model_name or "").strip()
        prod = name_components(model_name)
        canon = _normalize(e.name)
        if canon and whole == str(e.name or "").strip():
            return EXACT_CANONICAL
        if canon and _normalize(whole) == canon:
            return NORMALIZED_CANONICAL
        if canon and canon in prod:
            return COMPOUND_PIECE
        if {t for t in (_normalize(a) for a in e.aliases) if t} & prod:
            return RECORD_ALIAS
        comps = self.components(model_name)
        if self.neutral_normalization and ({canon} | {_normalize(a) for a in e.aliases}) & comps:
            return NEUTRAL_NORMALIZATION
        extra = {t for t in (_normalize(a) for a in self.source_backed_aliases.get(e.entity_id, ())) if t}
        if extra & comps:
            return SOURCE_BACKED_ALIAS
        return NO_MATCH


PRODUCTION = Matcher()


# ----------------------------------------------------------------------
# Eligibility: mirrors PersistentClinicalStore (see module docstring)
# ----------------------------------------------------------------------
def source_bounded_enabled(snap: CorpusSnapshot) -> bool:
    return lifecycle.source_bounded_retrieval_enabled(snap.environment)


def verified_source_ids(snap, source_ids):
    """_verified_source_ids: sources a human source-verified at their current version."""
    out = set()
    for sid in source_ids:
        src = snap.source(sid)
        if src is not None and src.review_status == SOURCE_VERIFIED and src.live_verification:
            out.add(sid)
    return out


def source_verified_entity_sources(snap, e):
    """_source_verified_entity_sources."""
    if e.review_status != SOURCE_VERIFIED or e.retired:
        return set()
    if not e.live_verification:
        return set()
    return verified_source_ids(snap, e.source_ids)


def endpoint_sources(snap, e):
    """_source_bounded_endpoint_sources."""
    if e.review_status == REVIEWED and e.clinical_ranking_eligible:
        return set(e.source_ids)
    return source_verified_entity_sources(snap, e)


def pattern_gate(snap, e):
    """(eligible, basis_or_reason) for one pattern under match_reviewed_patterns."""
    statuses = (REVIEWED, SOURCE_VERIFIED) if source_bounded_enabled(snap) else (REVIEWED,)
    if e.review_status not in statuses:
        if e.review_status == SOURCE_VERIFIED:
            return False, "SOURCE_BOUNDED_RETRIEVAL_DISABLED_IN_%s" % (snap.environment or "").upper()
        return False, "PATTERN_STATUS_%s" % e.review_status
    if e.review_status == SOURCE_VERIFIED:
        if not source_verified_entity_sources(snap, e):
            return False, "NO_LIVE_SOURCE_VERIFICATION_OR_VERIFIED_SOURCE"
        return True, "SOURCE_VERIFIED"
    return True, "CLINICAL_REVIEW"


@dataclass(frozen=True)
class PatternMatch:
    pattern_id: str
    pattern_name: str
    governance_basis: str
    explanation: str
    simulated: bool
    tier: str | None


def match_patterns(snap, model_name, matcher: Matcher = PRODUCTION, limit=3):
    """match_reviewed_patterns: eligible patterns, id order, first ``limit``."""
    if not name_components(model_name) and not matcher.components(model_name):
        return []
    out = []
    for e in snap.patterns:
        ok, basis = pattern_gate(snap, e)
        if not ok:
            continue
        how = matcher.explain(model_name, e)
        if how != NO_MATCH:
            out.append(PatternMatch(e.entity_id, e.name, basis, how, e.simulated, e.tier))
        if len(out) >= limit:
            break
    return out


def ineligible_name_matches(snap, model_name, matcher: Matcher = PRODUCTION):
    """Patterns that the name DOES name but that no gate lets through (diagnostic)."""
    out = []
    for e in snap.patterns:
        ok, reason = pattern_gate(snap, e)
        if not ok and matcher.explain(model_name, e) != NO_MATCH:
            out.append({"pattern_id": e.entity_id, "pattern_name": e.name, "reason": reason})
    return out


@dataclass(frozen=True)
class FormulaCandidate:
    formula_id: str
    name: str
    basis: str
    source_ids: tuple
    link_count: int
    pattern_ids: tuple
    formula_review_status: str
    source_scope: str | None
    simulated: bool
    tier: str | None


def _formula_ranking_eligible(e):
    return e is not None and e.entity_type == "formula" and e.review_status == REVIEWED and e.clinical_ranking_eligible


def _source_bounded_formula_candidates(snap, pattern_ids):
    out = {}
    for rel in snap.relationships:
        if rel.pattern_id not in pattern_ids or rel.relationship_type != PATTERN_FORMULA:
            continue
        if rel.review_status != SOURCE_VERIFIED or not rel.live_verification:
            continue
        pattern, formula = snap.entity(rel.pattern_id), snap.entity(rel.formula_id)
        if pattern is None or formula is None or formula.entity_type != "formula":
            continue
        shared = (verified_source_ids(snap, rel.evidence_source_ids)
                  & endpoint_sources(snap, pattern) & endpoint_sources(snap, formula))
        if not shared:
            continue
        item = out.setdefault(formula.entity_id,
                              {"count": 0, "patterns": [], "sources": set(), "formula": formula,
                               "simulated": False})
        item["count"] += 1
        item["patterns"].append(rel.pattern_id)
        item["sources"] |= shared
        item["simulated"] = item["simulated"] or rel.simulated or formula.simulated or pattern.simulated
        item.setdefault("scope", formula.source_scope or pattern.source_scope)
    result = []
    for eid, meta in sorted(out.items(), key=lambda kv: (-kv[1]["count"], kv[0])):
        f = meta["formula"]
        # The scope the source states for this link: the formula's own, else its pattern's.
        result.append(FormulaCandidate(eid, f.name, SOURCE_VERIFIED, tuple(sorted(meta["sources"])), meta["count"],
                                       tuple(meta["patterns"]), f.review_status, meta["scope"],
                                       meta["simulated"], f.tier))
    return result


def formula_candidates(snap, pattern_ids: Sequence[str]):
    """eligible_formula_candidates_for_patterns: reviewed links first, then source-bounded."""
    pattern_ids = [x for x in pattern_ids if x]
    if not pattern_ids:
        return []
    scored = {}
    for rel in snap.relationships:
        if rel.pattern_id not in pattern_ids or rel.relationship_type != PATTERN_FORMULA:
            continue
        if not (rel.review_status == REVIEWED and rel.clinical_ranking_eligible):
            continue
        formula = snap.entity(rel.formula_id)
        if not _formula_ranking_eligible(formula):
            continue
        item = scored.setdefault(formula.entity_id, {"count": 0, "patterns": [], "formula": formula})
        item["count"] += 1
        item["patterns"].append(rel.pattern_id)
    out = []
    for eid, meta in sorted(scored.items(), key=lambda kv: (-kv[1]["count"], kv[0]))[:3]:
        f = meta["formula"]
        out.append(FormulaCandidate(eid, f.name, "CLINICAL_REVIEW", tuple(sorted(f.source_ids)), meta["count"],
                                    tuple(meta["patterns"]), f.review_status, f.source_scope, f.simulated, f.tier))
    if len(out) < 3 and source_bounded_enabled(snap):
        seen = {c.formula_id for c in out}
        extra = [c for c in _source_bounded_formula_candidates(snap, pattern_ids) if c.formula_id not in seen]
        out.extend(extra[:3 - len(out)])
    return out


def relationship_block(snap, rel):
    """Why this PATTERN_FORMULA link yields no candidate: (outcome, reason)."""
    pattern, formula = snap.entity(rel.pattern_id), snap.entity(rel.formula_id)
    if formula is None or formula.entity_type != "formula":
        return FORMULA_INELIGIBLE, "FORMULA_MISSING"
    if rel.review_status == REVIEWED:
        if not rel.clinical_ranking_eligible:
            return RELATIONSHIP_INELIGIBLE, "REVIEWED_RELATIONSHIP_WITHOUT_VERIFIED_CLINICAL_ATTESTATION"
        if not _formula_ranking_eligible(formula):
            return FORMULA_INELIGIBLE, "FORMULA_NOT_CLINICALLY_RANKING_ELIGIBLE_%s" % formula.review_status
        return None, None
    if rel.review_status != SOURCE_VERIFIED:
        return RELATIONSHIP_INELIGIBLE, "RELATIONSHIP_STATUS_%s" % rel.review_status
    if not source_bounded_enabled(snap):
        return RELATIONSHIP_INELIGIBLE, "SOURCE_BOUNDED_RETRIEVAL_DISABLED"
    if not rel.live_verification:
        return RELATIONSHIP_INELIGIBLE, "NO_LIVE_RELATIONSHIP_VERIFICATION"
    evidence = verified_source_ids(snap, rel.evidence_source_ids)
    if not evidence:
        return RELATIONSHIP_INELIGIBLE, "EVIDENCE_SOURCE_NOT_VERIFIED"
    if not evidence & endpoint_sources(snap, pattern):
        return RELATIONSHIP_INELIGIBLE, "EVIDENCE_SOURCE_NOT_ATTACHED_TO_PATTERN"
    if not endpoint_sources(snap, formula):
        return FORMULA_INELIGIBLE, "FORMULA_NOT_SOURCE_VERIFIED_OR_NO_VERIFIED_SOURCE_%s" % formula.review_status
    if not evidence & endpoint_sources(snap, pattern) & endpoint_sources(snap, formula):
        return FORMULA_INELIGIBLE, "EVIDENCE_SOURCE_NOT_ATTACHED_TO_FORMULA"
    return None, None


def core_canonical_formula(snap, name):
    """Core formula_resolver_service._exact_formula_by_name: exact, stripped, exactly one row."""
    cleaned = (name or "").strip()
    return bool(cleaned) and sum(1 for n in snap.core_formula_names if n == cleaned) == 1


# ----------------------------------------------------------------------
# Full chain for one turn
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Hypothesis:
    name: str
    confidence: float | None = None


@dataclass(frozen=True)
class TurnInput:
    """Sanitized: pattern names, confidence, turn position, flags, formula ids. Nothing else."""
    ordinal: int
    hypotheses: tuple
    turn_count: int | None = None
    uncertainty_flags: tuple = ()
    recorded_formula_ids: tuple = ()
    recorded_state: str | None = None
    origin: str = "STAGING_TRACE"


@dataclass(frozen=True)
class TurnResult:
    ordinal: int
    origin: str
    outcome: str
    detail: str | None
    primary_match: bool
    any_match: bool
    hypothesis_explanations: tuple      # ((name, explanation, pattern_name|None), ...)
    matched_pattern_ids: tuple
    candidates: tuple
    core_formula: str | None
    severe_only: bool                   # governed path exists ONLY through SEVERE what-if objects

    @property
    def governed(self):
        return self.outcome in GOVERNED_OUTCOMES


def classify_turn(snap, turn: TurnInput, matcher: Matcher = PRODUCTION) -> TurnResult:
    """The whole chain, the way the engine and assembler walk it.

    Every hypothesis is matched; each matched hypothesis contributes its FIRST
    match's pattern id (engine.py); retrieval runs once over all of them
    (assembler.py); Core tries the candidates in order (formula_resolver_service).
    Readiness (HIGH missing fields) is a per-conversation fact the harness does
    not have; it measures corpus reach, not interview completeness.
    """
    explanations, pattern_ids, first_tiers = [], [], {}
    primary_match = False
    for i, h in enumerate(turn.hypotheses):
        matches = match_patterns(snap, h.name, matcher)
        if matches:
            m = matches[0]
            explanations.append((h.name, m.explanation, m.pattern_name))
            if m.pattern_id not in pattern_ids:
                pattern_ids.append(m.pattern_id)
            primary_match = primary_match or i == 0
        else:
            explanations.append((h.name, NO_MATCH, None))
    any_match = bool(pattern_ids)

    def result(outcome, detail=None, cands=(), core=None, severe_only=False):
        return TurnResult(turn.ordinal, turn.origin, outcome, detail, primary_match, any_match,
                          tuple(explanations), tuple(pattern_ids), tuple(cands), core, severe_only)

    if not any_match:
        blocked = [b for h in turn.hypotheses for b in ineligible_name_matches(snap, h.name, matcher)]
        if blocked:
            return result(PATTERN_INELIGIBLE, blocked[0]["reason"])
        return result(NO_PATTERN_MATCH)

    cands = formula_candidates(snap, pattern_ids)
    if not cands:
        rels = [r for r in snap.relationships
                if r.pattern_id in pattern_ids and r.relationship_type == PATTERN_FORMULA]
        if not rels:
            return result(NO_RELATIONSHIP)
        blocks = [relationship_block(snap, r) for r in rels]
        formula_blocks = [b for b in blocks if b[0] == FORMULA_INELIGIBLE]
        chosen = formula_blocks[0] if formula_blocks else blocks[0]
        return result(chosen[0] or RELATIONSHIP_INELIGIBLE, chosen[1])

    severe_only = all(c.tier == "SEVERE" for c in cands)
    core = next((c.name for c in cands if core_canonical_formula(snap, c.name)), None)
    if core:
        return result(FULL_PATH, None, cands, core, severe_only)
    return result(CORE_CANONICAL_MISSING, "NO_SINGLE_CORE_HERBAL_FORMULAS_ROW_WITH_EXACT_NAME", cands, None,
                  severe_only)


# ----------------------------------------------------------------------
# Diagnostics that never feed matching
# ----------------------------------------------------------------------
def _longest_common_substring(a, b):
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best


def nearby_canonical_names(model_name, canonical_names: Iterable[str], limit=3):
    """DIAGNOSTIC ONLY: canonical names sharing >= 2 consecutive characters. Never a match."""
    whole = _normalize(model_name)
    scored = []
    for c in set(canonical_names):
        n = _longest_common_substring(whole, _normalize(c))
        if n >= 2:
            scored.append((-n, c))
    return [c for _, c in sorted(scored)[:limit]]


def structural_relation(model_name, canonical_names: Iterable[str]):
    """Strategy 2 (canonical vocabulary grounding): structure only, no remapping."""
    comps = name_components(model_name)
    canon = {_normalize(c) for c in canonical_names}
    if comps & canon:
        return "USES_CANONICAL_TERM"
    whole = _normalize(model_name)
    if any(c and c in whole for c in canon):
        return "CONTAINS_CANONICAL_TERM_WITHOUT_BOUNDARY"
    if any(c and whole in c for c in canon):
        return "FRAGMENT_OF_CANONICAL_TERM"
    if any(_longest_common_substring(whole, c) >= 3 for c in canon):
        return "SHARES_3PLUS_CHARS_WITH_CANONICAL_TERM"
    if any(_longest_common_substring(whole, c) >= 2 for c in canon):
        return "SHARES_2_CHARS_WITH_CANONICAL_TERM"
    return "NO_CANONICAL_NEIGHBOUR"


# ----------------------------------------------------------------------
# Strategy 3: aliases require verbatim source evidence
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AliasProposal:
    pattern_name: str        # canonical, as in the corpus
    alias: str               # the name the model uses
    source_id: str | None = None
    verbatim_quote: str | None = None


def _squash(text):
    return _WS.sub("", unicodedata.normalize("NFKC", str(text or "")))


def evaluate_alias(proposal: AliasProposal, source_texts: Mapping[str, str], snap: CorpusSnapshot):
    """SOURCE_BACKED only if a verified source's own text names both, in one verbatim passage.

    Anything else -- including "they are obviously the same pattern" -- is a
    clinical equivalence claim, and needs evidence this harness does not accept.
    """
    if not proposal.source_id or not proposal.verbatim_quote:
        return CLINICAL_EQUIVALENCE_EVIDENCE_REQUIRED, "NO_SOURCE_EVIDENCE"
    src = snap.source(proposal.source_id)
    if src is None or src.review_status != SOURCE_VERIFIED or not src.live_verification:
        return CLINICAL_EQUIVALENCE_EVIDENCE_REQUIRED, "SOURCE_NOT_VERIFIED"
    text = source_texts.get(proposal.source_id)
    quote = _squash(proposal.verbatim_quote)
    if not text or quote not in _squash(text):
        return CLINICAL_EQUIVALENCE_EVIDENCE_REQUIRED, "QUOTE_NOT_FOUND_IN_SOURCE_TEXT"
    if _squash(proposal.alias) not in quote or _squash(proposal.pattern_name) not in quote:
        return CLINICAL_EQUIVALENCE_EVIDENCE_REQUIRED, "QUOTE_DOES_NOT_NAME_BOTH"
    return SOURCE_BACKED, None


def alias_matcher(snap, proposals, source_texts, *, base: Matcher = PRODUCTION):
    """A matcher carrying only the SOURCE_BACKED proposals, plus every verdict."""
    aliases, verdicts = {}, []
    by_name = {e.name: e.entity_id for e in snap.patterns}
    for p in proposals:
        verdict, reason = evaluate_alias(p, source_texts, snap)
        pid = by_name.get(p.pattern_name)
        if verdict == SOURCE_BACKED and pid is None:
            verdict, reason = CLINICAL_EQUIVALENCE_EVIDENCE_REQUIRED, "PATTERN_NOT_IN_CORPUS"
        verdicts.append({"pattern_name": p.pattern_name, "alias": p.alias, "verdict": verdict, "reason": reason})
        if verdict == SOURCE_BACKED:
            aliases.setdefault(pid, ())
            aliases[pid] = aliases[pid] + (p.alias,)
    return replace(base, name=base.name + "+source_backed_aliases", source_backed_aliases=aliases), verdicts
