"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P13: sanitized inputs, strategies and metrics.

Input discipline. A trace row enters the harness only as a ``TurnInput``:
pattern names, model confidence, turn position, uncertainty flags, recorded
formula ids and the recorded state. User ids, symptom text, the model's
``reasoning``, the consumer input, profile data and row ids never pass the
sanitizer; a pattern name that is not shaped like a pattern name (digits,
Latin letters, anything longer than a name) is replaced by a redaction token
rather than trusted.

Every number here is STAGING ENGINEERING TRACE COVERAGE: engineering and test
conversations on Staging, not a patient population and not clinical accuracy.
"""

from __future__ import annotations

import re
from collections import Counter

from app.services.coverage import harness as H
from app.services.coverage.whatif import (
    CONSUMER_PAIRS, FLU_SOURCE_ID, PAIRS, SEVERE, SEVERE_PAIRS, build_whatif, verify_pairs_against_source)
from app.services.knowledge.pattern_match import _BOUNDARY

COVERAGE_LABEL = "STAGING ENGINEERING TRACE COVERAGE"
REDACTED_NAME = "<NONCONFORMING_NAME_REDACTED>"

_NAME_OK = re.compile(r"^[㐀-鿿（），、；／：()/,;:\s·\-—～《》“”]{1,40}$")
_FLAG_OK = re.compile(r"^[A-Z][A-Z0-9_]{0,79}$")
_ID_OK = re.compile(r"^[A-Za-z0-9._:\-]{1,64}$")


# ----------------------------------------------------------------------
# Phase C: sanitized input
# ----------------------------------------------------------------------
def _clean_name(value):
    name = str(value or "").strip()
    if not name:
        return None
    return name if _NAME_OK.match(name) else REDACTED_NAME


def _clean_confidence(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, 3) if 0.0 <= v <= 1.0 else None


def sanitize_trace_row(row, ordinal):
    """One Core trace row -> TurnInput, or None when it carries no hypothesis.

    Accepts either the row's ``ai_response_summary`` fields at top level or
    nested under ``ai_response_summary``; reads nothing else from it.
    """
    if not isinstance(row, dict):
        return None
    summary = row.get("ai_response_summary") if isinstance(row.get("ai_response_summary"), dict) else row
    hyps = []
    for h in summary.get("pattern_hypotheses") or []:
        if not isinstance(h, dict):
            continue
        name = _clean_name(h.get("name"))
        if name:
            hyps.append(H.Hypothesis(name, _clean_confidence(h.get("confidence"))))
    if not hyps:
        return None
    flags = tuple(f for f in (summary.get("uncertainty_flags") or []) if isinstance(f, str) and _FLAG_OK.match(f))
    fids = tuple(str(c.get("formula_id")) for c in (summary.get("formula_candidates") or [])
                 if isinstance(c, dict) and c.get("formula_id") and _ID_OK.match(str(c.get("formula_id"))))
    turn = row.get("turn_count")
    state = row.get("recommendation_state")
    return H.TurnInput(ordinal=ordinal, hypotheses=tuple(hyps),
                       turn_count=turn if isinstance(turn, int) and 0 < turn < 100 else None,
                       uncertainty_flags=flags, recorded_formula_ids=fids,
                       recorded_state=state if isinstance(state, str) and _FLAG_OK.match(state) else None)


def sanitize_trace_rows(rows):
    out, redacted = [], 0
    for row in rows:
        t = sanitize_trace_row(row, len(out) + 1)
        if t is not None:
            redacted += sum(1 for h in t.hypotheses if h.name == REDACTED_NAME)
            out.append(t)
    return out, redacted


# Curated static inputs: what each matching rule must and must not do.
CURATED = (
    ("风寒束表", "exact canonical"),
    ("风寒束表证", "trailing 证"),
    (" 风寒束表 ", "surrounding whitespace"),
    ("风寒束表，肺气失宣", "compound, canonical first piece"),
    ("风寒束表（肺气失宣）", "compound in brackets"),
    ("风寒束表—肺气失宣", "dash separator (strategy 4 only)"),
    ("风寒 束表", "internal whitespace (strategy 4 only)"),
    ("风寒", "fragment: must not match"),
    ("风寒束肺", "different organ: must not match"),
    ("外感风寒束表", "prefixed: must not match"),
    ("风热犯卫", "source canonical"),
    ("风热犯肺", "model vocabulary; source says 风热犯卫"),
    ("表寒里热", "source canonical"),
    ("热毒袭肺", "source canonical"),
    ("气阴两虚，正气未复", "source canonical (recovery), itself compound"),
    ("气阴两虚", "fragment of a compound canonical"),
    ("毒热壅盛", "SEVERE source canonical"),
    ("毒热内陷，内闭外脱", "SEVERE source canonical"),
    ("肝阳上亢", "absent from corpus and source"),
)


def curated_turns():
    return [H.TurnInput(ordinal=i + 1, hypotheses=(H.Hypothesis(name, None),), origin="CURATED_STATIC")
            for i, (name, _why) in enumerate(CURATED)]


# ----------------------------------------------------------------------
# Families (lexical grouping of the PRIMARY name's first component)
# ----------------------------------------------------------------------
FAMILY_RULES = (
    ("风寒", re.compile(r"风寒|寒邪束")),
    ("风热", re.compile(r"风热")),
    ("风邪/燥", re.compile(r"风邪|燥")),
    ("肝胆/少阳", re.compile(r"肝|胆|少阳")),
    ("痰/肺热", re.compile(r"痰|肺热|邪热|热邪|寒饮")),
    ("脾胃/湿/食", re.compile(r"脾|胃|湿|食")),
    ("表证/外感", re.compile(r"表证|外感|卫表|肺卫|表寒里热")),
)
OTHER_FAMILY = "其他"


def family_of(name):
    first = next((p for p in _BOUNDARY.split(str(name or "").strip()) if p), "")
    for fam, rx in FAMILY_RULES:
        if rx.search(first):
            return fam
    return OTHER_FAMILY


def justified_families(turns, minimum_distinct=2):
    """A family is reported only if the trace uses >= ``minimum_distinct`` names in it."""
    names = {h.name for t in turns for h in t.hypotheses[:1]}
    per = Counter(family_of(n) for n in names)
    return {f for f, n in per.items() if n >= minimum_distinct and f != OTHER_FAMILY}


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def _ratio(count, denom):
    return {"count": count, "denominator": denom, "percent": round(100.0 * count / denom, 1) if denom else None}


def metrics(results):
    """Consumer metrics. A governed path that exists ONLY through SEVERE objects does not count."""
    n = len(results)
    consumer_governed = [r for r in results if r.governed and not r.severe_only]
    return {
        "PRIMARY_PATTERN_MATCH": _ratio(sum(r.primary_match for r in results), n),
        "ANY_HYPOTHESIS_MATCH": _ratio(sum(r.any_match for r in results), n),
        "FULL_GOVERNED_PATH": _ratio(len(consumer_governed), n),
        "FULL_PATH_TO_CORE_FORMULA": _ratio(sum(r.outcome == H.FULL_PATH for r in consumer_governed), n),
        "PATTERN_MATCH_ONLY": _ratio(sum(r.any_match and not r.governed for r in results), n),
        "SEVERE_ONLY_GOVERNED_PATH_NOT_CONSUMER_COVERAGE": sum(r.governed and r.severe_only for r in results),
        "outcomes": {o: sum(r.outcome == o for r in results) for o in H.OUTCOMES},
    }


def family_metrics(turns, results):
    fams = justified_families(turns)
    groups = {}
    for t, r in zip(turns, results):
        fam = family_of(t.hypotheses[0].name)
        groups.setdefault(fam if fam in fams else OTHER_FAMILY, []).append(r)
    return {f: metrics(rs) for f, rs in sorted(groups.items())}


def run(snap, turns, matcher=H.PRODUCTION, policy=H.PRIMARY_ANCHORED):
    return [H.classify_turn(snap, t, matcher, policy) for t in turns]


def retrieval_policy_comparison(snap, turns):
    """X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P15: the pooled rule it replaced vs the current one.

    "flagged" = turns the historical pooled rule sent down a governed path
    although their top-ranked hypothesis matched nothing (P14: 18 of 71);
    "legitimate" = turns whose top-ranked hypothesis itself reached one (22).
    """
    hist = run(snap, turns, policy=H.POOLED_ALL_HYPOTHESES_HISTORICAL)
    cur = run(snap, turns)
    flagged = [i for i, r in enumerate(hist) if r.governed and not r.primary_match]
    legit = [i for i, r in enumerate(hist) if r.governed and r.primary_match]
    n = len(turns)
    return {
        "current_policy": H.PRIMARY_ANCHORED,
        "historical_policy": H.POOLED_ALL_HYPOTHESES_HISTORICAL,
        "governed": {"current": _ratio(sum(r.governed for r in cur), n),
                     "historical": _ratio(sum(r.governed for r in hist), n)},
        "core": {"current": _ratio(sum(r.outcome == H.FULL_PATH for r in cur), n),
                 "historical": _ratio(sum(r.outcome == H.FULL_PATH for r in hist), n)},
        "flagged_secondary_derived": len(flagged),
        "flagged_still_governed_now": sum(1 for i in flagged if cur[i].governed),
        "legitimate_primary_governed": len(legit),
        "legitimate_preserved_now": sum(1 for i in legit if cur[i].governed),
        "secondary_governed_match_not_used_now": sum(1 for r in cur if r.detail == "SECONDARY_GOVERNED_MATCH_NOT_USED"),
        "primary_not_selected_now": sum(1 for r in cur if r.outcome == H.PRIMARY_NOT_SELECTED),
    }


# ----------------------------------------------------------------------
# Phase G-J: strategies, marginal value, safety signals
# ----------------------------------------------------------------------
def _canonical_pattern_names(snap):
    return [e.name for e in snap.patterns]


def derived_alias_proposals(snap, turns, matcher=H.PRODUCTION):
    """Names the model uses that sit close to a canonical name -- each one a CLAIM, not an alias."""
    canon = _canonical_pattern_names(snap)
    seen, out = set(), []
    for t in turns:
        for h in t.hypotheses:
            if h.name in seen or h.name == REDACTED_NAME or H.match_patterns(snap, h.name, matcher):
                continue
            seen.add(h.name)
            near = [c for c in H.nearby_canonical_names(h.name, canon, limit=1)
                    if H._longest_common_substring(H._normalize(h.name), H._normalize(c)) >= 3]
            if near:
                out.append(H.AliasProposal(pattern_name=near[0], alias=h.name))
    return out


def _influenza_scoped(snap, c):
    titles = [(snap.source(sid).title or "") if snap.source(sid) else "" for sid in c.source_ids]
    return "流行性感冒" in (c.source_scope or "") or any("流行性感冒" in t for t in titles)


def safety_signals(snap, results):
    """What the governed paths rest on. Identification only; no control is applied."""
    seen = {}
    for r in results:
        for c in r.candidates:
            key = (c.formula_id, c.name)
            row = seen.setdefault(key, {"formula_id": c.formula_id, "formula": c.name, "basis": c.basis,
                                        "clinical_review": "NOT_PERFORMED" if c.basis == H.SOURCE_VERIFIED
                                        else "ATTESTED", "source_ids": list(c.source_ids),
                                        "source_scope": c.source_scope, "formula_review_status":
                                        c.formula_review_status, "tier": c.tier, "simulated": c.simulated,
                                        "source_titles": [snap.source(sid).title for sid in c.source_ids
                                                          if snap.source(sid)],
                                        "turns": 0})
            row["turns"] += 1
    scoped = [r for r in results if r.governed and any(_influenza_scoped(snap, c) for c in r.candidates)]
    return {
        "governed_formulas": sorted(seen.values(), key=lambda x: (-x["turns"], x["formula_id"])),
        "governed_turns_resting_on_influenza_scoped_source": len(scoped),
        "scope_gate_in_matcher_or_retrieval": False,
    }


def recorded_cross_check(turns, results, governed_formula_ids):
    """Where Staging itself recorded a governed formula id, does the harness reach it too?"""
    recorded = [(t, r) for t, r in zip(turns, results) if set(t.recorded_formula_ids) & governed_formula_ids]
    agree = sum(1 for t, r in recorded if set(t.recorded_formula_ids) & {c.formula_id for c in r.candidates})
    harness_only = sum(1 for t, r in zip(turns, results)
                       if r.governed and not set(t.recorded_formula_ids) & governed_formula_ids)
    return {"recorded_governed_turns": len(recorded), "harness_agrees": agree,
            "harness_governed_but_not_recorded": harness_only,
            "note": "recorded turns predate or postdate corpus changes and pass a readiness gate the "
                    "harness does not model; disagreement here is expected, agreement is the check"}


def _delta(after, before):
    keys = ("PRIMARY_PATTERN_MATCH", "ANY_HYPOTHESIS_MATCH", "FULL_GOVERNED_PATH", "FULL_PATH_TO_CORE_FORMULA")
    return {k: after[k]["count"] - before[k]["count"] for k in keys}


def build_report(base, trace_turns, *, source_text=None, redacted_names=0, alias_evidence=()):
    """The whole P13 measurement over an immutable snapshot. No I/O."""
    source_texts = {FLU_SOURCE_ID: source_text} if source_text else {}
    s1, s1_prov = build_whatif(base, CONSUMER_PAIRS, label="S1_CANONICAL_EXPANSION")
    s1_core, _ = build_whatif(base, CONSUMER_PAIRS, label="S1_WITH_HYPOTHETICAL_CORE_ROWS", add_core_rows=True)
    s1_sev, sev_prov = build_whatif(base, PAIRS, label="S1_PLUS_SEVERE_NOT_CONSUMER")

    r0 = run(base, trace_turns)
    r1 = run(s1, trace_turns)
    r1c = run(s1_core, trace_turns)
    r1s = run(s1_sev, trace_turns)
    m0, m1 = metrics(r0), metrics(r1)

    # Strategy 2: structure of every name S1 still cannot match. No remapping.
    canon1 = _canonical_pattern_names(s1)
    unmatched = Counter(h.name for t in trace_turns for h in t.hypotheses
                        if not H.match_patterns(s1, h.name))
    structure = Counter()
    for name, n in unmatched.items():
        structure[H.structural_relation(name, canon1)] += n

    # Strategy 3: aliases, only if the source itself says so.
    proposals = list(alias_evidence) + derived_alias_proposals(s1, trace_turns)
    m3_0, verdicts0 = H.alias_matcher(base, proposals, source_texts)
    m3_1, verdicts1 = H.alias_matcher(s1, proposals, source_texts)
    # Strategy 4: neutral normalization.
    m4 = H.Matcher(name="neutral_normalization", neutral_normalization=True)

    strategies = {
        "S0_CURRENT": {"corpus": base.label, "matcher": "production", "policy": H.PRIMARY_ANCHORED,
                       "metrics": m0},
        "S0_HISTORICAL_POOLED": {"corpus": base.label, "matcher": "production",
                                 "policy": H.POOLED_ALL_HYPOTHESES_HISTORICAL,
                                 "metrics": metrics(run(base, trace_turns, policy=H.POOLED_ALL_HYPOTHESES_HISTORICAL))},
        "S1_CANONICAL_EXPANSION": {"corpus": s1.label, "matcher": "production", "metrics": m1},
        "S1_WITH_HYPOTHETICAL_CORE_ROWS": {"corpus": s1_core.label, "matcher": "production",
                                           "metrics": metrics(r1c)},
        "S2_VOCABULARY_GROUNDING_STRUCTURE": {"corpus": s1.label, "matcher": "none (analysis only)",
                                              "metrics": m1, "unmatched_hypothesis_mentions_by_structure":
                                              dict(structure.most_common())},
        "S3_SOURCE_BACKED_ALIASES_ON_CURRENT": {"corpus": base.label, "matcher": m3_0.name,
                                                "metrics": metrics(run(base, trace_turns, m3_0))},
        "S3_SOURCE_BACKED_ALIASES_ON_S1": {"corpus": s1.label, "matcher": m3_1.name,
                                           "metrics": metrics(run(s1, trace_turns, m3_1))},
        "S4_NEUTRAL_NORMALIZATION_ON_CURRENT": {"corpus": base.label, "matcher": m4.name,
                                                "metrics": metrics(run(base, trace_turns, m4))},
        "S4_NEUTRAL_NORMALIZATION_ON_S1": {"corpus": s1.label, "matcher": m4.name,
                                           "metrics": metrics(run(s1, trace_turns, m4))},
    }

    # Phase I: marginal value per pair.
    marginal = []
    for p in CONSUMER_PAIRS:
        alone, _ = build_whatif(base, (p,), label="S0+" + p.key)
        rest, _ = build_whatif(base, tuple(q for q in CONSUMER_PAIRS if q.key != p.key), label="S1-" + p.key)
        marginal.append({"pair": p.key, "pattern": p.pattern_name, "formula": p.formula_name, "tier": p.tier,
                         "added_to_current": _delta(metrics(run(alone, trace_turns)), m0),
                         "lost_if_removed_from_S1": _delta(m1, metrics(run(rest, trace_turns)))})
    severe = []
    for p in SEVERE_PAIRS:
        reached = [r for r in r1s if any(c.tier == SEVERE and c.name == p.formula_name for c in r.candidates)]
        severe.append({"pair": p.key, "pattern": p.pattern_name, "formula": p.formula_name,
                       "trace_turns_reaching_it": len(reached),
                       "reported_as": "SAFETY SIGNAL, NOT CONSUMER COVERAGE"})

    # Phase E: explanations over trace names (current and S1), and curated cases.
    def explain_names(snap):
        out = Counter()
        for t in trace_turns:
            for h in t.hypotheses:
                ms = H.match_patterns(snap, h.name)
                out[ms[0].explanation if ms else H.NO_MATCH] += 1
        return dict(out)

    curated = []
    cur_turns = curated_turns()
    for t, (name, why) in zip(cur_turns, CURATED):
        row = {"name": name, "why": why}
        for label, snap, matcher in (("S0", base, H.PRODUCTION), ("S1", s1, H.PRODUCTION),
                                     ("S1_SEVERE", s1_sev, H.PRODUCTION), ("S4_ON_S1", s1, m4)):
            r = H.classify_turn(snap, t, matcher)
            row[label] = {"explanation": r.hypothesis_explanations[0][1], "outcome": r.outcome,
                          "severe_only": r.severe_only}
        if row["S1"]["explanation"] == H.NO_MATCH:
            row["nearby_canonical_diagnostic_only"] = H.nearby_canonical_names(name, _canonical_pattern_names(s1_sev))
        curated.append(row)

    top_unmatched = [{"name": n, "mentions": c, "family": family_of(n),
                      "structure": H.structural_relation(n, canon1),
                      "nearby_canonical_diagnostic_only": H.nearby_canonical_names(n, canon1)}
                     for n, c in unmatched.most_common(15)]

    governed_ids = {e.entity_id for e in base.entities if e.entity_type == "formula"
                    and (e.review_status in (H.REVIEWED, H.SOURCE_VERIFIED))}
    return {
        "label": COVERAGE_LABEL,
        "input": {"trace_turns_with_hypotheses": len(trace_turns),
                  "distinct_hypothesis_names": len({h.name for t in trace_turns for h in t.hypotheses}),
                  "distinct_primary_names": len({t.hypotheses[0].name for t in trace_turns}),
                  "redacted_nonconforming_names": redacted_names,
                  "fields_read": ["pattern_hypotheses[].name", "pattern_hypotheses[].confidence", "turn_count",
                                  "uncertainty_flags", "formula_candidates[].formula_id",
                                  "recommendation_state"]},
        "corpus": {"environment": base.environment,
                   "source_bounded_retrieval_enabled": H.source_bounded_enabled(base),
                   "sources": len(base.sources), "patterns": len(base.patterns),
                   "formulas": sum(e.entity_type == "formula" for e in base.entities),
                   "relationships": len(base.relationships), "core_formula_rows": len(base.core_formula_names)},
        "whatif_pairs_source_locators_verified": verify_pairs_against_source(PAIRS, source_text)
        if source_text else "SOURCE_TEXT_NOT_SUPPLIED",
        "whatif_provenance": s1_prov + [r for r in sev_prov if r["tier"] == SEVERE],
        "strategies": strategies,
        "families_S0": family_metrics(trace_turns, r0),
        "families_S1": family_metrics(trace_turns, r1),
        "match_explanations": {"S0": explain_names(base), "S1": explain_names(s1)},
        "alias_verdicts": {"on_current": verdicts0, "on_S1": verdicts1},
        "marginal_value": marginal,
        "severe_pairs": severe,
        "safety": {"S0": safety_signals(base, r0), "S1": safety_signals(s1, r1),
                   "S1_PLUS_SEVERE": safety_signals(s1_sev, r1s)},
        "recorded_cross_check_S0": recorded_cross_check(trace_turns, r0, governed_ids),
        "retrieval_policy_comparison": retrieval_policy_comparison(base, trace_turns),
        "top_unmatched_after_S1": top_unmatched,
        "curated_static": curated,
        "simulated_objects": {"S1": s1.simulated_object_count, "S1_PLUS_SEVERE": s1_sev.simulated_object_count},
    }
