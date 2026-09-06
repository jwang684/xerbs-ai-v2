from __future__ import annotations
from app.schemas.reasoning import ConvergenceMetrics, ReasoningResponse

class PatternConvergenceEngine:
    """Deterministic multi-turn convergence calculation.

    It does not diagnose or alter TrustScore. It compares structured reasoning
    snapshots and exposes why an interview is or is not converging.
    """
    def evaluate(self, current: ReasoningResponse, previous: ReasoningResponse | None = None) -> ConvergenceMetrics:
        high=sum(1 for x in current.missing_information if x.priority=="HIGH")
        medium=sum(1 for x in current.missing_information if x.priority=="MEDIUM")
        low=sum(1 for x in current.missing_information if x.priority=="LOW")
        evidence_sufficiency=max(0.0,1.0-min(1.0,high*.22+medium*.11+low*.05))

        current_names={x.name for x in current.pattern_assessments if x.name}
        previous_names={x.name for x in previous.pattern_assessments if x.name} if previous else set()
        if previous is None:
            pattern_stability=0.5 if current_names else 0.0
            stable=[]; changed=sorted(current_names)
        else:
            union=current_names|previous_names
            overlap=current_names&previous_names
            pattern_stability=(len(overlap)/len(union)) if union else 0.0
            stable=sorted(overlap); changed=sorted(union-overlap)

        verified=max([x.model_confidence for x in current.pattern_assessments if x.corpus_match] or [0.0])
        contradiction_count=sum(len(x.contradictions) for x in current.pattern_assessments)
        contradiction_penalty=min(1.0,contradiction_count*.18)
        score=.34*evidence_sufficiency+.30*pattern_stability+.36*verified-.25*contradiction_penalty
        score=max(0.0,min(1.0,score))
        rationale=[]
        if high: rationale.append(f"{high} high-priority clinical information slot(s) remain unresolved")
        if stable: rationale.append("pattern hypothesis remained stable across turns: "+", ".join(stable))
        if changed: rationale.append("pattern hypothesis changed across turns: "+", ".join(changed))
        if verified: rationale.append("at least one hypothesis matches the reviewed clinical corpus")
        if contradiction_count: rationale.append(f"{contradiction_count} contradiction finding(s) reduce convergence")
        if not rationale: rationale.append("insufficient structured evidence for convergence")
        return ConvergenceMetrics(
            score=round(score,3), evidence_sufficiency=round(evidence_sufficiency,3),
            pattern_stability=round(pattern_stability,3), verified_pattern_strength=round(verified,3),
            contradiction_penalty=round(contradiction_penalty,3), stable_pattern_names=stable,
            changed_pattern_names=changed, contradiction_count=contradiction_count, rationale=rationale,
        )
