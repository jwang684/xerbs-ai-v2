"""Governed lifecycle, provenance classification, ranking eligibility (GOV2-C1).

The gap this closes
-------------------
``SafetyEngine.create_relationship`` wrote ``review_status='REVIEWED'`` as a
literal, and ``create_rule`` did the same. The ORM default was DRAFT; the code
overrode it. So a relationship's REVIEWED status was never conferred by any
review process anywhere -- it was a constant in a source file, and the ranking
layer trusted it. Production's one PATTERN_FORMULA edge is exactly that: no
reviewer, no review event, no version, status REVIEWED.

After this module, a relationship and a safety rule begin at DRAFT, may be
submitted, may be rejected or sent back, and **cannot reach REVIEWED at all**
inside this service. The approve transition is not "restricted" here; it is
absent. It becomes possible only when xerbs-core supplies a verified human
attestation, which is AI-GOV2's job -- see ``approve_requires_attestation``.

There is deliberately no machine bypass, not even a configurable one. A switch
that can be turned on is a switch that will be turned on.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.services.governance import canonical

# ----------------------------------------------------------------------
# States
# ----------------------------------------------------------------------
DRAFT = "DRAFT"
IN_REVIEW = "IN_REVIEW"
REVIEWED = "REVIEWED"
REJECTED = "REJECTED"
RETIRED = "RETIRED"

#: Every state a governed relationship or safety rule may hold. Mirrors the
#: vocabulary clinical entities already use, minus states they do not need.
GOVERNED_STATES = frozenset({DRAFT, IN_REVIEW, REVIEWED, REJECTED, RETIRED})

#: The state anything newly created starts in. Never REVIEWED.
INITIAL_STATE = DRAFT

#: Transitions this service can truthfully authorise on its own.
#:
#: CHANGES_REQUESTED is an action, not a state: like the entity lifecycle, it
#: returns the object to DRAFT. Approval is absent -- see below.
LOCAL_TRANSITIONS: dict[tuple[str, str], str] = {
    (DRAFT, "SUBMIT"): IN_REVIEW,
    (REJECTED, "SUBMIT"): IN_REVIEW,
    (IN_REVIEW, "REQUEST_CHANGES"): DRAFT,
    (IN_REVIEW, "REJECT"): REJECTED,
    (REVIEWED, "RETIRE"): RETIRED,
    (DRAFT, "RETIRE"): RETIRED,
    (REJECTED, "RETIRE"): RETIRED,
}

#: The one transition this service must never perform by itself.
ATTESTED_TRANSITION = (IN_REVIEW, "APPROVE")


class GovernanceLifecycleError(Exception):
    """A transition was refused."""


class HumanAttestationRequired(GovernanceLifecycleError):
    """Approval was attempted without a verified xerbs-core attestation.

    Raised unconditionally in GOV2-C1: no attestation verification exists yet,
    so every approval attempt fails closed. AI-GOV2 replaces the raise with a
    verified-attestation path; until then this is the whole implementation of
    "a machine may not approve".
    """


def next_state(current: str, action: str) -> str:
    """Resolve one locally-authorisable transition, or refuse."""
    if action == "APPROVE":
        raise HumanAttestationRequired(
            "IN_REVIEW -> REVIEWED requires a verified human clinical review "
            "attestation from xerbs-core. This service has no end-user "
            "identity and cannot establish that a human approved anything; a "
            "reviewer_role in a request body is a self-assertion, not "
            "authority. See docs and AI-GOV2.")
    target = LOCAL_TRANSITIONS.get((current, action))
    if target is None:
        raise GovernanceLifecycleError(
            "cannot %s a %s object" % (action, current))
    return target


def approve_requires_attestation() -> bool:
    """Always True in this service. Present so the seam is greppable."""
    return True


# ----------------------------------------------------------------------
# Governance provenance -- how an object came to hold the status it holds
# ----------------------------------------------------------------------
#: Authored under GOV2 and approved against a verified core attestation.
#: Nothing carries this yet; AI-GOV2 is what sets it.
ATTESTED = "ATTESTED"

#: Reviewed before GOV2 by a lifecycle that ran, with events and versions, but
#: where the submitter and the approver were the same identity. The process
#: happened; the independence did not.
LEGACY_SELF_REVIEWED = "LEGACY_SELF_REVIEWED"

#: Reviewed before GOV2 by two distinct identities. The strongest thing a
#: pre-GOV2 object can be, and still not a verified attestation.
LEGACY_INDEPENDENTLY_REVIEWED = "LEGACY_INDEPENDENTLY_REVIEWED"

#: Holds REVIEWED with no review record of any kind behind it. Production's
#: PATTERN_FORMULA edge is this: the status came from a literal in the code.
LEGACY_UNREVIEWED = "LEGACY_UNREVIEWED"

#: Authored under GOV2, not yet reviewed.
UNREVIEWED = "UNREVIEWED"

#: Was attested, and xerbs-core has since withdrawn that decision. Kept as a
#: distinct value rather than reverting to UNREVIEWED: "nobody has reviewed
#: this yet" and "someone reviewed it and then took it back" are different
#: facts, and the second one is the one a later reviewer needs to know.
ATTESTATION_REVOKED = "ATTESTATION_REVOKED"

GOVERNANCE_PROVENANCE = frozenset({
    ATTESTED, LEGACY_SELF_REVIEWED, LEGACY_INDEPENDENTLY_REVIEWED,
    LEGACY_UNREVIEWED, UNREVIEWED, ATTESTATION_REVOKED})

#: Provenance values that predate GOV2 and therefore carry no verified human
#: decision. Kept as a set rather than a string test so the ranking predicate
#: and the migration cannot drift apart.
LEGACY_PROVENANCE = frozenset({
    LEGACY_SELF_REVIEWED, LEGACY_INDEPENDENTLY_REVIEWED, LEGACY_UNREVIEWED})


# ----------------------------------------------------------------------
# Ranking eligibility
# ----------------------------------------------------------------------
#: GOV2-C1 policy: legacy rows keep the eligibility they already had.
#:
#: This phase is explicitly *pre-attestation*. Flipping this to False is the
#: whole of AI-GOV2-D, and it is not a tidy-up: it makes every pre-GOV2
#: REVIEWED relationship non-ranking-eligible until a human re-reviews it.
#: For Production that means 风热犯卫 -> 银翘散 stops producing a formula
#: candidate. That consequence is real, is stated in the phase report, and
#: must be scheduled deliberately rather than arriving as a side effect of
#: this commit.
#:
#: New objects are unaffected either way: they cannot reach REVIEWED at all,
#: so "lifecycle string alone" can never make a new object eligible.
LEGACY_ELIGIBILITY_GRANDFATHERED = True


def is_governed_object_ranking_eligible(
    *,
    review_status: str,
    governance_provenance: str | None,
    review_attestation_id: str | None,
    reviewed_evidence_source_count: int,
    retired_at: Any = None,
    has_verified_attested_approval: bool = False,
) -> bool:
    """THE eligibility rule for relationships and safety rules.

    One function, so ranking, safety screening and any future caller cannot
    each invent their own version. The lifecycle string is necessary and never
    sufficient.

    X1D-AIV2-ATTEST1 hardening: ``review_attestation_id`` being non-empty is
    also not sufficient. A column is just a column -- a direct UPDATE, a stray
    migration, or a future careless code path could fill it with any string.
    Eligibility therefore requires ``has_verified_attested_approval``, which
    the caller derives from a ``governed_object_review_event`` row recording
    APPROVED_BY_ATTESTATION at **this** version under **this** attestation.
    Only the verified-attestation transition writes that row, and it does not
    carry forward to a later version.
    """
    if retired_at is not None:
        return False
    if review_status != REVIEWED:
        return False

    if governance_provenance in LEGACY_PROVENANCE:
        # Pre-GOV2 rows. Grandfathered for now, explicitly and reversibly.
        return LEGACY_ELIGIBILITY_GRANDFATHERED

    # GOV2-era: durable proof of a verified human decision, or nothing.
    if not review_attestation_id:
        return False
    if not has_verified_attested_approval:
        return False
    return reviewed_evidence_source_count > 0


def relationship_content_hash(source_external_id: str, relationship_type: str,
                              target_external_id: str) -> str:
    return canonical.digest(canonical.relationship_subject(
        source_external_id=source_external_id,
        relationship_type=relationship_type,
        target_external_id=target_external_id))


def safety_rule_content_hash(*, target_external_id: str, rule_type: str,
                             trigger_term: str, severity: str, action: str,
                             message: str) -> str:
    return canonical.digest(canonical.safety_rule_subject(
        target_external_id=target_external_id, rule_type=rule_type,
        trigger_term=trigger_term, severity=severity, action=action,
        message=message))


def evidence_hash(evidence: Sequence[Mapping[str, Any]]) -> str:
    return canonical.evidence_hash(evidence)
