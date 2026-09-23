"""Governance V2 foundation (X1D-AIV2-GOV2-C1).

ai-v2 owns clinical object content, versions, semantic identity, evidence
provenance, lifecycle state, ranking eligibility and its own audit history.

xerbs-core owns authenticated HUMAN reviewer identity, reviewer authorization,
the human separation-of-duties decision, and the immutable attestation that
records it.

This package implements only the ai-v2 half. Nothing here can approve
anything: no machine principal in this service is, or may be represented as,
a human clinical reviewer.
"""
