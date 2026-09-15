"""X1D-KNOWLEDGE1B: the seam where local knowledge is asked, and misses are seen.

What this is
------------
Today the reasoning engine asks the corpus directly:

    self.corpus.search(name, ["pattern"], reviewed_only=True, limit=3)

That call is the entire local-knowledge lookup, and when it returns nothing the
case ends at PATTERN_NOT_VERIFIED_IN_REVIEWED_CORPUS. The miss is invisible:
nothing records which concept was unavailable, so there is no way to know what
knowledge would be worth acquiring.

This resolver sits in that exact position and does two things: it delegates the
lookup unchanged, and it records the miss. That is all. The returned value is
the store's own result, so the clinical outcome of every case is bit-for-bit
what it was before.

What it deliberately is not
---------------------------
Not a retriever. Not a verifier. Not a cache. It holds no knowledge state, no
evidence, and no authority: a miss stays a miss, and the existing gates decide
everything they decided yesterday.

The reason to build the seam before building anything behind it is that
external retrieval is expensive and the highest-risk claim kinds are exactly
the ones a naive implementation would fetch first. Measuring which claims are
actually missed, and how often, decides what is worth researching -- and gives
every later component one place to attach to that is already exercised in
production.

Ordering, for later phases
--------------------------
KNOWLEDGE1A defined local-first resolution as: canonical lookup, explicit
reviewed alias lookup, reviewed relationship lookup, verified-external cache,
freshness. Only the first two exist today -- PersistentClinicalStore.search
already matches snapshot aliases -- and both happen inside the delegated call.
The remaining steps attach here, behind this interface, without the reasoning
engine changing again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.services.knowledge.gap_recorder import CLAIM_PATTERN_ENTITY, record_gap
from app.services.knowledge.persistent_clinical import PersistentClinicalStore

logger = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    """What local knowledge had to say about one lookup.

    ``matches`` is the store's own return value, passed through untouched, so
    callers behave exactly as they did before this seam existed.
    """

    subject: str
    entity_types: Sequence[str]
    matches: List[Dict[str, Any]] = field(default_factory=list)
    gap_recorded: bool = False

    @property
    def resolved(self) -> bool:
        return bool(self.matches)


class KnowledgeResolver:
    """Local-knowledge lookup with miss observation.

    Behaviour-preserving by construction: every resolution returns the store
    result unchanged, and recording is fail-open, so neither a hit nor a miss
    can be altered by anything this class does.
    """

    def __init__(self, store: Optional[PersistentClinicalStore] = None,
                 record_gaps: bool = True):
        self.store = store or PersistentClinicalStore()
        # Kept switchable so tests can exercise the lookup path without
        # writing rows, and so a future operator can silence recording without
        # touching the clinical path.
        self.record_gaps = record_gaps

    def resolve_entity(
        self,
        subject: str,
        entity_types: Sequence[str],
        *,
        reviewed_only: bool = True,
        limit: int = 20,
        claim_type: str = CLAIM_PATTERN_ENTITY,
    ) -> ResolutionResult:
        matches = self.store.search(
            subject, list(entity_types), reviewed_only=reviewed_only,
            limit=limit)

        result = ResolutionResult(subject=subject,
                                  entity_types=tuple(entity_types),
                                  matches=matches)

        if not matches and self.record_gaps:
            result.gap_recorded = record_gap(
                claim_type=claim_type,
                subject=subject,
                entity_types=list(entity_types),
                reviewed_only=reviewed_only,
            )

        return result

    def search(
        self,
        subject: str,
        entity_types: Sequence[str],
        reviewed_only: bool = False,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Drop-in shape matching PersistentClinicalStore.search.

        Present so a call site can adopt the seam without changing how it reads
        the result, which is what keeps the reasoning engine's diff to one line.
        """
        return self.resolve_entity(
            subject, entity_types, reviewed_only=reviewed_only,
            limit=limit).matches
