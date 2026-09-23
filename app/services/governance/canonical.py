"""Canonical hashing for cross-service attestation subjects (GOV2-C1).

Why this file is small and pedantic
-----------------------------------
xerbs-core will one day record "a human approved THIS object, at THIS version,
with THIS content and THIS evidence". core does not hold the object -- only the
digest. So the digest is the entire binding, and two implementations that
disagree by one byte silently make every approval unverifiable.

The rules below are therefore stated exactly, versioned, and pinned by literal
test vectors in ``tests/vectors/canonical_hash_vectors.json`` that xerbs-core
can consume verbatim when CORE-ATTEST integration lands. Recomputing an
expected digest with this same code inside an assertion would prove nothing;
the vectors carry the expected values as constants.

The specification
-----------------
``SPEC_VERSION`` identifies it. Changing any rule below requires a new version.

1.  Encoding is UTF-8. Never escape non-ASCII: CJK is the majority of this
    corpus and ``\\uXXXX`` escaping would make the canonical form unreadable
    and locale-sensitive to review.
2.  Every string -- keys and values alike -- is Unicode-normalised to **NFC**
    before serialisation. The same 银翘散 typed on two platforms can arrive
    decomposed on one of them; without this, one object would have two digests.
3.  JSON object keys are sorted by Unicode code point.
4.  No insignificant whitespace: separators are ``(',', ':')``.
5.  Arrays that are semantically **sets** (evidence, indications, aliases,
    ingredient lists used as sets) are sorted by their canonical serialisation.
    Arrays that are semantically **ordered** keep their order. Which is which
    is declared per subject builder below, never guessed.
6.  ``None`` is omitted entirely rather than serialised as ``null``. "Absent"
    and "present but null" must not produce different digests for the same
    clinical meaning; omission is the single representation.
7.  Numbers are integers only. Floats are rejected outright rather than
    rounded, because no reviewed clinical subject field is a float and a
    silent repr change would move a digest.
8.  Excluded from every subject: database primary keys, ``created_at`` /
    ``updated_at``, environment-local ids, ``review_status``, reviewer
    identity, and the version counter. None of them is part of what a human
    read. The version is bound separately by the attestation, so it must not
    also perturb the hash.

Digests are lowercase SHA-256 hex.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

#: Bump this if any rule in the module docstring changes. xerbs-core pins it.
SPEC_VERSION = "xerbs-canonical-hash/1"

#: Relationship types whose semantic identity this module can build.
RELATIONSHIP_TYPES = ("PATTERN_FORMULA", "FORMULA_HERB")


class CanonicalisationError(ValueError):
    """The input cannot be canonicalised without guessing."""


def nfc(text: str) -> str:
    """NFC-normalise one string. The only normalisation this module performs."""
    if not isinstance(text, str):
        raise CanonicalisationError("expected str, got %r" % type(text).__name__)
    return unicodedata.normalize("NFC", text)


def _normalise(value: Any) -> Any:
    """Recursively apply rules 1, 2, 6 and 7. Ordering is applied by callers."""
    if value is None:
        return None
    if isinstance(value, str):
        return nfc(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise CanonicalisationError(
            "floats are not canonicalisable; no reviewed clinical field is a "
            "float, and rounding one silently would move the digest")
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if v is None:
                continue                      # rule 6: omit, never null
            out[nfc(str(k))] = _normalise(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value if v is not None]
    raise CanonicalisationError("unsupported type %r" % type(value).__name__)


def canonical_json(value: Any) -> str:
    """The canonical serialisation. This exact text is what gets hashed."""
    return json.dumps(
        _normalise(value),
        ensure_ascii=False,     # rule 1
        sort_keys=True,         # rule 3
        separators=(",", ":"),  # rule 4
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(value: Any) -> str:
    """Canonicalise then hash. The one entry point every subject builder uses."""
    return sha256_hex(canonical_json(value))


def sorted_set(values: Iterable[Any]) -> list:
    """Rule 5 for set-valued arrays: sort by canonical serialisation.

    Sorting by the canonical text rather than by the raw value keeps the order
    stable for mixed content and for CJK, where Python's default string order
    is code-point order on the *unnormalised* text.
    """
    items = [_normalise(v) for v in values if v is not None]
    return sorted(items, key=lambda v: json.dumps(
        v, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


# ----------------------------------------------------------------------
# Subject builders -- one per governed object type.
# ----------------------------------------------------------------------
# Each returns the canonical SUBJECT (a plain dict), so a caller can inspect,
# log or vector it. digest() turns it into the hash.

def source_subject(*, source_id: str, title: str, citation: str | None,
                   url: str | None, source_type: str) -> dict:
    """A Source is its own evidence, so it has no separate evidence hash."""
    return {
        "spec": SPEC_VERSION,
        "kind": "SOURCE",
        "source_id": source_id,
        "title": title,
        "citation": citation,
        "url": url,
        "source_type": source_type,
    }


#: Entity snapshot keys that are set-valued (rule 5). Everything else that is a
#: list keeps its order -- ingredients of a formula are written in a
#: conventional order and reordering them is a material change, not noise.
ENTITY_SET_FIELDS = ("aliases", "indications", "contraindications",
                     "interaction_flags", "exclusion_flags")

#: Snapshot keys that are never part of the reviewed subject (rule 8).
ENTITY_EXCLUDED_FIELDS = frozenset({
    "review_status", "version", "clinical_ranking_eligible", "sources",
    "pattern_id", "formula_id", "herb_id", "id", "migration_origin",
    "created_at", "updated_at", "retired_at", "superseded_by_id",
})


def entity_subject(*, external_id: str, entity_type: str,
                   snapshot: Mapping[str, Any]) -> dict:
    """The reviewed content of one clinical entity version.

    ``snapshot`` is a ``clinical_entity_version.snapshot`` row. The excluded
    keys above are dropped; set-valued keys are sorted; everything else is
    carried through as written.
    """
    content = {}
    for key, value in snapshot.items():
        if key in ENTITY_EXCLUDED_FIELDS or key == "entity_type":
            continue
        if value is None:
            continue
        if key in ENTITY_SET_FIELDS and isinstance(value, (list, tuple)):
            content[key] = sorted_set(value)
        else:
            content[key] = value
    return {
        "spec": SPEC_VERSION,
        "kind": "CLINICAL_ENTITY",
        "external_id": external_id,
        "entity_type": entity_type,
        "content": content,
    }


def relationship_subject(*, source_external_id: str, relationship_type: str,
                         target_external_id: str) -> dict:
    """A relationship's reviewed content IS its triple.

    There is nothing else to review: no free text, no attributes. That is why
    the relationship is immutable and a change of any element is a different
    relationship rather than an edit.
    """
    if relationship_type not in RELATIONSHIP_TYPES:
        raise CanonicalisationError(
            "unknown relationship_type %r" % (relationship_type,))
    return {
        "spec": SPEC_VERSION,
        "kind": "CLINICAL_RELATIONSHIP",
        "source_external_id": source_external_id,
        "relationship_type": relationship_type,
        "target_external_id": target_external_id,
    }


def safety_rule_subject(*, target_external_id: str, rule_type: str,
                        trigger_term: str, severity: str, action: str,
                        message: str) -> dict:
    """A safety rule's reviewed content, including what it does when it fires.

    severity and action are in the subject deliberately: changing a rule from
    WARN to BLOCK changes what a patient is allowed to buy, and must invalidate
    any prior approval.
    """
    return {
        "spec": SPEC_VERSION,
        "kind": "SAFETY_RULE",
        "target_external_id": target_external_id,
        "rule_type": rule_type,
        "trigger_term": trigger_term,
        "severity": severity,
        "action": action,
        "message": message,
    }


def evidence_subject(evidence: Sequence[Mapping[str, Any]]) -> dict:
    """The exact evidence set that was reviewed.

    Order in the database is not meaningful, so the set is sorted (rule 5):
    two identical evidence sets read back in different row orders must hash
    identically. ``locator`` is part of the subject -- "which section of the
    document" is part of what a reviewer read.
    """
    items = []
    for row in evidence:
        item = {
            "source_id": row["source_id"],
            "source_version": int(row.get("source_version") or 1),
        }
        locator = row.get("locator")
        if locator is not None:
            item["locator"] = locator
        items.append(item)
    return {
        "spec": SPEC_VERSION,
        "kind": "EVIDENCE_SET",
        "evidence": sorted_set(items),
    }


def evidence_hash(evidence: Sequence[Mapping[str, Any]]) -> str:
    """Digest of an evidence set. An empty set still hashes -- to a value that
    is deliberately NOT a sentinel, so "no evidence" cannot be confused with
    "not computed". Eligibility refuses empty evidence separately."""
    return digest(evidence_subject(evidence))
