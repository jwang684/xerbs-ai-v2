"""Semantic identity and machine principals (GOV2-C1).

Two separate problems, both about naming things durably.

Semantic identity
-----------------
A row's primary key here is ``pat-`` / ``frm-`` / ``rel-`` plus a uuid4. That
identifies a row in one database. It cannot identify the same clinical object
in another, which is exactly what a cross-service attestation and a future
governed Production -> Staging export both need.

So governed objects gain an ``external_id``:

  * entities: **explicitly supplied** by trusted authoring tooling. Not derived
    from ``name`` -- a display name is mutable, and deriving identity from it
    would mean renaming an entity silently forks its identity.
  * relationships and safety rules: **derived deterministically** from their
    endpoints' external ids. Those endpoints are themselves explicit, so the
    derivation never bottoms out in a mutable string.
  * sources: ``source_registry.source_id`` already is a semantic id
    (``nhsa-2024-YPSN202400007``). Reused as-is; no parallel field.

Legacy rows keep ``external_id IS NULL``. GOV2-C1 does not invent identities
for objects that were authored without one -- see the migration.

Principals
----------
Human reviewers live in xerbs-core. Everything that acts inside ai-v2 is a
machine, and is named as one. There is deliberately no ``CLINICAL_REVIEWER``
principal here: creating one to stand in for a person is the failure this
whole programme exists to undo.
"""

from __future__ import annotations

import re
import unicodedata

# ----------------------------------------------------------------------
# Namespaced subjects -- the format xerbs-core's attestation table validates.
# ----------------------------------------------------------------------
#: Shared with xerbs-core's ck_attestation_*_subject constraints. Keep in step.
SUBJECT_RE = re.compile(
    r"^(xerbs-core|xerbs-ai-v2):(admin|principal|service):"
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

AI_V2 = "xerbs-ai-v2"
CORE = "xerbs-core"


def principal_subject(name: str) -> str:
    """``xerbs-ai-v2:principal:<name>`` -- a machine identity in this service."""
    subject = "%s:principal:%s" % (AI_V2, name)
    if not SUBJECT_RE.match(subject):
        raise ValueError("invalid principal name %r" % (name,))
    return subject


def is_valid_subject(subject: str | None) -> bool:
    return bool(subject) and bool(SUBJECT_RE.match(subject))


#: The complete set of machine principals this service recognises.
#:
#: BOOTSTRAP        in-process corpus bootstrap (golden_corpus.py)
#: CORPUS_AUTHORING deliberate authoring tooling
#: MIGRATION        schema/data migrations acting on their own behalf
#: SERVICE_CORE     xerbs-core calling over the service boundary
#: LEGACY_UNKNOWN   historical rows whose actor was never recorded. NOT an
#:                  actor: a truthful statement that nobody knows who acted.
BOOTSTRAP = principal_subject("bootstrap")
CORPUS_AUTHORING = principal_subject("corpus-authoring")
MIGRATION = principal_subject("migration")
SERVICE_CORE = "%s:service:xerbs-core" % CORE
LEGACY_UNKNOWN = principal_subject("legacy-unknown")

MACHINE_PRINCIPALS = frozenset({
    BOOTSTRAP, CORPUS_AUTHORING, MIGRATION, SERVICE_CORE, LEGACY_UNKNOWN})

#: No principal in this service may approve. Stated as data so a test can
#: assert it rather than trusting that nobody adds one later.
PRINCIPALS_THAT_MAY_HUMAN_APPROVE: frozenset = frozenset()


def as_subject(actor: str | None) -> str:
    """Best-effort mapping of a historical free-text actor to a subject.

    Historical ``created_by`` values are arbitrary strings
    (``phase12c4b-automation``, ``xerbs-bootstrap-ingest``). They are not
    subjects and must not be dressed up as one. An actor that does not already
    look like a subject and cannot be safely expressed as a principal name is
    reported as LEGACY_UNKNOWN -- honest, and greppable.
    """
    if actor is None:
        return LEGACY_UNKNOWN
    actor = actor.strip()
    if is_valid_subject(actor):
        return actor
    candidate = "%s:principal:%s" % (AI_V2, actor)
    if SUBJECT_RE.match(candidate):
        return candidate
    return LEGACY_UNKNOWN


# ----------------------------------------------------------------------
# Semantic identity
# ----------------------------------------------------------------------
#: An entity external id is supplied, not generated. The grammar is permissive
#: about content and strict about shape, because the authoring side owns the
#: vocabulary and this service only has to store and compare it.
#:
#: CJK is explicitly allowed: the established X1D-E2E1 convention is
#: ``xerbs-core:canonical-formula:清肺排毒汤``. An ASCII-only grammar would
#: have rejected the one semantic identity this corpus already uses.
#:
#: What is forbidden is what would make identities ambiguous or unparseable:
#: whitespace (invisible differences), control characters, and ``|`` -- the
#: delimiter the derived relationship and safety-rule ids use.
EXTERNAL_ID_RE = re.compile(r"^[^\s\x00-\x1f\x7f|]{3,255}$")


class SemanticIdentityError(ValueError):
    """A semantic identity was missing, malformed, or could not be derived."""


def validate_external_id(external_id: str) -> str:
    """Validate and NFC-normalise one supplied semantic identity.

    Normalising here matters as much as it does for hashing: the same CJK
    identity typed on two platforms can arrive decomposed on one of them, and
    two spellings of one identity would defeat the unique index.
    """
    if not isinstance(external_id, str):
        raise SemanticIdentityError(
            "external_id must be a string, got %r" % type(external_id).__name__)
    normalised = unicodedata.normalize("NFC", external_id)
    if not EXTERNAL_ID_RE.match(normalised):
        raise SemanticIdentityError(
            "external_id %r is not a valid semantic identity; expected 3-255 "
            "characters with no whitespace, control characters or '|'"
            % (external_id,))
    return normalised


def relationship_external_id(source_external_id: str, relationship_type: str,
                             target_external_id: str) -> str:
    """``rel:<source>|<TYPE>|<target>``.

    Stable across evidence changes, metadata changes and new reviewed versions,
    because none of those appear in it. It changes only when the triple
    changes -- and a changed triple is a different relationship, not an edit.
    """
    for part in (source_external_id, target_external_id):
        if not part:
            raise SemanticIdentityError(
                "both endpoints need an external_id before a relationship can "
                "have a semantic identity; got %r -> %r"
                % (source_external_id, target_external_id))
        validate_external_id(part)
    if not relationship_type:
        raise SemanticIdentityError("relationship_type is required")
    return "rel:%s|%s|%s" % (source_external_id, relationship_type,
                             target_external_id)


def safety_rule_external_id(target_external_id: str, rule_type: str,
                            trigger_term: str) -> str:
    """``rule:<target>|<RULE_TYPE>|<nfc(trigger)>``.

    The trigger term is NFC-normalised and stripped so the same rule written on
    two platforms is one rule. It is not lower-cased: ``screen()`` already
    case-folds at match time, and folding here would merge rules that the
    authoring side deliberately distinguished.
    """
    if not target_external_id:
        raise SemanticIdentityError(
            "the target entity needs an external_id before a safety rule can "
            "have a semantic identity")
    validate_external_id(target_external_id)
    if not rule_type or not trigger_term or not trigger_term.strip():
        raise SemanticIdentityError("rule_type and trigger_term are required")
    trigger = unicodedata.normalize("NFC", trigger_term.strip())
    return "rule:%s|%s|%s" % (target_external_id, rule_type, trigger)
