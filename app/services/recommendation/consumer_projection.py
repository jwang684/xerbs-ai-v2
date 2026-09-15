"""X1D-LEGACYDIAG4.2: what of the model's reasoning a patient may read.

Why this exists
---------------
LEGACYDIAG2 asked the model for a full TCM reasoning envelope and kept it.
LEGACYDIAG4's forensic audit then found the uncomfortable result: Xerbs
generates more and better clinical reasoning than the legacy system ever did,
and shows the patient almost none of it. The envelope is written, persisted,
sent over the wire to core -- and dropped on the floor.

This is the allowlist that lets some of it through.

Where the boundary sits, and why here
-------------------------------------
In ai-v2, next to the envelope, not in core and not in the browser. The
envelope carries formula hypotheses with herb names, dosages, decoction
instructions and contraindications; those are the exact things the legacy
system leaked into a purchase, and the safest place to stop them is before
they are ever copied into something consumer-shaped. So this builds a new
object out of approved fields rather than removing fields from the envelope:
a field nobody listed here cannot reach a patient by being forgotten.

What it deliberately does not do
--------------------------------
Upgrade anything. The input is MODEL_GENERATED and the output is a display
projection of MODEL_GENERATED content. There is no path from here to the
corpus, to a relationship, to verification state, to a safety verdict, to
product resolution or to purchasability, and several tests exist to prove that
the path is missing rather than merely unused.

The treatment-principle guard
-----------------------------
``treatment_principle`` is the one approved field that could smuggle a
prescription. A principle is "辛温解表，宣肺止咳"; the same field could hold
"桂枝汤加减". The guard below drops the whole field when it looks like a
formula, a herb quantity, or anything the model separately proposed as a
formula. It is a heuristic, and heuristics are wrong sometimes -- so it is
built to fail toward showing less, which is the direction where being wrong
costs a sentence rather than a patient.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.schemas.reasoning import ClinicalReasoningEnvelope

# The eight principles, in the order a practitioner reads them. Only these keys
# are projected: a model that invents a tenth dimension does not get to display
# it, and the labels are ours rather than whatever the model called them.
EIGHT_PRINCIPLE_KEYS = (
    ("exterior_interior", "表里"),
    ("cold_heat", "寒热"),
    ("deficiency_excess", "虚实"),
    ("yin_yang", "阴阳"),
)

# Formula-shaped and dose-shaped writing. Used only to SUPPRESS, never to
# extract: matching means a field is dropped, not that anything is parsed.
_FORMULA_SHAPED = re.compile(
    r"(汤|丸|散|膏|丹|饮|冲剂|颗粒|口服液|片)\s*(加减|加味|合方)?$|"
    r"(汤|丸|散|膏|丹|饮)[，,、）)]|"
    r"\d+\s*(g|克|毫升|ml|剂|付|帖)|"
    r"(煎服|水煎|温服|饭后服|每日\s*\d|每次\s*\d)"
)

# Governance tokens look like THIS_AND_THIS. They are internal vocabulary and
# mean nothing to a patient, so they are filtered out of the uncertainty list
# rather than shown as if they were advice.
_INTERNAL_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]{3,}$")

MAX_ITEMS = 8
MAX_CHARS = 600


def _clean(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:MAX_CHARS] if text else None


def _clean_list(values: Any) -> List[str]:
    out: List[str] = []
    for item in (values or []):
        text = _clean(item)
        if text:
            out.append(text)
        if len(out) >= MAX_ITEMS:
            break
    return out


def mentions_treatment_product(text: Optional[str],
                               formula_names: Optional[List[str]] = None) -> bool:
    """Whether this text looks like a prescription rather than a principle.

    Suppression only. A false positive costs one displayed sentence; a false
    negative shows a patient an unverified formula, which is the failure the
    whole governance stack exists to prevent.
    """
    if not text:
        return False
    if _FORMULA_SHAPED.search(text):
        return True
    for name in (formula_names or []):
        cleaned = (name or "").strip()
        if cleaned and cleaned in text:
            return True
    return False


def project_eight_principle(raw: Any) -> List[Dict[str, str]]:
    """Only the four dimensions, only where the model actually said something."""
    if not isinstance(raw, dict):
        return []
    out: List[Dict[str, str]] = []
    for key, label in EIGHT_PRINCIPLE_KEYS:
        value = _clean(raw.get(key))
        if value:
            out.append({"dimension": label, "finding": value})
    return out


def project_patterns(hypotheses: Any) -> List[Dict[str, Any]]:
    """Descriptive only: name, role, and the findings for and against.

    No confidence. A number beside a pattern reads as certainty however it is
    labelled, and the epistemic status is already carried by the section's own
    wording. No status field either -- it is MODEL_HYPOTHESIS by construction,
    and repeating it per row invites the UI to display it as a badge.
    """
    out: List[Dict[str, Any]] = []
    for item in (hypotheses or []):
        name = _clean(getattr(item, "name", None))
        if not name:
            continue
        role = getattr(item, "role", "secondary")
        out.append({
            "pattern_name": name,
            "role": role if role in ("primary", "secondary") else "secondary",
            "supporting_findings": _clean_list(
                getattr(item, "supporting_findings", None)),
            "contradicting_findings": _clean_list(
                getattr(item, "contradicting_findings", None)),
        })
        if len(out) >= MAX_ITEMS:
            break
    # Primary first, otherwise the model's own order. Contradicting findings are
    # never dropped: hiding the evidence against a reading to make it look
    # tidier would be the one edit that makes this section dishonest.
    out.sort(key=lambda p: 0 if p["role"] == "primary" else 1)
    return out


def project_uncertainty(flags: Any) -> List[str]:
    """Model-written caveats, minus internal governance vocabulary."""
    return [text for text in _clean_list(flags)
            if not _INTERNAL_TOKEN.match(text)]


def build_consumer_reasoning(
    envelope: Optional[ClinicalReasoningEnvelope],
    fallback_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the allowlisted projection. Pure, so its limits are testable.

    Returns {} when there is nothing worth showing, so the UI renders no
    section rather than an empty heading.
    """
    if envelope is None:
        summary = _clean(fallback_summary)
        return {"summary": summary} if summary else {}

    formula_names = [getattr(f, "name", "") or ""
                     for f in (envelope.formula_hypotheses or [])]

    principle = _clean(envelope.treatment_principle)
    if mentions_treatment_product(principle, formula_names):
        principle = None

    projection: Dict[str, Any] = {
        "summary": _clean(envelope.clinical_summary) or _clean(fallback_summary),
        "eight_principle": project_eight_principle(
            envelope.eight_principle_differentiation),
        "pattern_hypotheses": project_patterns(envelope.pattern_hypotheses),
        "pathogenesis": _clean(envelope.pathogenesis),
        "treatment_principle": principle,
        "missing_information": _clean_list(envelope.missing_information),
        "uncertainty": project_uncertainty(envelope.uncertainty_flags),
    }
    # Drop empties so the UI never renders a heading over nothing.
    return {k: v for k, v in projection.items() if v}
