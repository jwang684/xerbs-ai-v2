"""X1D-PATIENT-DIAGNOSIS-FORMULA-E2E-P6: bounded matching of a model pattern name.

The model names a pattern the way a clinician writes it: "风寒束表，肺气失宣" --
the pattern, a comma, and a pathomechanism. The reviewed corpus holds the
source-backed canonical name, "风寒束表". The old lookup asked whether the
model's whole string occurs inside a corpus field, so the longer, more precise
model name never matched, while a vaguer fragment such as "风寒" did.

The rule here is component equality, nothing looser:

    * the model name is split only at punctuation and whitespace, which is how
      compound pattern names are written ("A，B", "A、B", "A（B）");
    * a component matches a reviewed name or alias only if it is EQUAL to it;
    * a single trailing "证" is ignored on both sides ("风寒束表证" names the same
      pattern as "风寒束表") -- it is the pattern suffix, not a synonym.

So "风寒束表" and "风寒束表，肺气失宣" match "风寒束表"; "风寒", "风寒束肺" and
"外感风寒束表" do not. No fragment, no character overlap, no fuzzy or semantic
comparison, and no alias that the reviewed record does not itself carry.
"""

from __future__ import annotations

import re
from typing import FrozenSet, Iterable

_BOUNDARY = re.compile(r"[，,、；;。．.:：/／|｜（）()【】\[\]\s]+")
_SUFFIX = "证"


def _normalize(term: str) -> str:
    term = str(term or "").strip().lower()
    # "证" alone is not a pattern; only strip it from a longer name.
    if term.endswith(_SUFFIX) and len(term) > len(_SUFFIX):
        term = term[: -len(_SUFFIX)].strip()
    return term


def name_components(model_name: str) -> FrozenSet[str]:
    """The whole name plus each punctuation-bounded component, normalized."""
    whole = str(model_name or "").strip()
    parts = [whole, *_BOUNDARY.split(whole)]
    return frozenset(p for p in (_normalize(x) for x in parts) if p)


def matches_reviewed_name(model_name: str, reviewed_terms: Iterable[str]) -> bool:
    """True when a reviewed name/alias equals the model name or one of its components."""
    terms = {t for t in (_normalize(x) for x in reviewed_terms) if t}
    return bool(terms & name_components(model_name))
