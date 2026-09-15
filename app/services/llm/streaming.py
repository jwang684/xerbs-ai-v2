"""X1D-LEGACYDIAG4.1: pull one display-only field out of a JSON stream.

Why this exists
---------------
LEGACYDIAG4 found that the legacy system's biggest UX advantage was not better
reasoning -- it was streaming. Its final call streamed Markdown, so the first
content appeared in about a second while the document built up in front of the
patient. Xerbs generates more, and better, clinical content than the legacy
system ever did, and then shows the patient a blank screen for 4 to 19 seconds
before revealing all of it at once.

The obstacle is that ai-v2 asks the provider for a JSON object, and half a JSON
object is not a clinical result. So this scanner does exactly one thing: it
follows the growing response and reports the text of the top-level "summary"
(or "interview_summary") string as it arrives.

That field and no other. ``summary`` is already projected to the consumer by
consumer_diagnosis_adapter.project_for_consumer, so streaming it shows the
patient something they were going to be shown anyway, a few seconds earlier.
Streaming anything else would be a back door around LEGACYDIAG4.2, which owns
the question of what more a patient may see.

What this is not
----------------
Not a JSON parser, and not a regex over partial JSON. It is a small state
machine that tracks one thing -- "am I inside the value of a top-level key I
care about?" -- and it is deliberately incapable of producing a clinical
object. The authoritative result still comes from json.loads over the complete
body, validated as it always was. Nothing here can make anything true.

Failure direction
-----------------
Fail silent. If the scanner cannot confidently tell where it is, it stops
emitting for the rest of the response and the turn degrades to progress-only
streaming with the final result unchanged. A display aid must never be able to
cost a diagnosis, and it must never guess: emitting the wrong text to a patient
is worse than emitting none.
"""

from __future__ import annotations

from typing import Optional

# Only these keys may ever be streamed, and only at the top level of the
# object. An allowlist rather than a blocklist: a new field added to the
# contract later is not streamable until someone decides it should be.
STREAMABLE_KEYS = ("summary", "interview_summary")

# Beyond this the scanner gives up rather than keep hunting through a body that
# clearly is not the shape it expects.
MAX_SCAN_CHARS = 200_000


class SummaryStreamScanner:
    """Follows a growing JSON body and yields new display text.

    Fed the whole accumulated body each time (not just the delta), because that
    is what the provider loop naturally has and it keeps the scanner free of
    assumptions about chunk boundaries -- a UTF-8 character or an escape
    sequence split across two chunks would otherwise be a correctness problem.
    """

    def __init__(self) -> None:
        self._emitted = 0          # characters of the value already reported
        self._value_start: Optional[int] = None
        self._disabled = False

    @property
    def active(self) -> bool:
        return not self._disabled

    def feed(self, body: str) -> str:
        """Return the newly available display text, or "" if there is none."""
        if self._disabled or not body:
            return ""
        if len(body) > MAX_SCAN_CHARS:
            self._disabled = True
            return ""
        try:
            if self._value_start is None:
                self._value_start = self._find_value_start(body)
                if self._value_start is None:
                    return ""
            text, complete = self._read_value(body, self._value_start)
            if text is None:
                return ""
            fresh = text[self._emitted:]
            self._emitted = len(text)
            if complete:
                # The field is finished; there is nothing further to stream and
                # continuing to scan would only risk wandering into other keys.
                self._disabled = True
            return fresh
        except Exception:  # noqa: BLE001 - a display aid never raises
            self._disabled = True
            return ""

    # ------------------------------------------------------------------
    def _find_value_start(self, body: str) -> Optional[int]:
        """Index just after the opening quote of a streamable key's value.

        Requires the key to sit at depth 1 -- directly inside the root object.
        A key of the same name nested inside some other structure is ignored,
        which is the whole reason this walks the body rather than searching it.
        """
        depth = 0
        index = 0
        length = len(body)
        while index < length:
            char = body[index]
            if char == '"':
                token, end = self._read_string(body, index)
                if token is None:
                    return None                      # truncated mid-key
                if depth == 1 and token in STREAMABLE_KEYS:
                    cursor = end
                    while cursor < length and body[cursor] in " \t\r\n":
                        cursor += 1
                    if cursor >= length:
                        return None                  # colon not arrived yet
                    if body[cursor] != ":":
                        index = end
                        continue
                    cursor += 1
                    while cursor < length and body[cursor] in " \t\r\n":
                        cursor += 1
                    if cursor >= length:
                        return None                  # value not arrived yet
                    if body[cursor] != '"':
                        # Not a string value. Not ours to stream.
                        self._disabled = True
                        return None
                    return cursor + 1
                index = end
                continue
            if char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
            index += 1
        return None

    def _read_string(self, body: str, quote_index: int):
        """Decode the JSON string starting at quote_index. (value, end) or (None, _)."""
        out = []
        index = quote_index + 1
        length = len(body)
        while index < length:
            char = body[index]
            if char == "\\":
                if index + 1 >= length:
                    return None, index               # escape split across chunks
                nxt = body[index + 1]
                if nxt == "u":
                    if index + 6 > length:
                        return None, index
                    try:
                        out.append(chr(int(body[index + 2:index + 6], 16)))
                    except ValueError:
                        return None, index
                    index += 6
                    continue
                out.append({"n": "\n", "t": "\t", "r": "\r", "b": "\b",
                            "f": "\f"}.get(nxt, nxt))
                index += 2
                continue
            if char == '"':
                return "".join(out), index + 1
            out.append(char)
            index += 1
        return None, index                           # string still open

    def _read_value(self, body: str, start: int):
        """Text of the value so far, and whether it has closed."""
        out = []
        index = start
        length = len(body)
        while index < length:
            char = body[index]
            if char == "\\":
                if index + 1 >= length:
                    # Escape split across chunks: report what is settled and
                    # wait. Reporting the backslash would show it to a patient.
                    return "".join(out), False
                nxt = body[index + 1]
                if nxt == "u":
                    if index + 6 > length:
                        return "".join(out), False
                    try:
                        out.append(chr(int(body[index + 2:index + 6], 16)))
                    except ValueError:
                        self._disabled = True
                        return None, False
                    index += 6
                    continue
                out.append({"n": "\n", "t": "\t", "r": "\r", "b": "\b",
                            "f": "\f"}.get(nxt, nxt))
                index += 2
                continue
            if char == '"':
                return "".join(out), True
            out.append(char)
            index += 1
        return "".join(out), False
