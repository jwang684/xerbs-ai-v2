import json
import logging
import time

import httpx

from app.services.llm.provider import (
    FULL_REASONING,
    INTERVIEW,
    LLMProvider,
    ProviderResult,
    ProviderUsage,
)
from app.services.llm.streaming import SummaryStreamScanner

logger = logging.getLogger(__name__)


# Upstream error bodies do not leave this module.
#
# Every failure here used to be re-raised as RuntimeError carrying the provider's
# raw response text. Base44GenerationService stores that string on the generation
# row (error_message) and returns it to xerbs-core, so an upstream body would be
# persisted in the database and shipped across a service boundary.
#
# That matters as soon as a real key exists: an OpenAI 401 body echoes a partial
# key back ("Incorrect API key provided: sk-...") along with account hints. While
# LLM_PROVIDER=mock there is no key and nothing to leak, which is exactly why
# this is fixed before the switch rather than after.
#
# The status code and failure class are kept -- they are what an operator needs
# and neither is sensitive. The body is logged, not returned.
class ProviderCallError(RuntimeError):
    """A provider call failed. The message is safe to persist and return."""


def _redact(text: str, *secrets: str) -> str:
    """Defence in depth for anything that does get logged."""
    out = str(text)
    for secret in secrets:
        if secret and len(secret) >= 8:
            out = out.replace(secret, "***REDACTED***")
    return out


SYSTEM_PROMPT = """You are a TCM clinical decision-support engine for licensed practitioner review.
Return JSON only. Do not claim certainty. Do not output a final prescription. Separate model confidence from clinical verification.

Required JSON object:
{
  "summary": "concise analysis",
  "pattern_hypotheses": [
    {
      "name": "...",
      "confidence": 0.0,
      "reasoning": "..."
    }
  ],
  "formula_candidates": [
    {
      "formula_id": null,
      "name": "...",
      "confidence": 0.0,
      "rationale": "...",
      "ingredients": [],
      "safety_flags": []
    }
  ],
  "uncertainty_flags": ["..."],
  "model_confidence": 0.0,
  "clinical_reasoning": {
    "clinical_summary": "1-2 sentences: the leading pattern and its key pathogenesis",
    "tcm_diagnosis_hypotheses": ["..."],
    "eight_principle_differentiation": {
      "cold_heat": "...", "exterior_interior": "...", "deficiency_excess": "..."
    },
    "pattern_hypotheses": [
      {
        "name": "...",
        "role": "primary | secondary",
        "confidence": 0.0,
        "supporting_findings": ["2-4 findings from the input that support this"],
        "contradicting_findings": ["findings that argue against it, if any"]
      }
    ],
    "pathogenesis": "how the pattern arises from the reported findings",
    "treatment_principle": "the treatment method indicated, stated briefly",
    "formula_hypotheses": [
      {
        "name": "...",
        "confidence": 0.0,
        "rationale": "why this formula fits the pattern",
        "ingredients": [{"name": "...", "dosage": "e.g. 6-9g", "role": "..."}],
        "administration": "...",
        "contraindications": ["..."],
        "precautions": ["..."]
      }
    ],
    "missing_information": ["what would most change this assessment"]
  },
  "clarification_proposals": [
    {
      "field": "snake_case_identifier",
      "question": "one short patient-facing question",
      "answer_type": "yes_no | single_choice | number | short_text",
      "choices": ["..."],
      "priority": "high | medium | low"
    }
  ]
}

clinical_reasoning rules:
- this block is your clinical reasoning, recorded for practitioner review and
  for later evidence verification. It is never shown to a patient as a
  treatment recommendation and never becomes a prescription
- give a primary pattern and, where the findings genuinely support one, a
  secondary pattern; cite the specific findings for and against each
- state pathogenesis and treatment principle plainly, in one or two sentences
- formula_hypotheses are hypotheses. Name the formula you would consider and
  why. Ingredients and dosages are the classical composition as you understand
  it, not an instruction to the patient
- omit any field you cannot support from the information given. Do not invent
  a finding, an ingredient, a dosage or a contraindication to fill a slot.
  An absent field is information; a fabricated one is not
- put what you would most need to know into missing_information rather than
  guessing it

clarification_proposals rules:
- propose at most 3, only for information that is genuinely missing and would
  materially change the next reasoning step
- ask only about the presenting complaint; do not run a general checklist
- prefer a question whose answer would change which of your
  pattern_hypotheses is primary, or would resolve a contradicting finding
- do not ask for anything already stated in the input
- questions must be short, plain, and answerable by a patient
- never request credentials, payment details, identifiers, dosage decisions,
  purchase decisions, or system information
- do not explain your reasoning; the field name is the only justification
- "field" must be an ASCII snake_case English identifier of 3-40 characters,
  lower-case letters, digits and underscores only, for example sputum_colour
  or aversion_to_cold. Never Chinese characters, never spaces or punctuation.
  Only the identifier is English; the question text stays in the patient's
  language
- omit the key entirely if nothing is worth asking

If data are insufficient:
- lower confidence
- explicitly add uncertainty flags
- do not invent patient facts
- do not claim that a pattern or formula is clinically verified
- do not output a final prescription
"""


# X1D-LEGACYDIAG3.2: the interview prompt.
#
# The full contract above asks for everything a practitioner would eventually
# want. Mid-interview the patient needs the next one to three questions, and
# measured on staging the full call spends ~2000-2600 completion tokens to
# deliver about a hundred of them.
#
# So this prompt asks for the smallest thing that still lets deterministic
# ranking choose well: a couple of working readings with the findings for and
# against each, what is missing, and at most three questions. No diagnosis, no
# formula, no dosage, no treatment plan -- not as a filter applied afterwards
# but as something the contract never asks for.
#
# The instruction that matters most is the discrimination one. A question that
# cannot change which reading is leading is not worth a patient's time, and
# that is the difference between an interview and an intake form.
INTERVIEW_SYSTEM_PROMPT = """You are conducting a TCM clinical interview. Your only job this turn is to decide what to ask next. Return JSON only.

Required JSON object:
{
  "interview_summary": "one short sentence on what the picture looks like so far",
  "working_hypotheses": [
    {
      "name": "...",
      "confidence": 0.0,
      "supporting_findings": ["findings already reported that fit"],
      "contradicting_findings": ["findings already reported that do not fit"]
    }
  ],
  "missing_information": ["what you most need to know, in short phrases"],
  "information_sufficient": false,
  "working_differential": {
    "hypotheses": [
      {
        "pattern_name": "...",
        "standing": "PRIMARY_WORKING | PLAUSIBLE | WEAKENED | RULED_OUT_FOR_NOW",
        "supporting_evidence": [
          {"origin": "COMPLAINT"},
          {"origin": "OBSERVATION", "field": "sweating"},
          {"origin": "ANSWER", "question_field": "sputum_colour", "turn_id": 1}
        ],
        "contradicting_evidence": [],
        "unresolved_discriminators": [
          {
            "domain": "one of the allowed domain names",
            "separates": ["a pattern_name above", "another pattern_name above"],
            "also_resolved_by": ["any OTHER allowed domains that would settle
                                  this same pair equally well"],
            "if_present_supports": ["which of those a POSITIVE answer favours"],
            "if_absent_supports": ["which of those a NEGATIVE answer favours"],
            "rationale": "one short line on why this answer would move things"
          }
        ]
      }
    ],
    "evidence_gaps": [
      {"domain": "sweat", "separates": ["pattern A", "pattern B"]}
    ]
  },
  "clarification_proposals": [
    {
      "field": "snake_case_identifier",
      "question": "one short patient-facing question",
      "answer_type": "yes_no | single_choice | number | short_text",
      "choices": ["..."],
      "priority": "high | medium | low"
    }
  ]
}

Work in this order, and do not skip to the end:

STEP 1  Re-read the patient evidence and decide which readings it now supports.
STEP 2  Among the readings that are still live (PRIMARY_WORKING or PLAUSIBLE),
        find the single most important unsettled competition -- the two whose
        difference matters most for what happens next.
STEP 3  For that competition, name the smallest set of findings the patient has
        NOT yet reported that would actually tell those two apart.
STEP 4  Map each of those findings to one of the allowed domain names below.
STEP 5  Write them as unresolved_discriminators, each naming the readings it
        separates and what a positive and a negative answer would each favour.
STEP 6  Only now write clarification_proposals, one per discriminator you
        found, in that order.

The question you are answering is NOT "what do I not know yet?" -- a checklist
can produce that and it is nearly the same list whatever the patient said. The
question is "which uncertainty BETWEEN MY OWN CURRENT READINGS matters most?"

A useful question is one whose different answers would move your readings in
different directions. If a finding would support the same reading whether it is
present or absent, it confirms rather than separates, and it is not worth one of
your three slots. Prefer the question that would most change the ordering.

How to choose the questions:
- give at most 3, and fewer when fewer would do. Two good questions beat four
- ask only about this complaint. Do not run a general intake checklist
- never ask about anything the input already states, and never about a domain
  listed under ALREADY_KNOWN. Read the input carefully first: information
  supplied by the patient is already known
- do not invent findings. If something was not reported, it is unknown, not absent
- questions must be short, plain, and answerable by a patient

working_hypotheses rules:
- two or three at most. They are provisional readings used to pick the next
  question, not a diagnosis
- cite findings actually present in the input, for and against each
- if the input supports only one reading, say so and give one

What this turn must NOT contain:
- no formula, no herb, no dosage, no administration, no treatment plan
- no final syndrome diagnosis and no claim that anything is verified
- no advice, no reassurance, no safety clearance

"field" must be an ASCII snake_case English identifier of 3-40 characters,
lower-case letters, digits and underscores only, for example sputum_colour or
aversion_to_cold. Never Chinese characters, never spaces or punctuation. Only
the identifier is English; the question text stays in the patient's language.

Set information_sufficient true only when another question would add nothing
material. It is advisory: deterministic governance decides what happens next.

working_differential rules:
- this is your working set of candidate readings, not a diagnosis. Two or three
  is usually right
- standing is ordinal and has no number attached. Use PRIMARY_WORKING for the
  reading that currently fits best, WEAKENED for one the evidence argues
  against, RULED_OUT_FOR_NOW for one that is currently implausible but could
  return if new information appeared. Nothing here is final
- evidence is a CITATION, never a statement. Cite only:
    {"origin": "COMPLAINT"}                         the patient's own words
    {"origin": "OBSERVATION", "field": "..."}       a field the patient filled in
    {"origin": "ANSWER", "question_field": "..."}   a question the patient answered
  A citation that does not resolve to something the patient actually supplied
  is discarded. Do not describe findings the patient never reported -- if a
  tongue or pulse was never described to you, it is unknown, not normal
- unresolved_discriminators are what you still need in order to tell your
  candidates apart. This is what the next question should be for. Each one must:
    * name in "separates" at least TWO DIFFERENT readings from your own
      hypotheses list above, both of which are still live. Naming a reading you
      have not listed, or naming the same one twice, describes a competition
      that is not happening and is discarded
    * give "if_present_supports" and "if_absent_supports" from those same
      names, and they must not be identical. If both answers favour the same
      reading, the finding confirms something rather than separating anything,
      and it will not earn a question slot
    * this is a demonstration, not a formality: it is how you show that asking
      would actually change your mind
    * list in "also_resolved_by" every OTHER allowed domain that would settle
      the same pair about as well. Name them even though you are not asking
      about them: which of the candidates is actually put to the patient is
      decided after you, and it is decided better when it can see all of them.
      Leave it empty only when nothing else would genuinely do
- the names you write in separates, also_resolved_by, if_present_supports and
  if_absent_supports must be copied EXACTLY from your own pattern_name values
  above. Not a paraphrase, not a translation, not a near-synonym, and never a
  reading you did not list. A name that is not one of yours describes a
  competition that is not happening and the whole discriminator is discarded.
  If a new reading is warranted, add it to the hypotheses list first; it can
  take part in a discriminator on a later turn
- if only ONE reading is live, say so and do not invent a second one to have
  something to separate. Report what would strengthen or weaken the one you
  have, leave unresolved_discriminators empty, and let the missing_information
  list carry the rest
- evidence_gaps name a domain and the two readings it would separate
- every "domain" you write, in unresolved_discriminators and evidence_gaps
  alike, MUST be one of these fifteen names, spelled exactly:
    cold_heat  sweat  head_body  excretion  diet  chest_abdomen  ear_eye
    thirst  past_illness  cause  sleep  onset_duration  nose  throat  sputum
  Use the nearest one rather than inventing a name: 鼻塞/流涕 is nose, 咽痛 is
  throat, 痰的颜色或多少 is sputum, 口渴 is thirst, 怕冷发热 is cold_heat. A
  domain outside this list cannot be acted on and is discarded

If PREVIOUS WORKING DIFFERENTIAL is supplied:
- it is your own earlier reasoning, not established fact and not patient
  information. Re-evaluate every entry against the patient evidence as it
  stands now
- you may strengthen, weaken, replace or drop any of them, and you may add
  alternatives that did not occur to you before
- do not carry a hypothesis forward merely because it was there before, and do
  not raise a standing without citing evidence you were not already citing
- agreeing with yourself is not evidence. If the answers you have since
  received fit a different reading better, lead with that one and say what
  moved. A differential that never changes across turns has learned nothing
- consistency with your earlier self is worth nothing here. Only the patient
  evidence counts

If ALREADY_KNOWN is supplied it lists domains the patient has already answered.
Do not propose a discriminator or a question in any of them -- that answer is
already on record, and asking again spends a slot and the patient's patience
for nothing.
"""


def _coerce_int(value) -> int | None:
    """Accept only a clean non-negative integer; anything else is unknown."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _extract_usage(response_data: dict) -> ProviderUsage | None:
    """Read the provider's own token accounting, or return None.

    Nothing is inferred. If the provider did not report a count, the count
    stays None -- estimating tokens from text length would produce a number
    indistinguishable from a measurement, and it would be wrong.

    The *_details objects are newer and model-dependent, so their absence is
    normal and must not be an error.
    """
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        return None

    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")

    result = ProviderUsage(
        prompt_tokens=_coerce_int(usage.get("prompt_tokens")),
        completion_tokens=_coerce_int(usage.get("completion_tokens")),
        total_tokens=_coerce_int(usage.get("total_tokens")),
        cached_input_tokens=_coerce_int(
            prompt_details.get("cached_tokens")
            if isinstance(prompt_details, dict) else None),
        reasoning_tokens=_coerce_int(
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, dict) else None),
    )
    return None if result.is_empty() else result


def _extract_clinical_reasoning(data: dict) -> dict:
    """Pull the reasoning envelope out of the model response, defensively.

    Returns a plain dict; the typed envelope is built later, where a
    validation failure can be contained. Richer contracts raise the chance of
    a malformed section, and a malformed *reasoning* section must never cost
    an otherwise valid governed result -- the clinical decision does not
    depend on it.
    """
    block = data.get("clinical_reasoning")
    return block if isinstance(block, dict) else {}


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, *, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model.strip()

    async def generate_interview(
        self,
        *,
        text_input: str,
        symptoms: list[str],
        language: str,
        on_display_text=None,
        carry_state: dict | None = None,
        known_domains: list | None = None,
    ) -> ProviderResult:
        """The small call. Same transport, different contract.

        Returns a ProviderResult so every existing mechanism -- usage
        accounting, latency, clarification proposals, the CLARIFY3 rejection
        record -- keeps working without a second pipeline. What differs is what
        is absent: formula_candidates and clinical_reasoning are empty here by
        construction, not by filtering.
        """
        # X1D-LEGACYDIAG4.4B: two sections, never one prose history.
        #
        # Flattening patient evidence and prior model reasoning into a single
        # narrative destroys provenance: the model can no longer tell what it
        # was told from what it previously guessed, and neither can anyone
        # reading the prompt afterwards.
        payload: dict = {
            "PATIENT_EVIDENCE": {
                "note": "Supplied by the patient. Authoritative input.",
                "text_input": text_input,
                "symptoms": symptoms,
                "language": language,
            },
        }
        if known_domains:
            # X1D-LEGACYDIAG4.5: domains the patient has already answered.
            # Server-side suppression is still the authority; this only stops
            # the model spending one of three proposals on a settled question.
            payload["ALREADY_KNOWN"] = {
                "note": ("The patient has already answered these domains. Do "
                         "not ask about them again."),
                "domains": sorted(known_domains),
            }
        if carry_state is not None:
            payload["PREVIOUS_WORKING_DIFFERENTIAL"] = {
                "note": ("Your own earlier reasoning. NOT established fact and "
                         "NOT patient information. Re-evaluate it."),
                "state": carry_state.get("prior"),
            }
            payload["CITABLE_EVIDENCE"] = {
                "note": ("The only things you may cite. A citation outside "
                         "this list is discarded."),
                **(carry_state.get("resolvable_evidence") or {}),
            }
        content = json.dumps(payload, ensure_ascii=False)
        data, usage, provider_latency_ms, first_delta_ms = await self._call(
            INTERVIEW_SYSTEM_PROMPT, content, on_display_text)

        raw_proposals = data.get("clarification_proposals")
        if not isinstance(raw_proposals, list):
            raw_proposals = []
        well_formed = [p for p in raw_proposals if isinstance(p, dict)]

        interview = {
            key: data.get(key)
            for key in ("interview_summary", "working_hypotheses",
                        "missing_information", "information_sufficient",
                        "working_differential")
            if data.get(key) is not None
        }

        return ProviderResult(
            summary=str(data.get("interview_summary") or ""),
            # Deliberately empty. An interview turn proposes no pattern to the
            # corpus and no formula to anyone.
            pattern_hypotheses=[],
            formula_candidates=[],
            uncertainty_flags=[],
            model_confidence=0.0,
            provider="openai-compatible",
            model=self.model,
            usage=usage,
            provider_latency_ms=provider_latency_ms,
            clarification_proposals=well_formed,
            clarification_proposals_discarded=len(raw_proposals) - len(well_formed),
            inference_purpose=INTERVIEW,
            interview=interview,
            first_delta_ms=first_delta_ms,
        )

    async def _call(self, system_prompt: str, content, on_display_text=None):
        """One provider round trip. Shared so both contracts use the same
        transport, the same error handling and the same redaction.

        X1D-LEGACYDIAG4.1: when ``on_display_text`` is supplied the call is
        streamed and the callback receives fragments of the response's summary
        field as they arrive. Everything else is unchanged -- the body is still
        accumulated in full, still parsed with json.loads, and still validated
        before it becomes anything. Streaming affects when the patient sees
        text, never what is true.

        Returns (data, usage, provider_latency_ms, first_delta_ms).
        provider_latency_ms keeps its existing meaning -- the whole provider
        call -- so the TELEMETRY1 series stays comparable across the change.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        if on_display_text is None:
            return await self._call_blocking(url, headers, payload)
        return await self._call_streaming(url, headers, payload, on_display_text)

    async def _call_blocking(self, url, headers, payload):
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.RequestError as exc:
                logger.error("LLM transport failure (%s): %s",
                             type(exc).__name__,
                             _redact(str(exc), self.api_key))
                raise ProviderCallError(
                    f"LLM provider unreachable ({type(exc).__name__})."
                ) from exc
        provider_latency_ms = (time.monotonic() - started) * 1000.0

        if response.is_error:
            logger.error("LLM provider returned %s: %s", response.status_code,
                         _redact(response.text, self.api_key)[:2000])
            raise ProviderCallError(
                f"LLM provider returned HTTP {response.status_code}.")

        try:
            response_data = response.json()
        except Exception as exc:
            logger.error("LLM provider returned a non-JSON body")
            raise ProviderCallError(
                "LLM provider returned a response that was not JSON.") from exc

        try:
            raw = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("LLM response lacked choices[0].message.content")
            raise ProviderCallError(
                "LLM provider response did not contain a message.") from exc

        return (self._decode(raw), _extract_usage(response_data),
                provider_latency_ms, None)

    async def _call_streaming(self, url, headers, payload, on_display_text):
        """Same request, read incrementally.

        stream_options.include_usage is required, not optional: without it a
        streamed call reports no token counts at all and the TELEMETRY1 cost
        record would silently become empty. Losing accounting to gain a
        progress bar is not a trade worth making, so if the final usage chunk
        never arrives the usage is reported as unknown rather than zero.
        """
        streamed = dict(payload)
        streamed["stream"] = True
        streamed["stream_options"] = {"include_usage": True}

        scanner = SummaryStreamScanner()
        body_parts: list[str] = []
        usage_payload = None
        first_delta_ms = None
        started = time.monotonic()

        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                async with client.stream("POST", url, headers=headers,
                                         json=streamed) as response:
                    if response.is_error:
                        detail = await response.aread()
                        logger.error(
                            "LLM provider returned %s: %s",
                            response.status_code,
                            _redact(detail.decode("utf-8", "replace"),
                                    self.api_key)[:2000])
                        raise ProviderCallError(
                            f"LLM provider returned HTTP {response.status_code}.")

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            event = json.loads(chunk)
                        except json.JSONDecodeError:
                            # One unreadable transport frame is not a clinical
                            # failure; the accumulated body is validated later
                            # and will fail then if it is genuinely broken.
                            continue

                        if isinstance(event.get("usage"), dict):
                            usage_payload = event

                        for choice in event.get("choices") or []:
                            piece = (choice.get("delta") or {}).get("content")
                            if not piece:
                                continue
                            if first_delta_ms is None:
                                first_delta_ms = (
                                    time.monotonic() - started) * 1000.0
                            body_parts.append(piece)
                            fresh = scanner.feed("".join(body_parts))
                            if fresh:
                                try:
                                    on_display_text(fresh)
                                except Exception:  # noqa: BLE001
                                    # A display sink must never be able to
                                    # interrupt generation.
                                    pass
            except ProviderCallError:
                raise
            except httpx.RequestError as exc:
                logger.error("LLM transport failure (%s): %s",
                             type(exc).__name__,
                             _redact(str(exc), self.api_key))
                raise ProviderCallError(
                    f"LLM provider unreachable ({type(exc).__name__})."
                ) from exc

        provider_latency_ms = (time.monotonic() - started) * 1000.0
        raw = "".join(body_parts)
        return (self._decode(raw),
                _extract_usage(usage_payload) if usage_payload else None,
                provider_latency_ms, first_delta_ms)

    def _decode(self, raw):
        """The one place a provider body becomes an object. Unchanged rules."""
        if not raw:
            raise ProviderCallError("LLM provider returned empty message content.")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned non-JSON content: %s", str(raw)[:2000])
            raise ProviderCallError(
                "LLM provider returned content that was not valid JSON."
            ) from exc
        if not isinstance(data, dict):
            raise ProviderCallError(
                "LLM provider returned JSON that was not an object.")
        return data

    async def generate_recommendation(
        self,
        *,
        text_input: str,
        symptoms: list[str],
        goals: list[str],
        constraints: list[str],
        image_data: str | None,
        language: str,
        on_display_text=None,
    ) -> ProviderResult:

        user_payload = {
            "text_input": text_input,
            "symptoms": symptoms,
            "goals": goals,
            "constraints": constraints,
            "language": language,
            "image_supplied": bool(image_data),
        }

        content: str | list[dict] = json.dumps(
            user_payload,
            ensure_ascii=False,
        )

        if image_data:
            content = [
                {
                    "type": "text",
                    "text": json.dumps(
                        user_payload,
                        ensure_ascii=False,
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    },
                },
            ]

        # X1D-LEGACYDIAG4.1: one transport for both contracts.
        #
        # This block used to carry its own copy of the request, the error
        # handling and the redaction. Routing it through _call means the
        # streaming path, the timeout, the redaction rules and the decode
        # rules are shared -- and that the full-reasoning turn, which is the
        # 15-19s one, can stream its summary without a second code path
        # existing to drift away from the first.
        data, usage, provider_latency_ms, first_delta_ms = await self._call(
            SYSTEM_PROMPT, content, on_display_text)


        raw_reasoning = _extract_clinical_reasoning(data)

        raw_proposals = data.get("clarification_proposals")
        if not isinstance(raw_proposals, list):
            raw_proposals = []
        well_formed = [p for p in raw_proposals if isinstance(p, dict)]
        discarded_proposals = len(raw_proposals) - len(well_formed)

        pattern_hypotheses = data.get("pattern_hypotheses", [])
        formula_candidates = data.get("formula_candidates", [])
        uncertainty_flags = data.get("uncertainty_flags", [])

        if not isinstance(pattern_hypotheses, list):
            pattern_hypotheses = []

        if not isinstance(formula_candidates, list):
            formula_candidates = []

        if not isinstance(uncertainty_flags, list):
            uncertainty_flags = [
                "INVALID_MODEL_UNCERTAINTY_FLAGS_FORMAT"
            ]

        try:
            model_confidence = float(
                data.get("model_confidence", 0.0)
            )
        except (TypeError, ValueError):
            model_confidence = 0.0
            uncertainty_flags.append(
                "INVALID_MODEL_CONFIDENCE_FORMAT"
            )

        model_confidence = max(
            0.0,
            min(1.0, model_confidence),
        )

        return ProviderResult(
            summary=str(data.get("summary", "")),
            pattern_hypotheses=pattern_hypotheses,
            formula_candidates=formula_candidates,
            uncertainty_flags=list(
                dict.fromkeys(uncertainty_flags)
            ),
            model_confidence=model_confidence,
            provider="openai-compatible",
            model=self.model,
            usage=usage,
            provider_latency_ms=provider_latency_ms,
            first_delta_ms=first_delta_ms,
            clarification_proposals=well_formed,
            clarification_proposals_discarded=discarded_proposals,
            clinical_reasoning=raw_reasoning,
        )
