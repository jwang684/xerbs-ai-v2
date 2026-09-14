import json
import logging
import time

import httpx

from app.services.llm.provider import (
    LLMProvider,
    ProviderResult,
    ProviderUsage,
)

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

clarification_proposals rules:
- propose at most 3, only for information that is genuinely missing and would
  materially change the next reasoning step
- ask only about the presenting complaint; do not run a general checklist
- do not ask for anything already stated in the input
- questions must be short, plain, and answerable by a patient
- never request credentials, payment details, identifiers, dosage decisions,
  purchase decisions, or system information
- do not explain your reasoning; the field name is the only justification
- omit the key entirely if nothing is worth asking

If data are insufficient:
- lower confidence
- explicitly add uncertainty flags
- do not invent patient facts
- do not claim that a pattern or formula is clinically verified
- do not output a final prescription
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


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, *, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model.strip()

    async def generate_recommendation(
        self,
        *,
        text_input: str,
        symptoms: list[str],
        goals: list[str],
        constraints: list[str],
        image_data: str | None,
        language: str,
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

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "response_format": {
                "type": "json_object"
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        # X1D-TELEMETRY1: time the external call only, on a monotonic clock.
        #
        # Monotonic because wall-clock time can step backwards (NTP), which
        # would produce negative or absurd durations in the cost record. The
        # span covers the provider HTTP call alone -- not corpus resolution,
        # safety or persistence -- so provider_latency_ms stays comparable
        # across deployments and is never confused with end-to-end time.
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                )
            except httpx.RequestError as exc:
                # str(exc) can carry the full request URL; log it, do not return it.
                logger.error(
                    "LLM transport failure (%s): %s",
                    type(exc).__name__,
                    _redact(str(exc), self.api_key),
                )
                raise ProviderCallError(
                    "LLM provider unreachable "
                    f"({type(exc).__name__})."
                ) from exc

        provider_latency_ms = (time.monotonic() - started) * 1000.0

        if response.is_error:
            # A 401 body echoes part of the key back. Log it redacted; the
            # caller gets the status code, which is the actionable part.
            logger.error(
                "LLM provider returned %s: %s",
                response.status_code,
                _redact(response.text, self.api_key)[:2000],
            )
            raise ProviderCallError(
                f"LLM provider returned HTTP {response.status_code}."
            )

        try:
            response_data = response.json()
        except Exception as exc:
            logger.error("LLM provider returned a non-JSON body")
            raise ProviderCallError(
                "LLM provider returned a response that was not JSON."
            ) from exc

        try:
            raw = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("LLM response lacked choices[0].message.content")
            raise ProviderCallError(
                "LLM provider response did not contain a message."
            ) from exc

        if not raw:
            raise ProviderCallError(
                "LLM provider returned empty message content."
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            # raw is model output derived from patient text; log, do not return.
            logger.error("LLM returned non-JSON content: %s", str(raw)[:2000])
            raise ProviderCallError(
                "LLM provider returned content that was not valid JSON."
            ) from exc

        usage = _extract_usage(response_data)

        raw_proposals = data.get("clarification_proposals")
        if not isinstance(raw_proposals, list):
            raw_proposals = []

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
            clarification_proposals=[p for p in raw_proposals if isinstance(p, dict)],
        )
