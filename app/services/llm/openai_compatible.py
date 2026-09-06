import json

import httpx

from app.services.llm.provider import LLMProvider, ProviderResult


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
  "model_confidence": 0.0
}

If data are insufficient:
- lower confidence
- explicitly add uncertainty flags
- do not invent patient facts
- do not claim that a pattern or formula is clinically verified
- do not output a final prescription
"""


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

        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                )
            except httpx.RequestError as exc:
                raise RuntimeError(
                    f"OpenAI-compatible provider request failed: {exc}"
                ) from exc

        if response.is_error:
            error_text = response.text

            try:
                error_json = response.json()
                error_text = json.dumps(
                    error_json,
                    ensure_ascii=False,
                )
            except Exception:
                pass

            raise RuntimeError(
                f"OpenAI API error {response.status_code}: {error_text}"
            )

        try:
            response_data = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI API returned invalid JSON: {response.text}"
            ) from exc

        try:
            raw = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "OpenAI API response did not contain "
                "choices[0].message.content. "
                f"Response: {json.dumps(response_data, ensure_ascii=False)}"
            ) from exc

        if not raw:
            raise RuntimeError(
                "OpenAI API returned an empty message content."
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenAI model returned content that was not valid JSON. "
                f"Raw content: {raw}"
            ) from exc

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
        )
