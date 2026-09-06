import json
import httpx
from app.services.llm.provider import LLMProvider, ProviderResult


SYSTEM_PROMPT = """You are a TCM clinical decision-support engine for licensed practitioner review.
Return JSON only. Do not claim certainty. Do not output a final prescription. Separate model confidence from clinical verification.
Required JSON object:
{
  "summary": "concise analysis",
  "pattern_hypotheses": [{"name":"...","confidence":0.0,"reasoning":"..."}],
  "formula_candidates": [{"formula_id":null,"name":"...","confidence":0.0,"rationale":"...","ingredients":[],"safety_flags":[]}],
  "uncertainty_flags": ["..."],
  "model_confidence": 0.0
}
If data are insufficient, lower confidence and explicitly add uncertainty flags. Do not invent patient facts."""


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, *, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def generate_recommendation(self, *, text_input: str, symptoms: list[str], goals: list[str], constraints: list[str], image_data: str | None, language: str) -> ProviderResult:
        user_payload = {
            "text_input": text_input,
            "symptoms": symptoms,
            "goals": goals,
            "constraints": constraints,
            "language": language,
            "image_supplied": bool(image_data),
        }
        content: str | list[dict] = json.dumps(user_payload, ensure_ascii=False)
        if image_data:
            content = [
                {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ]
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
        data = json.loads(raw)
        return ProviderResult(
            summary=str(data.get("summary", "")),
            pattern_hypotheses=list(data.get("pattern_hypotheses", [])),
            formula_candidates=list(data.get("formula_candidates", [])),
            uncertainty_flags=list(data.get("uncertainty_flags", [])),
            model_confidence=float(data.get("model_confidence", 0.0)),
            provider="openai-compatible",
            model=self.model,
        )
