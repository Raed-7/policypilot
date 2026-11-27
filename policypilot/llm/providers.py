import httpx

from policypilot.llm.base import LLMError


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        raise LLMError(f"request failed: {exc}") from exc
    if response.status_code >= 400:
        raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


class GroqProvider:
    name = "groq"
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, timeout: float = 60.0):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self.model, self._key, self._temperature, self._timeout = model, api_key, temperature, timeout

    def complete(self, system: str, prompt: str) -> str:
        data = _post(
            self.url,
            {
                "model": self.model,
                "temperature": self._temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
            {"Authorization": f"Bearer {self._key}"},
            self._timeout,
        )
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response: {data}") from exc


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, timeout: float = 60.0):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.model, self._key, self._temperature, self._timeout = model, api_key, temperature, timeout

    def complete(self, system: str, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        data = _post(
            url,
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": self._temperature},
            },
            {"x-goog-api-key": self._key},
            self._timeout,
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response: {data}") from exc


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, temperature: float = 0.0, timeout: float = 120.0):
        self.model, self._url, self._temperature, self._timeout = model, base_url.rstrip("/"), temperature, timeout

    def complete(self, system: str, prompt: str) -> str:
        data = _post(
            f"{self._url}/api/chat",
            {
                "model": self.model,
                "stream": False,
                "options": {"temperature": self._temperature},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
            {},
            self._timeout,
        )
        try:
            return data["message"]["content"].strip()
        except KeyError as exc:
            raise LLMError(f"unexpected response: {data}") from exc
