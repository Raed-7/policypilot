import httpx
import pytest

from policypilot.config import Settings
from policypilot.llm import LLMError, build_provider
from policypilot.llm.providers import GeminiProvider, GroqProvider, OllamaProvider


class FakeResponse:
    def __init__(self, status: int, payload: dict):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.fixture
def capture(monkeypatch):
    calls = {}

    def install(status: int, payload: dict):
        def fake_post(url, json, headers, timeout):
            calls.update(url=url, json=json, headers=headers)
            return FakeResponse(status, payload)

        monkeypatch.setattr(httpx, "post", fake_post)
        return calls

    return install


def test_groq_request_and_response(capture):
    calls = capture(200, {"choices": [{"message": {"content": " Answer [1] "}}]})
    out = GroqProvider("key", "llama").complete("sys", "prompt")
    assert out == "Answer [1]"
    assert calls["headers"]["Authorization"] == "Bearer key"
    assert calls["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert calls["json"]["temperature"] == 0.0


def test_gemini_request_and_response(capture):
    calls = capture(200, {"candidates": [{"content": {"parts": [{"text": "Hello "}, {"text": "[1]"}]}}]})
    assert GeminiProvider("key", "gemini-x").complete("sys", "prompt") == "Hello [1]"
    assert calls["url"].endswith("/models/gemini-x:generateContent")
    assert calls["headers"]["x-goog-api-key"] == "key"
    assert calls["json"]["systemInstruction"]["parts"][0]["text"] == "sys"


def test_ollama_request_and_response(capture):
    calls = capture(200, {"message": {"content": "Hi"}})
    assert OllamaProvider("http://ollama:11434/", "llama3.2").complete("s", "p") == "Hi"
    assert calls["url"] == "http://ollama:11434/api/chat"
    assert calls["json"]["stream"] is False


def test_http_error_raises_llm_error(capture):
    capture(429, {"error": "rate limited"})
    with pytest.raises(LLMError, match="429"):
        GroqProvider("key", "m").complete("s", "p")


def test_malformed_response_raises_llm_error(capture):
    capture(200, {"unexpected": True})
    with pytest.raises(LLMError):
        GeminiProvider("key", "m").complete("s", "p")


def test_missing_key_raises():
    with pytest.raises(LLMError):
        GroqProvider("", "m")


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, None),
        ({"groq_api_key": "g"}, "groq"),
        ({"gemini_api_key": "x"}, "gemini"),
        ({"groq_api_key": "g", "gemini_api_key": "x"}, "groq"),
        ({"llm_provider": "ollama"}, "ollama"),
        ({"llm_provider": "extractive", "groq_api_key": "g"}, None),
    ],
)
def test_provider_selection(overrides, expected):
    provider = build_provider(Settings(_env_file=None, **overrides))
    assert (provider.name if provider else None) == expected
