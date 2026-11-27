from policypilot.config import Settings
from policypilot.llm.base import LLMProvider
from policypilot.llm.providers import GeminiProvider, GroqProvider, OllamaProvider


def build_provider(settings: Settings) -> LLMProvider | None:
    """Return the configured LLM, or None to use the extractive answerer."""
    choice = settings.llm_provider
    if choice == "auto":
        if settings.groq_api_key:
            choice = "groq"
        elif settings.gemini_api_key:
            choice = "gemini"
        else:
            return None
    common = {"temperature": settings.llm_temperature, "timeout": settings.llm_timeout}
    if choice == "groq":
        return GroqProvider(settings.groq_api_key, settings.groq_model, **common)
    if choice == "gemini":
        return GeminiProvider(settings.gemini_api_key, settings.gemini_model, **common)
    if choice == "ollama":
        return OllamaProvider(settings.ollama_url, settings.ollama_model, **common)
    return None
