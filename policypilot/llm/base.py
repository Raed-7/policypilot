from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, prompt: str) -> str: ...
