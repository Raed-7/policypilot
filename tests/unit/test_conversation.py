import pytest

from policypilot.conversation import contextualise, is_follow_up
from policypilot.llm.base import LLMError


class FakeLLM:
    def __init__(self, reply: str | None = None):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.reply is None:
            raise LLMError("unavailable")
        return self.reply


@pytest.mark.parametrize(
    "question",
    ["What about part-time staff?", "And for London?", "Does that include bank holidays?", "Is it paid?"],
)
def test_detects_follow_ups(question):
    assert is_follow_up(question)


@pytest.mark.parametrize(
    "question",
    [
        "How many days of annual leave do I get?",
        "What is the mileage rate for using my own car on business trips to clients?",
        "",
    ],
)
def test_standalone_questions_are_not_follow_ups(question):
    assert not is_follow_up(question)


def test_standalone_question_is_unchanged():
    q = "How many days of annual leave do I get?"
    assert contextualise(q, ["What is the mileage rate?"]) == q


def test_follow_up_without_history_is_unchanged():
    assert contextualise("What about London?", []) == "What about London?"


def test_follow_up_is_combined_with_previous_question():
    query = contextualise("What about London?", ["What is the maximum hotel rate?"])
    assert query == "What is the maximum hotel rate? What about London?"


def test_llm_rewrite_is_used_when_available():
    llm = FakeLLM('"What is the maximum hotel rate in London?"')
    query = contextualise("What about London?", ["What is the maximum hotel rate?"], llm)
    assert query == "What is the maximum hotel rate in London?"
    assert "What is the maximum hotel rate?" in llm.prompts[0]


def test_llm_failure_falls_back_to_concatenation():
    query = contextualise("What about London?", ["What is the maximum hotel rate?"], FakeLLM(None))
    assert query == "What is the maximum hotel rate? What about London?"
