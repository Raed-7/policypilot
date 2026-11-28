"""Turn follow-up questions into standalone search queries using the conversation so far."""

import re

from policypilot.llm.base import LLMError, LLMProvider

_LEADING_CUES = re.compile(
    r"^(and|also|or|but|so|then|what about|how about|what if|same for|is that|is it|does that|does it|do they|"
    r"are they|can i also|and if|what else|anything else|who do i|where do i|how do i)\b",
    re.IGNORECASE,
)
_REFERENCES = re.compile(r"\b(it|its|that|this|those|these|they|them|there|then|one|ones|same|any)\b", re.IGNORECASE)
_WORDS = re.compile(r"[\w£$%'-]+")

REWRITE_PROMPT = """Rewrite the user's latest question so it can be understood without the conversation.
Keep the user's wording where possible, resolve references like "it", "that" or "what about", and add nothing else.
Reply with the rewritten question only.

Earlier questions:
{history}

Latest question: {question}
Standalone question:"""


def is_follow_up(question: str) -> bool:
    """Heuristic: short questions that open with a continuation cue or lean on a reference word."""
    words = _WORDS.findall(question)
    if not words or len(words) > 12:
        return False
    if _LEADING_CUES.match(question.strip()):
        return True
    return len(words) <= 7 and bool(_REFERENCES.search(question))


def contextualise(question: str, history: list[str] | None, llm: LLMProvider | None = None) -> str:
    """Return the query to search with: the question itself, or a version that carries the earlier context."""
    history = [h.strip() for h in (history or []) if h and h.strip()]
    if not history or not is_follow_up(question):
        return question
    if llm is not None:
        try:
            rewritten = llm.complete(
                "You rewrite follow-up questions as standalone questions.",
                REWRITE_PROMPT.format(history="\n".join(f"- {h}" for h in history[-3:]), question=question),
            ).strip().strip('"')
            if rewritten and len(rewritten) < 400:
                return rewritten
        except LLMError:
            pass
    return f"{history[-1]} {question}"
