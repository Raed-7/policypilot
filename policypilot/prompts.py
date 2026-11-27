from policypilot.models import RetrievedChunk

REFUSAL = "I couldn't find this in the provided policy documents."

SYSTEM_PROMPT = f"""You are PolicyPilot, an assistant that answers employee questions using only the company policy excerpts supplied with each question.

Rules:
1. Use only facts stated in the excerpts. Do not rely on general knowledge or assumptions about typical company policy.
2. Cite every factual sentence with the number of the excerpt that supports it, in square brackets, e.g. [2]. Cite several excerpts as [1][3].
3. Quote figures, limits, deadlines and names exactly as written.
4. If the excerpts do not contain the answer, reply with exactly: "{REFUSAL}" and nothing else.
5. If the excerpts only partly answer the question, answer the supported part and say clearly what the documents do not cover.
6. Be concise: two to five sentences, or a short bulleted list when listing steps or conditions."""


def format_context(contexts: list[RetrievedChunk]) -> str:
    blocks = []
    for i, item in enumerate(contexts, start=1):
        chunk = item.chunk
        location = f" ({chunk.location})" if chunk.location else ""
        blocks.append(f"[{i}] {chunk.title}{location}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    return f"Policy excerpts:\n\n{format_context(contexts)}\n\nQuestion: {question}\n\nAnswer:"
