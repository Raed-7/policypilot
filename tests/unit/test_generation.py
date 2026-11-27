from policypilot.generation import ExtractiveGenerator, LLMGenerator, extract_citations, is_refusal, strip_citations
from policypilot.models import Chunk, RetrievedChunk
from policypilot.prompts import REFUSAL, SYSTEM_PROMPT, build_user_prompt


def _ctx(text: str, page: int | None = None, heading: str = "") -> RetrievedChunk:
    chunk = Chunk(chunk_id=text[:8], doc_id="d", source="s.pdf", title="Expenses", text=text, position=0, page=page, heading=heading)
    return RetrievedChunk(chunk=chunk, score=0.8, dense_score=0.8)


def test_extract_citations_dedupes_and_ignores_out_of_range():
    assert extract_citations("A [2]. B [1][2]. C [1, 3]. D [9].", n_contexts=3) == [2, 1, 3]


def test_strip_citations():
    assert strip_citations("Rate is 45p [1]. Then 25p [2][3].") == "Rate is 45p. Then 25p."


def test_refusal_detection():
    assert is_refusal(REFUSAL)
    assert is_refusal(f'"{REFUSAL}"')
    assert not is_refusal("The rate is 45p per mile [1].")


def test_prompt_numbers_contexts_with_locations():
    prompt = build_user_prompt("What is the rate?", [_ctx("Mileage is 45p.", page=2, heading="Mileage")])
    assert "[1] Expenses (p. 2, Mileage)\nMileage is 45p." in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert REFUSAL in SYSTEM_PROMPT


class FakeProvider:
    name, model = "fake", "fake-1"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.reply


def test_llm_generator_passes_context_to_provider():
    provider = FakeProvider("It is 45p [1].")
    result = LLMGenerator(provider).generate("Rate?", [_ctx("Mileage is 45p per mile.")])
    assert result.text == "It is 45p [1]."
    assert result.provider == "fake"
    assert "Mileage is 45p per mile." in provider.calls[0][1]


def test_llm_generator_skips_call_without_context():
    provider = FakeProvider("unused")
    assert LLMGenerator(provider).generate("Rate?", []).text == REFUSAL
    assert provider.calls == []


def test_extractive_answer_quotes_best_sentence_with_citation(embedder):
    contexts = [
        _ctx("Hotels are capped at £150 per night in London. Book them through TripDesk."),
        _ctx("Mileage is paid at 45p per mile for business journeys."),
    ]
    result = ExtractiveGenerator(embedder, margin=0.0).generate("What is the mileage rate per mile?", contexts)
    assert result.text == "Mileage is paid at 45p per mile for business journeys. [2]"


def test_extractive_strips_heading_prefix(embedder):
    ctx = _ctx("Mileage and taxis: You can claim 45p per mile.", heading="Mileage and taxis")
    result = ExtractiveGenerator(embedder).generate("mileage claim", [ctx])
    assert result.text.startswith("You can claim 45p per mile.")


def test_extractive_without_usable_sentences_refuses(embedder):
    assert ExtractiveGenerator(embedder).generate("anything", [_ctx("Too short.")]).text == REFUSAL


def test_evidence_analysis_reports_best_similarity(embedder):
    gen = ExtractiveGenerator(embedder)
    strong = gen.analyse("mileage rate per mile", [_ctx("The mileage rate is 45p per mile.")])
    weak = gen.analyse("company car policy", [_ctx("The mileage rate is 45p per mile.")])
    assert strong.best_similarity > weak.best_similarity
    assert gen.analyse("x", []).best_similarity == 0.0
