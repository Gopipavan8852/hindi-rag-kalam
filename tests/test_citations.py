from src.generator import NOT_FOUND_TEXT, build_messages
from src.rag import Answer, RagPipeline, format_answer, format_citation
from src.vector_store import Hit


def make_hit(score=0.83):
    return Hit(47, "1997: भारत रत्न", 16, 14, "पुरस्कार एवं सम्मान", "table", 1 - score, score)


def test_citation_comes_from_metadata():
    assert format_citation(make_hit()) == 'page: 16 · section: "पुरस्कार एवं सम्मान" · chunk_id: 47 · score: 0.83'


def test_prompt_contains_context_and_question():
    messages = build_messages("प्रश्न?", [make_hit()])
    assert "[1] (page 16)" in messages[1]["content"] and "प्रश्न?" in messages[1]["content"]
    assert "ONLY" in messages[0]["content"]


class FakeRetriever:
    def __init__(self, hits): self.hits = hits
    def search(self, question, k): return self.hits


def test_low_score_returns_not_found_without_calling_llm():
    def must_not_run(messages): raise AssertionError("LLM should not be called")
    answer = RagPipeline(FakeRetriever([make_hit(0.3)]), must_not_run).ask("capital of France?")
    assert answer.text == NOT_FOUND_TEXT and answer.answered is False


def test_good_score_calls_llm_and_formats_sources():
    answer = RagPipeline(FakeRetriever([make_hit()]), lambda m: "1997").ask("भारत रत्न कब?")
    text = format_answer(answer)
    assert answer.answered and "Answer: 1997" in text and "chunk_id: 47" in text