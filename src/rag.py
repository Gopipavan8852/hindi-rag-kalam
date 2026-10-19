"""The full pipeline: question -> retrieve -> (gate) -> generate -> answer + citations."""
from dataclasses import dataclass
from typing import Callable

from src.config import MIN_SCORE, TOP_K
from src.generator import ask_ollama, build_messages, is_not_found, not_found_text
from src.retriever import Retriever
from src.vector_store import Hit


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    answered: bool  # False when we said "not found"


def format_citation(hit: Hit) -> str:
    """Build the citation purely from stored metadata (never typed by hand)."""
    return (f'page: {hit.page} · section: "{hit.section_title}" · '
            f"chunk_id: {hit.chunk_id} · score: {hit.score:.2f}")


def format_answer(answer: Answer) -> str:
    heading = "Sources:" if answer.answered else "Closest passages (not used: no answer found):"
    lines = [f"Query : {answer.question}", f"Answer: {answer.text}", heading]
    lines += [f"  [{i}] {format_citation(h)}" for i, h in enumerate(answer.hits, start=1)]
    return "\n".join(lines)


class RagPipeline:
    def __init__(self, retriever: Retriever,
                 generate: Callable[[list[dict]], str] = ask_ollama) -> None:
        self.retriever = retriever
        self.generate = generate

    def ask(self, question: str, k: int = TOP_K) -> Answer:
        hits = self.retriever.search(question, k)
        if not hits or hits[0].score < MIN_SCORE:
            return Answer(question, not_found_text(question), hits, answered=False)
        text = self.generate(build_messages(question, hits))
        return Answer(question, text, hits, answered=not is_not_found(text))