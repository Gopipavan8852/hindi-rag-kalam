"""Ask questions: python -m scripts.ask "your question"   (or no argument for a prompt loop)"""
import sys

from src.embeddings import Embedder
from src.rag import RagPipeline, format_answer
from src.retriever import Retriever
from src.vector_store import VectorStore


def main() -> None:
    pipeline = RagPipeline(Retriever(Embedder(), VectorStore()))
    if len(sys.argv) > 1:
        print(format_answer(pipeline.ask(" ".join(sys.argv[1:]))))
        return
    while True:
        question = input("\nQuestion (empty to quit): ").strip()
        if not question:
            break
        print(format_answer(pipeline.ask(question)))


if __name__ == "__main__":
    main()