"""Question -> embedding -> top-k chunks."""
from src.config import TOP_K
from src.embeddings import Embedder
from src.vector_store import Hit, VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(self, question: str, k: int = TOP_K) -> list[Hit]:
        return self.store.search(self.embedder.embed_query(question), k)