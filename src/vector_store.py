"""ChromaDB wrapper: store chunks with metadata, search by cosine similarity."""
from dataclasses import dataclass
from pathlib import Path

import chromadb

from src.chunker import Chunk
from src.config import CHROMA_DIR, COLLECTION_NAME


@dataclass
class Hit:
    chunk_id: int
    text: str
    page: int
    section_no: int
    section_title: str
    kind: str
    distance: float  # cosine distance: 0 = identical direction
    score: float     # cosine similarity = 1 - distance


class VectorStore:
    def __init__(self, path: Path = CHROMA_DIR, name: str = COLLECTION_NAME) -> None:
        self._client = chromadb.PersistentClient(path=str(path))
        self._name = name
        self._collection = self._client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        """Delete everything so re-ingesting never creates duplicates."""
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.add(
            ids=[str(c.chunk_id) for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"chunk_id": c.chunk_id, "page": c.page, "section_no": c.section_no,
                        "section_title": c.section_title, "kind": c.kind} for c in chunks],
        )

    def search(self, query_embedding: list[float], k: int) -> list[Hit]:
        result = self._collection.query(query_embeddings=[query_embedding], n_results=k)
        hits = []
        for text, meta, dist in zip(result["documents"][0], result["metadatas"][0],
                                    result["distances"][0]):
            hits.append(Hit(meta["chunk_id"], text, meta["page"], meta["section_no"],
                            meta["section_title"], meta["kind"], dist, 1.0 - dist))
        return hits