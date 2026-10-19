"""Local multilingual embeddings (intfloat/multilingual-e5-small)."""
from src.config import EMBEDDING_MODEL, PASSAGE_PREFIX, QUERY_PREFIX


class Embedder:
    """Loads the model on first use (the first run downloads ~470 MB once)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, query: str) -> list[float]:
        return self._encode([QUERY_PREFIX + query])[0]