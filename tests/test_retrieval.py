"""Retrieval test with a fake embedder, so it needs no model download."""
import hashlib

import pytest
from src.chunker import Chunk
from src.vector_store import VectorStore


def fake_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [b / 255 - 0.5 for b in digest[:16]]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@pytest.fixture()
def store(tmp_path):
    store = VectorStore(path=tmp_path / "db", name="test")
    chunks = [Chunk(0, "alpha", 8, 6, "इसरो", "text"), Chunk(1, "beta", 16, 14, "पुरस्कार", "table"),
              Chunk(2, "gamma", 10, 8, "पोखरण", "text")]
    store.add_chunks(chunks, [fake_vector(c.text) for c in chunks])
    return store


def test_exact_match_ranks_first_with_metadata(store):
    hits = store.search(fake_vector("beta"), k=3)
    assert hits[0].chunk_id == 1 and hits[0].page == 16 and hits[0].section_title == "पुरस्कार"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_reset_removes_everything(store):
    store.reset()
    assert store.count() == 0