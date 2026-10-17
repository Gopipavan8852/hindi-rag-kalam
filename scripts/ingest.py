"""Build the vector database: python -m scripts.ingest"""
from src.chunker import build_chunks
from src.config import CHROMA_DIR, PDF_PATH
from src.embeddings import Embedder
from src.pdf_loader import load_pages
from src.vector_store import VectorStore


def main() -> None:
    pages = load_pages(PDF_PATH)
    chunks = build_chunks(pages)
    print(f"Read {len(pages)} pages -> {len(chunks)} chunks "
          f"({sum(c.kind == 'table' for c in chunks)} table rows)")
    print("Embedding (first run downloads the model, please wait)...")
    embeddings = Embedder().embed_passages([c.embed_text for c in chunks])
    store = VectorStore()
    store.reset()
    store.add_chunks(chunks, embeddings)
    print(f"Stored {store.count()} chunks in {CHROMA_DIR}")
    for c in chunks[:3] + chunks[-2:]:
        print(f"  chunk {c.chunk_id} | page {c.page} | {c.section_title} | {c.text[:70]}...")


if __name__ == "__main__":
    main()