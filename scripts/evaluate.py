import sys
from pathlib import Path

from src.config import PROJECT_ROOT, TOP_K
from src.embeddings import Embedder
from src.rag import RagPipeline, format_citation
from src.retriever import Retriever
from src.vector_store import VectorStore

# (query, language, pages whose text contains the answer)
QUERIES = [
    ("SLV-III ने किस उपग्रह को कक्षा में स्थापित किया और किस वर्ष?", "Hindi", {8, 17, 20}),
    ("कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?", "Hindi", {16, 17, 20, 22}),
    ("पोखरण-II में कलाम की क्या भूमिका थी?", "Hindi", {10}),
    ("Which institution did Kalam attend to study aeronautical engineering?", "English", {6, 17, 20}),
    ("Who co-wrote the autobiography, and in what year was it published?", "English", {3, 17, 22}),
    ("How and where did Kalam die in 2015?", "English", {15, 17, 22}),
]
EXTRA_NOT_IN_DOCUMENT = "What is the capital of France?"


def main() -> None:
    retrieval_only = "--retrieval-only" in sys.argv
    retriever = Retriever(Embedder(), VectorStore())
    pipeline = RagPipeline(retriever)
    out = ["# Evaluation results", "", f"top-k = {TOP_K}", ""]
    hits_found = 0
    for number, (query, language, good_pages) in enumerate(QUERIES, start=1):
        if retrieval_only:
            hits, answer_text = retriever.search(query, TOP_K), "(retrieval only)"
        else:
            result = pipeline.ask(query)
            hits, answer_text = result.hits, result.text
        rank = next((i for i, h in enumerate(hits, 1) if h.page in good_pages), None)
        hits_found += rank is not None
        out += [f"## {number}. {query}  ({language})", f"**Answer:** {answer_text}", "",
                f"**Retrieval hit:** {'yes, rank ' + str(rank) if rank else 'NO'}", "",
                "**Sources:**"]
        out += [f"{i}. {format_citation(h)}" for i, h in enumerate(hits, 1)] + [""]
        print(f"Q{number} [{language}] hit={'rank ' + str(rank) if rank else 'NO'}")
    out.insert(3, f"**Retrieval hit-rate: {hits_found}/{len(QUERIES)}**")
    if not retrieval_only:
        negative = pipeline.ask(EXTRA_NOT_IN_DOCUMENT)
        out += ["## Extra: question not in the document", f"**Query:** {EXTRA_NOT_IN_DOCUMENT}",
                f"**Answer:** {negative.text}", f"**Answered flag:** {negative.answered}", ""]
    path = Path(PROJECT_ROOT) / "evaluation" / "results.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"Hit-rate {hits_found}/{len(QUERIES)}. Wrote {path}")


if __name__ == "__main__":
    main()