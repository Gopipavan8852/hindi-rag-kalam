import pytest
from src.chunker import build_chunks, pack_sentences, split_sentences, table_row_to_text
from src.config import CHUNK_MAX_CHARS, PDF_PATH
from src.pdf_loader import PageText, load_pages


def test_split_on_danda_and_question_mark():
    text = "कलाम का जन्म हुआ। वे वैज्ञानिक थे? हाँ! ए.पी.जे. अब्दुल कलाम।"
    assert split_sentences(text) == ["कलाम का जन्म हुआ।", "वे वैज्ञानिक थे?", "हाँ!",
                                     "ए.पी.जे. अब्दुल कलाम।"]


def test_pack_respects_max_and_overlaps():
    sentences = [f"वाक्य संख्या {i} है।" for i in range(30)]
    chunks = pack_sentences(sentences, max_chars=60, overlap=1)
    assert all(len(c) <= 60 for c in chunks)
    assert chunks[0].split("।")[-2].strip() + "।" in chunks[1]  # last sentence repeated


def test_long_sentence_is_never_cut():
    long = "क" * 900 + "।"
    assert pack_sentences([long], 500, 1) == [long]


def test_table_row_text():
    assert table_row_to_text(["वर्ष", "सम्मान"], ["1997", "भारत रत्न"]) == "वर्ष: 1997; सम्मान: भारत रत्न"
    assert table_row_to_text(["वर्ष", "सम्मान", "टिप्पणी"], ["1997", "पुरस्कार", "—"]) == "1997; पुरस्कार"


def test_cover_pages_skipped_and_metadata_set():
    pages = [PageText(1, None, "", ["cover"]),
             PageText(5, 3, "शीर्षक", ["पहला वाक्य है। दूसरा वाक्य है।"])]
    chunks = build_chunks(pages)
    assert [(c.chunk_id, c.page, c.section_no, c.section_title) for c in chunks] == [(0, 5, 3, "शीर्षक")]
    assert chunks[0].embed_text.startswith("शीर्षक: ")


@pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF missing")
def test_real_pdf_chunks():
    chunks = build_chunks(load_pages(PDF_PATH))
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert all(c.page >= 3 for c in chunks)
    assert all(len(c.text) <= CHUNK_MAX_CHARS or c.kind == "text" for c in chunks)
    bharat = [c for c in chunks if c.page == 16 and c.kind == "table" and "भारत रत्न" in c.text]
    assert bharat and "1997" in bharat[0].text