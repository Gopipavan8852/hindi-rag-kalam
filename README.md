# Hindi Document Question Answering Using RAG

Answers questions (Hindi or English) about A.P.J. Abdul Kalam from a supplied Hindi PDF.
Fully local and free: no paid APIs, no cloud inference.

**Stack:** Python · poppler `pdftotext` · multilingual-e5-small (Sentence Transformers) ·
ChromaDB · Ollama with `qwen3:4b-instruct`.

## Results (6 required queries)
Retrieval: the correct page was ranked first for 6/6 queries (English to Hindi included).
Answers: 6/6 correct against the PDF. Full output with citations: `evaluation/results.md`.

| # | Query (language) | Answer | Top source |
|---|---|---|---|
| 1 | SLV-III satellite and year (HI) | Rohini, 1980 | page 8, chunk 24 |
| 2 | Bharat Ratna year (HI) | 1997 | page 16, chunk 56 |
| 3 | Pokhran-II role (HI) | Scientific adviser and DRDO head, central organizational role | page 10, chunk 32 |
| 4 | Institution for aeronautical engineering (EN) | Madras Institute of Technology | page 6, chunk 13 |
| 5 | Co-author and year (EN) | Arun Tiwari, 1999 | page 17, chunk 72 |
| 6 | How and where Kalam died (EN) | 27 July 2015, Shillong, during a lecture | page 17, chunk 75 |

Off-topic questions (France, FIFA 2018, cake, Indian capital in Hindi) were all refused.
Typical time per question on my laptop (8 GB RAM,11th Gen Intel(R) Core(TM)): about [11] seconds,
including model loading.

## 1. Chunking strategy
- Prose is split on the Hindi danda (।), "?" and "!". Not on ".", because the text has
  names like ए.पी.जे.
- Sentences are packed into chunks of at most 500 characters with 1 sentence of overlap,
  and a sentence is never cut in half. Each section is about one page, so 500 characters
  keeps one idea per chunk; the overlap protects facts on a boundary.
- Every table row is its own chunk ("वर्ष: 1997; सम्मान: भारत रत्न; ..."), so tables are
  searchable.
- The section title is prepended to the text that is embedded.
- Result: 106 chunks (49 table rows, 57 text chunks). Cover and contents pages are skipped.
- I did not experiment with other chunk sizes or overlaps. Retrieval was already 6/6, and
  with six queries I could not measure a difference honestly.

## 2. Why ChromaDB
Local, installed with pip, persistent, no server to run, and it stores page, section and
chunk_id next to each vector, so citations are produced from metadata, not written by hand.

## 3. How Hindi was handled
- **Extraction.** I compared five extractors on this PDF (printed from Chrome with Type 3
  fonts): pypdf and pdfplumber drop characters, PyMuPDF doubles vowel signs and drops
  repeated letters, pypdfium2 inserts stray spaces. Only poppler's `pdftotext` gave
  readable Hindi, so the loader uses it (table rows kept on one line with `-layout`).
- **Normalization.** Unicode NFC, removal of invisible junk characters, UTF-8 decoded
  explicitly.
- **Embeddings.** `intfloat/multilingual-e5-small` with the required `query:` / `passage:`
  prefixes. English queries retrieved the correct Hindi chunks (3/3).
- **Answer language.** The code detects Devanagari in the question and instructs the model
  to answer in that language. Without this, English questions were answered in Hindi.
- **Page numbers.** The PDF has a cover and a contents page, so printed section N is on
  physical page N+2. Citations use the physical PDF page.

## 4. Limitations and what I would improve
- **Model size matters.** With `qwen2.5:1.5b`, retrieval was already 6/6 and the correct
  chunks were retrieved, but 2 of 6 answers were wrong (Pokhran-II role, co-author) and one
  had a misspelled place name. Moving to `qwen3:4b-instruct` and generalizing the prompt
  fixed all six. I changed the model and prompt after seeing results, and six queries is a
  very small test set. `qwen2.5:7b` failed with an HTTP 500 on my 8 GB machine.
- **Stray spaces from extraction.** Poppler leaves some spaces inside words
  ("कें द्रीय"), and the model sometimes copies them into answers.
- **Scores are not reliable for refusal.** E5 similarity is compressed: off-topic questions
  scored 0.74 to 0.79 and real questions 0.79 to 0.89, with overlap. So `MIN_SCORE = 0.70`
  rarely fires, and refusals come from the model following the prompt. Tested on only four
  off-topic questions.
- **Citations show retrieved evidence, not verified evidence.** All four retrieved chunks
  are listed (rank 1 is the one to trust); I do not check sentence by sentence which one
  was used.
- **Answers can be slightly incomplete** (query 6 omits that he collapsed and his age).
- **Next steps:** hybrid keyword + vector search, a re-ranker, a footer/noise filter (one
  chunk is just the PDF footer), a larger evaluation set with expected answers, and a
  small UI.

## 5. How to run (Windows PowerShell)
1. Install Python 3.12, Git, [Ollama](https://ollama.com) and poppler for Windows
   (unzip it, then either add its `Library\bin` folder to PATH or set
   `$env:POPPLER_BIN = "<path to Library\bin>"`). Check with `pdftotext -v`.
2. `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. `ollama pull qwen3:4b-instruct` (about 2.5 GB; keep the Ollama app running)
5. Put `Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf` in `data/` (included in this repo)
6. `python -m pytest -q`
7. `python -m scripts.ingest` (first run downloads the embedding model, about 470 MB)
8. `python -m scripts.evaluate` (writes `evaluation/results.md`; add `--retrieval-only`
   to skip the LLM)
9. `python -m scripts.ask "your question"`, or `python -m scripts.ask` for a prompt loop

Models and licenses (verify on the model pages): multilingual-e5-small is MIT;
Qwen3-4B is Apache 2.0.