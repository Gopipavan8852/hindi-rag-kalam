"""Central settings. Every other module imports paths and constants from here."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "kalam_hi"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"  # MIT license, runs locally
LLM_MODEL = "qwen2.5:3b"                            # via Ollama, fits in 8 GB RAM
CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP_SENTENCES = 1
TOP_K = 4

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT_SECONDS = 180
QUERY_PREFIX = "query: "      # multilingual-e5 expects these prefixes
PASSAGE_PREFIX = "passage: "
MIN_SCORE = 0.70              # below this top score we answer "not found"; tune with evaluate.py