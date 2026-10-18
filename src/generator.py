"""Answer generation with a local LLM served by Ollama (plain HTTP, no extra library)."""
import json
import urllib.error
import urllib.request

from src.config import LLM_MODEL, OLLAMA_TIMEOUT_SECONDS, OLLAMA_URL
from src.vector_store import Hit

NOT_FOUND_TEXT = "दस्तावेज़ में इस प्रश्न का उत्तर नहीं मिला। (Not found in the document.)"

SYSTEM_PROMPT = (
    "You answer questions about A.P.J. Abdul Kalam using ONLY the numbered context passages. "
    "The passages are in Hindi. Rules: "
    "1) Use only facts that appear in the passages; never use outside knowledge. "
    "2) Reply in the same language as the question (Hindi question -> Hindi answer, "
    "English question -> English answer). "
    "3) Be short: one or two sentences. "
    f"4) If the passages do not contain the answer, reply exactly: {NOT_FOUND_TEXT}"
)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context = "\n".join(f"[{i}] (page {h.page}) {h.text}" for i, h in enumerate(hits, start=1))
    user = f"Context passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def ask_ollama(messages: list[dict], model: str = LLM_MODEL) -> str:
    """Send the chat to the local Ollama server and return the reply text."""
    payload = json.dumps({"model": model, "messages": messages, "stream": False,
                          "options": {"temperature": 0}}).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))["message"]["content"].strip()
    except urllib.error.HTTPError as error:
        # The server answered, but with an error. Show Ollama's own explanation.
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama answered with HTTP {error.code} for model '{model}': {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach Ollama ({error.reason}). Start the Ollama app and run "
            f"`ollama pull {model}`."
        ) from error