"""Answer generation with a local LLM served by Ollama (plain HTTP, no extra library)."""
import json
import re
import urllib.error
import urllib.request

from src.config import LLM_MODEL, OLLAMA_TIMEOUT_SECONDS, OLLAMA_URL
from src.vector_store import Hit

NOT_FOUND_HI = "दस्तावेज़ में इस प्रश्न का उत्तर नहीं मिला।"
NOT_FOUND_EN = "The answer was not found in the document."
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def is_hindi(question: str) -> bool:
    """True if the question contains Devanagari letters (otherwise treat it as English)."""
    return len(_DEVANAGARI.findall(question)) >= 3


def not_found_text(question: str) -> str:
    """The refusal sentence, in the language of the question."""
    return NOT_FOUND_HI if is_hindi(question) else NOT_FOUND_EN


def is_not_found(answer_text: str) -> bool:
    """True if the model refused, in either language."""
    return NOT_FOUND_HI in answer_text or NOT_FOUND_EN in answer_text


def build_system_prompt(question: str) -> str:
    language = "Hindi (Devanagari script)" if is_hindi(question) else "English"
    extra = ("" if is_hindi(question) else
             " The passages are Hindi, so translate the facts into English; keep names such as "
             "ISRO, DRDO, SLV-III as they are, and write Hindi person names in English letters "
             "(for example अरुण तिवारी -> Arun Tiwari).")
    return (
        "You answer questions about A.P.J. Abdul Kalam using ONLY the numbered context passages. "
        "The whole document is about Kalam, so when a question asks who else was involved "
        "(co-author, teacher, mentor, colleague), name the OTHER person, not Kalam. "
        "Rules: 1) Use only facts that appear in the passages; never use outside knowledge. "
        f"2) Write the answer in {language} only.{extra} "
        "3) Answer exactly what is asked (who, when, where, which, what role) in one or two "
        "sentences, using the passage sentence that states it directly. "
        f"4) If the passages do not contain the answer, reply exactly: {not_found_text(question)}"
    )


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context = "\n".join(f"[{i}] (page {h.page}) {h.text}" for i, h in enumerate(hits, start=1))
    user = f"Context passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    return [{"role": "system", "content": build_system_prompt(question)},
            {"role": "user", "content": user}]


def ask_ollama(messages: list[dict], model: str = LLM_MODEL) -> str:
    """Send the chat to the local Ollama server and return the reply text."""
    payload = json.dumps({"model": model, "messages": messages, "stream": False,
                          "options": {"temperature": 0}}).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            text = json.loads(response.read().decode("utf-8"))["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama answered with HTTP {error.code} for model '{model}': {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach Ollama ({error.reason}). Start the Ollama app and run "
            f"`ollama pull {model}`."
        ) from error