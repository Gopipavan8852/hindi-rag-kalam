import re
from dataclasses import dataclass

from src.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_SENTENCES
from src.pdf_loader import TABLE_SEPARATOR, PageText

_SENTENCE_END = re.compile(r"(?<=[।?!])\s+")
_EMPTY_CELLS = {"—", "-", "–", ""}


@dataclass
class Chunk:
    chunk_id: int
    text: str
    page: int
    section_no: int
    section_title: str
    kind: str  # "text" or "table"

    @property
    def embed_text(self) -> str:
        """Text sent to the embedding model (title + body)."""
        return f"{self.section_title}: {self.text}"


def split_sentences(text: str) -> list[str]:
    """Split on danda / question mark / exclamation mark."""
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def pack_sentences(sentences: list[str], max_chars: int, overlap: int) -> list[str]:
    """Group sentences into chunks of <= max_chars, repeating `overlap` sentences."""
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap > 0 else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def table_row_to_text(headers: list[str], cells: list[str]) -> str:
    """Turn ['1997','भारत रत्न','...'] into 'वर्ष: 1997; सम्मान: भारत रत्न; ...'."""
    cells = [c for c in cells if c not in _EMPTY_CELLS]
    if len(cells) == len(headers):
        return "; ".join(f"{h}: {c}" for h, c in zip(headers, cells))
    return "; ".join(cells)


def _group_lines(lines: list[str]) -> list[tuple[bool, list[str]]]:
    """Group consecutive lines into (is_table, lines) blocks."""
    groups: list[tuple[bool, list[str]]] = []
    for line in lines:
        is_table = TABLE_SEPARATOR in line
        if groups and groups[-1][0] == is_table:
            groups[-1][1].append(line)
        else:
            groups.append((is_table, [line]))
    return groups


def build_chunks(pages: list[PageText]) -> list[Chunk]:
    """Create chunks for every section page (cover and contents pages are skipped)."""
    chunks: list[Chunk] = []

    def add(text: str, page: PageText, kind: str) -> None:
        chunks.append(Chunk(len(chunks), text, page.page, page.section_no,
                            page.section_title, kind))

    for page in pages:
        if page.section_no is None:
            continue
        for is_table, lines in _group_lines(page.lines):
            if is_table:
                headers = lines[0].split(TABLE_SEPARATOR)
                for row in lines[1:]:
                    add(table_row_to_text(headers, row.split(TABLE_SEPARATOR)), page, "table")
            else:
                sentences = split_sentences(" ".join(lines))
                for text in pack_sentences(sentences, CHUNK_MAX_CHARS, CHUNK_OVERLAP_SENTENCES):
                    add(text, page, "text")
    return chunks