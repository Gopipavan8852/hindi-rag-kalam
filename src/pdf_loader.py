import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.text_cleaner import clean_text, parse_heading

TABLE_SEPARATOR = " | "
_WIDE_GAP = re.compile(r" {3,}")  


@dataclass
class PageText:
    page: int                 
    section_no: int | None    
    section_title: str
    lines: list[str]          


def _find_pdftotext() -> str:
    """Find pdftotext on PATH, or in the folder named by the POPPLER_BIN variable."""
    found = shutil.which("pdftotext")
    if found is None and os.environ.get("POPPLER_BIN"):
        found = shutil.which("pdftotext", path=os.environ["POPPLER_BIN"])
    if found is None:
        raise RuntimeError(
            "pdftotext (poppler) not found. Add poppler's bin folder to PATH, "
            "or set the POPPLER_BIN environment variable to that folder."
        )
    return found


def _run_pdftotext(pdf_path: Path) -> list[str]:
    """Return the text of each page (pages are separated by form-feed characters)."""
    result = subprocess.run(
        [_find_pdftotext(), "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
        capture_output=True, check=True,
    )
    raw = result.stdout.decode("utf-8")
    return raw.split("\f")[:-1]  # last element is empty after the final form-feed


def _clean_line(line: str) -> str:
    """Turn wide column gaps into ' | ' and collapse other spacing."""
    cells = [clean_text(c) for c in _WIDE_GAP.split(line.strip())]
    return TABLE_SEPARATOR.join(c for c in cells if c)


def load_pages(pdf_path: Path) -> list[PageText]:
    """Read every page; cover/TOC pages get section_no=None."""
    pages: list[PageText] = []
    for index, text in enumerate(_run_pdftotext(pdf_path), start=1):
        lines = [_clean_line(l) for l in text.splitlines()]
        lines = [l for l in lines if l]
        heading = parse_heading(lines[0]) if lines else None
        if heading:
            pages.append(PageText(index, heading[0], heading[1], lines[1:]))
        else:
            pages.append(PageText(index, None, "", lines))
    return pages