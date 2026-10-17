"""Clean and normalize Devanagari text extracted from the PDF."""
import re
import unicodedata

# Dependent vowel signs, anusvara/visarga, nukta and virama (U+0900-U+0903, U+093A-U+094F).
# A correct Hindi word never repeats one of these back to back, but some PDF
# extractors emit them twice (e.g. "सम्माान" for "सम्मान").
_DOUBLED_SIGN = re.compile(r"([\u0900-\u0903\u093A-\u094F])\1+")
_JUNK_CHARS = re.compile(r"[\x00\u00ad\u200b\ufeff]")  # NUL, soft hyphen, zero-width space, BOM
_HEADING = re.compile(r"^\s*(\d{1,2})\s*[·.]\s*(.+?)\s*$")


def fix_doubled_signs(text: str) -> str:
    """Collapse repeated vowel signs, e.g. 'पुरस्काार' -> 'पुरस्कार'."""
    return _DOUBLED_SIGN.sub(r"\1", text)


def clean_text(text: str) -> str:
    """NFC-normalize, remove junk characters, fix doubled signs, tidy spaces."""
    text = unicodedata.normalize("NFC", text)
    text = _JUNK_CHARS.sub("", text)
    text = fix_doubled_signs(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_heading(line: str) -> tuple[int, str] | None:
    """Return (section_number, title) for a line like '14 · पुरस्कार एवं सम्मान'."""
    match = _HEADING.match(line)
    return (int(match.group(1)), match.group(2)) if match else None