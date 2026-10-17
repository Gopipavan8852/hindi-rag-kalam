import pytest
from src.config import PDF_PATH
from src.pdf_loader import load_pages

pytestmark = pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF missing")
PAGES = load_pages(PDF_PATH) if PDF_PATH.exists() else []


def test_page_and_section_mapping():
    assert len(PAGES) == 22
    p16 = PAGES[15]
    assert (p16.page, p16.section_no) == (16, 14)
    assert "पुरस्कार" in p16.section_title


def test_table_row_kept_on_one_line():
    assert "1997 | भारत रत्न | भारत का सर्वोच्च नागरिक सम्मान।" in PAGES[15].lines


def test_no_doubled_vowel_signs():
    assert all("ाा" not in l and "ूू" not in l for p in PAGES for l in p.lines)


def test_key_facts_present_on_expected_pages():
    joined = lambda n: " ".join(PAGES[n - 1].lines)
    assert "रोहिणी" in joined(8) and "1980" in joined(8)
    assert "मद्रास इंस्टीट्यूट" in joined(6).replace("  ", " ") or "एमआईटी" in joined(6)
    assert "27 जुलाई 2015" in joined(15)