import unicodedata
from src.text_cleaner import clean_text, fix_doubled_signs, parse_heading


def test_doubled_signs_fixed():
    assert fix_doubled_signs("सम्माान") == "सम्मान"
    assert fix_doubled_signs("पुरस्काार") == "पुरस्कार"
    assert fix_doubled_signs("अक्टूूबर") == "अक्टूबर"


def test_valid_text_unchanged():
    assert clean_text("भारत रत्न") == "भारत रत्न"


def test_nfc_and_junk_removed():
    out = clean_text("क\u093c\x00")
    assert out == unicodedata.normalize("NFC", "क\u093c")


def test_parse_heading():
    assert parse_heading("14 · पुरस्कार एवं सम्मान") == (14, "पुरस्कार एवं सम्मान")
    assert parse_heading("कलाम को भारत") is None