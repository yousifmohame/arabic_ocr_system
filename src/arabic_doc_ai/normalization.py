from __future__ import annotations

import re
import unicodedata

# Arabic-Indic and Eastern Arabic-Indic digits -> Western digits.
_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

_TATWEEL = "ـ"
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_SPACES = re.compile(r"\s+")


def normalize_arabic_for_matching(text: str) -> str:
    """Return a search/matching copy; never overwrite the OCR raw text."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DIGIT_TRANSLATION)
    text = text.replace(_TATWEEL, "")
    text = _DIACRITICS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    return _SPACES.sub(" ", text).strip()


def normalize_digits(text: str) -> str:
    """Normalize Arabic-Indic digits without changing other characters."""
    return text.translate(_DIGIT_TRANSLATION)
