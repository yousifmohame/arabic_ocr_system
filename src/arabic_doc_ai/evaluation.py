from __future__ import annotations

import re
import unicodedata

# Arabic-Indic and Eastern Arabic-Indic digits -> Western digits.
_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_WHITESPACE = re.compile(r"\s+")
_BIDI_CONTROLS = dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"), None)
_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})


def _clean_text(text: str) -> str:
    """Strict OCR scoring: normalize Unicode composition and whitespace only.

    Arabic letters, hamza forms, digits, and punctuation remain significant.
    CER then ignores whitespace, while WER treats whitespace as token separators.
    """
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(_BIDI_CONTROLS)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_arabic_for_metrics(text: str) -> str:
    """Conservative normalization for a second, more tolerant OCR metric.

    It folds Arabic/Persian digit forms, tatweel, vowel marks, and common alef
    variants. It deliberately does NOT fold ى/ي or other potentially meaningful
    letter distinctions. Raw OCR text is never changed by this function.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_BIDI_CONTROLS)
    text = text.translate(_DIGIT_TRANSLATION)
    text = text.replace("ـ", "")
    text = _DIACRITICS.sub("", text)
    text = text.translate(_ALEF_VARIANTS)
    return _WHITESPACE.sub(" ", text).strip()


def edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance using O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            cost = 0 if ref_char == hyp_char else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))
        previous = current
    return previous[-1]


def _cer_from_clean(reference: str, hypothesis: str) -> float:
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def _wer_from_clean(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    """Strict Character Error Rate; whitespace is ignored."""
    return _cer_from_clean(_clean_text(reference), _clean_text(hypothesis))


def wer(reference: str, hypothesis: str) -> float:
    """Strict Word Error Rate after Unicode/whitespace normalization."""
    return _wer_from_clean(_clean_text(reference), _clean_text(hypothesis))


def normalized_cer(reference: str, hypothesis: str) -> float:
    """CER after conservative Arabic orthographic normalization."""
    return _cer_from_clean(
        normalize_arabic_for_metrics(reference),
        normalize_arabic_for_metrics(hypothesis),
    )


def normalized_wer(reference: str, hypothesis: str) -> float:
    """WER after conservative Arabic orthographic normalization."""
    return _wer_from_clean(
        normalize_arabic_for_metrics(reference),
        normalize_arabic_for_metrics(hypothesis),
    )
