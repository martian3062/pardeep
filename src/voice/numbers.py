"""Recover digits that speech recognition turned into words.

The voice loop asked "2017 june me kya kar raha tha" and Whisper heard
"तु अगा तत्रा जून में" — the year was gone, so the date filter never fired and the
twin said it did not remember a year it holds 2,100 photos from. The reply was
honest and the retrieval was correct; the number simply never survived the
microphone.

Spoken Hindi and Punjabi build years in two ways: as a plain number
("दो हज़ार सत्रह") or digit by digit ("दो शून्य एक सात"). Both are handled, and a
number is only rewritten when the result lands in the range the archive actually
covers — otherwise "सत्रह लोग आए" becomes a year.
"""
from __future__ import annotations

import re
from datetime import datetime

_MIN_YEAR, _MAX_YEAR = 1990, datetime.now().year + 1

# Devanagari and Gurmukhi share most of these once transliterated by Whisper,
# which writes Punjabi speech in either script depending on the utterance.
UNITS = {
    "शून्य": 0, "सिफर": 0, "ਸਿਫ਼ਰ": 0, "zero": 0,
    "एक": 1, "ਇੱਕ": 1, "ਇਕ": 1,
    "दो": 2, "ਦੋ": 2,
    "तीन": 3, "ਤਿੰਨ": 3,
    "चार": 4, "ਚਾਰ": 4,
    "पांच": 5, "पाँच": 5, "ਪੰਜ": 5,
    "छह": 6, "छे": 6, "ਛੇ": 6,
    "सात": 7, "ਸੱਤ": 7,
    "आठ": 8, "ਅੱਠ": 8,
    "नौ": 9, "ਨੌਂ": 9, "ਨੌ": 9,
}

TEENS_AND_TENS = {
    "दस": 10, "ਦਸ": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14,
    "पंद्रह": 15, "सोलह": 16, "सत्रह": 17, "सतरह": 17, "ਸਤਾਰਾਂ": 17,
    "अठारह": 18, "ਅਠਾਰਾਂ": 18, "उन्नीस": 19, "ਉੱਨੀ": 19,
    "बीस": 20, "ਵੀਹ": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26,
}

HUNDRED = {"सौ", "ਸੌ"}
THOUSAND = {"हज़ार", "हजार", "ਹਜ਼ਾਰ", "हज़ार"}

_WORD = re.compile(r"[\wऀ-ॿ਀-੿]+", re.UNICODE)


def _value(token: str) -> int | None:
    token = token.strip().lower()
    if token in UNITS:
        return UNITS[token]
    if token in TEENS_AND_TENS:
        return TEENS_AND_TENS[token]
    return None


def _year_from_words(tokens: list[str]) -> int | None:
    """Read a run of number words as a year, both spoken forms."""
    total = 0
    current = 0
    digits = ""
    saw_scale = False

    for tok in tokens:
        low = tok.lower()
        val = _value(low)
        if val is not None:
            current = current * 100 + val if current and val < 10 and not saw_scale else (current + val if current and saw_scale else val)
            digits += str(val) if val < 10 else str(val)
            continue
        if low in THOUSAND:
            total += (current or 1) * 1000
            current = 0
            saw_scale = True
            continue
        if low in HUNDRED:
            total += (current or 1) * 100
            current = 0
            saw_scale = True
            continue
        return None

    if saw_scale:
        candidate = total + current
    elif digits and len(digits) == 4:
        candidate = int(digits)  # spelled out digit by digit
    else:
        return None
    return candidate if _MIN_YEAR <= candidate <= _MAX_YEAR else None


def recover_years(text: str) -> str:
    """Rewrite spoken years back into digits, leaving everything else alone."""
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        # a year is at most four number words ("दो हज़ार सत्रह", "दो शून्य एक सात")
        matched = False
        for span in (4, 3, 2):
            if i + span > len(tokens):
                continue
            window = [t.strip(",.।?!") for t in tokens[i : i + span]]
            year = _year_from_words(window)
            if year is not None:
                out.append(str(year))
                i += span
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1
    return " ".join(out)
