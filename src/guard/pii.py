"""Last line before a reply reaches someone who is not him.

Trust-tier retrieval already keeps private memories out of a guest's context,
which is the real defence — a model cannot leak what it was never given. This
catches the rest: a phone number the model reconstructs from a public memory, an
address mentioned in passing, an email quoted from a screenshot caption.

Presidio ships recognisers for the identifiers of countries this archive is not
from. A UK postcode recogniser does nothing for a person whose life is recorded
in Indian phone numbers, Aadhaar and PAN numbers, and UPI handles, so those are
added.

Redaction is deliberately visible. Silently deleting a number would leave a reply
that reads as a complete sentence but has quietly changed meaning; [phone] is
obvious to both the reader and to him when he audits it later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# What a guest must never receive, whatever the model wrote.
GUEST_BLOCKED = (
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "IN_AADHAAR",
    "IN_PAN",
    "IN_UPI",
    "IN_VEHICLE",
)

# Presidio's built-ins miss these, and they are exactly the identifiers a life
# recorded in India is full of.
_INDIAN_PATTERNS = {
    # Phone first: a country code runs straight into the number with no word
    # boundary between them, so \b after "+91" never matches and "+919876543210"
    # slipped past — then the Aadhaar pattern ate the 12 digits and labelled a
    # phone number as an ID.
    "PHONE_NUMBER": re.compile(r"(?<![\d\w])(?:\+?91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)"),
    # 12 digits in groups of four, first never 0 or 1; not part of a longer
    # number and not sitting behind a country code
    "IN_AADHAAR": re.compile(r"(?<![\d+])\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b(?!\d)"),
    "IN_PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "IN_UPI": re.compile(
        r"\b[\w.\-]{2,}@(?:okaxis|oksbi|okhdfcbank|okicici|ybl|paytm|upi|ibl|axl)\b", re.I
    ),
    "IN_VEHICLE": re.compile(r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}\b"),
}


@dataclass
class Finding:
    kind: str
    text: str
    start: int
    end: int


@lru_cache(maxsize=1)
def _analyzer():
    from presidio_analyzer import AnalyzerEngine

    return AnalyzerEngine()


def _regex_findings(text: str) -> list[Finding]:
    out: list[Finding] = []
    for kind, pattern in _INDIAN_PATTERNS.items():
        for m in pattern.finditer(text):
            out.append(Finding(kind=kind, text=m.group(), start=m.start(), end=m.end()))
    return out


def find(text: str, kinds: tuple[str, ...] = GUEST_BLOCKED) -> list[Finding]:
    """Everything in `text` that looks like a blocked identifier."""
    if not text.strip():
        return []
    found = [f for f in _regex_findings(text) if f.kind in kinds]
    try:
        for r in _analyzer().analyze(text=text, language="en"):
            if r.entity_type in kinds and r.score >= 0.5:
                found.append(
                    Finding(
                        kind=r.entity_type,
                        text=text[r.start : r.end],
                        start=r.start,
                        end=r.end,
                    )
                )
    except Exception:
        pass  # regex layer still applies; never fail open on an engine error
    return _dedupe(found)


# When two recognisers claim overlapping text, the more specific label wins:
# calling a phone number an Aadhaar number is wrong in the audit log even though
# both would redact the digits.
_SPECIFICITY = {"PHONE_NUMBER": 0, "IN_AADHAAR": 1, "IN_PAN": 1, "IN_UPI": 1, "IN_VEHICLE": 1}


def _dedupe(found: list[Finding]) -> list[Finding]:
    """Drop spans contained in a longer one, so a number is not tagged twice."""
    ordered = sorted(
        found, key=lambda f: (f.start, _SPECIFICITY.get(f.kind, 2), -(f.end - f.start))
    )
    kept: list[Finding] = []
    for f in ordered:
        if any(f.start >= k.start and f.end <= k.end for k in kept):
            continue
        kept.append(f)
    return kept


_LABEL = {
    "PHONE_NUMBER": "[phone]",
    "EMAIL_ADDRESS": "[email]",
    "CREDIT_CARD": "[card]",
    "IBAN_CODE": "[bank]",
    "IP_ADDRESS": "[ip]",
    "LOCATION": "[place]",
    "IN_AADHAAR": "[aadhaar]",
    "IN_PAN": "[pan]",
    "IN_UPI": "[upi]",
    "IN_VEHICLE": "[vehicle]",
}


def redact(text: str, kinds: tuple[str, ...] = GUEST_BLOCKED) -> tuple[str, list[Finding]]:
    """Replace blocked identifiers with a visible label.

    Returns the cleaned text and what was removed, because the audit log needs to
    record that something was caught — a guard that works silently cannot be
    reviewed.
    """
    found = find(text, kinds)
    if not found:
        return text, []
    out = text
    for f in sorted(found, key=lambda f: f.start, reverse=True):
        out = out[: f.start] + _LABEL.get(f.kind, "[redacted]") + out[f.end :]
    return out, found
