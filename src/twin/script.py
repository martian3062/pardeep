"""Which script a message is written in.

The prompt already said "match the language he used", and the twin ignored it:
asked "Tell me honestly, what are you worst at?" in plain English, it answered
in Devanagari. A general instruction competes with a persona card full of
Devanagari and Gurmukhi examples and with retrieved memories that are mostly
Hinglish, and it loses. Naming the script explicitly per turn does not.

He genuinely writes in all three, so this reports what he used rather than
picking a house style.
"""
from __future__ import annotations

import re

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_GURMUKHI = re.compile(r"[਀-੿]")
_LATIN = re.compile(r"[A-Za-z]")

# Words that are Hindi/Punjabi wearing Latin letters. A message full of these is
# Hinglish, not English, and answering it in formal English would be as wrong as
# answering English in Devanagari.
_ROMAN_HINGLISH = {
    "yaar", "bhai", "bro", "hai", "hain", "nahi", "nahin", "kya", "kaise", "kaisa",
    "mujhe", "mera", "meri", "tera", "teri", "tu", "tum", "aap", "kar", "karna",
    "raha", "rha", "tha", "thi", "hoga", "chal", "thik", "theek", "acha", "accha",
    "koi", "kuch", "abhi", "phir", "fir", "bas", "matlab", "yeh", "woh", "wo",
    "ke", "ka", "ki", "ko", "se", "me", "mein", "par", "aur", "toh", "to",
    "ho", "hu", "hun", "gya", "gaya", "diya", "liya", "pta", "pata", "bol",
}


def script_of(text: str) -> str:
    """One of: devanagari | gurmukhi | hinglish | english | unknown."""
    if not text or not text.strip():
        return "unknown"
    dev = len(_DEVANAGARI.findall(text))
    gur = len(_GURMUKHI.findall(text))
    lat = len(_LATIN.findall(text))

    if gur and gur >= dev:
        return "gurmukhi"
    # a few English loanwords inside a Devanagari sentence is still Devanagari
    if dev and dev * 3 >= lat:
        return "devanagari"
    if not lat:
        return "unknown"

    words = re.findall(r"[a-z']+", text.lower())
    if words:
        hinglish = sum(1 for w in words if w in _ROMAN_HINGLISH)
        if hinglish / len(words) >= 0.2:
            return "hinglish"
    return "english"


_INSTRUCTION = {
    "english": (
        "He wrote to you in plain English. Reply in English. Do not switch to "
        "Hindi or Punjabi script, however natural that feels."
    ),
    "hinglish": (
        "He wrote in Roman-script Hinglish. Reply the same way — Latin letters, "
        "Hindi/Punjabi words. Do not switch to Devanagari."
    ),
    "devanagari": "He wrote in Devanagari. Reply in Devanagari, mixing English words as he does.",
    "gurmukhi": "He wrote in Gurmukhi Punjabi. Reply in Gurmukhi.",
}


def language_rule(text: str) -> str:
    """The per-turn line telling the model which script to answer in."""
    return _INSTRUCTION.get(script_of(text), "")
