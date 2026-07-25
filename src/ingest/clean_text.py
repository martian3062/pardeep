"""Repair Whisper transcription artifacts on noisy phone audio.

Even with temperature fallback and a compression-ratio threshold, Whisper still
gets stuck repeating an n-gram ("कर दो कर दो कर दो ..." x12) on compressed
Punjabi/Hindi call audio. The underlying speech was the phrase said once, so
collapsing runs recovers the real utterance instead of discarding the segment.
"""
from __future__ import annotations

import re

# Whisper's placeholder for speech it cannot map to text
_FOREIGN = re.compile(r"\bforeign\b", re.IGNORECASE)
# hallucinated boilerplate learned from subtitle training data
_BOILERPLATE = re.compile(
    r"(thanks? for watching|please subscribe|terima kasih|"
    r"amara\.org|subtitles? by|www\.[\w.]+)",
    re.IGNORECASE,
)


def collapse_repeats(text: str, max_repeat: int = 2, max_ngram: int = 6) -> str:
    """Collapse any n-gram repeated more than max_repeat times down to max_repeat.

    Shortest n-gram first: "कर दो" x12 must collapse on the 2-word unit, not on a
    6-word window that already contains three copies of it. Natural speech
    repetition (up to max_repeat) survives untouched.
    """
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        collapsed = False
        for n in range(1, min(max_ngram, (len(words) - i) // 2) + 1):
            gram = words[i : i + n]
            count, j = 1, i + n
            while words[j : j + n] == gram:
                count += 1
                j += n
            if count > max_repeat:
                out.extend(gram * max_repeat)
                i = j
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def clean_transcript(text: str) -> str:
    """Full cleanup: drop artifacts, collapse loops, normalize whitespace.

    Returns "" for segments that are pure noise, so callers can drop them.
    """
    text = _FOREIGN.sub(" ", text)
    text = _BOILERPLATE.sub(" ", text)
    text = collapse_repeats(text)
    text = re.sub(r"\s+", " ", text).strip()
    # sub-word leftovers after artifact removal carry no information
    if len(text.split()) <= 1 and len(text) <= 3:
        return ""
    return text
