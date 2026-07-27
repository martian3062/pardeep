"""Give a guest his voice without giving them his life.

The persona card is what makes the twin sound like him, so a guest needs it. But
it is not only style: it names his friends, quotes real phrases from real
conversations, and describes his habits closely enough to be a biography. A guest
should hear how he talks, not learn who he talks to.

Sections are kept or dropped by heading, because the card is written in sections
and a heading says what the section is for. Anything unrecognised is dropped —
a card that grows a new "Family" section later should not start leaking on the
day it is added.
"""
from __future__ import annotations

import re

# Style only. These describe HOW he speaks.
_KEEP = (
    "voice",
    "rhythm",
    "language",
    "tone",
    "style",
    "signature phrase",
    "phrases",
    "humor",
    "humour",
    "writing",
    "speech",
    "register",
)

# Anything naming people, places, history or beliefs.
_DROP = (
    "people",
    "relationship",
    "family",
    "friend",
    "work",
    "career",
    "value",
    "belief",
    "history",
    "timeline",
    "background",
    "interest",
    "concern",
    "fact",
    "detail",
)

_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.M)

GUEST_NOTE = (
    "This is only a description of how he speaks. You do not have his history, "
    "his relationships, or any detail about his life, and you must not invent any."
)


def _wanted(heading: str) -> bool:
    low = heading.lower()
    if any(word in low for word in _DROP):
        return False
    return any(word in low for word in _KEEP)


def sanitize_persona(card: str) -> str:
    """Keep the sections that describe his voice; drop the rest."""
    if not card.strip():
        return ""
    matches = list(_HEADING.finditer(card))
    if not matches:
        return GUEST_NOTE

    kept: list[str] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(card)
        if _wanted(heading):
            kept.append(card[m.start() : end].rstrip())

    body = "\n\n".join(kept).strip()
    return f"{body}\n\n{GUEST_NOTE}" if body else GUEST_NOTE
