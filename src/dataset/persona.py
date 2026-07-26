"""Extract the persona card: HOW Pardeep talks.

The card is prepended as the system prompt in every generation and in the SFT
examples, so the model starts each reply already inside his voice.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from rich.console import Console

from ..config import REPO_ROOT
from .llm import ask, format_messages, sample_my_messages

console = Console()

PROMPT = """Below are real messages written or spoken by ONE person (Pardeep), taken from his
WhatsApp chats and transcribed phone calls. He is an Indian engineering student / early-career
AI-bioinformatics person from Punjab who mixes Punjabi, Hindi and English.

Your task: write a PERSONA CARD that another language model will use as its system prompt to
reply exactly as him. Study the samples and capture what is actually there — not a flattering
or generic description.

Cover:
1. **Voice** — sentence length and rhythm, punctuation/capitalisation habits, how he opens and
   closes conversations.
2. **Language mixing** — when he uses Punjabi vs Hindi vs English, whether he types Roman or
   native script, which words he never translates.
3. **Signature phrases** — actual recurring words, fillers, greetings, exclamations. Quote them
   verbatim; these matter more than adjectives.
4. **Tone & humour** — how he jokes, teases, complains, encourages.
5. **Register shifts** — how he differs with family vs friends vs formal/work calls.
6. **Recurring topics & concerns** — what he keeps coming back to.
7. **Don'ts** — things he never does (that a generic assistant would), e.g. over-formality,
   emoji habits he lacks, corporate phrasing.

Write it as direct second-person instructions ("You write in short bursts...", "You say 'yaar'
when..."). Be concrete and quote real examples. 500-900 words. Output ONLY the markdown card,
no preamble.

=== MESSAGES ===
{samples}
"""


def extract_persona(
    conn: sqlite3.Connection, model: str = "claude-opus-4-5", n_samples: int = 400
) -> Path:
    msgs = sample_my_messages(conn, n=n_samples)
    if not msgs:
        raise SystemExit("No messages by you in the corpus yet")
    console.print(f"Analysing {len(msgs)} of your messages ...")
    card = ask(PROMPT.format(samples=format_messages(msgs)), model=model, max_tokens=4000)

    out = REPO_ROOT / "src" / "twin" / "persona.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card.strip() + "\n", encoding="utf-8")
    return out
