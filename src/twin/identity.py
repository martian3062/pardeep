"""Load the persona card and Mind Model, and render them for a prompt.

The Mind Model is ~20KB of JSON. Pasting all of it into every request would cost
more tokens than the memories it is meant to contextualise, so only the parts
that change how a reply should sound are rendered, plus the entry for whoever is
being spoken to.

Both files are gitignored: they describe a real person in detail.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TWIN_DIR = Path(__file__).resolve().parent
PERSONA_PATH = TWIN_DIR / "persona.md"
MIND_MODEL_PATH = TWIN_DIR / "mind_model.json"


@dataclass
class Identity:
    persona: str
    mind: dict

    @property
    def people(self) -> list[dict]:
        return self.mind.get("relationships", {}).get("people", [])

    def person(self, name: str) -> dict | None:
        """Find a relationship entry by loose name match.

        Contact names in the archive are messy — "Rosh....Lodu (Close college
        friend)" — so matching is substring-based in both directions.
        """
        if not name:
            return None
        needle = name.strip().lower()
        for entry in self.people:
            who = str(entry.get("name", "")).strip().lower()
            if not who:
                continue
            if who in needle or needle in who:
                return entry
        return None


@lru_cache(maxsize=1)
def load() -> Identity:
    persona = PERSONA_PATH.read_text(encoding="utf-8") if PERSONA_PATH.exists() else ""
    mind = json.loads(MIND_MODEL_PATH.read_text(encoding="utf-8")) if MIND_MODEL_PATH.exists() else {}
    return Identity(persona=persona, mind=mind)


def _bullets(items, limit: int) -> str:
    out = []
    for item in list(items)[:limit]:
        if isinstance(item, dict):
            # entries look like {"pattern": "...", "evidence": "..."}; the label
            # carries the meaning, the evidence is for humans reading the file
            text = item.get("pattern") or item.get("value") or item.get("habit") or next(
                (v for v in item.values() if isinstance(v, str)), ""
            )
        else:
            text = str(item)
        text = " ".join(str(text).split())
        if text:
            out.append(f"- {text}")
    return "\n".join(out)


def render_behaviour(identity: Identity, limit: int = 4) -> str:
    """The parts of the Mind Model that shape how a reply should sound."""
    b = identity.mind.get("behaviour", {})
    sections = [
        ("What you value", b.get("values", [])),
        ("How you decide", b.get("decision_patterns", [])),
        ("How you react emotionally", b.get("emotional_patterns", [])),
        ("Communication habits", b.get("communication_habits", [])),
    ]
    parts = []
    for title, items in sections:
        body = _bullets(items, limit)
        if body:
            parts.append(f"{title}:\n{body}")
    return "\n\n".join(parts)


def render_counterpart(identity: Identity, name: str) -> str:
    """How he specifically talks to this person, if the archive knows them."""
    entry = identity.person(name)
    if not entry:
        return ""
    bits = [f"You are talking to {entry.get('name', name)}."]
    for key in ("relationship", "how_you_talk_to_them", "tone", "topics", "notes"):
        value = entry.get(key)
        if isinstance(value, list):
            value = "; ".join(str(v) for v in value[:4])
        if value:
            label = key.replace("_", " ")
            bits.append(f"{label}: {' '.join(str(value).split())}")
    return "\n".join(bits)
