"""Turn the message store into retrievable memories.

Individual messages are too small to be useful on their own ("Nope", "han bss")
— meaning lives in the exchange. Conversations are therefore chunked into
windows that keep both sides together, so retrieval returns a scene rather than
a fragment.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

from rich.console import Console

from ..dataset.build_sft import counterpart_label, load_relationship_map
from .store import Memory, MemoryStore

console = Console()

# Anything obviously private stays out of guest-visible memory (Phase 9).
SENSITIVE_HINTS = (
    "password", "otp", "account number", "aadhaar", "pan card", "cvv",
    "bank", "salary", "loan", "upi", "पासवर्ड", "खाता",
)


def infer_trust(text: str, source: str) -> str:
    lowered = text.lower()
    if any(h in lowered for h in SENSITIVE_HINTS):
        return "secret"
    return "personal"  # nothing is public by default; promotion is deliberate


def chunk_conversation(
    rows: list[sqlite3.Row], max_chars: int = 900, max_gap_hours: float = 6.0
) -> list[tuple[str, datetime]]:
    """Group consecutive turns into readable scenes, breaking on long silences."""
    chunks: list[tuple[str, datetime]] = []
    buf: list[str] = []
    start: datetime | None = None
    prev: datetime | None = None
    size = 0

    for row in rows:
        ts = datetime.fromisoformat(row["timestamp"])
        gap_hours = ((ts - prev).total_seconds() / 3600) if prev else 0.0
        if buf and (size >= max_chars or gap_hours > max_gap_hours):
            chunks.append(("\n".join(buf), start or ts))
            buf, size, start = [], 0, None
        who = "Me" if row["speaker"] == "me" else (row["speaker_name"] or "Them")
        line = f"{who}: {row['text']}"
        buf.append(line)
        size += len(line)
        start = start or ts
        prev = ts

    if buf:
        chunks.append(("\n".join(buf), start or datetime.now()))
    return chunks


def build_episodic(
    conn: sqlite3.Connection,
    store: MemoryStore,
    exclude_sources: tuple[str, ...] = ("video",),
    min_chars: int = 60,
) -> int:
    relationships = load_relationship_map()
    placeholders = ",".join("?" * len(exclude_sources)) if exclude_sources else ""
    where = f"WHERE source NOT IN ({placeholders})" if exclude_sources else ""
    conv_ids = [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT conversation_id FROM messages {where}", list(exclude_sources)
        )
    ]

    memories: list[Memory] = []
    for conv_id in conv_ids:
        rows = conn.execute(
            f"""SELECT speaker, speaker_name, text, timestamp, source
                FROM messages WHERE conversation_id = ? {where.replace('WHERE', 'AND')}
                ORDER BY timestamp""",
            [conv_id, *exclude_sources],
        ).fetchall()
        if not rows:
            continue
        who = counterpart_label(conv_id, relationships)
        for text, ts in chunk_conversation(rows):
            if len(text) < min_chars:
                continue
            memories.append(
                Memory(
                    text=f"[conversation with {who}]\n{text}",
                    kind="episodic",
                    source=rows[0]["source"],
                    timestamp=ts,
                    conversation_id=conv_id,
                    counterpart=who,
                    trust=infer_trust(text, rows[0]["source"]),
                )
            )
    console.print(f"  built {len(memories)} episodic chunks from {len(conv_ids)} conversations")
    return store.add(memories)


def build_facts(store: MemoryStore) -> int:
    """Index the Mind Model as retrievable facts: relationships, values,
    decision patterns and life phases the twin can consult about itself."""
    import json

    from ..config import REPO_ROOT

    path = REPO_ROOT / "src" / "twin" / "mind_model.json"
    if not path.exists():
        console.print("  [yellow]no mind_model.json — run `dataset mind` first[/yellow]")
        return 0
    doc = json.loads(path.read_text(encoding="utf-8"))
    generated = datetime.fromisoformat(doc.get("generated_at", datetime.now().isoformat()))
    memories: list[Memory] = []

    for person in doc.get("relationships", {}).get("people", []) or []:
        text = (
            f"{person.get('contact')} is {person.get('likely_relationship')}. "
            f"How I talk to them: {person.get('how_he_talks_to_them')}. "
            f"Typical topics: {', '.join(person.get('typical_topics') or [])}"
        )
        memories.append(
            Memory(text=text, kind="fact", source="mind_model", timestamp=generated,
                   counterpart=str(person.get("contact", "")), trust="personal")
        )

    behaviour = doc.get("behaviour", {})
    for key in ("values", "motivations", "decision_patterns", "emotional_patterns",
                "communication_habits", "recurring_concerns", "interests"):
        for item in behaviour.get(key) or []:
            memories.append(
                Memory(text=f"[{key.replace('_', ' ')}] {item}", kind="fact",
                       source="mind_model", timestamp=generated, trust="personal")
            )

    for phase in doc.get("trajectory", {}).get("phases", []) or []:
        memories.append(
            Memory(
                text=f"[life phase {phase.get('period')}] {phase.get('what_they_were_doing')}",
                kind="fact", source="mind_model", timestamp=generated, trust="personal",
            )
        )

    console.print(f"  built {len(memories)} facts from the Mind Model")
    return store.add(memories)
