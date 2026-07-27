"""Record a day, then make it part of what the twin knows.

A diary entry only matters if it can be recalled later, so indexing runs in the
same breath as capture. Entries carry today's date, which puts them at the recent
end of the timeline where the recency tilt gives them a small edge over a
nine-year-old chat about the same subject — which is right, since when he asks
"what did I do this week" he means this week.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

from ..memory.store import Memory, MemoryStore
from . import store as st


def capture_text(conn: sqlite3.Connection, text: str, day: str | None = None) -> int:
    entry = st.Entry(text=text, source="text")
    if day:
        entry.day = day
    return st.add_entry(conn, entry)


def capture_voice(conn: sqlite3.Connection, device: int | None = None) -> tuple[int, str] | None:
    """Speak the entry. Transcription is local; nothing recorded leaves."""
    from ..voice.listen import Ears

    ears = Ears(device=device)
    utt = ears.listen()
    if utt is None:
        return None
    text, lang = ears.transcribe(utt)
    if not text.strip():
        return None
    entry_id = st.add_entry(
        conn, st.Entry(text=text, source="voice", lang=lang, audio_seconds=utt.seconds)
    )
    return entry_id, text


def index_new(conn: sqlite3.Connection, store: MemoryStore | None = None) -> int:
    """Push captured entries into the memory store."""
    rows = st.unindexed(conn)
    if not rows:
        return 0
    store = store or MemoryStore()
    memories = []
    ids = []
    for entry_id, day, text, _lang in rows:
        memories.append(
            Memory(
                text=f"Diary, {datetime.fromisoformat(day).strftime('%d %b %Y')}: {text}",
                kind="diary",
                source="diary",
                timestamp=datetime.fromisoformat(day),
                conversation_id=f"diary:{entry_id}",
                # a diary is the most private thing here; guests never see it
                trust="secret",
            )
        )
        ids.append(entry_id)
    added = store.add(memories)
    store.ensure_fts_index(rebuild=True)
    st.mark_indexed(conn, ids)
    return added
