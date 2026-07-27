"""Push captioned photos into the same LanceDB store as chats and calls.

A photo memory reads like a diary line — "12 Aug 2017, camera: four people on a
motorcycle outside a dhaba" — so a single query can pull back conversations and
pictures from the same week.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from rich.console import Console

from ..memory.store import Memory, MemoryStore
from . import photos as ph

console = Console()


def memory_text(taken_at: datetime | None, folder: str, caption: str) -> str:
    """Compose the retrievable sentence. Undated photos simply omit the date
    rather than claiming a wrong one."""
    parts = []
    if taken_at:
        parts.append(taken_at.strftime("%d %b %Y"))
    if folder and folder.lower() not in ("camera", "camera roll", "sent"):
        parts.append(folder)
    prefix = ", ".join(parts)
    return f"{prefix}: {caption}" if prefix else caption


def run(db_path: Path, store: MemoryStore | None = None) -> int:
    conn = sqlite3.connect(db_path)
    conn.executescript(ph.SCHEMA)

    rows = conn.execute(
        """SELECT path, taken_at, date_source, folder_hint, caption
           FROM photos WHERE caption IS NOT NULL ORDER BY taken_at"""
    ).fetchall()
    conn.close()

    if not rows:
        console.print("[yellow]No captions yet — run `caption` first.[/yellow]")
        return 0

    store = store or MemoryStore()
    memories: list[Memory] = []
    undated = 0
    for path, taken_at, source, folder, caption in rows:
        dated = source in ph.TRUSTED_SOURCES and taken_at
        when = datetime.fromisoformat(taken_at) if dated else None
        if not dated:
            undated += 1
        memories.append(
            Memory(
                text=memory_text(when, folder or "", caption),
                kind="photo",
                source="photo",
                # a photo with no believable date gets the epoch, so recency
                # weighting ranks it last instead of pretending it is from today
                timestamp=when or datetime(1970, 1, 1),
                conversation_id=str(path),
                trust="personal",
            )
        )

    added = store.add(memories)
    console.print(f"[green]Indexed {added} photo memories[/green] ({undated} undated)")
    return added
