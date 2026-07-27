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

# Sentinel for photos with no believable date: old enough to sort last, but not
# the epoch. datetime(1970,1,1).timestamp() raises OSError on Windows in any
# timezone east of UTC — local midnight maps to a negative time_t the CRT
# rejects — and Memory.to_row calls .timestamp() on every row.
UNDATED = datetime(1980, 1, 1)


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
                # no believable date: sort last rather than claim it is recent
                timestamp=when or UNDATED,
                conversation_id=str(path),
                trust="personal",
            )
        )

    # Captioning is incremental, so `index` gets re-run as more photos finish.
    # MemoryStore.add is a plain append, so without this every re-run would
    # duplicate every photo memory already in the store.
    removed = _drop_existing_photos(store)
    added = store.add(memories)
    console.print(
        f"[green]Indexed {added} photo memories[/green] "
        f"({undated} undated" + (f", replaced {removed} existing" if removed else "") + ")"
    )
    return added


def _drop_existing_photos(store: MemoryStore) -> int:
    """Remove previously indexed photo memories so re-indexing replaces them."""
    table = store._open()
    if table is None:
        return 0
    try:
        before = table.count_rows()
        table.delete("kind = 'photo'")
        return before - table.count_rows()
    except Exception:  # older LanceDB without delete(): fall back to appending
        return 0
