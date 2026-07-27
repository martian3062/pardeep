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


def memory_text(
    taken_at: datetime | None, folder: str, caption: str, face_count: int = -1
) -> str:
    """Compose the retrievable sentence. Undated photos simply omit the date
    rather than claiming a wrong one.

    A photo containing people is marked as such. Asked to describe an old photo,
    the twin once answered that it had none while holding 1,307 — memes win vague
    photo queries because their captions are dense with text, while a real
    photograph's caption is one plain sentence. Saying "photo of N people" gives
    those queries something to match that a joke image does not have.
    """
    parts = []
    if taken_at:
        parts.append(taken_at.strftime("%d %b %Y"))
    if folder and folder.lower() not in ("camera", "camera roll", "sent"):
        parts.append(folder)
    prefix = ", ".join(parts)
    body = f"{prefix}: {caption}" if prefix else caption
    if face_count > 0:
        who = "1 person" if face_count == 1 else f"{face_count} people"
        body = f"[photo of {who}] {body}"
    return body


def run(db_path: Path, store: MemoryStore | None = None, cpu: bool = False) -> int:
    conn = sqlite3.connect(db_path)
    ph.ensure_schema(conn)

    rows = conn.execute(
        """SELECT path, taken_at, date_source, folder_hint, caption,
                  COALESCE(face_count, -1)
           FROM photos WHERE caption IS NOT NULL AND dupe_of IS NULL
           ORDER BY taken_at"""
    ).fetchall()
    conn.close()

    if not rows:
        console.print("[yellow]No captions yet — run `caption` first.[/yellow]")
        return 0

    if store is None:
        # captioning owns the GPU for hours; embedding on the CPU lets memories
        # be indexed while it runs instead of waiting for it to finish
        from ..memory.store import Embedder

        store = MemoryStore(embedder=Embedder(device="cpu") if cpu else None)
    memories: list[Memory] = []
    undated = 0
    with_people = 0
    for path, taken_at, source, folder, caption, face_count in rows:
        dated = source in ph.TRUSTED_SOURCES and taken_at
        when = datetime.fromisoformat(taken_at) if dated else None
        if not dated:
            undated += 1
        if face_count > 0:
            with_people += 1
        memories.append(
            Memory(
                text=memory_text(when, folder or "", caption, face_count),
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
    # keyword search only sees rows that are in the full-text index
    store.ensure_fts_index(rebuild=True)
    console.print(
        f"[green]Indexed {added} photo memories[/green] "
        f"({with_people} with people, {undated} undated"
        + (f", replaced {removed} existing" if removed else "")
        + ")"
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
