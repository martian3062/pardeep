"""Litestar API for the twin.

    uv run litestar --app app.api.main:app run --port 8100 --reload

The heavy objects — the embedding model, the LanceDB handle, the persona — load
once at startup and are reused, because loading bge-m3 per request would put
several seconds on every message.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src  # noqa: F401,E402  — pins model caches to E:\cache before torch loads

from litestar import Litestar, get, post  # noqa: E402
from litestar.config.cors import CORSConfig  # noqa: E402
from litestar.exceptions import NotFoundException  # noqa: E402
from litestar.response import File  # noqa: E402

from src.twin.chat import Twin  # noqa: E402
from src.twin.retrieve import gather  # noqa: E402

from .photos import resolve_photo  # noqa: E402

# The UI is a separate dev server; in production both sit behind one origin.
cors = CORSConfig(allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"])

_twin: Twin | None = None


def twin() -> Twin:
    global _twin
    if _twin is None:
        _twin = Twin()
    return _twin


@dataclass
class ChatRequest:
    message: str
    counterpart: str = ""
    guest: bool = False
    backend: str = "auto"


@dataclass
class MemoryOut:
    text: str
    kind: str
    timestamp: str
    counterpart: str
    photo_path: str
    score: float


@dataclass
class ChatResponse:
    reply: str
    backend: str
    model: str
    memories: list[MemoryOut]


def _to_out(snippets) -> list[MemoryOut]:
    return [
        MemoryOut(
            text=s.text,
            kind=s.kind,
            timestamp=s.timestamp,
            counterpart=s.counterpart,
            # only photo memories carry a file the UI can render
            photo_path=s.source_ref if s.kind == "photo" else "",
            score=round(s.score, 4),
        )
        for s in snippets
    ]


@post("/api/chat")
async def chat(data: ChatRequest) -> ChatResponse:
    t = twin()
    t.guest = data.guest
    reply = t.reply(data.message, counterpart=data.counterpart, backend=data.backend)
    return ChatResponse(
        reply=reply.text,
        backend=reply.backend,
        model=reply.model,
        memories=_to_out(reply.memories),
    )


@get("/api/memory")
async def memory_search(q: str, limit: int = 12, guest: bool = False) -> list[MemoryOut]:
    """Search memory directly — the inspector view, without generating a reply."""
    return _to_out(gather(q, store=twin().store, limit=limit, guest=guest))


@get("/api/photo")
async def photo(path: str) -> File:
    resolved = resolve_photo(path)
    if resolved is None:
        raise NotFoundException("not an archive photo")
    return File(path=resolved, filename=resolved.name, content_disposition_type="inline")


@get("/api/stats")
async def stats() -> dict:
    from src.twin.router import available_backends

    store = twin().store
    table = store._open()
    counts: dict[str, int] = {}
    if table is not None:
        df = table.to_pandas()
        counts = {k: int(v) for k, v in df["kind"].value_counts().to_dict().items()}
    return {
        "memories": store.count(),
        "by_kind": counts,
        "backends": available_backends(),
    }


@post("/api/reset")
async def reset() -> dict:
    """Forget the conversation, keep the memories."""
    twin().history.clear()
    return {"ok": True}


@dataclass
class DiaryRequest:
    text: str
    day: str = ""


@dataclass
class DiaryResponse:
    entry_id: int
    indexed: int
    reaction: str


@post("/api/diary")
async def diary_write(data: DiaryRequest) -> DiaryResponse:
    """Save a day, index it, and let the twin answer it."""
    from src.diary import capture
    from src.diary import store as st

    conn = st.connect()
    entry_id = capture.capture_text(conn, data.text, day=data.day or None)
    indexed = capture.index_new(conn, store=twin().store)
    reaction = twin().reply(f"Aaj ka mera diary entry: {data.text}").text
    return DiaryResponse(entry_id=entry_id, indexed=indexed, reaction=reaction)


@get("/api/diary")
async def diary_list(limit: int = 30) -> list[dict]:
    from src.diary import store as st

    conn = st.connect()
    conn.row_factory = __import__("sqlite3").Row
    rows = conn.execute(
        "SELECT id, day, created_at, source, text FROM entries ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


@dataclass
class CorrectionRequest:
    prompt: str
    said: str
    instead: str = ""
    verdict: str = "down"
    counterpart: str = ""


@post("/api/feedback")
async def feedback(data: CorrectionRequest) -> dict:
    """Log how he would have said it — the only signal that can retrain the twin."""
    from src.diary import store as st

    conn = st.connect()
    fb_id = st.add_feedback(
        conn,
        st.Feedback(
            prompt=data.prompt,
            rejected=data.said,
            chosen=data.instead,
            verdict="down" if data.instead else data.verdict,
            counterpart=data.counterpart,
        ),
    )
    return {"id": fb_id, "trainable": bool(data.instead.strip())}


@get("/api/diary/stats")
async def diary_stats() -> dict:
    from src.diary import store as st

    return st.stats(st.connect())


app = Litestar(
    route_handlers=[
        chat,
        memory_search,
        photo,
        stats,
        reset,
        diary_write,
        diary_list,
        diary_stats,
        feedback,
    ],
    cors_config=cors,
)
