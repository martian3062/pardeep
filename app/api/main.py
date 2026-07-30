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
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src  # noqa: F401,E402  — pins model caches to E:\cache before torch loads

from litestar import Litestar, get, post  # noqa: E402
from litestar.config.cors import CORSConfig  # noqa: E402
from litestar.datastructures import UploadFile  # noqa: E402
from litestar.enums import MediaType, RequestEncodingType  # noqa: E402
from litestar.exceptions import NotFoundException  # noqa: E402
from litestar.params import Body  # noqa: E402
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


@dataclass
class MoodRequest:
    mood: str
    score: float


@post("/api/mood")
async def mood_report(data: MoodRequest) -> dict:
    """Receive a mood reading from the browser.

    Only the label and a confidence arrive here. The camera frames are processed
    by MediaPipe as WASM inside the page and never sent anywhere.
    """
    from src.twin import mood as md

    conn = md.connect()
    md.record(conn, data.mood, data.score)
    now = md.current(conn)
    return {
        "ok": True,
        "mood": now.label if now else "",
        "affects_replies": bool(now and now.usable),
    }


@get("/api/record/questions")
async def record_questions() -> dict:
    """The session plan, minus anything already answered."""
    from src.voice import session as ss

    conn = ss.connect()
    done = ss.answered_ids(conn)
    plan = [q for q in ss.build_questions(seed=7) if q["id"] not in done]
    return {"questions": plan, "progress": ss.progress(conn)}


@post("/api/record/clip", media_type=MediaType.JSON)
async def record_clip(
    data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
    question_id: str,
    question: str,
    lang: str,
) -> dict:
    """Receive one recorded answer, measure it, and say immediately if it is bad.

    Quality is judged now rather than after the session, because the archive's
    fatal flaw — a 48kHz container holding 1.6kHz of real audio — is invisible
    unless something measures it.
    """
    from src.voice import session as ss

    audio = await data.read()
    conn = ss.connect()
    clip_id, quality, problem = ss.save_clip(
        conn, audio, {"id": question_id, "text": question, "lang": lang}
    )
    return {
        "id": clip_id,
        "accepted": quality.ok,
        "problem": problem,
        "seconds": round(quality.seconds, 1),
        "sample_rate": quality.sample_rate,
        "bandwidth_hz": round(quality.bandwidth_hz),
        "rms": round(quality.rms, 4),
        "progress": ss.progress(conn),
    }


@get("/api/record/progress")
async def record_progress() -> dict:
    from src.voice import session as ss

    return ss.progress(ss.connect())


@post("/api/record/export")
async def record_export() -> dict:
    from src.voice import session as ss

    conn = ss.connect()
    path = ss.export_manifest(conn)
    return {"manifest": str(path), "progress": ss.progress(conn)}


@get("/api/mood")
async def mood_now() -> dict:
    from src.twin import mood as md

    conn = md.connect()
    now = md.current(conn)
    return {
        "mood": now.label if now else "",
        "score": round(now.score, 3) if now else 0.0,
        "fresh": bool(now and now.fresh),
        "timeline": md.timeline(conn),
    }


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
        mood_report,
        mood_now,
        record_questions,
        record_clip,
        record_progress,
        record_export,
    ],
    cors_config=cors,
)
