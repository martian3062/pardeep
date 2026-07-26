"""Unified message store (SQLite). All sources normalize into the `messages` table.

Idempotency: files are tracked by sha256 in `ingest_files`; individual rows are
deduped by `content_hash` (INSERT OR IGNORE), so re-running any ingest is safe.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    speaker TEXT NOT NULL CHECK (speaker IN ('me', 'other')),
    speaker_name TEXT,
    text TEXT NOT NULL,
    audio_ref TEXT,
    lang TEXT,
    is_media INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_messages_conv_ts ON messages(conversation_id, timestamp);

-- Call recordings: transcript segments awaiting speaker labels from diarization.
CREATE TABLE IF NOT EXISTS call_segments (
    id INTEGER PRIMARY KEY,
    audio_ref TEXT NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT,
    lang TEXT,
    UNIQUE (audio_ref, start_sec, end_sec)
);

CREATE TABLE IF NOT EXISTS ingest_files (
    path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    item_count INTEGER NOT NULL
);
"""


class Message(BaseModel):
    source: str
    conversation_id: str
    timestamp: datetime
    speaker: Literal["me", "other"]
    speaker_name: str | None = None
    text: str
    audio_ref: str | None = None
    lang: str | None = None
    is_media: bool = False

    def content_hash(self) -> str:
        key = "\x1f".join(
            [
                self.source,
                self.conversation_id,
                self.timestamp.isoformat(),
                self.speaker_name or "",
                self.text,
            ]
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def already_ingested(conn: sqlite3.Connection, path: Path, sha256: str) -> bool:
    row = conn.execute(
        "SELECT sha256 FROM ingest_files WHERE path = ?", (str(path),)
    ).fetchone()
    return row is not None and row["sha256"] == sha256


def record_file(conn: sqlite3.Connection, path: Path, sha256: str, item_count: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO ingest_files (path, sha256, ingested_at, item_count) VALUES (?, ?, ?, ?)",
        (str(path), sha256, datetime.now(timezone.utc).isoformat(), item_count),
    )
    conn.commit()


def insert_messages(conn: sqlite3.Connection, messages: Iterable[Message]) -> int:
    cur = conn.executemany(
        """INSERT OR IGNORE INTO messages
           (source, conversation_id, timestamp, speaker, speaker_name, text, audio_ref, lang, is_media, content_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                m.source,
                m.conversation_id,
                m.timestamp.isoformat(),
                m.speaker,
                m.speaker_name,
                m.text,
                m.audio_ref,
                m.lang,
                int(m.is_media),
                m.content_hash(),
            )
            for m in messages
        ],
    )
    conn.commit()
    return cur.rowcount


def insert_call_segments(
    conn: sqlite3.Connection,
    audio_ref: str,
    segments: Iterable[tuple[float, float, str]],
    lang: str | None,
) -> int:
    cur = conn.executemany(
        """INSERT OR IGNORE INTO call_segments (audio_ref, start_sec, end_sec, text, lang)
           VALUES (?, ?, ?, ?, ?)""",
        [(audio_ref, s, e, t, lang) for s, e, t in segments],
    )
    conn.commit()
    return cur.rowcount


def pending_call_files(conn: sqlite3.Connection) -> list[str]:
    """Call recordings that have transcript segments but no speaker labels yet."""
    return [
        row["audio_ref"]
        for row in conn.execute(
            "SELECT DISTINCT audio_ref FROM call_segments WHERE speaker IS NULL"
        )
    ]


def fetch_segments(conn: sqlite3.Connection, audio_ref: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM call_segments WHERE audio_ref = ? ORDER BY start_sec",
        (audio_ref,),
    ).fetchall()


def call_language_map(conn: sqlite3.Connection) -> dict[str, str]:
    """Most common detected language per call file, from the current transcripts."""
    rows = conn.execute(
        """SELECT audio_ref, lang, COUNT(*) n FROM call_segments
           WHERE lang IS NOT NULL GROUP BY audio_ref, lang ORDER BY n DESC"""
    )
    out: dict[str, str] = {}
    for row in rows:
        out.setdefault(row["audio_ref"], row["lang"])
    return out


def reset_call_transcripts(conn: sqlite3.Connection) -> tuple[int, int]:
    """Drop all call transcripts + their file records so calls re-transcribe.
    Diarization embedding caches are keyed by audio path and stay valid."""
    segs = conn.execute("DELETE FROM call_segments").rowcount
    conn.execute("DELETE FROM messages WHERE source = 'call'")
    files = conn.execute("DELETE FROM ingest_files WHERE path LIKE '%\\calls\\%'").rowcount
    conn.commit()
    return segs, files


def reset_call_labels(conn: sqlite3.Connection) -> tuple[int, int]:
    """Clear diarization results so labeling can be re-run from cached embeddings.
    Returns (segments_unlabeled, messages_deleted)."""
    segs = conn.execute("UPDATE call_segments SET speaker = NULL").rowcount
    # video-derived rows are promoted by the same pass, so they must be cleared
    # too or relabeling leaves stale duplicates behind
    msgs = conn.execute(
        "DELETE FROM messages WHERE source IN ('call', 'video')"
    ).rowcount
    conn.commit()
    return segs, msgs


def label_segments(conn: sqlite3.Connection, labels: Iterable[tuple[str, int]]) -> None:
    """labels: (speaker, segment_id) pairs."""
    conn.executemany("UPDATE call_segments SET speaker = ? WHERE id = ?", list(labels))
    conn.commit()


def stats(conn: sqlite3.Connection) -> dict:
    out: dict = {"by_source": {}, "by_speaker": {}}
    for row in conn.execute(
        "SELECT source, COUNT(*) n FROM messages GROUP BY source ORDER BY n DESC"
    ):
        out["by_source"][row["source"]] = row["n"]
    for row in conn.execute("SELECT speaker, COUNT(*) n FROM messages GROUP BY speaker"):
        out["by_speaker"][row["speaker"]] = row["n"]
    out["messages"] = sum(out["by_source"].values())
    out["conversations"] = conn.execute(
        "SELECT COUNT(DISTINCT conversation_id) FROM messages"
    ).fetchone()[0]
    out["call_segments"] = conn.execute("SELECT COUNT(*) FROM call_segments").fetchone()[0]
    out["files_ingested"] = conn.execute("SELECT COUNT(*) FROM ingest_files").fetchone()[0]
    return out
