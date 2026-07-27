"""Where each day and each correction is kept.

The archive is finished — it stops at the day it was exported. Everything the
twin learns from here comes through this file: what he did today, and where the
twin got him wrong. Both are kept as plain rows so a training batch is a query,
not a migration.

Feedback is stored as preference pairs rather than scores. A thumbs-down alone
says something was wrong but not what right looks like; the rewrite box is the
part that can actually train a model, and DPO wants exactly (prompt, chosen,
rejected).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path("data/processed/diary.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY,
    day TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,          -- voice | text
    lang TEXT,
    text TEXT NOT NULL,
    audio_seconds REAL,
    indexed_at TEXT                -- when it reached the memory store
);
CREATE INDEX IF NOT EXISTS idx_entries_day ON entries(day);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    prompt TEXT NOT NULL,          -- what was asked
    context TEXT,                  -- memories the twin had, for reproducibility
    rejected TEXT NOT NULL,        -- what the twin said
    chosen TEXT,                   -- how he would have said it; NULL for a bare thumbs-down
    verdict TEXT NOT NULL,         -- up | down
    counterpart TEXT,
    exported_at TEXT               -- set once it lands in a training batch
);
CREATE INDEX IF NOT EXISTS idx_feedback_export ON feedback(exported_at);

CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY,
    period TEXT NOT NULL,          -- e.g. 2026-W31
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(period)
);
"""


@dataclass
class Entry:
    text: str
    source: str = "text"
    lang: str = ""
    audio_seconds: float | None = None
    day: str = field(default_factory=lambda: date.today().isoformat())


@dataclass
class Feedback:
    prompt: str
    rejected: str
    verdict: str  # up | down
    chosen: str = ""
    context: str = ""
    counterpart: str = ""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def add_entry(conn: sqlite3.Connection, entry: Entry) -> int:
    cur = conn.execute(
        """INSERT INTO entries (day, created_at, source, lang, text, audio_seconds)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            entry.day,
            datetime.now().isoformat(timespec="seconds"),
            entry.source,
            entry.lang,
            entry.text.strip(),
            entry.audio_seconds,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def add_feedback(conn: sqlite3.Connection, fb: Feedback) -> int:
    cur = conn.execute(
        """INSERT INTO feedback (created_at, prompt, context, rejected, chosen, verdict, counterpart)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            fb.prompt,
            fb.context,
            fb.rejected,
            fb.chosen or None,
            fb.verdict,
            fb.counterpart,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def unindexed(conn: sqlite3.Connection) -> list[tuple[int, str, str, str]]:
    return [
        (r[0], r[1], r[2], r[3])
        for r in conn.execute(
            "SELECT id, day, text, lang FROM entries WHERE indexed_at IS NULL ORDER BY day"
        )
    ]


def mark_indexed(conn: sqlite3.Connection, ids: list[int]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany("UPDATE entries SET indexed_at = ? WHERE id = ?", [(now, i) for i in ids])
    conn.commit()


def entries_for(conn: sqlite3.Connection, day: str) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM entries WHERE day = ? ORDER BY created_at", (day,)
    ).fetchall()


def pending_pairs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Preference pairs not yet used in a training batch.

    Only rows with a rewrite are usable: DPO needs something better to prefer,
    and a bare thumbs-down gives it nothing to move toward.
    """
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """SELECT * FROM feedback
           WHERE exported_at IS NULL AND chosen IS NOT NULL AND length(trim(chosen)) > 0
           ORDER BY created_at"""
    ).fetchall()


def mark_exported(conn: sqlite3.Connection, ids: list[int]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany("UPDATE feedback SET exported_at = ? WHERE id = ?", [(now, i) for i in ids])
    conn.commit()


def stats(conn: sqlite3.Connection) -> dict:
    def one(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0] or 0)

    return {
        "entries": one("SELECT count(*) FROM entries"),
        "days": one("SELECT count(DISTINCT day) FROM entries"),
        "unindexed": one("SELECT count(*) FROM entries WHERE indexed_at IS NULL"),
        "feedback": one("SELECT count(*) FROM feedback"),
        "pairs_ready": one(
            "SELECT count(*) FROM feedback WHERE exported_at IS NULL "
            "AND chosen IS NOT NULL AND length(trim(chosen)) > 0"
        ),
        "thumbs_down": one("SELECT count(*) FROM feedback WHERE verdict = 'down'"),
    }
