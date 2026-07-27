"""How he seems right now, and what the twin does about it.

The camera never reaches this process. MediaPipe runs in the browser as WASM and
sends only a handful of numbers — the video stays on the machine it was captured
on, which is the same rule the rest of the project follows for raw data.

A mood signal is weak evidence. Someone squinting at a bright screen scores as
"stressed", and a face at rest scores as bored. So mood adjusts *tone* and is
never treated as fact: the twin is told he looks tired, not that he is tired, and
is told not to bring it up unless it fits.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("data/processed/mood.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS mood_samples (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    mood TEXT NOT NULL,
    score REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'camera'
);
CREATE INDEX IF NOT EXISTS idx_mood_time ON mood_samples(created_at);
"""

MOODS = ("happy", "tired", "stressed", "surprised", "neutral")

# A sample below this is noise — a blink, a glance away, a moment of squinting.
MIN_CONFIDENCE = 0.35
# Older than this and it says nothing about the conversation happening now.
FRESH_FOR = timedelta(minutes=10)


@dataclass
class Mood:
    label: str
    score: float
    at: datetime

    @property
    def fresh(self) -> bool:
        return datetime.now() - self.at <= FRESH_FOR

    @property
    def usable(self) -> bool:
        return self.fresh and self.score >= MIN_CONFIDENCE and self.label != "neutral"


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def record(conn: sqlite3.Connection, mood: str, score: float, source: str = "camera") -> int:
    if mood not in MOODS:
        raise ValueError(f"unknown mood {mood!r}")
    cur = conn.execute(
        "INSERT INTO mood_samples (created_at, mood, score, source) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), mood, float(score), source),
    )
    conn.commit()
    return int(cur.lastrowid)


def current(conn: sqlite3.Connection) -> Mood | None:
    row = conn.execute(
        "SELECT mood, score, created_at FROM mood_samples ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return Mood(label=row[0], score=float(row[1]), at=datetime.fromisoformat(row[2]))


def timeline(conn: sqlite3.Connection, days: int = 7) -> list[tuple[str, str, int]]:
    """Per-day mood counts — what a weekly digest would report."""
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    return [
        (r[0], r[1], int(r[2]))
        for r in conn.execute(
            """SELECT substr(created_at, 1, 10) day, mood, count(*)
               FROM mood_samples WHERE created_at >= ? AND score >= ?
               GROUP BY day, mood ORDER BY day""",
            (since, MIN_CONFIDENCE),
        )
    ]


_TONE = {
    "tired": (
        "He looks tired right now. Keep it short and do not pile on questions. "
        "Do not announce that he looks tired unless it comes up naturally."
    ),
    "stressed": (
        "He looks tense right now. Be direct and useful, skip the banter, and do "
        "not add pressure. Do not diagnose his mood out loud."
    ),
    "happy": "He seems in a good mood. Match it — he is more playful when he is like this.",
    "surprised": "He looks caught off guard. Answer plainly before anything else.",
}


def tone_rule(mood: Mood | None) -> str:
    """The line added to the system prompt, or nothing when the signal is weak."""
    if mood is None or not mood.usable:
        return ""
    return _TONE.get(mood.label, "")
