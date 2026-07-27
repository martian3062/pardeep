"""A record of everything the twin has said to someone who is not him.

Without this, guest mode is a promise. With it, he can read back exactly what his
twin told people, which of those replies had something stripped out of them, and
what was refused — the difference between believing the guardrails work and being
able to check.

Owner conversations are not logged. He does not need surveillance of himself, and
a log of his own chats would be one more copy of his private life sitting on disk.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("data/processed/audit.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS guest_turns (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    guest_label TEXT,
    message TEXT NOT NULL,
    reply TEXT NOT NULL,
    outcome TEXT NOT NULL,        -- answered | refused | redacted
    blocked_reason TEXT,
    redactions TEXT,              -- comma-separated kinds removed
    memories_used INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_audit_time ON guest_turns(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_outcome ON guest_turns(outcome);
"""


@dataclass
class Turn:
    message: str
    reply: str
    outcome: str
    guest_label: str = ""
    blocked_reason: str = ""
    redactions: tuple[str, ...] = ()
    memories_used: int = 0


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def record(conn: sqlite3.Connection, turn: Turn) -> int:
    cur = conn.execute(
        """INSERT INTO guest_turns
           (created_at, guest_label, message, reply, outcome, blocked_reason,
            redactions, memories_used)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            turn.guest_label,
            turn.message,
            turn.reply,
            turn.outcome,
            turn.blocked_reason or None,
            ",".join(turn.redactions) or None,
            turn.memories_used,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def recent(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM guest_turns ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()


def digest(conn: sqlite3.Connection, days: int = 7) -> dict:
    """What the twin told others this week — the thing worth reading."""
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    def count(where: str, args: tuple = ()) -> int:
        return int(
            conn.execute(
                f"SELECT count(*) FROM guest_turns WHERE created_at >= ? {where}",
                (since, *args),
            ).fetchone()[0]
        )

    kinds: dict[str, int] = {}
    for (row,) in conn.execute(
        "SELECT redactions FROM guest_turns WHERE created_at >= ? AND redactions IS NOT NULL",
        (since,),
    ):
        for kind in row.split(","):
            kinds[kind] = kinds.get(kind, 0) + 1

    return {
        "turns": count(""),
        "answered": count("AND outcome = 'answered'"),
        "refused": count("AND outcome = 'refused'"),
        "redacted": count("AND outcome = 'redacted'"),
        "redaction_kinds": kinds,
    }
