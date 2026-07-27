"""Turn corrections into training data, and days into digests.

The twin was trained once on a finished archive. What keeps it from drifting into
a museum piece is this: every time it answers in a way he would not have, he
rewrites it, and that pair becomes something DPO can learn from.

Quality over volume. A month of thumbs with no rewrites teaches nothing, so only
pairs with an actual alternative are exported, and near-identical pairs are
dropped because a preference model learns nothing from "prefer X over almost X".
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

from . import store as st

DPO_DIR = Path("data/datasets")

# Below this the two answers differ by a comma. DPO on such a pair pushes the
# model toward noise.
MIN_DIFFERENCE = 0.12


def _too_similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio() > (1.0 - MIN_DIFFERENCE)


def export_pairs(conn: sqlite3.Connection, out_dir: Path = DPO_DIR) -> tuple[Path | None, int]:
    """Write pending corrections as a DPO batch."""
    rows = st.pending_pairs(conn)
    usable, ids, skipped = [], [], 0
    for r in rows:
        chosen, rejected = (r["chosen"] or "").strip(), (r["rejected"] or "").strip()
        if not chosen or _too_similar(chosen, rejected):
            skipped += 1
            ids.append(r["id"])  # mark handled so it is not reconsidered forever
            continue
        usable.append(
            {
                "prompt": r["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "counterpart": r["counterpart"] or "",
            }
        )
        ids.append(r["id"])

    if not usable:
        st.mark_exported(conn, ids)
        return None, skipped

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    path = out_dir / f"dpo_{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in usable:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    st.mark_exported(conn, ids)
    return path, skipped


def week_of(day: date) -> str:
    iso = day.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def entries_in_week(conn: sqlite3.Connection, anchor: date) -> list[sqlite3.Row]:
    start = anchor - timedelta(days=anchor.weekday())
    end = start + timedelta(days=7)
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM entries WHERE day >= ? AND day < ? ORDER BY day, created_at",
        (start.isoformat(), end.isoformat()),
    ).fetchall()


def build_digest(conn: sqlite3.Connection, anchor: date | None = None) -> str | None:
    """Summarise a week in his own voice, and keep it as a memory of its own.

    A digest is worth storing separately from the entries because retrieval over
    seven scattered fragments answers "what happened Tuesday", while the digest
    answers "how was the week" — a question the fragments cannot.
    """
    anchor = anchor or date.today()
    rows = entries_in_week(conn, anchor)
    if not rows:
        return None

    from ..twin.chat import Twin

    joined = "\n".join(f"{r['day']}: {r['text']}" for r in rows)
    twin = Twin()
    reply = twin.reply(
        "Ye mere is hafte ke diary entries hain. Inko apne andaz me chhota sa "
        "summary bana — kya hua, kaisa raha, kya pending hai. Koi cheez mat "
        "banana jo likhi nahi hai.\n\n" + joined
    )
    text = reply.text.strip()
    conn.execute(
        """INSERT INTO digests (period, created_at, text) VALUES (?, ?, ?)
           ON CONFLICT(period) DO UPDATE SET text = excluded.text, created_at = excluded.created_at""",
        (week_of(anchor), datetime.now().isoformat(timespec="seconds"), text),
    )
    conn.commit()
    return text
