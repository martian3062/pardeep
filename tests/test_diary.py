"""The diary is how the twin keeps learning after the archive ends."""
import sqlite3
from datetime import date

import pytest

from src.diary import learn
from src.diary import store as st


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript(st.SCHEMA)
    return c


def test_an_entry_is_stored_against_a_day(conn):
    st.add_entry(conn, st.Entry(text="lab me poora din gaya", day="2026-07-27"))
    rows = st.entries_for(conn, "2026-07-27")
    assert len(rows) == 1
    assert rows[0]["text"] == "lab me poora din gaya"


def test_entries_wait_to_be_indexed_then_stop(conn):
    a = st.add_entry(conn, st.Entry(text="one"))
    st.add_entry(conn, st.Entry(text="two"))
    assert len(st.unindexed(conn)) == 2
    st.mark_indexed(conn, [a])
    assert len(st.unindexed(conn)) == 1


def test_only_corrections_with_a_rewrite_can_train(conn):
    """A thumbs-down says something was wrong but not what right looks like.
    DPO needs something to prefer."""
    st.add_feedback(conn, st.Feedback(prompt="q", rejected="bad answer", verdict="down"))
    st.add_feedback(
        conn,
        st.Feedback(prompt="q", rejected="bad answer", chosen="aise bol yaar", verdict="down"),
    )
    assert st.stats(conn)["feedback"] == 2
    assert len(st.pending_pairs(conn)) == 1


def test_export_writes_a_dpo_batch(conn, tmp_path):
    st.add_feedback(
        conn,
        st.Feedback(
            prompt="kal kya kiya",
            rejected="I apologise, I do not have that information available.",
            chosen="kuch nahi yaar, ghar pe hi tha",
            verdict="down",
        ),
    )
    path, skipped = learn.export_pairs(conn, out_dir=tmp_path)
    assert path is not None and skipped == 0
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "ghar pe hi tha" in lines[0]


def test_a_pair_that_barely_differs_is_not_exported(conn, tmp_path):
    """Preferring X over almost-X teaches the model noise."""
    st.add_feedback(
        conn,
        st.Feedback(
            prompt="q",
            rejected="haan bhai aa raha hun",
            chosen="haan bhai aa raha hu",
            verdict="down",
        ),
    )
    path, skipped = learn.export_pairs(conn, out_dir=tmp_path)
    assert path is None
    assert skipped == 1


def test_exported_pairs_are_not_exported_twice(conn, tmp_path):
    st.add_feedback(
        conn,
        st.Feedback(prompt="q", rejected="formal assistant reply", chosen="chal thik hai", verdict="down"),
    )
    first, _ = learn.export_pairs(conn, out_dir=tmp_path)
    assert first is not None
    second, _ = learn.export_pairs(conn, out_dir=tmp_path)
    assert second is None  # nothing left pending


def test_week_label_is_iso():
    assert learn.week_of(date(2026, 7, 27)) == "2026-W31"


def test_entries_in_week_covers_monday_to_sunday(conn):
    for day in ("2026-07-19", "2026-07-20", "2026-07-26", "2026-07-27"):
        st.add_entry(conn, st.Entry(text=day, day=day))
    rows = learn.entries_in_week(conn, date(2026, 7, 22))  # Wed of that week
    days = {r["day"] for r in rows}
    assert days == {"2026-07-20", "2026-07-26"}  # Mon 20th .. Sun 26th
