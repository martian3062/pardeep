"""Mood shifts tone. It is never treated as fact."""
import sqlite3
from datetime import datetime, timedelta

import pytest

from src.twin import mood as md


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript(md.SCHEMA)
    return c


def test_a_reading_is_stored_and_read_back(conn):
    md.record(conn, "tired", 0.62)
    now = md.current(conn)
    assert now.label == "tired"
    assert now.usable


def test_an_unknown_label_is_refused(conn):
    with pytest.raises(ValueError):
        md.record(conn, "furious", 0.9)


def test_a_weak_reading_does_not_change_the_reply():
    """Someone squinting at a bright screen scores as stressed. A low-confidence
    guess must not have the twin softening its tone at random."""
    weak = md.Mood(label="stressed", score=0.12, at=datetime.now())
    assert not weak.usable
    assert md.tone_rule(weak) == ""


def test_a_stale_reading_is_ignored():
    """A face read an hour ago says nothing about this conversation."""
    old = md.Mood(label="tired", score=0.9, at=datetime.now() - timedelta(hours=2))
    assert not old.fresh
    assert md.tone_rule(old) == ""


def test_neutral_says_nothing():
    assert md.tone_rule(md.Mood("neutral", 0.9, datetime.now())) == ""


def test_no_camera_at_all_is_fine():
    assert md.tone_rule(None) == ""


def test_the_rule_shifts_tone_without_asserting_the_mood():
    """The twin should adapt, not diagnose him out loud."""
    rule = md.tone_rule(md.Mood("tired", 0.7, datetime.now()))
    assert "short" in rule.lower()
    assert "not announce" in rule.lower() or "do not" in rule.lower()


def test_timeline_groups_by_day(conn):
    md.record(conn, "happy", 0.7)
    md.record(conn, "happy", 0.8)
    md.record(conn, "tired", 0.6)
    rows = md.timeline(conn)
    counts = {(d, m): n for d, m, n in rows}
    today = datetime.now().date().isoformat()
    assert counts[(today, "happy")] == 2
    assert counts[(today, "tired")] == 1
