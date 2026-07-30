"""The recording session has to catch bad audio while he is still sitting there."""
import math
import sqlite3
import struct
import wave

import numpy as np
import pytest

from src.voice import session as ss


def write_wav(path, freqs, seconds=5.0, sr=48000, amp=0.3):
    """A tone mixture, so the measured bandwidth is known in advance."""
    t = np.arange(int(sr * seconds)) / sr
    x = sum(np.sin(2 * math.pi * f * t) for f in freqs) / len(freqs) * amp
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript(ss.SCHEMA)
    return c


def test_full_band_audio_is_accepted(tmp_path):
    p = write_wav(tmp_path / "good.wav", [200, 900, 3000, 8000, 11000])
    q = ss.measure(p)
    assert q.sample_rate == 48000
    assert q.bandwidth_hz >= ss.MIN_BANDWIDTH_HZ
    assert q.ok, q.problem


def test_telephone_audio_is_rejected(tmp_path):
    """Regression: every call recording in the archive is a 48kHz container with
    ~1.6kHz of real audio, which is why 32 hours of it cannot train a voice. The
    session must catch that during the session, not after."""
    p = write_wav(tmp_path / "phone.wav", [200, 600, 1200])
    q = ss.measure(p)
    assert q.sample_rate == 48000  # the container lies
    assert q.bandwidth_hz < ss.MIN_BANDWIDTH_HZ  # the audio does not
    assert not q.ok
    assert "real audio" in q.problem


def test_too_quiet_is_rejected(tmp_path):
    p = write_wav(tmp_path / "quiet.wav", [200, 3000, 9000], amp=0.002)
    q = ss.measure(p)
    assert not q.ok
    assert "quiet" in q.problem


def test_too_short_is_rejected(tmp_path):
    p = write_wav(tmp_path / "short.wav", [200, 3000, 9000], seconds=1.2)
    q = ss.measure(p)
    assert not q.ok
    assert "short" in q.problem


def test_clipping_is_rejected(tmp_path):
    sr, seconds = 48000, 5.0
    t = np.arange(int(sr * seconds)) / sr
    x = np.sin(2 * math.pi * 300 * t) * 3.0  # driven far past full scale
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    p = tmp_path / "clipped.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    q = ss.measure(p)
    assert not q.ok
    assert "clipping" in q.problem


def test_a_long_answer_is_a_warning_not_a_rejection(tmp_path):
    """He talks in long bursts; a 100s answer is usable, just awkward."""
    p = write_wav(tmp_path / "long.wav", [200, 3000, 9000], seconds=95)
    q = ss.measure(p)
    assert q.ok
    assert "long" in q.problem  # still surfaced to him


# --------------------------------------------------------------- questions


def test_every_language_and_register_is_covered_early():
    """A session that ends after 20 questions must still span the range, since a
    voice model only reproduces registers it heard."""
    qs = ss.build_questions(seed=1)
    head = qs[:20]
    assert {q["lang"] for q in head} >= set(ss.REQUIRED_LANGS)
    assert {q["register"] for q in head} >= set(ss.REQUIRED_REGISTERS)


def test_question_ids_are_unique():
    qs = ss.build_questions(seed=2)
    assert len({q["id"] for q in qs}) == len(qs)


def test_friend_placeholders_are_filled():
    qs = ss.build_questions(seed=3)
    assert not any("{friend}" in q["text"] for q in qs)


def test_the_bank_is_big_enough_for_an_hour():
    """At roughly 40s an answer, 45 minutes needs ~65 questions."""
    assert len(ss.build_questions()) >= 40


# ------------------------------------------------------------------ store


def test_progress_counts_only_accepted_clips(conn, tmp_path):
    good = write_wav(tmp_path / "a.wav", [200, 3000, 9000], seconds=10)
    bad = write_wav(tmp_path / "b.wav", [200, 600], seconds=10)
    q = {"id": "q001", "text": "?", "lang": "hi"}
    ss.CLIP_DIR = tmp_path  # keep the test off the real corpus
    ss.save_clip(conn, good.read_bytes(), q)
    ss.save_clip(conn, bad.read_bytes(), {**q, "id": "q002"})
    p = ss.progress(conn)
    assert p["clips"] == 1
    assert p["rejected"] == 1
    assert p["minutes"] > 0


def test_answered_questions_are_not_asked_again(conn, tmp_path):
    good = write_wav(tmp_path / "c.wav", [200, 3000, 9000], seconds=8)
    ss.CLIP_DIR = tmp_path
    ss.save_clip(conn, good.read_bytes(), {"id": "q007", "text": "?", "lang": "en"})
    assert "q007" in ss.answered_ids(conn)
