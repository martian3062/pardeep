import sqlite3
from datetime import datetime
from pathlib import Path

from src.vision import photos as ph
from src.vision.index_photos import memory_text


def test_facebook_and_screenshots_rank_lowest():
    assert ph.folder_rank(Path(r"E:\d\oppoa5s\Facebook\x.jpg")) == 0
    assert ph.folder_rank(Path(r"E:\d\old_7yr\photo\Screenshots\x.png")) == 0
    assert ph.folder_rank(Path(r"E:\d\oppoa5s\scrss\x.jpg")) == 0


def test_his_own_photos_rank_highest():
    assert ph.folder_rank(Path(r"E:\d\old_7yr\photo\camera\x.jpg")) == 3
    assert ph.folder_rank(Path(r"E:\d\DCIM\Camera\x.jpg")) == 3
    # images he SENT are his, unlike the received ones in the same tree
    assert ph.folder_rank(Path(r"E:\d\backu\WhatsApp Images\Sent\x.jpg")) == 3


def test_received_whatsapp_ranks_below_sent():
    received = ph.folder_rank(Path(r"E:\d\photo\WhatsApp Images\IMG-20190421-WA0001.jpg"))
    sent = ph.folder_rank(Path(r"E:\d\photo\WhatsApp Images\Sent\IMG-20190421-WA0002.jpg"))
    assert received < sent


def test_mtime_date_is_not_trusted(tmp_path):
    """Regression: archive copies carry today's mtime, which once dated 3,549
    photos to the day they were pulled off Drive."""
    f = tmp_path / "no_date_here.jpg"
    f.write_bytes(b"x" * 30_000)
    dt, source = ph.photo_date(f)
    assert source == "mtime"
    assert not ph.Photo(f, dt, source, 3, "camera").date_trusted


def test_filename_date_beats_mtime(tmp_path):
    f = tmp_path / "IMG-20190421-WA0001.jpg"
    f.write_bytes(b"x" * 30_000)
    dt, source = ph.photo_date(f)
    assert source == "filename"
    assert (dt.year, dt.month, dt.day) == (2019, 4, 21)


def _seed(conn, rows):
    conn.executescript(ph.SCHEMA)
    conn.executemany(
        "INSERT INTO photos (path, taken_at, date_source, rank, folder_hint) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()


def test_impute_dates_uses_folder_median():
    conn = sqlite3.connect(":memory:")
    _seed(
        conn,
        [
            ("a.jpg", "2017-06-01T10:00:00", "exif", 3, "wedding"),
            ("b.jpg", "2017-06-02T10:00:00", "exif", 3, "wedding"),
            ("c.jpg", "2017-06-03T10:00:00", "exif", 3, "wedding"),
            ("d.jpg", "2026-07-27T10:00:00", "mtime", 3, "wedding"),
        ],
    )
    assert ph.impute_dates(conn) == 1
    taken, source = conn.execute("SELECT taken_at, date_source FROM photos WHERE path='d.jpg'").fetchone()
    assert taken.startswith("2017-06")
    assert source == "folder_median"


def test_impute_skips_folders_without_enough_dated_siblings():
    conn = sqlite3.connect(":memory:")
    _seed(
        conn,
        [
            ("a.jpg", "2017-06-01T10:00:00", "exif", 3, "loose"),
            ("d.jpg", "2026-07-27T10:00:00", "mtime", 3, "loose"),
        ],
    )
    assert ph.impute_dates(conn) == 0
    assert conn.execute("SELECT date_source FROM photos WHERE path='d.jpg'").fetchone()[0] == "mtime"


def test_memory_text_omits_generic_folder_names():
    text = memory_text(datetime(2017, 8, 12), "camera", "four people outside a dhaba")
    assert text == "12 Aug 2017: four people outside a dhaba"


def test_memory_text_keeps_meaningful_folder():
    text = memory_text(datetime(2017, 8, 12), "photos smj5", "a classroom")
    assert "photos smj5" in text


def test_memory_text_without_date_makes_no_claim():
    text = memory_text(None, "camera", "a dog on a roof")
    assert text == "a dog on a roof"
