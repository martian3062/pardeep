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
            (r"E:\a\wedding\a.jpg", "2017-06-01T10:00:00", "exif", 3, "wedding"),
            (r"E:\a\wedding\b.jpg", "2017-06-02T10:00:00", "exif", 3, "wedding"),
            (r"E:\a\wedding\c.jpg", "2017-06-03T10:00:00", "exif", 3, "wedding"),
            (r"E:\a\wedding\d.jpg", "2026-07-27T10:00:00", "mtime", 3, "wedding"),
        ],
    )
    assert ph.impute_dates(conn) == 1
    taken, source = conn.execute(
        r"SELECT taken_at, date_source FROM photos WHERE path='E:\a\wedding\d.jpg'"
    ).fetchone()
    assert taken.startswith("2017-06")
    assert source == "folder_median"


def test_impute_skips_folders_without_enough_dated_siblings():
    conn = sqlite3.connect(":memory:")
    _seed(
        conn,
        [
            (r"E:\a\loose\a.jpg", "2017-06-01T10:00:00", "exif", 3, "loose"),
            (r"E:\a\loose\d.jpg", "2026-07-27T10:00:00", "mtime", 3, "loose"),
        ],
    )
    assert ph.impute_dates(conn) == 0
    assert (
        conn.execute(r"SELECT date_source FROM photos WHERE path='E:\a\loose\d.jpg'").fetchone()[0]
        == "mtime"
    )


def test_impute_does_not_merge_same_named_folders_in_different_trees():
    """Regression: grouping by bare folder name merged nine names that appear in
    several trees — 'Camera' alone spans 2016-2024 across three directories — and
    777 of 2,642 imputed dates came from an unrelated directory's median."""
    conn = sqlite3.connect(":memory:")
    _seed(
        conn,
        [
            (r"E:\old\Camera\a.jpg", "2016-01-01T10:00:00", "exif", 3, "Camera"),
            (r"E:\old\Camera\b.jpg", "2016-01-02T10:00:00", "exif", 3, "Camera"),
            (r"E:\old\Camera\c.jpg", "2016-01-03T10:00:00", "exif", 3, "Camera"),
            (r"D:\new\Camera\x.jpg", "2024-06-01T10:00:00", "exif", 3, "Camera"),
            (r"D:\new\Camera\y.jpg", "2024-06-02T10:00:00", "exif", 3, "Camera"),
            (r"D:\new\Camera\z.jpg", "2024-06-03T10:00:00", "exif", 3, "Camera"),
            (r"D:\new\Camera\u.jpg", "2026-07-27T10:00:00", "mtime", 3, "Camera"),
        ],
    )
    assert ph.impute_dates(conn) == 1
    taken = conn.execute(
        r"SELECT taken_at FROM photos WHERE path='D:\new\Camera\u.jpg'"
    ).fetchone()[0]
    # must take its OWN directory's 2024 median, not the 2016 folder's
    assert taken.startswith("2024-06")


def test_memory_text_omits_generic_folder_names():
    text = memory_text(datetime(2017, 8, 12), "camera", "four people outside a dhaba")
    assert text == "12 Aug 2017: four people outside a dhaba"


def test_memory_text_keeps_meaningful_folder():
    text = memory_text(datetime(2017, 8, 12), "photos smj5", "a classroom")
    assert "photos smj5" in text


def test_memory_text_without_date_makes_no_claim():
    text = memory_text(None, "camera", "a dog on a roof")
    assert text == "a dog on a roof"


def test_a_photo_with_people_says_so():
    """Regression: asked to describe an old photo the twin said it had none,
    while holding 1,307. Memes win vague photo queries because their captions are
    text-dense; a real photograph's is one plain sentence."""
    text = memory_text(datetime(2017, 8, 12), "camera", "two men beside a locomotive", 2)
    assert text.startswith("[photo of 2 people]")
    assert "12 Aug 2017" in text


def test_one_person_is_singular():
    assert memory_text(None, "", "a child smiling", 1).startswith("[photo of 1 person]")


def test_an_image_with_no_faces_is_not_labelled():
    assert not memory_text(None, "", "a meme with text", 0).startswith("[photo of")
    assert not memory_text(None, "", "not yet scanned", -1).startswith("[photo of")


def _write_jpeg_with_exif(path, *, original=None, modified=None):
    """A real JPEG carrying the requested EXIF date tags."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), "gray")
    exif = img.getexif()
    if modified:
        exif[306] = modified  # IFD0 DateTime — when the file was last written
    if original:
        exif.get_ifd(0x8769)[36867] = original  # SubIFD DateTimeOriginal — capture
    img.save(path, exif=exif, quality=95)


def test_capture_time_beats_file_modified_time(tmp_path):
    """Regression: Image.getexif() returns IFD0 only, whose DateTime is when the
    file was last WRITTEN. Reading it instead of SubIFD DateTimeOriginal dated a
    photo shot on 2017-01-27 to 2016-01-01, the clock of the app that re-saved it."""
    f = tmp_path / "shot.jpg"
    _write_jpeg_with_exif(f, original="2017:01:27 17:09:26", modified="2016:01:01 06:35:36")
    capture, modified = ph.exif_datetimes(f)
    assert capture == datetime(2017, 1, 27, 17, 9, 26)
    assert modified == datetime(2016, 1, 1, 6, 35, 36)

    dt, source = ph.photo_date(f)
    assert source == "exif"
    assert dt == datetime(2017, 1, 27, 17, 9, 26)


def test_filename_beats_a_resave_timestamp(tmp_path):
    """A Picasa re-save leaves only IFD0 DateTime. The filename was written at
    capture, so it is the better date."""
    f = tmp_path / "IMG-20190323-WA0001.jpg"
    _write_jpeg_with_exif(f, modified="2020:04:07 06:56:16")
    dt, source = ph.photo_date(f)
    assert source == "filename"
    assert dt.date() == datetime(2019, 3, 23).date()


def test_resave_timestamp_still_beats_mtime(tmp_path):
    f = tmp_path / "nodate.jpg"
    _write_jpeg_with_exif(f, modified="2018:05:05 12:00:00")
    dt, source = ph.photo_date(f)
    assert source == "exif_modified"
    assert dt == datetime(2018, 5, 5, 12, 0, 0)
    # good enough to place on the timeline, not good enough to seed a folder median
    assert source in ph.TRUSTED_SOURCES
    assert not ph.Photo(f, dt, source, 3, "camera").date_trusted


def test_duplicates_are_captioned_once(tmp_path):
    """The archive was merged from several phone backups, so the same picture
    appears as 'IMG_x.jpg', 'IMG_x (1).jpg', and again under camera/collage/.
    149 of 1,456 captionable photos were redundant."""
    same = b"\xff\xd8" + b"identical bytes" * 2000
    other = b"\xff\xd8" + b"a different picture" * 2000
    a = tmp_path / "IMG_1.jpg"
    b = tmp_path / "IMG_1 (1).jpg"
    c = tmp_path / "collage"
    c.mkdir()
    c = c / "IMG_1.jpg"
    d = tmp_path / "IMG_2.jpg"
    for p, data in ((a, same), (b, same), (c, same), (d, other)):
        p.write_bytes(data)

    conn = sqlite3.connect(":memory:")
    _seed(
        conn,
        [(str(p), "2017-06-01T10:00:00", "exif", 3, "camera") for p in (a, b, c, d)],
    )
    assert ph.find_duplicates(conn) == 2  # three copies collapse to one canonical

    pending = ph.pending_captions(conn)
    assert len(pending) == 2  # one of the trio, plus the genuinely different photo
    assert str(a) in {p for _, p in pending}  # shortest path is canonical


def test_dedupe_reuses_a_caption_already_computed(tmp_path):
    same = b"\xff\xd8" + b"identical" * 3000
    a = tmp_path / "IMG_1.jpg"
    b = tmp_path / "IMG_1 (1).jpg"
    a.write_bytes(same)
    b.write_bytes(same)

    conn = sqlite3.connect(":memory:")
    _seed(conn, [(str(p), "2017-06-01T10:00:00", "exif", 3, "camera") for p in (a, b)])
    # the copy got captioned first; the canonical must inherit it, not redo it
    conn.execute("UPDATE photos SET caption = 'a dog on a roof' WHERE path = ?", (str(b),))
    conn.commit()

    ph.find_duplicates(conn)
    assert ph.pending_captions(conn) == []
    assert (
        conn.execute("SELECT caption FROM photos WHERE path = ?", (str(a),)).fetchone()[0]
        == "a dog on a roof"
    )


def test_undated_sentinel_survives_timestamp_conversion():
    """Regression: datetime(1970,1,1).timestamp() raises OSError on Windows in
    any timezone east of UTC, and Memory.to_row calls .timestamp() on every row.
    165 undated photos would have crashed `index` after ~1,280 rows were written."""
    from src.vision.index_photos import UNDATED

    assert UNDATED.timestamp() > 0  # would raise OSError for the epoch on IST/Windows
    assert UNDATED.year < 2010  # still older than any real photo, so it sorts last
