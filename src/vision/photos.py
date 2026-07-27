"""Turn personal photos into dated, searchable memories.

The archive holds ~7,000 images, but most are not memories: Facebook caches,
screenshots, and forwarded WhatsApp images outnumber his own photographs several
times over. The video experiment already showed what happens when that is
ignored — the twin learned song lyrics and Korean drama. So photos are ranked by
provenance first, and only the ones plausibly taken by or sent by him are worth
captioning on a 6GB GPU.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..ingest.transcribe import timestamp_from_name

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

# Higher rank = more likely to be a real personal memory.
_FOLDER_RANK = (
    (re.compile(r"(camera|dcim|100nikon|my ?photo|pardeep|captured)", re.I), 3),
    (re.compile(r"whatsapp images[\\/]sent", re.I), 3),  # images HE sent
    (re.compile(r"(photos? smj|picasa|collage|wedding)", re.I), 2),
    (re.compile(r"whatsapp images", re.I), 1),  # received: mostly forwards
    (re.compile(r"(facebook|/fb/|screenshot|scrss|telegram|download)", re.I), 0),
)


@dataclass
class Photo:
    path: Path
    taken_at: datetime | None
    date_source: str  # exif | filename | mtime | none
    rank: int
    folder_hint: str

    @property
    def usable(self) -> bool:
        return self.rank >= 2

    @property
    def date_trusted(self) -> bool:
        """Capture-accurate: good enough to seed a folder's median date."""
        return self.date_source in ("exif", "filename")


# Good enough to place a memory on the timeline. Wider than date_trusted: a
# re-save date is the wrong moment but the right era, whereas mtime is the day
# the archive was copied and says nothing about the photo at all.
TRUSTED_SOURCES = ("exif", "filename", "exif_modified", "folder_median")


def folder_rank(path: Path) -> int:
    s = str(path)
    for pattern, rank in _FOLDER_RANK:
        if pattern.search(s):
            return rank
    return 1


def folder_hint(path: Path, root: Path) -> str:
    """The parent folder name, which often says what the photos are ('wedding')."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return rel.parent.name or rel.parent.as_posix()


_EXIF_SUBIFD = 0x8769  # ExifOffset — DateTimeOriginal lives here, not in IFD0
_DATETIME_ORIGINAL = 36867
_DATETIME_DIGITIZED = 36868
_DATETIME_MODIFIED = 306  # IFD0 "DateTime": when the file was last WRITTEN


def _parse_exif_dt(raw) -> datetime | None:
    try:
        return datetime.strptime(str(raw)[:19], "%Y:%m:%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def exif_datetimes(path: Path) -> tuple[datetime | None, datetime | None]:
    """Return (capture time, file-modified time) from EXIF.

    Image.getexif() returns IFD0 only, whose `DateTime` tag is when the file was
    last written — a photo re-saved by a frame app or Picasa carries that app's
    clock. The real capture time is `DateTimeOriginal` in the Exif SubIFD, and
    reading only IFD0 dated 23 photos up to a year wrong: one shot on
    2017-01-27 was stamped 2016-01-01 by a re-save on a phone with a reset clock.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None, None
            modified = _parse_exif_dt(exif.get(_DATETIME_MODIFIED))
            try:
                sub = exif.get_ifd(_EXIF_SUBIFD)
            except Exception:
                sub = {}
            capture = _parse_exif_dt(sub.get(_DATETIME_ORIGINAL)) or _parse_exif_dt(
                sub.get(_DATETIME_DIGITIZED)
            )
            return capture, modified
    except Exception:
        return None, None


def photo_date(path: Path) -> tuple[datetime | None, str]:
    """Best available date, in order of how much it can be believed.

    1. EXIF DateTimeOriginal — stamped by the camera at the shutter.
    2. The filename (IMG_20170127_170925, IMG-20190421-WA0001) — also written at
       capture, so it beats any timestamp describing when the file was written.
    3. EXIF DateTime — a re-save, which for an edited photo is the editing app's
       clock, not the moment being remembered.
    4. mtime — the day the archive was pulled off Drive. Recorded, never trusted:
       believing it dated 3,549 photos to this week, the same failure that once
       landed every call recording on a single afternoon.
    """
    capture, modified = exif_datetimes(path)
    if capture:
        return capture, "exif"
    if dt := timestamp_from_name(path.name):
        return dt, "filename"
    if modified:
        return modified, "exif_modified"
    try:
        return datetime.fromtimestamp(path.stat().st_mtime), "mtime"
    except OSError:
        return None, "none"


def scan(roots: list[Path]) -> list[Photo]:
    photos: list[Photo] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
                continue
            if p.stat().st_size < 20_000:  # thumbnails and icons
                continue
            taken_at, source = photo_date(p)
            photos.append(
                Photo(
                    path=p,
                    taken_at=taken_at,
                    date_source=source,
                    rank=folder_rank(p),
                    folder_hint=folder_hint(p, root),
                )
            )
    return photos


SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    taken_at TEXT,
    date_source TEXT NOT NULL,
    rank INTEGER NOT NULL,
    folder_hint TEXT,
    caption TEXT,
    captioned_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_photos_taken ON photos(taken_at);
"""


def save(conn: sqlite3.Connection, photos: list[Photo]) -> int:
    conn.executescript(SCHEMA)
    cur = conn.executemany(
        """INSERT OR IGNORE INTO photos (path, taken_at, date_source, rank, folder_hint)
           VALUES (?, ?, ?, ?, ?)""",
        [
            (
                str(p.path),
                p.taken_at.isoformat() if p.taken_at else None,
                p.date_source,
                p.rank,
                p.folder_hint,
            )
            for p in photos
        ],
    )
    conn.commit()
    return cur.rowcount


def redate(conn: sqlite3.Connection) -> int:
    """Recompute every stored date from the files, keeping captions.

    Paths are already in the table, so this re-reads dates without re-walking the
    archive — the way to apply a fix to the date logic without losing hours of
    captioning work.
    """
    conn.executescript(SCHEMA)
    rows = conn.execute("SELECT id, path, taken_at, date_source FROM photos").fetchall()
    changed = 0
    for photo_id, path_str, old_taken, old_source in rows:
        taken, source = photo_date(Path(path_str))
        new_taken = taken.isoformat() if taken else None
        if (new_taken, source) == (old_taken, old_source):
            continue
        conn.execute(
            "UPDATE photos SET taken_at = ?, date_source = ? WHERE id = ?",
            (new_taken, source, photo_id),
        )
        changed += 1
    conn.commit()
    return changed


def reset_imputed(conn: sqlite3.Connection) -> int:
    """Undo a previous imputation by re-reading each file's mtime.

    Imputation overwrites taken_at, so re-running it would otherwise compound on
    its own guesses. The original mtime is still on disk, which makes the
    operation repeatable.
    """
    conn.executescript(SCHEMA)
    rows = conn.execute(
        "SELECT id, path FROM photos WHERE date_source = 'folder_median'"
    ).fetchall()
    restored = 0
    for photo_id, path_str in rows:
        try:
            mtime = datetime.fromtimestamp(Path(path_str).stat().st_mtime)
        except OSError:
            continue
        conn.execute(
            "UPDATE photos SET taken_at = ?, date_source = 'mtime' WHERE id = ?",
            (mtime.isoformat(), photo_id),
        )
        restored += 1
    conn.commit()
    return restored


def impute_dates(conn: sqlite3.Connection, min_siblings: int = 3) -> int:
    """Give mtime-only photos the median date of their containing directory.

    A directory is usually one period of life — a camera roll from 2017, a
    wedding album. When neighbours carry real EXIF dates, their median is a far
    better guess than the day the archive was copied off Drive. Directories
    without enough dated neighbours keep `date_source='mtime'` and stay out of
    the timeline.

    Grouping is by FULL parent path, not folder name. Nine folder names in this
    archive appear in several trees at once — "Camera" alone names three
    different directories spanning 2016 to 2024 — and grouping by name merged
    them, so 777 of 2,642 imputed dates came from an unrelated directory's
    median.
    """
    conn.executescript(SCHEMA)
    rows = conn.execute(
        "SELECT id, path, taken_at, date_source FROM photos WHERE taken_at IS NOT NULL"
    ).fetchall()

    dated: dict[str, list[str]] = {}
    undated: dict[str, list[int]] = {}
    for photo_id, path_str, taken_at, source in rows:
        parent = str(Path(path_str).parent)
        if source in ("exif", "filename"):
            dated.setdefault(parent, []).append(taken_at)
        elif source == "mtime":
            undated.setdefault(parent, []).append(photo_id)

    updated = 0
    for parent, ids in undated.items():
        siblings = sorted(dated.get(parent, []))
        if len(siblings) < min_siblings:
            continue
        median = siblings[len(siblings) // 2]
        conn.executemany(
            "UPDATE photos SET taken_at = ?, date_source = 'folder_median' WHERE id = ?",
            [(median, i) for i in ids],
        )
        updated += len(ids)
    conn.commit()
    return updated


def pending_captions(conn: sqlite3.Connection, min_rank: int = 2) -> list[tuple[int, str]]:
    conn.executescript(SCHEMA)
    return [
        (row[0], row[1])
        for row in conn.execute(
            "SELECT id, path FROM photos WHERE caption IS NULL AND rank >= ? ORDER BY taken_at",
            (min_rank,),
        )
    ]
