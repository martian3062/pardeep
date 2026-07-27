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
        """mtime on an archive copy is the day it was downloaded, not taken."""
        return self.date_source in ("exif", "filename")


TRUSTED_SOURCES = ("exif", "filename", "folder_median")


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


def exif_datetime(path: Path) -> datetime | None:
    """Camera timestamp, the most trustworthy date a photo carries."""
    try:
        from PIL import Image, ExifTags

        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            for key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
                raw = tags.get(key)
                if raw:
                    return datetime.strptime(str(raw)[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def photo_date(path: Path) -> tuple[datetime | None, str]:
    """EXIF first, then the filename (IMG-20190421-WA0001), then file mtime.

    Returns the source alongside the date because it decides whether the date can
    be believed: mtime on an archive copy is the day it was pulled off Drive, not
    the day it was taken. Trusting mtime is what once dated every call memory to
    the same afternoon.
    """
    if dt := exif_datetime(path):
        return dt, "exif"
    if dt := timestamp_from_name(path.name):
        return dt, "filename"
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


def impute_dates(conn: sqlite3.Connection, min_siblings: int = 3) -> int:
    """Give mtime-only photos the median date of their folder.

    A folder is usually one period of life — a camera roll from 2017, a wedding
    album. When neighbours carry real EXIF dates, their median is a far better
    guess than the day the archive was copied off Drive. Folders without enough
    dated neighbours keep `date_source='mtime'` and stay out of the timeline.
    """
    conn.executescript(SCHEMA)
    folders = conn.execute(
        """SELECT folder_hint, count(*) FROM photos
           WHERE date_source IN ('exif','filename') AND folder_hint <> ''
           GROUP BY 1 HAVING count(*) >= ?""",
        (min_siblings,),
    ).fetchall()

    updated = 0
    for folder, _ in folders:
        dates = [
            r[0]
            for r in conn.execute(
                """SELECT taken_at FROM photos
                   WHERE folder_hint = ? AND date_source IN ('exif','filename')
                   ORDER BY taken_at""",
                (folder,),
            )
        ]
        median = dates[len(dates) // 2]
        cur = conn.execute(
            """UPDATE photos SET taken_at = ?, date_source = 'folder_median'
               WHERE folder_hint = ? AND date_source = 'mtime'""",
            (median, folder),
        )
        updated += cur.rowcount
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
