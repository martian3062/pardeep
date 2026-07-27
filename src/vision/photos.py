"""Turn personal photos into dated, searchable memories.

The archive holds ~7,000 images, but most are not memories: Facebook caches,
screenshots, and forwarded WhatsApp images outnumber his own photographs several
times over. The video experiment already showed what happens when that is
ignored — the twin learned song lyrics and Korean drama. So photos are ranked by
provenance first, and only the ones plausibly taken by or sent by him are worth
captioning on a 6GB GPU.
"""
from __future__ import annotations

import hashlib
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


# A forwarded joke is not a memory of his life, but it looks like one to
# retrieval: meme captions are dense with quoted text, watermarks and named
# people, while a real photograph's caption is one plain sentence. Asked to
# describe an old photo, the twin answered that it had none while holding 1,307.
#
# Face count does NOT separate them — memes contain faces too, which is why
# boosting photos-with-people left the defect in place. What separates them is
# the furniture of a forward: a watermark, overlay text, a screenshot frame.
_FORWARD_SIGNALS = (
    re.compile(r"\b(?:FB|WWW)\.[A-Z0-9]+\.(?:COM|IN)\b", re.I),
    re.compile(r"\bwatermark(?:ed)?\b", re.I),
    re.compile(r"\b(?:meme|joke|cartoon|comic strip|caption reads)\b", re.I),
    # the captioner varies the wording — "overlaid Hindi text", "text overlay at
    # the bottom", "superimposed white text" — so a word is allowed in between,
    # which an adjacent-only pattern missed and let a meme through
    re.compile(r"\b(?:overlaid|overlay(?:ing)?|superimposed)\s+(?:\w+\s+){0,2}text\b", re.I),
    re.compile(r"\btext\s+(?:overlay|superimposed)\b|\bimpact font\b", re.I),
    re.compile(r"\btext (?:at the (?:top|bottom)|across the (?:top|image))\b", re.I),
    re.compile(r"\bscreenshot\b|\bwhatsapp (?:chat|conversation)\b|\bsocial media post\b", re.I),
    re.compile(r"\b(?:fake )?tweet\b|\bfacebook post\b|\binstagram post\b", re.I),
    re.compile(r"\bgood\s*morning\b|\bgreeting card\b|\bmotivational (?:poster|quote)\b", re.I),
)


def looks_forwarded(caption: str) -> bool:
    """Does this caption describe a forward rather than a moment?"""
    if not caption:
        return False
    return any(rx.search(caption) for rx in _FORWARD_SIGNALS)


_TABLE = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    taken_at TEXT,
    date_source TEXT NOT NULL,
    rank INTEGER NOT NULL,
    folder_hint TEXT,
    caption TEXT,
    captioned_at TEXT,
    dupe_of INTEGER
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_photos_taken ON photos(taken_at);
CREATE INDEX IF NOT EXISTS idx_photos_dupe ON photos(dupe_of);
"""

# columns added after the table first shipped; CREATE TABLE IF NOT EXISTS will
# not add them to a database that already holds hours of captioning work
_ADDED_COLUMNS = {"dupe_of": "INTEGER", "is_forward": "INTEGER"}


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_TABLE)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(photos)")}
    for column, decl in _ADDED_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE photos ADD COLUMN {column} {decl}")
    conn.executescript(_INDEXES)
    conn.commit()


SCHEMA = _TABLE  # kept for callers that only need the table definition


def save(conn: sqlite3.Connection, photos: list[Photo]) -> int:
    ensure_schema(conn)
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
    ensure_schema(conn)
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
    ensure_schema(conn)
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
    ensure_schema(conn)
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


def find_duplicates(conn: sqlite3.Connection, min_rank: int = 2) -> int:
    """Point byte-identical photos at one canonical copy.

    The archive was assembled from several phone backups, so the same picture
    shows up as "IMG_x.jpg" and "IMG_x (1).jpg", and again under camera/collage/.
    149 of 1,456 captionable photos are redundant — 45 minutes of GPU spent
    describing the same images twice, and a duplicate memory for each.

    Files are grouped by size first and hashed only within colliding groups, so
    almost nothing is read.
    """
    ensure_schema(conn)
    rows = conn.execute(
        "SELECT id, path FROM photos WHERE rank >= ? ORDER BY length(path), path",
        (min_rank,),
    ).fetchall()

    by_size: dict[int, list[tuple[int, str]]] = {}
    for photo_id, path_str in rows:
        try:
            by_size.setdefault(Path(path_str).stat().st_size, []).append((photo_id, path_str))
        except OSError:
            continue

    by_hash: dict[str, list[int]] = {}
    for items in by_size.values():
        if len(items) < 2:
            continue  # a unique size cannot have a byte-identical twin
        for photo_id, path_str in items:
            try:
                digest = hashlib.blake2b(Path(path_str).read_bytes(), digest_size=16).hexdigest()
            except OSError:
                continue
            by_hash.setdefault(digest, []).append(photo_id)

    marked = 0
    for ids in by_hash.values():
        if len(ids) < 2:
            continue
        canonical, *copies = ids  # shortest path wins: the original, not "x (1)"
        # carry any caption across so a copy already described is not redone
        caption = conn.execute(
            f"SELECT caption FROM photos WHERE id IN ({','.join('?' * len(ids))}) "
            "AND caption IS NOT NULL LIMIT 1",
            ids,
        ).fetchone()
        if caption:
            conn.execute(
                "UPDATE photos SET caption = ? WHERE id = ? AND caption IS NULL",
                (caption[0], canonical),
            )
        conn.executemany(
            "UPDATE photos SET dupe_of = ? WHERE id = ?", [(canonical, c) for c in copies]
        )
        marked += len(copies)
    conn.commit()
    return marked


def flag_forwards(conn: sqlite3.Connection) -> int:
    """Mark captioned photos that describe a forward rather than a moment."""
    ensure_schema(conn)
    rows = conn.execute("SELECT id, caption FROM photos WHERE caption IS NOT NULL").fetchall()
    flags = [(1 if looks_forwarded(c) else 0, i) for i, c in rows]
    conn.executemany("UPDATE photos SET is_forward = ? WHERE id = ?", flags)
    conn.commit()
    return sum(f for f, _ in flags)


def pending_captions(conn: sqlite3.Connection, min_rank: int = 2) -> list[tuple[int, str]]:
    ensure_schema(conn)
    return [
        (row[0], row[1])
        for row in conn.execute(
            """SELECT id, path FROM photos
               WHERE caption IS NULL AND rank >= ? AND dupe_of IS NULL
               ORDER BY taken_at""",
            (min_rank,),
        )
    ]
