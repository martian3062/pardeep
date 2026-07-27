"""Serve archive photos to the UI, and only archive photos.

The API answers with file paths that came out of the memory store, and the
browser has to fetch the image itself. That endpoint takes a path from the
client, which is exactly the shape of a directory-traversal hole: without a
check, `?path=../../../.ssh/id_rsa` reads whatever the process can read. So a
requested path is resolved and must land inside one of the known archive roots.
"""
from __future__ import annotations

from pathlib import Path

# the same roots the scanner walks
ALLOWED_ROOTS = [
    Path("data/raw/_dump").resolve(),
    Path("D:/iqoo").resolve(),
    Path("D:/iqooz6data").resolve(),
]

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def resolve_photo(raw_path: str) -> Path | None:
    """Return the real file for a requested path, or None if it is not ours.

    Rejects traversal, symlinks pointing outside, and anything that is not an
    image — a caller should never be able to turn this into a general file read.
    """
    if not raw_path:
        return None
    try:
        candidate = Path(raw_path).resolve(strict=True)
    except (OSError, RuntimeError):
        return None

    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        return None
    if not candidate.is_file():
        return None

    for root in ALLOWED_ROOTS:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        return candidate
    return None
