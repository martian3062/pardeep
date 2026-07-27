"""The photo endpoint takes a path from the browser. It must not become a file read."""
from pathlib import Path

import pytest

from app.api import photos


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / "dump"
    (root / "camera").mkdir(parents=True)
    pic = root / "camera" / "IMG_1.jpg"
    pic.write_bytes(b"\xff\xd8fake jpeg")
    secret = tmp_path / "secret.txt"
    secret.write_text("private key")
    monkeypatch.setattr(photos, "ALLOWED_ROOTS", [root.resolve()])
    return root, pic, secret


def test_serves_a_real_archive_photo(archive):
    _, pic, _ = archive
    assert photos.resolve_photo(str(pic)) == pic.resolve()


def test_rejects_traversal_out_of_the_archive(archive):
    root, _, secret = archive
    attack = str(root / "camera" / ".." / ".." / "secret.txt")
    assert photos.resolve_photo(attack) is None


def test_rejects_an_absolute_path_outside_the_archive(archive):
    _, _, secret = archive
    assert photos.resolve_photo(str(secret)) is None


def test_rejects_a_non_image_even_inside_the_archive(archive):
    root, _, _ = archive
    note = root / "camera" / "notes.txt"
    note.write_text("something")
    assert photos.resolve_photo(str(note)) is None


def test_rejects_missing_and_empty_paths(archive):
    root, _, _ = archive
    assert photos.resolve_photo("") is None
    assert photos.resolve_photo(str(root / "camera" / "nope.jpg")) is None


def test_rejects_a_directory(archive):
    root, _, _ = archive
    assert photos.resolve_photo(str(root / "camera")) is None


def test_real_roots_are_configured():
    """The shipped configuration should point at the actual archives."""
    assert any("_dump" in str(p) for p in photos.ALLOWED_ROOTS)
