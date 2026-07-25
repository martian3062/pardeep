"""Audio transcription via faster-whisper (local, GPU with CPU fallback).

- voice notes -> `messages` rows (speaker=me) + copied refs for TTS cloning later
- call recordings -> `call_segments` rows (speaker labeled by the diarization pass later)
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

from rich.console import Console

from . import db

console = Console()

AUDIO_EXTS = {".opus", ".ogg", ".mp3", ".m4a", ".wav", ".aac", ".flac", ".wma", ".amr"}

# WhatsApp voice-note filenames embed the date: PTT-20230726-WA0001.opus
_WA_DATE = re.compile(r"(?:PTT|AUD)-(\d{4})(\d{2})(\d{2})-", re.IGNORECASE)


def iter_audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def load_model(model_size: str = "large-v3-turbo"):
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    try:
        model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        console.print(f"[green]Whisper {model_size} loaded on GPU (int8_float16, batched)[/green]")
    except Exception as exc:
        console.print(f"[yellow]GPU load failed ({exc}); falling back to CPU int8[/yellow]")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return BatchedInferencePipeline(model)


def transcribe_file(model, path: Path, language: str | None = None):
    segments, info = model.transcribe(
        str(path),
        language=language,
        vad_filter=True,
        beam_size=5,
        batch_size=8,
    )
    return [(s.start, s.end, s.text.strip()) for s in segments], info


def file_timestamp(path: Path) -> datetime:
    m = _WA_DATE.search(path.name)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return datetime.fromtimestamp(path.stat().st_mtime)


def ingest_voice_notes(
    conn: sqlite3.Connection, root: Path, model_size: str, language: str | None
) -> tuple[int, int]:
    """Returns (files_processed, messages_inserted)."""
    files = iter_audio_files(root)
    if not files:
        console.print(f"[yellow]No audio files under {root}[/yellow]")
        return 0, 0
    model = load_model(model_size)
    processed = inserted = 0
    for path in files:
        sha = db.file_sha256(path)
        if db.already_ingested(conn, path, sha):
            continue
        # claim BEFORE processing: if this file hard-crashes the process, the
        # restart sees it as done (item_count=-1 = quarantined) and moves on
        db.record_file(conn, path, sha, -1)
        try:
            segs, info = transcribe_file(model, path, language)
        except Exception as exc:
            console.print(f"  [yellow]skipped {path.name}: {exc}[/yellow]")
            db.record_file(conn, path, sha, 0)
            continue
        text = " ".join(t for _, _, t in segs).strip()
        n = 0
        if text:
            n = db.insert_messages(
                conn,
                [
                    db.Message(
                        source="voice_note",
                        conversation_id="voice_notes",
                        timestamp=file_timestamp(path),
                        speaker="me",
                        speaker_name=None,
                        text=text,
                        audio_ref=str(path),
                        lang=info.language,
                    )
                ],
            )
        db.record_file(conn, path, sha, n)
        processed += 1
        inserted += n
        console.print(f"  {path.name}: [{info.language}] {text[:80]}")
    return processed, inserted


def ingest_calls(
    conn: sqlite3.Connection, root: Path, model_size: str, language: str | None
) -> tuple[int, int]:
    """Returns (files_processed, segments_inserted)."""
    files = iter_audio_files(root)
    if not files:
        console.print(f"[yellow]No audio files under {root}[/yellow]")
        return 0, 0
    model = load_model(model_size)
    processed = inserted = 0
    for path in files:
        sha = db.file_sha256(path)
        if db.already_ingested(conn, path, sha):
            continue
        db.record_file(conn, path, sha, -1)  # claim first — see ingest_voice_notes
        try:
            segs, info = transcribe_file(model, path, language)
        except Exception as exc:
            console.print(f"  [yellow]skipped {path.name}: {exc}[/yellow]")
            db.record_file(conn, path, sha, 0)
            continue
        n = db.insert_call_segments(conn, str(path), segs, info.language)
        db.record_file(conn, path, sha, n)
        processed += 1
        inserted += n
        console.print(f"  {path.name}: {n} segments [{info.language}]")
    return processed, inserted
