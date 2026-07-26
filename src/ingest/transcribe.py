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

# Recorders stamp the real date into the filename; file mtime is only the day the
# archive was downloaded, so parsing the name is what gives the twin a truthful
# timeline ("what was I doing in May 2025?") instead of dating everything to the
# day it was ingested.
_FILENAME_DATES = (
    # call recorder: "Roshan 2025-10-14 13-38-46.m4a"  (the bulk of the corpus)
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[ _](\d{2})-(\d{2})-(\d{2})"), "YMDhms"),
    # older recorder: "919876075331_2020_09_20_15_34_02_out.mp3"
    (re.compile(r"_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_"), "YMDhms"),
    # phone recorder: "REC20191026134952.mp3", "VID20220816183243.mp4".
    # Must precede the date-only WhatsApp pattern below, which would otherwise
    # match the same prefix and silently drop the time.
    (re.compile(r"(?:REC|VID)(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", re.IGNORECASE), "YMDhms"),
    # WhatsApp media: "PTT-20230726-WA0001.opus"
    (re.compile(r"(?:PTT|AUD|VID|IMG)[-_]?(\d{4})(\d{2})(\d{2})", re.IGNORECASE), "YMD"),
    # TalkerACR: "phone_20241201-211909.amr"
    (re.compile(r"_(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})"), "YMDhms"),
    # amr recorder: "Khusraj3-2012161436.amr" -> yy mm dd hh mm
    (re.compile(r"[-_](\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?:\D|$)"), "ymdhm"),
)


def timestamp_from_name(name: str) -> datetime | None:
    """Extract the recording time from a filename, or None if it has none."""
    for pattern, layout in _FILENAME_DATES:
        m = pattern.search(name)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        try:
            if layout == "YMDhms":
                dt = datetime(g[0], g[1], g[2], g[3], g[4], g[5])
            elif layout == "YMD":
                dt = datetime(g[0], g[1], g[2])
            else:  # ymdhm — two-digit year
                dt = datetime(2000 + g[0], g[1], g[2], g[3], g[4])
        except ValueError:
            continue  # e.g. month 13 from a coincidental digit run
        # guard against random digit runs matching: the archive spans ~2015-now
        if 2015 <= dt.year <= datetime.now().year + 1:
            return dt
    return None


def iter_audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


# Whisper mislabels romanized Hindi/Punjabi phone audio as neighbouring or
# unrelated languages; collapse those onto what the user actually speaks.
LANG_MAP = {"ur": "hi", "sa": "hi", "ne": "hi", "mr": "hi", "bn": "hi", "gu": "hi"}
ALLOWED_LANGS = {"en", "hi", "pa"}


def language_hint(detected: str | None) -> str | None:
    """Map a previous auto-detection onto a forced language, or None to re-detect."""
    if not detected:
        return None
    lang = LANG_MAP.get(detected, detected)
    return lang if lang in ALLOWED_LANGS else None


def load_model(model_size: str = "large-v3"):
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    try:
        model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        console.print(f"[green]Whisper {model_size} loaded on GPU (int8_float16, batched)[/green]")
    except Exception as exc:
        console.print(f"[yellow]GPU load failed ({exc}); falling back to CPU int8[/yellow]")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return BatchedInferencePipeline(model)


def transcribe_file(model, path: Path, language: str | None = None, batch_size: int = 8):
    """Compressed phone audio makes Whisper hallucinate: it loops phrases and
    emits 'foreign'/boilerplate on unintelligible speech. These thresholds make
    it drop a decode and retry at a higher temperature instead of looping."""
    segments, info = model.transcribe(
        str(path),
        language=language,
        vad_filter=True,
        beam_size=5,
        batch_size=batch_size,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,  # repetition detector
        log_prob_threshold=-1.0,  # low-confidence decode -> retry
        no_speech_threshold=0.6,  # silence -> emit nothing
    )
    return [(s.start, s.end, s.text.strip()) for s in segments], info


def file_timestamp(path: Path) -> datetime:
    """When the recording happened — from the filename if it says, else mtime."""
    return timestamp_from_name(path.name) or datetime.fromtimestamp(path.stat().st_mtime)


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
    conn: sqlite3.Connection,
    root: Path,
    model_size: str,
    language: str | None,
    lang_hints: dict[str, str] | None = None,
) -> tuple[int, int]:
    """Returns (files_processed, segments_inserted).

    lang_hints maps audio path -> language detected on an earlier pass; forcing
    it beats re-detecting on noisy phone audio.
    """
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
        forced = language or language_hint((lang_hints or {}).get(str(path)))
        try:
            segs, info = transcribe_file(model, path, forced)
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
