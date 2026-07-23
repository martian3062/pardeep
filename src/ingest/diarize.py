"""Phase 1b: figure out who is speaking in call recordings, and which speaker is ME.

Flow:
  1. `enroll`  — build my voice fingerprint (averaged speaker embedding) from voice notes;
                 also copies the longest clean clips aside as TTS-cloning reference audio.
  2. `diarize` — for every call with unlabeled segments: pyannote diarization -> speaker
                 turns; embed each speaker's audio; cosine-match against my fingerprint;
                 label segments me/other; merge consecutive segments and promote them
                 into the unified `messages` table.

Heavy imports (torch, pyannote) stay inside functions so the rest of the CLI works
without the `diarize` dependency group installed.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np
from rich.console import Console

from ..config import REPO_ROOT
from . import db
from .audio_io import TARGET_SR, duration_sec, load_audio
from .transcribe import file_timestamp, iter_audio_files

console = Console()

# ---------------------------------------------------------------- pure helpers


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def assign_speaker(
    start: float, end: float, turns: list[tuple[float, float, str]]
) -> str | None:
    """Label a whisper segment with the diarized speaker that overlaps it most."""
    overlap: dict[str, float] = {}
    for t_start, t_end, label in turns:
        ov = min(end, t_end) - max(start, t_start)
        if ov > 0:
            overlap[label] = overlap.get(label, 0.0) + ov
    return max(overlap, key=overlap.get) if overlap else None


def merge_utterances(
    segments: list[tuple[float, float, str, str]], gap: float = 2.0
) -> list[tuple[str, float, str]]:
    """(start, end, speaker, text) segments -> merged (speaker, start, text) utterances.

    Consecutive segments by the same speaker separated by < gap seconds join into one.
    """
    merged: list[tuple[str, float, str]] = []
    cur_speaker, cur_start, cur_end, cur_text = None, 0.0, 0.0, ""
    for start, end, speaker, text in segments:
        if speaker == cur_speaker and start - cur_end < gap:
            cur_text += " " + text
            cur_end = end
        else:
            if cur_speaker is not None and cur_text.strip():
                merged.append((cur_speaker, cur_start, cur_text.strip()))
            cur_speaker, cur_start, cur_end, cur_text = speaker, start, end, text
    if cur_speaker is not None and cur_text.strip():
        merged.append((cur_speaker, cur_start, cur_text.strip()))
    return merged


# ------------------------------------------------------------------ model glue


def _hf_token() -> str:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "HF_TOKEN missing. Create a free token at huggingface.co/settings/tokens,\n"
            "accept the terms on the model pages of pyannote/speaker-diarization-3.1\n"
            "AND pyannote/segmentation-3.0, then put HF_TOKEN=... in .env"
        )
    return token


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_embedder(cfg: dict):
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained(
        cfg["diarization"]["embedding_model"], use_auth_token=_hf_token()
    )
    return Inference(model, window="whole", device=_device())


def _embed(inference, waveform: np.ndarray) -> np.ndarray:
    import torch

    return np.asarray(
        inference({"waveform": torch.from_numpy(waveform), "sample_rate": TARGET_SR})
    )


# ------------------------------------------------------------------ enrollment


def enroll(voice_notes_dir: Path, cfg: dict) -> Path:
    """Average my voice-note embeddings into a fingerprint; copy TTS reference clips."""
    dcfg = cfg["diarization"]
    files = iter_audio_files(voice_notes_dir)
    if not files:
        raise SystemExit(f"No audio files under {voice_notes_dir}")

    inference = _load_embedder(cfg)
    embeddings: list[np.ndarray] = []
    durations: list[tuple[float, Path]] = []
    for path in files[: dcfg["max_enroll_files"]]:
        wav = load_audio(path)
        dur = duration_sec(wav)
        if dur < dcfg["min_enroll_sec"]:
            continue
        emb = _embed(inference, wav)
        embeddings.append(emb / np.linalg.norm(emb))
        durations.append((dur, path))
        console.print(f"  enrolled {path.name} ({dur:.1f}s)")
    if not embeddings:
        raise SystemExit(
            f"No voice notes >= {dcfg['min_enroll_sec']}s found — nothing to enroll from"
        )

    profile = np.mean(embeddings, axis=0)
    profile /= np.linalg.norm(profile)
    out = REPO_ROOT / cfg["paths"]["voice_profile"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, profile)
    out.with_suffix(".json").write_text(
        json.dumps(
            {"files": len(embeddings), "model": dcfg["embedding_model"]}, indent=2
        )
    )

    # longest clips -> TTS cloning reference
    ref_dir = REPO_ROOT / cfg["paths"]["tts_reference_dir"]
    ref_dir.mkdir(parents=True, exist_ok=True)
    for _, path in sorted(durations, reverse=True)[: cfg["tts"]["reference_clips"]]:
        shutil.copy2(path, ref_dir / path.name)

    console.print(
        f"[bold green]Voice profile from {len(embeddings)} notes -> {out}[/bold green]\n"
        f"TTS reference clips -> {ref_dir}"
    )
    return out


# ----------------------------------------------------------------- diarization


def _speaker_embeddings(
    inference, waveform: np.ndarray, turns: list[tuple[float, float, str]],
    max_audio_sec: float = 60.0,
) -> dict[str, np.ndarray]:
    """One embedding per diarized speaker, from up to max_audio_sec of their longest turns."""
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for start, end, label in turns:
        if end - start >= 1.0:
            by_speaker.setdefault(label, []).append((start, end))

    out: dict[str, np.ndarray] = {}
    for label, spans in by_speaker.items():
        spans.sort(key=lambda s: s[1] - s[0], reverse=True)
        crops, total = [], 0.0
        for start, end in spans:
            crops.append(waveform[:, int(start * TARGET_SR) : int(end * TARGET_SR)])
            total += end - start
            if total >= max_audio_sec:
                break
        out[label] = _embed(inference, np.concatenate(crops, axis=1))
    return out


def diarize_calls(conn: sqlite3.Connection, cfg: dict) -> tuple[int, int]:
    """Label + promote every call recording with pending (unlabeled) segments.

    Returns (files_processed, messages_promoted).
    """
    dcfg = cfg["diarization"]
    pending = db.pending_call_files(conn)
    if not pending:
        console.print("[yellow]No pending call recordings (run `audio --kind call` first)[/yellow]")
        return 0, 0

    profile_path = REPO_ROOT / cfg["paths"]["voice_profile"]
    if not profile_path.exists():
        raise SystemExit("Voice profile missing — run `enroll data\\raw\\voice_notes` first")
    profile = np.load(profile_path)

    from pyannote.audio import Pipeline

    console.print(f"Loading {dcfg['model']} ...")
    pipeline = Pipeline.from_pretrained(dcfg["model"], use_auth_token=_hf_token())
    pipeline.to(_device())
    inference = _load_embedder(cfg)

    import torch

    processed = promoted = 0
    for audio_ref in pending:
        path = Path(audio_ref)
        if not path.exists():
            console.print(f"  [red]{path.name}: file moved/deleted, skipping[/red]")
            continue
        wav = load_audio(path)
        annotation = pipeline(
            {"waveform": torch.from_numpy(wav), "sample_rate": TARGET_SR}
        )
        turns = [
            (turn.start, turn.end, label)
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
        if not turns:
            console.print(f"  [yellow]{path.name}: no speech found[/yellow]")
            continue

        sims = {
            label: cosine(profile, emb)
            for label, emb in _speaker_embeddings(inference, wav, turns).items()
        }
        best = max(sims, key=sims.get)
        if sims[best] < dcfg["similarity_threshold"]:
            console.print(
                f"  [yellow]{path.name}: no speaker matches your voice "
                f"(best {sims[best]:.2f} < {dcfg['similarity_threshold']}) — left unlabeled[/yellow]"
            )
            continue

        segments = db.fetch_segments(conn, audio_ref)
        labels = []
        labeled: list[tuple[float, float, str, str]] = []
        for seg in segments:
            who = assign_speaker(seg["start_sec"], seg["end_sec"], turns)
            speaker = "me" if who == best else "other"
            labels.append((speaker, seg["id"]))
            labeled.append((seg["start_sec"], seg["end_sec"], speaker, seg["text"]))
        db.label_segments(conn, labels)

        base_ts = file_timestamp(path)
        lang = segments[0]["lang"] if segments else None
        n = db.insert_messages(
            conn,
            [
                db.Message(
                    source="call",
                    conversation_id=f"call:{path.stem}",
                    timestamp=base_ts + timedelta(seconds=start),
                    speaker=speaker,
                    text=text,
                    audio_ref=audio_ref,
                    lang=lang,
                )
                for speaker, start, text in merge_utterances(
                    labeled, gap=dcfg["merge_gap_sec"]
                )
            ],
        )
        processed += 1
        promoted += n
        console.print(
            f"  {path.name}: me={sims[best]:.2f} similarity, "
            f"{len(sims)} speakers, {n} messages promoted"
        )
    return processed, promoted
