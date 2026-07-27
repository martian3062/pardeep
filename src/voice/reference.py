"""Build clean single-speaker reference audio for cloning his voice.

Half the clips set aside during ingestion are call recordings, which contain the
other person too. Cloning from those would blend two voices into one that
belongs to nobody. The enrolled voice embedding from Phase 1b already knows what
he sounds like, so segments are kept only when they match it.

What a cloning model wants is a few seconds of clean, continuous speech — not
the longest file. So candidates are scored on speaker match and loudness
stability, and the best few win.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# pyannote 3.x checkpoints predate torch 2.6's weights_only default; same trusted
# models the diarization pass already loads.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

REFERENCE_DIR = Path("data/processed/tts_reference")
VOICE_EMBED = Path("data/processed/me_voice.npy")
OUT_DIR = Path("data/processed/voice_clone")

TARGET_SR = 24000  # what most current cloning models expect
MIN_SECONDS = 4.0
MAX_SECONDS = 12.0

# The enrolled embedding was built from voice notes, so voice notes match it far
# more closely than phone calls do: the same speaker scores ~0.65 on a note and
# ~0.25 on a compressed call. Judging calls by the note bar rejected every one of
# 3,920 segments diarization had already confirmed as him. Calls are held to the
# threshold diarization itself was validated at.
MIN_SIMILARITY_NOTES = 0.55
MIN_SIMILARITY_CALLS = 0.40
MIN_SIMILARITY = MIN_SIMILARITY_NOTES


@dataclass
class Clip:
    path: Path
    start: float
    end: float
    similarity: float
    rms: float
    speech_ratio: float = 1.0

    @property
    def seconds(self) -> float:
        return self.end - self.start

    @property
    def score(self) -> float:
        # Speaker match alone picked windows that were his voice but barely
        # speech: two of six transcribed to nothing at all, and one to the
        # "सब्सक्राइब" artifact Whisper hallucinates over noise. A cloning model
        # copies whatever it is given, so continuous clear speech is the point.
        return self.similarity * self.speech_ratio * min(1.0, self.rms / 0.05)


# Below this the window is mostly silence, breath or background, whatever the
# speaker embedding says about it.
MIN_SPEECH_RATIO = 0.6


def _vad():
    import torch

    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    return model, utils[0]  # get_speech_timestamps


def speech_ratio(audio: np.ndarray, sr: int, vad=None) -> float:
    """Fraction of a window that silero-VAD considers speech."""
    import torch

    model, get_speech_timestamps = vad or _vad()
    # silero expects 16k
    tensor = torch.from_numpy(audio).float()
    if sr != 16000:
        import torchaudio

        tensor = torchaudio.functional.resample(tensor, sr, 16000)
    try:
        stamps = get_speech_timestamps(tensor, model, sampling_rate=16000)
    except Exception:
        return 1.0  # never let a VAD failure discard everything
    total = sum(s["end"] - s["start"] for s in stamps)
    return float(total) / max(1, tensor.numel())


def load_audio(path: Path, sr: int = TARGET_SR) -> np.ndarray:
    """Mono float32 at `sr`.

    Decoding goes through the ingest loader, which runs PyAV in a fresh
    subprocess: importing torch first replaces the FFmpeg DLLs PyAV binds to, and
    every decode afterwards fails with InvalidDataError.
    """
    from ..ingest.audio_io import load_audio_isolated

    wav = load_audio_isolated(path, target_sr=sr)
    audio = np.asarray(wav, dtype=np.float32).squeeze()
    peak = np.abs(audio).max() if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.95  # comparable level across phones and years
    return audio


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def _embedder():
    """The same model that enrolled his voice, so the embeddings are comparable."""
    import torch
    from dotenv import load_dotenv

    from ..ingest.diarize import _patch_speechbrain_lazy_modules

    _patch_speechbrain_lazy_modules()  # k2_fsa is Linux-only and breaks the import
    load_dotenv()

    from pyannote.audio import Inference, Model

    token = os.getenv("HF_TOKEN")
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Inference(model, window="whole", device=torch.device(device))


def windows(total: float, size: float, hop: float):
    start = 0.0
    while start + size <= total:
        yield start, start + size
        start += hop


def find_clips(
    files: list[Path],
    me: np.ndarray,
    window_sec: float = 8.0,
    hop_sec: float = 4.0,
) -> list[Clip]:
    """Slide over each file and keep the windows that sound like him."""
    import torch
    from pyannote.audio.core.io import AudioFile

    inference = _embedder()
    vad = _vad()
    found: list[Clip] = []
    for path in files:
        try:
            audio = load_audio(path, TARGET_SR)
        except Exception:
            continue
        duration = len(audio) / TARGET_SR
        if duration < MIN_SECONDS:
            continue
        for start, end in windows(duration, min(window_sec, duration), hop_sec):
            chunk = audio[int(start * TARGET_SR) : int(end * TARGET_SR)]
            level = rms(chunk)
            if level < 0.02:  # near-silence embeds to noise
                continue
            talking = speech_ratio(chunk, TARGET_SR, vad)
            if talking < MIN_SPEECH_RATIO:
                continue
            try:
                tensor = torch.from_numpy(chunk).unsqueeze(0)
                emb = inference({"waveform": tensor, "sample_rate": TARGET_SR})
                emb = np.asarray(emb).squeeze()
            except Exception:
                continue
            sim = cosine(emb, me)
            if sim >= MIN_SIMILARITY:
                found.append(Clip(path, start, end, sim, level, talking))
    return sorted(found, key=lambda c: c.score, reverse=True)


CALL_DB = Path("data/processed/messages.db")


def from_call_segments(
    me: np.ndarray,
    min_sec: float = 6.0,
    max_sec: float = 12.0,
    min_chars: int = 40,
    langs: tuple[str, ...] = ("hi", "pa", "en"),
    per_lang: int = 40,
    min_similarity: float = MIN_SIMILARITY_CALLS,
) -> list[Clip]:
    """Pull reference audio from the diarized calls instead of the voice notes.

    The twelve clips set aside during ingestion turned out to be poor: even after
    filtering for speech, Whisper read one as Malayalam and produced nothing at
    all for another. Meanwhile diarization had already isolated 3,920 segments of
    him talking — 32.7 hours — each with a transcript proving it is speech and a
    language tag. A segment with a real sentence in it is far better evidence of
    clean audio than a window that merely embeds like him.

    Candidates are still checked against the enrolled embedding, because
    diarization is not perfect and one wrong segment poisons a cloned voice.
    """
    import sqlite3

    conn = sqlite3.connect(CALL_DB)
    rows: list[tuple] = []
    for lang in langs:
        rows.extend(
            conn.execute(
                """SELECT audio_ref, start_sec, end_sec, text FROM call_segments
                   WHERE speaker = 'me' AND lang = ?
                     AND (end_sec - start_sec) BETWEEN ? AND ?
                     AND length(text) > ?
                   ORDER BY length(text) DESC LIMIT ?""",
                (lang, min_sec, max_sec, min_chars, per_lang),
            ).fetchall()
        )
    conn.close()

    inference = _embedder()
    vad = _vad()
    found: list[Clip] = []
    for audio_ref, start, end, _text in rows:
        path = Path(audio_ref)
        if not path.exists():
            continue
        try:
            audio = load_audio(path, TARGET_SR)
        except Exception:
            continue
        chunk = audio[int(start * TARGET_SR) : int(end * TARGET_SR)]
        if chunk.size < int(MIN_SECONDS * TARGET_SR):
            continue
        level = rms(chunk)
        if level < 0.02:
            continue
        talking = speech_ratio(chunk, TARGET_SR, vad)
        if talking < MIN_SPEECH_RATIO:
            continue
        try:
            import torch

            emb = inference(
                {"waveform": torch.from_numpy(chunk).unsqueeze(0), "sample_rate": TARGET_SR}
            )
            sim = cosine(np.asarray(emb).squeeze(), me)
        except Exception:
            continue
        if sim >= min_similarity:
            found.append(Clip(path, start, end, sim, level, talking))
    return sorted(found, key=lambda c: c.score, reverse=True)


def export(clips: list[Clip], out_dir: Path = OUT_DIR, keep: int = 6) -> list[Path]:
    """Write the best clips as wav, one per source file where possible.

    Taking the top N outright would take several windows of the same recording
    and clone from one moment of one day; spreading across files samples how he
    actually sounds.
    """
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.wav"):
        old.unlink()

    chosen: list[Clip] = []
    seen: set[Path] = set()
    for clip in clips:
        if clip.path not in seen:
            chosen.append(clip)
            seen.add(clip.path)
        if len(chosen) >= keep:
            break
    for clip in clips:  # top up if there were fewer distinct sources than `keep`
        if len(chosen) >= keep:
            break
        if clip not in chosen:
            chosen.append(clip)

    written = []
    manifest = []
    for i, clip in enumerate(chosen, 1):
        audio = load_audio(clip.path, TARGET_SR)
        segment = audio[int(clip.start * TARGET_SR) : int(clip.end * TARGET_SR)]
        dest = out_dir / f"ref_{i:02d}.wav"
        sf.write(str(dest), segment, TARGET_SR)
        written.append(dest)
        manifest.append(
            {
                "file": dest.name,
                "source": str(clip.path),
                "start": round(clip.start, 2),
                "end": round(clip.end, 2),
                "similarity": round(clip.similarity, 3),
                "speech_ratio": round(clip.speech_ratio, 3),
                "rms": round(clip.rms, 4),
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return written
