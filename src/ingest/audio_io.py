"""Decode any audio format (.opus/.m4a/.amr/...) to a mono float32 waveform via PyAV.

Used instead of torchaudio/soundfile because WhatsApp/call-recorder formats vary wildly
and PyAV (bundled with faster-whisper) decodes them all.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TARGET_SR = 16000


def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Returns shape (1, samples) float32 in [-1, 1] at target_sr."""
    import av

    container = av.open(str(path))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
    chunks: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray())
    for rf in resampler.resample(None):  # flush
        chunks.append(rf.to_ndarray())
    container.close()
    if not chunks:
        return np.zeros((1, 0), dtype=np.float32)
    audio = np.concatenate(chunks, axis=1).astype(np.float32) / 32768.0
    return audio


def duration_sec(waveform: np.ndarray, sr: int = TARGET_SR) -> float:
    return waveform.shape[1] / sr
