"""Decode any audio format (.opus/.m4a/.amr/...) to a mono float32 waveform via PyAV.

Used instead of torchaudio/soundfile because WhatsApp/call-recorder formats vary wildly
and PyAV (bundled with faster-whisper) decodes them all.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TARGET_SR = 16000


def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Returns shape (1, samples) float32 in [-1, 1] at target_sr.

    Tolerant of corrupt frames (common in old phone-backup recordings): bad
    packets are skipped and every decodable stretch of audio is kept.
    """
    import av

    container = av.open(str(path))
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
    chunks: list[np.ndarray] = []
    try:
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue  # corrupt packet — skip, keep going
            for frame in frames:
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray())
    except av.error.InvalidDataError:
        pass  # corrupt container tail — keep what we salvaged
    try:
        for rf in resampler.resample(None):  # flush
            chunks.append(rf.to_ndarray())
    except av.error.InvalidDataError:
        pass
    container.close()
    if not chunks:
        return np.zeros((1, 0), dtype=np.float32)
    audio = np.concatenate(chunks, axis=1).astype(np.float32) / 32768.0
    return audio


def duration_sec(waveform: np.ndarray, sr: int = TARGET_SR) -> float:
    return waveform.shape[1] / sr


def load_audio_isolated(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode in a fresh subprocess.

    torch/torchaudio bundle their own FFmpeg DLLs on Windows; once they are
    imported, in-process PyAV decoding breaks (avcodec_send_packet errors).
    The diarization path imports torch, so it must decode out-of-process.
    """
    import subprocess
    import sys
    import tempfile

    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "src.ingest.audio_io", str(path), str(out), str(target_sr)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"decode failed: {proc.stderr.strip()[-200:]}")
        return np.load(out)
    finally:
        out.unlink(missing_ok=True)


if __name__ == "__main__":
    import sys

    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    sr = int(sys.argv[3]) if len(sys.argv) > 3 else TARGET_SR
    np.save(dst, load_audio(src, sr))
