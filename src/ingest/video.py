"""Extract audio tracks from personal videos so they feed the speech pipeline.

Most videos in the archive are forwarded/downloaded clips (Telegram folders,
memes, movie snippets) where Pardeep never speaks. Transcribing them blindly
would teach the twin strangers' words, so extracted audio is routed through the
same diarization + voice-fingerprint matching as call recordings: segments that
do not match his enrolled voice are labelled `other` and never become training
targets, and videos where he never speaks contribute nothing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()

VIDEO_EXTS = {".mp4", ".mov", ".3gp", ".avi", ".mkv", ".webm", ".m4v"}
TARGET_SR = 16000


def iter_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def extract_audio(src: Path, dst: Path, target_sr: int = TARGET_SR) -> float:
    """Decode a video's audio track to mono 16k wav. Returns duration in seconds
    (0.0 when the file has no audio stream). Runs out-of-process: PyAV shares
    FFmpeg symbols with torch and breaks once torch is loaded."""
    code = (
        "import sys, av, numpy as np, wave\n"
        "src, dst, sr = sys.argv[1], sys.argv[2], int(sys.argv[3])\n"
        "c = av.open(src)\n"
        "if not c.streams.audio:\n"
        "    print(0.0); raise SystemExit\n"
        "res = av.AudioResampler(format='s16', layout='mono', rate=sr)\n"
        "chunks = []\n"
        "for packet in c.demux(c.streams.audio[0]):\n"
        "    try:\n"
        "        frames = packet.decode()\n"
        "    except Exception:\n"
        "        continue\n"
        "    for f in frames:\n"
        "        for rf in res.resample(f):\n"
        "            chunks.append(rf.to_ndarray())\n"
        "for rf in res.resample(None):\n"
        "    chunks.append(rf.to_ndarray())\n"
        "c.close()\n"
        "if not chunks:\n"
        "    print(0.0); raise SystemExit\n"
        "a = np.concatenate(chunks, axis=1).astype(np.int16)\n"
        "w = wave.open(dst, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)\n"
        "w.writeframes(a.tobytes()); w.close()\n"
        "print(a.shape[1] / sr)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(src), str(dst), str(target_sr)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-200:])
    try:
        return float(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0.0


def extract_all(
    root: Path, out_dir: Path, min_seconds: float = 3.0
) -> tuple[int, int, float]:
    """Returns (videos_seen, audio_written, total_seconds)."""
    videos = iter_videos(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    total = 0.0
    for i, path in enumerate(videos, 1):
        # keep the source folder in the name: it is the only clue about origin
        rel = path.relative_to(root).with_suffix("")
        safe = "__".join(rel.parts)[:120]
        dst = out_dir / f"{safe}.wav"
        if dst.exists():
            written += 1
            continue
        try:
            secs = extract_audio(path, dst)
        except Exception as exc:
            console.print(f"  [yellow]{path.name}: {exc}[/yellow]")
            continue
        if secs < min_seconds:
            dst.unlink(missing_ok=True)
            continue
        written += 1
        total += secs
        if i % 25 == 0 or i == len(videos):
            console.print(f"  {i}/{len(videos)} scanned, {written} with usable audio")
    return len(videos), written, total
