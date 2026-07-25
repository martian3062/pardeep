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

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rich.console import Console

from ..config import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")
# pyannote 3.x checkpoints predate torch 2.6's weights_only default; they are
# official trusted models, so allow the legacy load path.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def _patch_speechbrain_lazy_modules() -> None:
    """torch's op registration inspects all loaded modules (hasattr __file__),
    which triggers speechbrain's lazy k2_fsa import — a Linux-only dep that
    ImportErrors on Windows and kills the pyannote Pipeline import. Make lazy
    modules answer dunder introspection with AttributeError instead of
    importing."""
    try:
        from speechbrain.utils import importutils
    except ImportError:
        return
    if getattr(importutils.LazyModule, "_dunder_safe", False):
        return
    orig = importutils.LazyModule.__getattr__

    def safe_getattr(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return orig(self, attr)

    importutils.LazyModule.__getattr__ = safe_getattr
    importutils.LazyModule._dunder_safe = True
from . import db
from .audio_io import TARGET_SR, duration_sec, load_audio_isolated as load_audio
from .clean_text import clean_transcript
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
        try:
            wav = load_audio(path)
            dur = duration_sec(wav)
            if dur < dcfg["min_enroll_sec"]:
                continue
            emb = _embed(inference, wav)
        except Exception as exc:
            console.print(f"  [yellow]skipped {path.name}: {exc}[/yellow]")
            continue
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
            crop = waveform[:, int(start * TARGET_SR) : int(end * TARGET_SR)]
            if crop.shape[1] > 0:
                crops.append(crop)
                total += end - start
            if total >= max_audio_sec:
                break
        if crops:
            out[label] = _embed(inference, np.concatenate(crops, axis=1))
    return out


def _cache_file(audio_ref: str) -> Path:
    d = REPO_ROOT / "data" / "processed" / "diarize_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha1(audio_ref.encode("utf-8")).hexdigest() + ".npz")


def _analyze_pending(conn: sqlite3.Connection, cfg: dict) -> None:
    """Pass 1: diarize + embed every pending call ONCE; cache to disk so
    clustering/labeling never has to re-run the expensive audio analysis."""
    dcfg = cfg["diarization"]
    todo = [
        ref
        for ref in db.pending_call_files(conn)
        if not _cache_file(ref).exists() and Path(ref).exists()
    ]
    if not todo:
        return

    _patch_speechbrain_lazy_modules()
    import torch
    from pyannote.audio import Pipeline

    console.print(f"Loading {dcfg['model']} ... ({len(todo)} calls to analyze)")
    pipeline = Pipeline.from_pretrained(dcfg["model"], use_auth_token=_hf_token())
    pipeline.to(_device())
    inference = _load_embedder(cfg)

    for i, audio_ref in enumerate(todo, 1):
        path = Path(audio_ref)
        try:
            wav = load_audio(path)
            annotation = pipeline(
                {"waveform": torch.from_numpy(wav), "sample_rate": TARGET_SR}
            )
            turns = [
                (turn.start, turn.end, label)
                for turn, _, label in annotation.itertracks(yield_label=True)
            ]
            embs = _speaker_embeddings(inference, wav, turns)
        except Exception as exc:
            console.print(f"  [yellow][{i}/{len(todo)}] {path.name}: analysis failed ({exc})[/yellow]")
            np.savez(_cache_file(audio_ref), emb_labels=np.array([], dtype="U8"))
            continue
        labels = list(embs)
        np.savez(
            _cache_file(audio_ref),
            turn_bounds=np.array([[s, e] for s, e, _ in turns], dtype=np.float64),
            turn_labels=np.array([l for _, _, l in turns], dtype="U16"),
            emb_labels=np.array(labels, dtype="U16"),
            emb_matrix=(
                np.stack([embs[l].ravel() for l in labels])
                if labels
                else np.zeros((0, 1), dtype=np.float32)
            ),
        )
        console.print(f"  [{i}/{len(todo)}] {path.name}: {len(turns)} turns, {len(labels)} speakers")


def _load_cache(audio_ref: str):
    f = _cache_file(audio_ref)
    if not f.exists():
        return None
    d = np.load(f)
    if len(d["emb_labels"]) == 0:
        return None
    turns = [
        (float(s), float(e), str(l)) for (s, e), l in zip(d["turn_bounds"], d["turn_labels"])
    ]
    embs = {str(l): v for l, v in zip(d["emb_labels"], d["emb_matrix"])}
    return turns, embs


def _cluster_speakers(calls: dict[str, tuple], cluster_threshold: float) -> list[dict]:
    """Greedy cosine clustering of speaker embeddings across calls, sorted by
    how many distinct calls each cluster appears in (descending)."""
    clusters: list[dict] = []
    for ref, (_, embs) in calls.items():
        for emb in embs.values():
            v = emb / (np.linalg.norm(emb) or 1.0)
            best, best_sim = None, cluster_threshold
            for cl in clusters:
                sim = float(np.dot(cl["centroid"], v))
                if sim > best_sim:
                    best, best_sim = cl, sim
            if best is None:
                clusters.append({"sum": v.copy(), "centroid": v.copy(), "calls": {ref}})
            else:
                best["sum"] += v
                best["centroid"] = best["sum"] / np.linalg.norm(best["sum"])
                best["calls"].add(ref)
    return sorted(clusters, key=lambda c: len(c["calls"]), reverse=True)


def _pick_owner(
    clusters: list[dict], n_calls: int, memo: np.ndarray | None
) -> tuple[np.ndarray, float, str]:
    """The phone owner is the voice in the most calls. FALLBACK: if no cluster
    convincingly spans the corpus (coverage < 50%), pick among the 5 widest
    clusters the one closest to the voice-memo fingerprint."""
    top = clusters[0]
    coverage = len(top["calls"]) / n_calls
    if coverage >= 0.5 or memo is None:
        return top["centroid"], coverage, "coverage"
    candidates = clusters[:5]
    by_memo = max(candidates, key=lambda c: float(np.dot(c["centroid"], memo)))
    return by_memo["centroid"], len(by_memo["calls"]) / n_calls, "memo-fallback"


def _refine_owner_em(
    calls: dict[str, tuple], init: np.ndarray, iters: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy clustering fragments the owner's voice across varying phone-audio
    conditions (measured: only ~40% coverage, 379 clusters). This EM refinement
    uses the far stronger prior that the phone's owner speaks in EVERY call:
    repeatedly pick each call's best-matching speaker, then re-average.

    Validated: starting from the acoustic top cluster and from the independent
    voice-memo fingerprint converge to the same centroid (cosine 1.00), which is
    what confirms the identified voice really is the owner.

    Returns (centroid, per-call best-match similarities).
    """
    c = init / (np.linalg.norm(init) or 1.0)
    for _ in range(iters):
        picks = [
            max(embs.values(), key=lambda v: float(np.dot(c, v / (np.linalg.norm(v) or 1.0))))
            for _, embs in calls.values()
            if embs
        ]
        if not picks:
            break
        new = np.mean([p / (np.linalg.norm(p) or 1.0) for p in picks], axis=0)
        new /= np.linalg.norm(new) or 1.0
        converged = float(np.dot(new, c)) > 0.9999
        c = new
        if converged:
            break
    sims = np.array(
        [
            max(float(np.dot(c, v / (np.linalg.norm(v) or 1.0))) for v in embs.values())
            for _, embs in calls.values()
            if embs
        ]
    )
    return c, sims


def diarize_calls(conn: sqlite3.Connection, cfg: dict) -> tuple[int, int]:
    """Two-pass: (1) analyze+cache all pending calls, (2) cluster speakers
    across calls to find the owner's voice, then label + promote.

    Returns (files_processed, messages_promoted).
    """
    dcfg = cfg["diarization"]
    _analyze_pending(conn, cfg)

    pending = db.pending_call_files(conn)
    calls: dict[str, tuple] = {}
    for ref in pending:
        cached = _load_cache(ref)
        if cached:
            calls[ref] = cached
    if not calls:
        console.print("[yellow]No analyzable pending calls[/yellow]")
        return 0, 0

    memo = None
    profile_path = REPO_ROOT / cfg["paths"]["voice_profile"]
    if profile_path.exists():
        memo = np.load(profile_path).ravel()
        memo = memo / np.linalg.norm(memo)

    clusters = _cluster_speakers(calls, dcfg.get("cluster_threshold", 0.55))
    seed, coverage, method = _pick_owner(clusters, len(calls), memo)
    console.print(
        f"  seed cluster spans {coverage:.0%} of {len(calls)} calls (by {method})"
    )

    centroid, sims = _refine_owner_em(calls, seed)
    console.print(
        f"[bold]Owner voice (EM-refined): per-call match mean {sims.mean():.2f}, "
        f"{(sims >= dcfg.get('min_me_sim', 0.35)).mean():.0%} of calls above threshold[/bold]"
    )
    if memo is not None:
        # independent confirmation: does EM from the memo fingerprint agree?
        alt, _ = _refine_owner_em(calls, memo)
        console.print(
            f"  vs voice-memo fingerprint: {float(np.dot(centroid, memo)):.2f} | "
            f"EM-from-memos agreement: {float(np.dot(centroid, alt)):.2f} "
            f"(1.00 = independent methods identify the same voice)"
        )

    min_sim = dcfg.get("min_me_sim", 0.35)
    processed = promoted = 0
    for ref, (turns, embs) in calls.items():
      try:
        path = Path(ref)
        if not path.exists():
            continue
        sims = {
            l: float(np.dot(centroid, e / (np.linalg.norm(e) or 1.0)))
            for l, e in embs.items()
        }
        best = max(sims, key=sims.get)
        me_present = sims[best] >= min_sim

        segments = db.fetch_segments(conn, ref)
        labels = []
        labeled: list[tuple[float, float, str, str]] = []
        for seg in segments:
            who = assign_speaker(seg["start_sec"], seg["end_sec"], turns)
            speaker = "me" if (me_present and who == best) else "other"
            labels.append((speaker, seg["id"]))
            labeled.append((seg["start_sec"], seg["end_sec"], speaker, seg["text"]))
        db.label_segments(conn, labels)

        base_ts = file_timestamp(path)
        lang = segments[0]["lang"] if segments else None
        # segments keep the raw ASR text; messages (the training corpus) get
        # Whisper's repetition loops and artifacts cleaned out
        utterances = [
            (speaker, start, clean_transcript(text))
            for speaker, start, text in merge_utterances(labeled, gap=dcfg["merge_gap_sec"])
        ]
        n = db.insert_messages(
            conn,
            [
                db.Message(
                    source="call",
                    conversation_id=f"call:{path.stem}",
                    timestamp=base_ts + timedelta(seconds=start),
                    speaker=speaker,
                    text=text,
                    audio_ref=ref,
                    lang=lang,
                )
                for speaker, start, text in utterances
                if text
            ],
        )
        processed += 1
        promoted += n
        tag = f"me={sims[best]:.2f}" if me_present else f"[yellow]owner absent (best {sims[best]:.2f})[/yellow]"
        console.print(f"  {path.name}: {tag}, {len(sims)} speakers, {n} messages")
      except Exception as exc:
        console.print(f"  [yellow]{Path(ref).name}: labeling failed ({exc}) — skipped[/yellow]")
    return processed, promoted
