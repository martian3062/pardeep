"""Resumable, low-memory model download.

The default multi-worker download holds several 10MB chunks in flight per file and
died with a MemoryError partway through an 8.6GB VLM when the machine was under
memory pressure. One worker, small chunks, and a retry loop turn a slow link into
a slow-but-finishing download; HuggingFace resumes from the .incomplete blobs, so
a crash costs seconds rather than gigabytes.

    python scripts/fetch_model.py Qwen/Qwen3-VL-4B-Instruct
"""
from __future__ import annotations

import os
import sys
import time

# must precede any huggingface import so the cache lands on E:\cache
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src  # noqa: F401  (sets HF_HOME / clears TRANSFORMERS_CACHE)

from huggingface_hub import constants, snapshot_download

# 10MB chunks × workers is what ran the machine out of RAM
constants.DOWNLOAD_CHUNK_SIZE = 1024 * 1024

# weights + config + tokenizer only: no .bin duplicates, no ONNX, no demo assets
ALLOW = ["*.safetensors", "*.json", "*.txt", "*.model", "*.py"]
IGNORE = ["*.bin", "*.onnx", "*.msgpack", "*.h5", "original/*", "*.gguf"]


def fetch(repo_id: str, attempts: int = 40) -> str:
    for attempt in range(1, attempts + 1):
        try:
            path = snapshot_download(
                repo_id,
                allow_patterns=ALLOW,
                ignore_patterns=IGNORE,
                max_workers=1,
                etag_timeout=60,
            )
            print(f"\nOK  {repo_id}\n    {path}")
            return path
        except Exception as exc:  # noqa: BLE001 — any failure is worth retrying
            wait = min(30, 3 * attempt)
            print(f"[{attempt}/{attempts}] {type(exc).__name__}: {exc} — retrying in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"gave up on {repo_id} after {attempts} attempts")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for repo in sys.argv[1:]:
        fetch(repo)
