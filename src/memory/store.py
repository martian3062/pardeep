"""Local memory store: what the twin KNOWS, as opposed to how it talks.

LanceDB is embedded (a folder on disk, no server) and bge-m3 embeds Hinglish,
Punjabi and English in one shared space, so a question typed in English can
retrieve a Punjabi phone conversation. Everything runs on this machine —
memories are the most private artefact in the project.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import REPO_ROOT

MEMORY_DIR = REPO_ROOT / "data" / "processed" / "memory"
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
TABLE = "memories"

# tiers used by the Phase 9 guardrails: guests only ever retrieve `public`
TRUST_TIERS = ("public", "personal", "secret")


@dataclass
class Memory:
    text: str
    kind: str  # episodic | fact | persona | diary
    source: str
    timestamp: datetime
    conversation_id: str = ""
    counterpart: str = ""
    trust: str = "personal"

    def to_row(self, vector: list[float]) -> dict:
        return {
            "vector": vector,
            "text": self.text,
            "kind": self.kind,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "ts_epoch": self.timestamp.timestamp(),
            "conversation_id": self.conversation_id,
            "counterpart": self.counterpart,
            "trust": self.trust,
        }


class Embedder:
    """bge-m3 on the laptop GPU when it fits, else CPU."""

    def __init__(self, model_name: str = EMBED_MODEL, device: str | None = None):
        os.environ.setdefault("HF_HOME", r"E:\cache\huggingface")
        from sentence_transformers import SentenceTransformer

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        vecs = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]


class MemoryStore:
    def __init__(self, path: Path = MEMORY_DIR, embedder: Embedder | None = None):
        import lancedb

        path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(path))
        self._embedder = embedder
        self._table = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def _open(self):
        if self._table is None and TABLE in self.db.table_names():
            self._table = self.db.open_table(TABLE)
        return self._table

    def add(self, memories: list[Memory], batch_size: int = 64) -> int:
        if not memories:
            return 0
        added = 0
        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]
            vectors = self.embedder.encode([m.text for m in batch])
            rows = [m.to_row(v) for m, v in zip(batch, vectors)]
            table = self._open()
            if table is None:
                self._table = self.db.create_table(TABLE, data=rows)
            else:
                table.add(rows)
            added += len(rows)
        return added

    def count(self) -> int:
        table = self._open()
        return table.count_rows() if table is not None else 0

    def search(
        self,
        query: str,
        limit: int = 8,
        kinds: tuple[str, ...] = (),
        max_trust: str = "secret",
        recency_halflife_days: float = 540.0,
    ) -> list[dict]:
        """Semantic search with a recency tilt.

        Old memories stay reachable — a 2019 conversation is still part of who he
        is — but when several are similar the newer one should surface first, so
        relevance is scaled by exponential recency rather than filtered by date.
        """
        table = self._open()
        if table is None:
            return []
        allowed = TRUST_TIERS[: TRUST_TIERS.index(max_trust) + 1]
        conditions = [f"trust IN ({', '.join(repr(t) for t in allowed)})"]
        if kinds:
            conditions.append(f"kind IN ({', '.join(repr(k) for k in kinds)})")

        vector = self.embedder.encode([query])[0]
        rows = (
            table.search(vector)
            .where(" AND ".join(conditions), prefilter=True)
            .limit(limit * 4)
            .to_list()
        )
        now = datetime.now(timezone.utc).timestamp()
        for r in rows:
            age_days = max(0.0, (now - r.get("ts_epoch", now)) / 86400)
            similarity = 1.0 - r.get("_distance", 0.0)
            r["score"] = similarity * (0.5 ** (age_days / recency_halflife_days))
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:limit]
