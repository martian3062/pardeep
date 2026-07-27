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

    # bge-m3 accepts 8192 tokens, but conversation chunks are ~300; leaving the
    # default window makes every batch allocate for the worst case and OOMs a
    # 6GB laptop card partway through indexing.
    MAX_TOKENS = 512

    def __init__(self, model_name: str = EMBED_MODEL, device: str | None = None):
        os.environ.setdefault("HF_HOME", r"E:\cache\huggingface")
        from sentence_transformers import SentenceTransformer

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = self.MAX_TOKENS

    def encode(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        try:
            vecs = self.model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or self.device == "cpu":
                raise
            # finish on CPU rather than lose an hour of indexing to one long batch
            import torch

            torch.cuda.empty_cache()
            self.model = self.model.to("cpu")
            self.device = "cpu"
            print("  embedding OOM on GPU — continuing on CPU")
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
        recency_weight: float = 0.25,
        time_range: tuple[float, float] | None = None,
    ) -> list[dict]:
        """Semantic search with a recency tilt that can only ever nudge.

        Two wrong shapes came before this one. Multiplying similarity by
        `0.5**(age/halflife)` scaled a 2017 memory to 0.014 of itself, so nothing
        from the early archive could win and the twin could not recall a photo at
        all. Adding a flat 0.15 then over-corrected: real similarities here run
        0.1–0.4, so the bonus was not a tiebreaker but the dominant term, and a
        chat that merely said "college" beat the actual classroom photograph.

        Scaling by `1 + w·recency` keeps the bonus proportional to how good the
        match already is. A brand-new memory gains at most 25%, which reorders
        near-ties and nothing else.
        """
        table = self._open()
        if table is None:
            return []
        allowed = TRUST_TIERS[: TRUST_TIERS.index(max_trust) + 1]
        conditions = [f"trust IN ({', '.join(repr(t) for t in allowed)})"]
        if kinds:
            conditions.append(f"kind IN ({', '.join(repr(k) for k in kinds)})")
        if time_range:
            # a question naming a year is asking about that year, not about
            # whatever happens to be worded similarly
            lo, hi = time_range
            conditions.append(f"ts_epoch >= {lo} AND ts_epoch < {hi}")

        where = " AND ".join(conditions)
        fetch = max(limit * 4, 24)

        vector = self.embedder.encode([query])[0]
        dense = table.search(vector).where(where, prefilter=True).limit(fetch).to_list()

        now = datetime.now(timezone.utc).timestamp()
        for r in dense:
            r["similarity"] = 1.0 - r.get("_distance", 0.0)

        rows = self._fuse(dense, self._keyword_search(table, query, where, fetch))

        for r in rows:
            age_days = max(0.0, (now - r.get("ts_epoch", now)) / 86400)
            recency = 0.5 ** (age_days / recency_halflife_days)
            r["score"] = r["similarity"] * (1.0 + recency_weight * recency)
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:limit]

    def ensure_fts_index(self, rebuild: bool = False) -> bool:
        """Build the full-text index keyword search needs. Safe to call often."""
        table = self._open()
        if table is None:
            return False
        try:
            if rebuild or "text_idx" not in {i.name for i in table.list_indices()}:
                table.create_fts_index("text", replace=True, use_tantivy=False)
            return True
        except Exception:
            return False

    def _keyword_search(self, table, query: str, where: str, fetch: int) -> list[dict]:
        """Literal matches, to cover what embeddings miss.

        A Hinglish question buries its subject: in "college classroom ki koi
        photo hai kya mere paas", the scaffolding outweighs the two words that
        matter, and the classroom photographs lost to chats that merely said
        "college". Keyword search finds the word "classroom" regardless.
        """
        try:
            return (
                table.search(query, query_type="fts")
                .where(where, prefilter=True)
                .limit(fetch)
                .to_list()
            )
        except Exception:
            return []  # no FTS index yet: dense results alone are still valid

    @staticmethod
    def _fuse(dense: list[dict], keyword: list[dict], k: int = 60) -> list[dict]:
        """Reciprocal rank fusion: agreement between two rankings beats a strong
        showing in either one alone."""
        if not keyword:
            return dense
        fused: dict[str, dict] = {}
        best = max((r["similarity"] for r in dense), default=1.0) or 1.0

        for rank, row in enumerate(dense):
            key = f"{row.get('timestamp')}|{row.get('text', '')[:80]}"
            row["_rrf"] = 1.0 / (k + rank + 1)
            fused[key] = row
        for rank, row in enumerate(keyword):
            key = f"{row.get('timestamp')}|{row.get('text', '')[:80]}"
            bonus = 1.0 / (k + rank + 1)
            if key in fused:
                fused[key]["_rrf"] += bonus
            else:
                row["_rrf"] = bonus
                row["similarity"] = 0.0  # dense never saw it; RRF carries it
                fused[key] = row

        rows = list(fused.values())
        # map fused rank back onto the similarity scale so downstream recency
        # weighting and the photo-intent boost keep working unchanged
        top = max(r["_rrf"] for r in rows)
        for r in rows:
            r["similarity"] = max(r.get("similarity", 0.0), best * (r["_rrf"] / top))
        return rows
