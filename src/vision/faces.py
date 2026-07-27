"""Who is in each photo — and whether the photo is of anyone at all.

Two jobs from one pass over the archive.

The first is the obvious one: a photo memory that knows who is in it can answer
"show me pictures with Karan", and the twin can say who it is looking at rather
than "two men".

The second fixes a defect the twenty-question evaluation exposed. Asked to
describe an old photo, the twin replied that it had none — while holding 1,307.
Memes win vague photo queries because their captions are dense with text, while a
real photograph's caption is one plain sentence. Faces separate them cleanly:
a picture of people is a memory, a watermarked joke usually is not. So face count
becomes a signal at retrieval time, not just a label.

Everything runs locally on ONNX. No image leaves the machine.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_NAME = "buffalo_l"
MODEL_ROOT = Path(r"E:\cache\insightface")
OWNER_EMBED = Path("data/processed/me_face.npy")

# buffalo_l returns a 512-d embedding; cosine above this is the same person in
# practice for family-album photos, which are lit and posed inconsistently.
SAME_PERSON = 0.42
# Detections smaller than this are usually bystanders or background faces.
MIN_FACE_PX = 40

SCHEMA = """
CREATE TABLE IF NOT EXISTS photo_faces (
    id INTEGER PRIMARY KEY,
    photo_id INTEGER NOT NULL,
    bbox TEXT NOT NULL,
    det_score REAL,
    embedding BLOB NOT NULL,
    person TEXT,
    UNIQUE(photo_id, bbox)
);
CREATE INDEX IF NOT EXISTS idx_faces_photo ON photo_faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON photo_faces(person);
"""

ADDED_COLUMNS = {"face_count": "INTEGER", "faces_at": "TEXT"}


@dataclass
class Face:
    bbox: tuple[int, int, int, int]
    det_score: float
    embedding: np.ndarray

    @property
    def size(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return min(x2 - x1, y2 - y1)


def ensure_schema(conn: sqlite3.Connection) -> None:
    from . import photos as ph

    ph.ensure_schema(conn)
    conn.executescript(SCHEMA)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(photos)")}
    for column, decl in ADDED_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE photos ADD COLUMN {column} {decl}")
    conn.commit()


def analyser():
    """buffalo_l on CPU by default — detection is fast and the GPU is often busy."""
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name=MODEL_NAME, root=str(MODEL_ROOT), providers=providers)
    app.prepare(ctx_id=0 if "CUDA" in providers[0] else -1, det_size=(640, 640))
    return app


def detect(app, path: Path) -> list[Face]:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return []
    out = []
    for f in app.get(img):
        x1, y1, x2, y2 = (int(v) for v in f.bbox)
        face = Face(
            bbox=(x1, y1, x2, y2),
            det_score=float(f.det_score),
            embedding=np.asarray(f.normed_embedding, dtype=np.float32),
        )
        if face.size >= MIN_FACE_PX:
            out.append(face)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def save_faces(conn: sqlite3.Connection, photo_id: int, faces: list[Face]) -> None:
    from datetime import datetime

    conn.executemany(
        """INSERT OR IGNORE INTO photo_faces (photo_id, bbox, det_score, embedding)
           VALUES (?, ?, ?, ?)""",
        [
            (photo_id, ",".join(map(str, f.bbox)), f.det_score, f.embedding.tobytes())
            for f in faces
        ],
    )
    conn.execute(
        "UPDATE photos SET face_count = ?, faces_at = ? WHERE id = ?",
        (len(faces), datetime.now().isoformat(timespec="seconds"), photo_id),
    )
    conn.commit()


def pending(conn: sqlite3.Connection, min_rank: int = 2) -> list[tuple[int, str]]:
    ensure_schema(conn)
    return [
        (r[0], r[1])
        for r in conn.execute(
            """SELECT id, path FROM photos
               WHERE faces_at IS NULL AND rank >= ? AND dupe_of IS NULL
               ORDER BY taken_at""",
            (min_rank,),
        )
    ]


def load_embeddings(conn: sqlite3.Connection) -> list[tuple[int, int, np.ndarray]]:
    rows = conn.execute("SELECT id, photo_id, embedding FROM photo_faces").fetchall()
    return [(r[0], r[1], np.frombuffer(r[2], dtype=np.float32)) for r in rows]


def cluster(
    embeddings: list[tuple[int, int, np.ndarray]], threshold: float = SAME_PERSON
) -> dict[int, int]:
    """Greedy agglomeration into person groups.

    Proper clustering would need every pairwise distance; with a few thousand
    faces the greedy pass against running centroids is close enough and finishes
    in seconds. Groups are ordered by size, so person 0 is whoever appears most —
    which in a personal archive is almost always the owner.
    """
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    assignment: dict[int, int] = {}

    for face_id, _photo_id, emb in embeddings:
        best, best_sim = -1, 0.0
        for i, c in enumerate(centroids):
            sim = cosine(emb, c)
            if sim > best_sim:
                best, best_sim = i, sim
        if best >= 0 and best_sim >= threshold:
            n = counts[best]
            centroids[best] = (centroids[best] * n + emb) / (n + 1)
            counts[best] = n + 1
            assignment[face_id] = best
        else:
            centroids.append(emb.copy())
            counts.append(1)
            assignment[face_id] = len(centroids) - 1

    order = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    rank_of = {old: new for new, old in enumerate(order)}
    return {fid: rank_of[grp] for fid, grp in assignment.items()}
