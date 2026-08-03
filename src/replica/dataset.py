"""Turn 20-30 of his photos into a training bundle an identity model can learn from.

The model can only be as exact as the dataset is clean, so every photo passes
three gates before it is allowed in:

- sharp enough and the face large enough — a soft 400px face teaches soft skin
- exactly one dominant face — a group shot teaches the model to average people
- actually him — checked against the archive identity (person_00, 204 faces from
  Phase 8), which catches a stray photo of someone else before it poisons a
  training run that costs GPU-hours

Captions matter more than usual here because of the dastar. An identity LoRA
binds whatever is constant across the dataset into the trigger token; whatever
varies must be *named* so it stays controllable. His turban colour varies photo
to photo — naming the colour in each caption keeps "which dastar" promptable
instead of the model welding one colour onto his face.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REF_DIR = Path("data/raw/face_ref")
BUNDLE_DIR = Path("data/processed/replica/bundle")
GT_PATH = Path("data/processed/replica/ground_truth.npz")
PHOTOS_DB = Path("data/processed/photos.db")

# The token the model will learn to mean "him". Rare on purpose: a real word
# would drag its existing meaning into every generation.
TRIGGER = "pxrdp man"

MIN_FACE_PX = 384          # below this the crop upscales and teaches mush
MIN_SHARPNESS = 60.0       # variance of Laplacian on the face crop
# person_00 is mostly 2017-2020; he looks different now. The gate exists to catch
# a *different person*, not to punish ageing — so reject only when clearly not
# him, and warn in between.
IDENTITY_REJECT = 0.22
IDENTITY_WARN = 0.42
TRAIN_SIZE = 1024


@dataclass
class RefPhoto:
    path: Path
    accepted: bool
    reason: str = ""
    warning: str = ""
    face_px: int = 0
    sharpness: float = 0.0
    identity_sim: float = 0.0
    embedding: np.ndarray | None = None
    crop_path: Path | None = None
    caption: str = ""


@dataclass
class Curation:
    photos: list[RefPhoto] = field(default_factory=list)

    @property
    def accepted(self) -> list[RefPhoto]:
        return [p for p in self.photos if p.accepted]

    @property
    def rejected(self) -> list[RefPhoto]:
        return [p for p in self.photos if not p.accepted]


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard cheap focus measure."""
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def archive_centroid() -> np.ndarray | None:
    """His identity as the archive knows it — the wrong-person tripwire."""
    import sqlite3

    if not PHOTOS_DB.exists():
        return None
    conn = sqlite3.connect(PHOTOS_DB)
    rows = conn.execute(
        "SELECT embedding FROM photo_faces WHERE person = 'person_00'"
    ).fetchall()
    conn.close()
    if not rows:
        return None
    embs = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    centroid = embs.mean(axis=0)
    return centroid / np.linalg.norm(centroid)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def judge(
    face_px: int,
    sharp: float,
    identity_sim: float | None,
    n_faces: int,
) -> tuple[bool, str, str]:
    """The gate, as a pure function so the thresholds are testable."""
    if n_faces == 0:
        return False, "no face found", ""
    if n_faces > 1:
        return False, f"{n_faces} faces — needs to be a solo photo", ""
    if face_px < MIN_FACE_PX:
        return False, f"face only {face_px}px — too small to train on", ""
    if sharp < MIN_SHARPNESS:
        return False, f"blurry (sharpness {sharp:.0f})", ""
    if identity_sim is not None and identity_sim < IDENTITY_REJECT:
        return False, f"does not match his identity (sim {identity_sim:.2f})", ""
    warning = ""
    if identity_sim is not None and identity_sim < IDENTITY_WARN:
        warning = f"low archive match ({identity_sim:.2f}) — fine if the photo is recent"
    return True, "", warning


def _crop_face(img, bbox, margin: float = 0.55, size: int = TRAIN_SIZE):
    """Square crop centred on the face with generous margin — the dastar is part
    of his identity and cropping it off would train a bareheaded stranger."""
    import cv2

    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + margin)
    h, w = img.shape[:2]
    x1c, x2c = int(max(0, cx - half)), int(min(w, cx + half))
    y1c, y2c = int(max(0, cy - half * 1.15)), int(min(h, cy + half))  # extra headroom up top
    crop = img[y1c:y2c, x1c:x2c]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)


def curate(src: Path = REF_DIR, out: Path = BUNDLE_DIR) -> Curation:
    import cv2

    from ..vision import faces as fc

    app = fc.analyser()
    centroid = archive_centroid()
    out.mkdir(parents=True, exist_ok=True)

    result = Curation()
    images = sorted(
        p for p in src.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    )
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            result.photos.append(RefPhoto(path, False, reason="unreadable"))
            continue
        found = fc.detect(app, path)
        # judge on the dominant face; secondary tiny faces (posters, reflections)
        # only disqualify when genuinely comparable in size
        found.sort(key=lambda f: f.size, reverse=True)
        comparable = [f for f in found if found and f.size > found[0].size * 0.5]
        face = found[0] if found else None
        gray_sharp = 0.0
        sim = None
        if face is not None:
            x1, y1, x2, y2 = face.bbox
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                gray_sharp = sharpness(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            if centroid is not None:
                sim = cosine(face.embedding, centroid)

        ok, reason, warning = judge(
            face.size if face else 0, gray_sharp, sim, len(comparable) if found else 0
        )
        photo = RefPhoto(
            path=path,
            accepted=ok,
            reason=reason,
            warning=warning,
            face_px=face.size if face else 0,
            sharpness=gray_sharp,
            identity_sim=sim or 0.0,
            embedding=face.embedding if face else None,
        )
        if ok and face is not None:
            crop_img = _crop_face(img, face.bbox)
            photo.crop_path = out / f"{path.stem}.png"
            cv2.imwrite(str(photo.crop_path), crop_img)
        result.photos.append(photo)
    return result


# ------------------------------------------------------------------ captions


def caption_template(description: str) -> str:
    """One caption line. The trigger leads; the variable scene follows."""
    description = " ".join(description.split()).strip().rstrip(".")
    return f"photo of {TRIGGER}, {description}" if description else f"photo of {TRIGGER}"


CAPTION_PROMPT = (
    "Describe this photo for image-model training in ONE short line: the colour of the "
    "turban if worn, clothing, lighting, background, and expression. Do NOT describe his "
    "face, age, or identity. Example: 'wearing a navy blue turban and grey hoodie, indoor "
    "warm light, plain wall behind, slight smile'."
)


def caption_all(curation: Curation, use_vlm: bool = True) -> None:
    """Caption accepted crops. Qwen3-VL (already local from Phase 4b) names what
    varies; the fallback template at least keeps the trigger consistent."""
    captioner = None
    if use_vlm:
        try:
            from ..vision.caption import Captioner

            captioner = Captioner()
        except Exception:
            captioner = None

    for photo in curation.accepted:
        desc = ""
        if captioner is not None and photo.crop_path:
            try:
                desc = captioner.caption(photo.crop_path, hint="") or ""
                # the VLM answers in sentences; keep it one line
                desc = desc.split(".")[0][:140]
            except Exception:
                desc = ""
        photo.caption = caption_template(desc)
        if photo.crop_path:
            photo.crop_path.with_suffix(".txt").write_text(photo.caption, encoding="utf-8")


# ------------------------------------------------------------------- bundle


def export(curation: Curation, out: Path = BUNDLE_DIR) -> Path:
    """Ground truth + manifest. The same photos that train the model also define
    the bar it will be judged against — they are the only honest definition of
    'exactly him' that exists."""
    accepted = [p for p in curation.accepted if p.embedding is not None]
    if accepted:
        np.savez(
            GT_PATH,
            embeddings=np.stack([p.embedding for p in accepted]),
            files=np.array([p.path.name for p in accepted]),
        )
    manifest = {
        "trigger": TRIGGER,
        "accepted": [
            {
                "file": p.path.name,
                "crop": p.crop_path.name if p.crop_path else None,
                "caption": p.caption,
                "face_px": p.face_px,
                "sharpness": round(p.sharpness, 1),
                "identity_sim": round(p.identity_sim, 3),
                "warning": p.warning,
            }
            for p in curation.accepted
        ],
        "rejected": [{"file": p.path.name, "reason": p.reason} for p in curation.rejected],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def pack(bundle: Path = BUNDLE_DIR, archive: Path | None = None) -> Path:
    """One file to scp to the GPU box."""
    archive = archive or bundle.parent / "replica_bundle"
    return Path(shutil.make_archive(str(archive), "zip", bundle))
