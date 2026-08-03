"""Is the generated face actually him? Measured, not eyeballed.

"Exact replica" is turned into a number the same way the voice work turned
"clean audio" into bandwidth: his own reference photos are embedded, and each
one is scored against the centroid of the others (leave-one-out). That
distribution is what *he himself* scores as himself across lighting, expression
and years — the only honest definition of "indistinguishable" available.

A generated image passes when it scores like a real photo of him does. The
acceptance bar is calibrated from his data, not hard-coded, because a fixed
threshold would be either flattering or impossible depending on the camera his
photos came from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .dataset import GT_PATH, cosine

REPORT_DIR = Path("data/processed/replica/reports")


@dataclass
class Calibration:
    loo_sims: np.ndarray  # each real photo vs the centroid of the others

    @property
    def median(self) -> float:
        return float(np.median(self.loo_sims))

    @property
    def p25(self) -> float:
        return float(np.percentile(self.loo_sims, 25))

    @property
    def floor(self) -> float:
        """A single generated image below this is not a picture of him."""
        return float(np.percentile(self.loo_sims, 5))


@dataclass
class Score:
    file: str
    similarity: float
    faces: int
    passes: bool
    note: str = ""


@dataclass
class Report:
    calibration_median: float
    calibration_p25: float
    scores: list[Score] = field(default_factory=list)

    @property
    def generated_median(self) -> float:
        vals = [s.similarity for s in self.scores if s.faces == 1]
        return float(np.median(vals)) if vals else 0.0

    @property
    def verdict(self) -> str:
        """Pass = the battery's median lands inside his real-photo distribution."""
        if not self.scores:
            return "empty"
        usable = [s for s in self.scores if s.faces == 1]
        if len(usable) < len(self.scores) * 0.8:
            return "fail: too many images without a single clean face"
        if self.generated_median >= self.calibration_p25:
            return "pass"
        return (
            f"fail: generated median {self.generated_median:.3f} below his "
            f"real-photo p25 {self.calibration_p25:.3f}"
        )


def load_ground_truth() -> np.ndarray:
    if not GT_PATH.exists():
        raise FileNotFoundError(
            f"{GT_PATH} missing — run `python -m src.replica.run prep` first"
        )
    return np.load(GT_PATH)["embeddings"]


def calibrate(embeddings: np.ndarray) -> Calibration:
    """Leave-one-out: how much does a real photo of him look like the rest of
    him? Generated images are held to this, not to perfection — his own photos
    do not score 1.0 against each other either."""
    sims = []
    n = len(embeddings)
    for i in range(n):
        rest = np.delete(embeddings, i, axis=0).mean(axis=0)
        sims.append(cosine(embeddings[i], rest))
    return Calibration(loo_sims=np.asarray(sims))


def score_dir(generated: Path, embeddings: np.ndarray | None = None) -> Report:
    """Score every image in a directory of generations against him."""
    from ..vision import faces as fc

    embeddings = embeddings if embeddings is not None else load_ground_truth()
    cal = calibrate(embeddings)
    centroid = embeddings.mean(axis=0)

    app = fc.analyser()
    report = Report(calibration_median=cal.median, calibration_p25=cal.p25)

    for path in sorted(generated.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        found = fc.detect(app, path)
        found.sort(key=lambda f: f.size, reverse=True)
        if not found:
            report.scores.append(Score(path.name, 0.0, 0, False, "no face"))
            continue
        sim = cosine(found[0].embedding, centroid)
        report.scores.append(
            Score(
                file=path.name,
                similarity=round(sim, 4),
                faces=len(found),
                passes=sim >= cal.floor,
                note="" if sim >= cal.floor else "below the floor — not him",
            )
        )
    return report


def contact_sheet(
    generated: Path,
    report: Report,
    ref_dir: Path | None = None,
    out: Path | None = None,
) -> Path:
    """An HTML grid of generations beside real photos — the check ArcFace cannot do.

    buffalo_l aligns and crops the face region, which largely EXCLUDES headwear:
    a checkpoint can score 0.70 cosine while mangling the dastar's wrap, colour
    or layering. The number gates the face; only eyes gate the turban. The sheet
    puts every battery image next to real reference photos so that judgement
    takes thirty seconds instead of a folder crawl.
    """
    from .dataset import REF_DIR

    ref_dir = ref_dir or REF_DIR
    out = out or REPORT_DIR / "contact_sheet.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    refs = sorted(
        p for p in ref_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )[:6] if ref_dir.exists() else []

    def img(path: Path, label: str, cls: str = "") -> str:
        return (
            f'<figure class="{cls}"><img src="{path.resolve().as_uri()}" loading="lazy">'
            f"<figcaption>{label}</figcaption></figure>"
        )

    cells = [img(p, "REAL", "real") for p in refs]
    by_name = {s.file: s for s in report.scores}
    for p in sorted(generated.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        s = by_name.get(p.name)
        label = f"{s.similarity:.3f}" if s else "?"
        cls = "" if (s and s.passes) else "fail"
        cells.append(img(p, label, cls))

    out.write_text(
        "<!doctype html><meta charset='utf-8'><title>replica contact sheet</title>"
        "<style>body{background:#111;color:#ddd;font:13px system-ui;margin:16px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}"
        "figure{margin:0}img{width:100%;border-radius:6px;display:block}"
        "figcaption{padding:3px 2px;color:#9a9}"
        ".real img{outline:3px solid #2c5cff}.real figcaption{color:#7af}"
        ".fail img{outline:3px solid #c0392b}.fail figcaption{color:#f88}</style>"
        "<h2>Blue = real photos. Judge the dastar and beard here — the similarity "
        "score cannot see them.</h2>"
        f"<div class='grid'>{''.join(cells)}</div>",
        encoding="utf-8",
    )
    return out


def save(report: Report, name: str = "") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"verify_{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "verdict": report.verdict,
                "caveat": (
                    "similarity gates the FACE only — buffalo_l crops out headwear, "
                    "so the dastar must be judged on the contact sheet, by eye"
                ),
                "generated_median": report.generated_median,
                "calibration": {
                    "median": report.calibration_median,
                    "p25": report.calibration_p25,
                },
                "scores": [s.__dict__ for s in report.scores],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out
