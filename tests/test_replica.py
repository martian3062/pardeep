"""The replica is only as exact as its gates and its yardstick."""
import numpy as np

from src.replica import dataset as ds
from src.replica import verify as vf


# ------------------------------------------------------------------ the gate


def test_a_clean_solo_photo_passes():
    ok, reason, warning = ds.judge(face_px=600, sharp=200.0, identity_sim=0.7, n_faces=1)
    assert ok and not reason and not warning


def test_group_photos_are_rejected():
    """Training on a group shot teaches the model to average people."""
    ok, reason, _ = ds.judge(face_px=600, sharp=200.0, identity_sim=0.7, n_faces=3)
    assert not ok and "3 faces" in reason


def test_small_faces_are_rejected():
    ok, reason, _ = ds.judge(face_px=200, sharp=200.0, identity_sim=0.7, n_faces=1)
    assert not ok and "small" in reason


def test_blur_is_rejected():
    """A soft face teaches soft skin — the training-data lesson from every
    earlier phase, applied here before GPU-hours are spent."""
    ok, reason, _ = ds.judge(face_px=600, sharp=12.0, identity_sim=0.7, n_faces=1)
    assert not ok and "blurry" in reason


def test_a_different_person_is_rejected():
    """The tripwire this gate exists for: one stray photo of someone else would
    poison the whole identity."""
    ok, reason, _ = ds.judge(face_px=600, sharp=200.0, identity_sim=0.1, n_faces=1)
    assert not ok and "identity" in reason


def test_aged_but_plausible_gets_a_warning_not_a_rejection():
    """person_00 is mostly 2017-2020; he looks different now. Low-but-plausible
    similarity warns instead of rejecting a genuinely recent photo."""
    ok, reason, warning = ds.judge(face_px=600, sharp=200.0, identity_sim=0.30, n_faces=1)
    assert ok and not reason
    assert "recent" in warning


def test_no_identity_reference_still_gates_on_quality():
    ok, _, _ = ds.judge(face_px=600, sharp=200.0, identity_sim=None, n_faces=1)
    assert ok


# ---------------------------------------------------------------- captions


def test_caption_leads_with_the_trigger():
    line = ds.caption_template("wearing a navy turban and grey hoodie, warm light")
    assert line.startswith(f"photo of {ds.TRIGGER},")
    assert "navy turban" in line


def test_empty_description_still_yields_a_valid_caption():
    assert ds.caption_template("") == f"photo of {ds.TRIGGER}"


def test_trigger_is_not_a_real_word():
    """A real word would drag its meaning into every generation."""
    for tok in ds.TRIGGER.split():
        assert tok == "man" or tok not in {
            "photo", "person", "face", "portrait", "male", "sikh", "singh",
        }


# ------------------------------------------------------------- calibration


def _base_identity(dim=64, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim)
    return base / np.linalg.norm(base)


def _draws(base, n=20, spread=0.04, seed=0):
    """Unit vectors near `base` — one person across lighting and years.

    The spread has to stay well under 1/sqrt(dim)·dim tolerance: per-dimension
    noise of 0.15 over 64 dims has norm ≈ 1.2, which *swamps* a unit identity
    and models twenty strangers, not one person. The first version of this
    fixture made exactly that mistake and failed its own tests.
    """
    rng = np.random.default_rng(seed)
    embs = base + rng.normal(scale=spread, size=(n, len(base)))
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)


def test_leave_one_out_scores_high_for_one_person():
    cal = vf.calibrate(_draws(_base_identity()))
    assert cal.median > 0.8
    assert cal.floor <= cal.p25 <= cal.median


def test_generated_battery_that_matches_him_passes():
    him = _base_identity()
    embs = _draws(him, seed=0)
    cal = vf.calibrate(embs)
    report = vf.Report(calibration_median=cal.median, calibration_p25=cal.p25)
    centroid = embs.mean(axis=0)
    for i, e in enumerate(_draws(him, n=10, seed=1)):  # same person, fresh draws
        report.scores.append(vf.Score(f"gen_{i}.png", vf.cosine(e, centroid), 1, True))
    assert report.verdict == "pass"


def test_a_different_face_fails_the_battery():
    him = _base_identity(seed=0)
    embs = _draws(him)
    cal = vf.calibrate(embs)
    report = vf.Report(calibration_median=cal.median, calibration_p25=cal.p25)
    centroid = embs.mean(axis=0)
    stranger = _base_identity(seed=123)
    for i, e in enumerate(_draws(stranger, n=10, seed=5)):
        report.scores.append(vf.Score(f"gen_{i}.png", vf.cosine(e, centroid), 1, False))
    assert report.verdict.startswith("fail")


def test_faceless_generations_fail_the_battery_wholesale():
    """A model that stopped producing clean single faces has failed even if the
    few faces it makes score well."""
    embs = _draws(_base_identity())
    cal = vf.calibrate(embs)
    report = vf.Report(calibration_median=cal.median, calibration_p25=cal.p25)
    report.scores.append(vf.Score("ok.png", cal.median, 1, True))
    for i in range(5):
        report.scores.append(vf.Score(f"bad_{i}.png", 0.0, 0, False, "no face"))
    assert "clean face" in report.verdict
