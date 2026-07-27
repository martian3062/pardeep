"""The recency tilt must nudge the ranking, never decide it."""
from datetime import datetime, timezone

WEIGHT = 0.25
HALFLIFE = 540.0


def score(similarity: float, age_days: float, halflife: float = HALFLIFE, weight: float = WEIGHT) -> float:
    """Mirror of MemoryStore.search's scoring, isolated from LanceDB."""
    return similarity * (1.0 + weight * (0.5 ** (age_days / halflife)))


def test_a_much_better_old_match_beats_a_weak_recent_one():
    """Regression: scoring once multiplied similarity by 0.5**(age/540), scaling a
    2017 memory to 0.014 of itself. Every photo memory is from those years, so
    the twin could never recall a picture."""
    old_perfect = score(0.95, age_days=3300)
    recent_weak = score(0.20, age_days=1)
    assert old_perfect > recent_weak
    assert (0.95 * 0.5 ** (3300 / 540)) < (0.20 * 0.5 ** (1 / 540))  # the old form


def test_a_flat_bonus_lets_a_weaker_recent_match_win():
    """Regression: the additive fix over-corrected. Real similarities here run
    0.1-0.4, so a flat +0.15 was not a tiebreaker but the dominant term — worth
    more than the entire similarity of a decent match. A recent chat that merely
    said "college" could outrank the actual classroom photograph.

    Scaling instead of adding keeps the bonus proportional, so the better match
    stays ahead.
    """
    photo_sim, photo_age = 0.137, 826  # 2024-04-22 classroom photo
    chat_sim, chat_age = 0.09, 60  # a recent chat mentioning college

    def additive(s, a):
        return s + 0.15 * (0.5 ** (a / HALFLIFE))

    assert additive(chat_sim, chat_age) > additive(photo_sim, photo_age)  # went wrong
    assert score(photo_sim, photo_age) > score(chat_sim, chat_age)  # correct now


def test_recency_still_breaks_a_tie():
    assert score(0.8, age_days=10) > score(0.8, age_days=2000)


def test_the_bonus_is_bounded_by_its_weight():
    """A brand-new memory gains at most 25%, so it cannot outrank a match that is
    better by more than that."""
    assert score(0.50, age_days=0) == 0.50 * 1.25
    assert score(0.70, age_days=100_000) > score(0.50, age_days=0)


def test_an_old_photo_keeps_its_similarity():
    oldest = datetime(2017, 1, 1, tzinfo=timezone.utc)
    age = (datetime(2026, 7, 27, tzinfo=timezone.utc) - oldest).days
    assert score(0.9, age_days=age) >= 0.9  # under the old scheme this was 0.013
