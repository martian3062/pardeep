"""The recency tilt must break ties, not decide the ranking."""
from datetime import datetime, timezone


def score(similarity: float, age_days: float, halflife: float = 540.0, weight: float = 0.15) -> float:
    """Mirror of MemoryStore.search's scoring, isolated from LanceDB."""
    return similarity + weight * (0.5 ** (age_days / halflife))


def test_a_much_better_old_match_beats_a_weak_recent_one():
    """Regression: scoring multiplied similarity by 0.5**(age/540), which scaled a
    2017 memory by ~0.014. Every photo memory is from those years, so the twin
    could never recall a picture — a perfect 9-year-old match lost to noise."""
    old_perfect = score(0.95, age_days=3300)  # a 2017 photo that answers the query
    recent_weak = score(0.20, age_days=1)  # yesterday, barely related
    assert old_perfect > recent_weak

    # the old multiplicative form got this backwards
    assert (0.95 * 0.5 ** (3300 / 540)) < (0.20 * 0.5 ** (1 / 540))


def test_recency_still_breaks_a_tie():
    """The tilt's actual purpose: equally relevant, prefer the newer one."""
    assert score(0.8, age_days=10) > score(0.8, age_days=2000)


def test_recency_bonus_is_bounded():
    """A brand-new memory can gain at most `weight`, so it cannot outrank a
    match that is better by more than that."""
    newest = score(0.50, age_days=0)
    better_but_ancient = score(0.70, age_days=100_000)
    assert better_but_ancient > newest
    assert newest - 0.50 <= 0.15 + 1e-9


def test_age_of_the_oldest_archive_photo_still_scores_usefully():
    oldest = datetime(2017, 1, 1, tzinfo=timezone.utc)
    age = (datetime(2026, 7, 27, tzinfo=timezone.utc) - oldest).days
    # under the old scheme this was ~0.013 of its similarity; now it keeps it
    assert score(0.9, age_days=age) > 0.9
