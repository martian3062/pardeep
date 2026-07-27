"""A question that names a time must be answered from that time."""
from datetime import datetime

from src.twin.when import parse_time_range

NOW = datetime(2026, 7, 27, 12, 0, 0)


def test_bare_year():
    """Regression: asked "2017 june me kya kar raha tha", retrieval returned
    eight recent chats and nothing from 2017, so the twin said it did not
    remember a year it holds 2,100 photos from."""
    lo, hi = parse_time_range("2017 me kya kar raha tha", NOW)
    assert lo == datetime(2017, 1, 1)
    assert hi == datetime(2018, 1, 1)


def test_month_and_year_narrows_to_that_month():
    lo, hi = parse_time_range("2017 june me kya kar raha tha? kuch yaad hai?", NOW)
    assert lo == datetime(2017, 6, 1)
    assert hi == datetime(2017, 7, 1)


def test_december_does_not_overflow_the_year():
    lo, hi = parse_time_range("december 2019 ki photos dikha", NOW)
    assert lo == datetime(2019, 12, 1)
    assert hi == datetime(2020, 1, 1)


def test_a_span_of_years():
    lo, hi = parse_time_range("2018 se 2020 ke beech kya hua", NOW)
    assert lo == datetime(2018, 1, 1)
    assert hi == datetime(2021, 1, 1)


def test_no_time_mentioned():
    assert parse_time_range("mera dost kaun hai", NOW) is None


def test_numbers_that_are_not_years_are_ignored():
    """Prices, marks and phone digits must not be read as dates."""
    assert parse_time_range("usne 1500 rupees mange the", NOW) is None
    assert parse_time_range("80 marks aaye the", NOW) is None


def test_hinglish_relative_time():
    lo, hi = parse_time_range("pichle saal kya kar raha tha", NOW)
    assert (NOW - lo).days == 365
    assert hi > NOW


def test_childhood_maps_to_the_old_end_of_the_archive():
    lo, hi = parse_time_range("bachpan ki photos", NOW)
    assert lo.year <= 2010
    assert hi.year <= NOW.year - 5


def test_month_alone_is_too_ambiguous_to_filter():
    """"june me" without a year would wrongly exclude every other June."""
    assert parse_time_range("june me garmi thi", NOW) is None
