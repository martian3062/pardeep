"""A year spoken aloud must still reach the date filter."""
from src.voice.numbers import recover_years


def test_hindi_thousand_form():
    """Regression: the voice loop asked about 2017 and Whisper wrote the year as
    words, so the date filter never fired and the twin said it did not remember a
    year it holds 2,100 photos from."""
    assert recover_years("दो हज़ार सत्रह जून में क्या कर रहा था") == "2017 जून में क्या कर रहा था"


def test_digit_by_digit_form():
    assert recover_years("दो शून्य एक सात में") == "2017 में"


def test_punjabi_form():
    assert recover_years("ਦੋ ਹਜ਼ਾਰ ਸਤਾਰਾਂ ਵਿੱਚ") == "2017 ਵਿੱਚ"


def test_a_plain_count_is_not_turned_into_a_year():
    """"सत्रह लोग" is seventeen people, not the year 17."""
    assert recover_years("सत्रह लोग आए थे") == "सत्रह लोग आए थे"


def test_a_number_outside_the_archive_is_left_alone():
    assert "1200" not in recover_years("बारह सौ रुपये लगे")


def test_text_without_numbers_is_untouched():
    s = "यार कल मिलते हैं फिर"
    assert recover_years(s) == s


def test_year_in_the_middle_of_a_sentence():
    out = recover_years("मुझे दो हज़ार बीस वाला याद है")
    assert "2020" in out
    assert out.startswith("मुझे")
    assert out.endswith("वाला याद है")
