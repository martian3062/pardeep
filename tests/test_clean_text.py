from src.ingest.clean_text import clean_transcript, collapse_repeats


def test_collapse_word_loop():
    assert collapse_repeats("कर दो " * 12) == "कर दो कर दो"
    assert collapse_repeats("नहीं " * 8) == "नहीं नहीं"


def test_collapse_phrase_loop():
    text = "I don't know what to say. " * 4
    assert collapse_repeats(text) == "I don't know what to say. I don't know what to say."


def test_natural_repetition_survives():
    # people really do say things twice — only runs beyond max_repeat collapse
    assert collapse_repeats("haan haan bhai") == "haan haan bhai"
    assert collapse_repeats("acha acha theek hai") == "acha acha theek hai"


def test_normal_speech_untouched():
    s = "kal milte hai yaar, movie chalein?"
    assert collapse_repeats(s) == s
    assert clean_transcript(s) == s


def test_clean_strips_artifacts():
    assert clean_transcript("foreign foreign kaise ho") == "kaise ho"
    assert clean_transcript("Thanks for watching!") == ""
    assert clean_transcript("Terima kasih") == ""


def test_clean_pure_noise_returns_empty_or_single():
    assert clean_transcript("foreign") == ""
    assert clean_transcript("no no no no no no no no") == "no no"


def test_clean_collapses_and_normalizes():
    assert clean_transcript("  बहुत  बहुत बहुत बहुत बहुत  ") == "बहुत बहुत"
