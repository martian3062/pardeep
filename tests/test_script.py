"""He writes in three scripts. A reply must come back in the one he used."""
from src.twin.script import language_rule, script_of


def test_plain_english_is_english():
    """Regression: asked "Tell me honestly, what are you worst at?" in English,
    the twin answered in Devanagari. A general "match his language" rule loses
    against a persona card full of Devanagari examples."""
    assert script_of("Tell me honestly, what are you worst at?") == "english"
    assert "English" in language_rule("Tell me honestly, what are you worst at?")


def test_roman_hinglish_is_not_english():
    assert script_of("chal thik hai bro, kal milte hain fir") == "hinglish"
    assert script_of("2017 june me kya kar raha tha? kuch yaad hai?") == "hinglish"
    assert "Devanagari" in language_rule("kya kar raha tha yaar")


def test_devanagari():
    assert script_of("हां यार, कल मिलते हैं फिर") == "devanagari"


def test_devanagari_with_english_loanwords_stays_devanagari():
    """He code-switches constantly; a few Latin words do not change the script."""
    assert script_of("train यार्ड में photos पड़ी है मेरे पास") == "devanagari"


def test_gurmukhi():
    assert script_of("ਯਾਰ ਪਿਛਲੇ ਸਾਲ ਦਾ ਕਿਹੜਾ ਖਾਸ ਪੁੱਛ ਰਿਹਾ") == "gurmukhi"
    assert "Gurmukhi" in language_rule("ਯਾਰ ਕੀ ਹਾਲ ਹੈ")


def test_english_technical_question_is_still_english():
    assert script_of("What model are you running for inference?") == "english"


def test_empty_and_symbols():
    assert script_of("") == "unknown"
    assert script_of("??") == "unknown"
    assert language_rule("") == ""
