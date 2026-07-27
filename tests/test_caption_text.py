"""Captions should say what is in a photo, not what isn't."""
from src.vision.caption import strip_absence


def test_drops_the_most_common_boilerplate():
    """'No other people or readable text are visible.' appeared in 75 of 931
    captions — noise that makes unrelated photos embed alike."""
    text = (
        "A young child wearing a bright orange turban smiles at the camera. "
        "No other people or readable text are visible."
    )
    assert strip_absence(text) == "A young child wearing a bright orange turban smiles at the camera."


def test_drops_absence_variants():
    for tail in (
        "No readable text or specific occasion is visible.",
        "No readable text or occasion is visible.",
        "No text or readable labels are present.",
        "No other people are visible.",
        "Only one person is visible.",
        "There is no readable text.",
        "No readable text or specific occasion is discernible.",
    ):
        out = strip_absence(f"Two men stand beside a green locomotive. {tail}")
        assert out == "Two men stand beside a green locomotive.", tail


def test_keeps_everything_that_describes_the_photo():
    text = (
        "A baby in a white shirt with red text sits on a striped surface. "
        "The text on the shirt reads 'KERS DELIC' and 'Old Style.'"
    )
    assert strip_absence(text) == text


def test_absence_in_the_middle_is_removed_without_mangling_the_rest():
    text = (
        "A man stands in a field. No readable text is visible. "
        "A tractor is parked behind him."
    )
    out = strip_absence(text)
    assert "A man stands in a field." in out
    assert "A tractor is parked behind him." in out
    assert "readable text" not in out


def test_a_caption_that_is_only_absence_is_left_alone():
    """Better a weak caption than an empty memory."""
    text = "No readable text or specific occasion is visible."
    assert strip_absence(text) == text


def test_does_not_eat_a_real_observation_starting_with_no():
    text = "A crowd fills the street. Nobody is wearing a helmet in the photo."
    assert "Nobody is wearing a helmet" in strip_absence(text)
