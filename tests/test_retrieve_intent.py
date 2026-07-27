"""What a photo question is actually asking for."""
from src.twin.retrieve import wants_own_photos, wants_photos


def test_a_question_mentioning_photos_wants_photos():
    assert wants_photos("koi photo hai mere paas jisme log hain?")
    assert wants_photos("meri koi purani photo describe kar")
    assert wants_photos("show me the wedding pics")
    assert wants_photos("कोई फोटो है क्या")


def test_a_question_about_something_else_does_not():
    assert not wants_photos("kal kya kiya tha")
    assert not wants_photos("mera sabse close dost kaun hai")


def test_asking_for_his_own_photo_excludes_forwards():
    """Regression: asked to describe an old photo of his, retrieval returned five
    forwarded memes and one photograph, and the twin concluded it held none of
    his own — while holding 1,079 that are not forwards. Demoting was not enough:
    a meme caption quoting a punchline matches a vague request far better than
    "a green field of tall grass" does."""
    assert wants_own_photos("meri koi purani photo describe kar")
    assert wants_own_photos("apni koi photo dikha")
    assert wants_own_photos("show me my old pictures")


def test_asking_about_forwards_in_general_still_includes_them():
    """He did share them; a question that is not about HIS photo keeps them."""
    assert not wants_own_photos("koi funny meme photo hai")
    assert not wants_own_photos("photos of a wedding")
