"""How a turn is turned into a memory search."""
from src.twin.chat import Turn, Twin


class _Store:
    """Stands in for LanceDB — Twin only needs something to hand to gather()."""


def _twin_with_history(turns):
    twin = Twin.__new__(Twin)  # skip __init__: it would load bge-m3 and the persona
    twin.store = _Store()
    twin.guest = False
    twin.history = list(turns)
    return twin


def test_a_question_that_stands_alone_is_searched_alone():
    """Regression: asked about college straight after a question about a 2017
    trip, retrieval borrowed the previous turn, returned that trip's train
    photos, and the twin said it had no classroom photos while holding several."""
    twin = _twin_with_history(
        [Turn("user", "2017 june wali trip yaad hai? kahan gaye the"), Turn("assistant", "...")]
    )
    query = twin._retrieval_query("college me kya chal raha tha? classroom ki photo yaad hai?")
    assert "2017" not in query
    assert query.startswith("college")


def test_a_bare_follow_up_borrows_the_previous_turn():
    """"phir?" on its own retrieves nothing."""
    twin = _twin_with_history(
        [Turn("user", "2017 june me kya kar raha tha"), Turn("assistant", "...")]
    )
    query = twin._retrieval_query("phir?")
    assert "2017" in query
    assert "phir?" in query


def test_no_history_is_harmless():
    twin = _twin_with_history([])
    assert twin._retrieval_query("phir?") == "phir?"


def test_only_user_turns_are_borrowed():
    """The twin's own words are its output, not evidence of what to search for."""
    twin = _twin_with_history(
        [Turn("user", "rohan kaun hai"), Turn("assistant", "college ka dost hai steam locomotive")]
    )
    query = twin._retrieval_query("aur?")
    assert "locomotive" not in query
    assert "rohan" in query
