import numpy as np

from src.ingest.diarize import assign_speaker, cosine, merge_utterances


def test_cosine():
    a = np.array([1.0, 0.0])
    assert cosine(a, np.array([1.0, 0.0])) == 1.0
    assert cosine(a, np.array([0.0, 1.0])) == 0.0
    assert cosine(a, np.array([0.0, 0.0])) == 0.0  # zero vector -> 0, no crash


TURNS = [
    (0.0, 5.0, "SPEAKER_00"),
    (5.0, 8.0, "SPEAKER_01"),
    (8.0, 12.0, "SPEAKER_00"),
]


def test_assign_speaker_max_overlap():
    assert assign_speaker(0.5, 4.0, TURNS) == "SPEAKER_00"
    assert assign_speaker(5.2, 7.8, TURNS) == "SPEAKER_01"
    # spans both: 4.5-5 (0.5s of 00) vs 5-7 (2s of 01)
    assert assign_speaker(4.5, 7.0, TURNS) == "SPEAKER_01"
    assert assign_speaker(20.0, 25.0, TURNS) is None


def test_merge_utterances():
    segs = [
        (0.0, 2.0, "me", "haan bhai"),
        (2.5, 4.0, "me", "bol kya scene hai"),   # gap 0.5 < 2 -> merges
        (7.0, 9.0, "me", "acha ok"),             # gap 3.0 > 2 -> new utterance
        (9.5, 11.0, "other", "kal milte hai"),
    ]
    merged = merge_utterances(segs, gap=2.0)
    assert merged == [
        ("me", 0.0, "haan bhai bol kya scene hai"),
        ("me", 7.0, "acha ok"),
        ("other", 9.5, "kal milte hai"),
    ]


def test_merge_utterances_empty():
    assert merge_utterances([]) == []
