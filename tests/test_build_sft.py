import json

from src.dataset.build_sft import (
    build_examples,
    is_low_quality,
    merge_consecutive,
    write_splits,
)


def test_merge_consecutive_runs():
    turns = [("other", "hi"), ("me", "hey"), ("me", "kya haal"), ("other", "badhiya")]
    assert [(s, t) for s, t, _ in merge_consecutive(turns)] == [
        ("other", "hi"),
        ("me", "hey\nkya haal"),
        ("other", "badhiya"),
    ]


def test_merge_consecutive_keeps_last_timestamp():
    # the merged turn is dated by its final message, which is what incremental
    # builds use to decide whether an example is new
    turns = [
        ("me", "hey", "2025-01-01T10:00:00"),
        ("me", "kya haal", "2025-01-01T10:05:00"),
    ]
    assert merge_consecutive(turns) == [("me", "hey\nkya haal", "2025-01-01T10:05:00")]


def test_low_quality_filters():
    assert is_low_quality("[media]", 2, 250) == "empty_or_media"
    assert is_low_quality("ok", 2, 250) == "too_short"
    assert is_low_quality("word " * 300, 2, 250) == "too_long"
    assert is_low_quality("no no no no no no no no", 2, 250) == "repetitive"
    assert is_low_quality("haan yaar kal milte hai", 2, 250) is None


def test_build_examples_shape_and_context():
    convs = {
        "c1": [
            ("other", "bhai kya kar raha"),
            ("me", "kuch nahi yaar bas timepass"),
            ("other", "movie chale?"),
            ("me", "haan chal pakka"),
        ]
    }
    examples, stats = build_examples(convs, context_turns=6, system_prompt="PERSONA")
    assert stats.examples == 2
    first = examples[0]["messages"]
    assert first[0]["role"] == "system"
    assert first[0]["content"].startswith("PERSONA")  # + person-aware preamble
    assert first[1]["role"] == "user"  # the other person speaks
    assert first[-1] == {"role": "assistant", "content": "kuch nahi yaar bas timepass"}
    # the second example carries the earlier turns as context
    assert len(examples[1]["messages"]) > len(first)


def test_first_turn_and_contextless_replies_skipped():
    convs = {"c1": [("me", "starting the chat myself"), ("me", "another one")]}
    examples, _ = build_examples(convs)
    assert examples == []


def test_duplicates_deduped():
    convs = {
        "c1": [("other", "khana khaya?"), ("me", "haan bhai kha liya")],
        "c2": [("other", "khana khaya?"), ("me", "haan bhai kha liya")],
    }
    examples, stats = build_examples(convs)
    assert len(examples) == 1
    assert stats.dropped.get("duplicate") == 1


def test_eval_split_never_swallowed_by_one_huge_conversation(tmp_path):
    """Regression: conversation sizes are wildly uneven — picking eval
    conversations blindly put the dominant chat in eval and starved training."""
    examples = [{"conversation_id": "big", "messages": [{"role": "user", "content": "x"}]}] * 900
    examples += [
        {"conversation_id": f"small{i}", "messages": [{"role": "user", "content": "y"}]}
        for i in range(100)
    ]
    n_train, n_eval = write_splits(examples, tmp_path, eval_fraction=0.05)
    assert n_train + n_eval == 1000
    assert n_eval <= 100  # eval stays a small slice
    assert n_train >= 900  # the dominant conversation stays in training


def test_write_splits_emits_valid_jsonl(tmp_path):
    examples = [
        {"conversation_id": f"c{i}", "messages": [{"role": "assistant", "content": f"m{i}"}]}
        for i in range(40)
    ]
    write_splits(examples, tmp_path, eval_fraction=0.1)
    for name in ("sft_train.jsonl", "sft_eval.jsonl"):
        for line in (tmp_path / name).read_text(encoding="utf-8").splitlines():
            assert "messages" in json.loads(line)
