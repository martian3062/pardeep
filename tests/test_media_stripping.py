import json

from src.dataset.build_sft import build_examples, strip_media_markers, write_splits


def test_strips_media_placeholder():
    assert strip_media_markers("[media] haan bhai kal milte hai") == "haan bhai kal milte hai"
    assert strip_media_markers("dekh ye [media] kaisa hai") == "dekh ye kaisa hai"
    assert strip_media_markers("[media]") == ""


def test_normal_text_untouched():
    s = "kal milte hai yaar, movie chalein?"
    assert strip_media_markers(s) == s


def test_merged_reply_loses_the_placeholder(tmp_path):
    """Regression: consecutive-message merging spliced '[media]' into real
    replies, so 9.4% of targets taught the twin to answer literally '[media]'."""
    convs = {
        "whatsapp:Roshan": [
            ("other", "oye kal ka plan?"),
            ("me", "[media]"),
            ("me", "ye dekh, kal chalte hai"),
        ]
    }
    examples, _ = build_examples(convs)
    assert examples, "a real reply should survive"
    target = examples[-1]["messages"][-1]["content"]
    assert "[media]" not in target
    assert "kal chalte hai" in target


def test_media_only_reply_is_dropped():
    convs = {"whatsapp:X": [("other", "photo bhej"), ("me", "[media]")]}
    examples, stats = build_examples(convs)
    assert examples == []
    assert stats.dropped.get("empty_or_media") == 1


def test_no_media_tokens_anywhere_in_written_dataset(tmp_path):
    convs = {
        "whatsapp:A": [
            ("other", "hi"),
            ("me", "[media] theek hai bhai chalte hai kal subah"),
        ]
    }
    examples, _ = build_examples(convs)
    write_splits(examples, tmp_path, eval_fraction=0.0)
    text = (tmp_path / "sft_train.jsonl").read_text(encoding="utf-8")
    for line in text.splitlines():
        row = json.loads(line)
        assert all("[media]" not in m["content"] for m in row["messages"])
