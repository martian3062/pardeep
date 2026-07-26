from src.dataset.build_sft import build_examples, counterpart_label

RELATIONSHIPS = {
    "roshan": "Roshan (very close friend)",
    "mummy airtel": "Mummy Airtel (mother)",
}


def test_known_contact_gets_relationship():
    assert counterpart_label("call:Roshan 2025-10-14 13-38-46", RELATIONSHIPS) == (
        "Roshan (very close friend)"
    )


def test_timestamp_stripped_from_call_ids():
    assert counterpart_label("call:Karan Cu 2025-02-04 16-35-47", {}) == "Karan Cu"


def test_contact_sort_prefixes_cleaned():
    # phone address books carry sort-order junk in the name
    assert counterpart_label("whatsapp:.Jaya", {}) == "Jaya"
    assert counterpart_label("call:-...Pachi 2025-03-16 14-43-01", {}) == "Pachi"


def test_unsaved_numbers_are_not_leaked_as_names():
    assert counterpart_label("call:+917007567191 2025-02-21 17-53-58", {}) == "an unsaved number"


def test_examples_carry_the_counterpart_in_the_system_prompt():
    convs = {
        "call:Mummy Airtel 2025-11-19 13-16-27": [
            ("other", "khana khaya beta?"),
            ("me", "haan mummy kha liya"),
        ]
    }
    examples, stats = build_examples(convs, system_prompt="PERSONA", relationships=RELATIONSHIPS)
    system = examples[0]["messages"][0]
    assert system["role"] == "system"
    assert "PERSONA" in system["content"]
    assert "You are talking to Mummy Airtel (mother)." in system["content"]


def test_person_aware_can_be_disabled():
    convs = {"call:Roshan 2025-10-14 13-38-46": [("other", "oye"), ("me", "haan bhai bol")]}
    examples, _ = build_examples(convs, system_prompt="", relationships={})
    # without a relationship map the raw contact name is still supplied
    assert "You are talking to Roshan." in examples[0]["messages"][0]["content"]
