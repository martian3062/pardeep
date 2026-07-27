from src.ingest import whatsapp as wa

ME = ["Pardeep"]


def test_edited_annotation_stripped(tmp_path):
    """Regression: WhatsApp appends '<This message was edited>' to the text, and
    18.5% of training examples carried one — the twin learned to emit it."""
    f = tmp_path / "WhatsApp Chat with Roshan.txt"
    f.write_text(
        "2/16/24, 9:52 AM - Roshan: kal aa raha hai?\n"
        "2/16/24, 9:53 AM - Pardeep: haan bhai aa raha hun <This message was edited>\n",
        encoding="utf-8",
    )
    msgs = wa.parse_file(f, ME)
    assert msgs[-1].text == "haan bhai aa raha hun"
    assert all("edited" not in m.text for m in msgs)


def test_attached_annotation_stripped(tmp_path):
    f = tmp_path / "WhatsApp Chat with A.txt"
    f.write_text(
        "2/16/24, 9:52 AM - A: hi\n"
        "2/16/24, 9:53 AM - Pardeep: dekh ye <attached: 00000042-PHOTO.jpg> kaisa hai\n",
        encoding="utf-8",
    )
    msgs = wa.parse_file(f, ME)
    assert msgs[-1].text == "dekh ye kaisa hai"


def test_real_angle_brackets_survive(tmp_path):
    # only WhatsApp's own annotations are removed, not the user's typing
    f = tmp_path / "WhatsApp Chat with B.txt"
    f.write_text(
        "2/16/24, 9:52 AM - B: hi\n2/16/24, 9:53 AM - Pardeep: 5 < 10 hai na\n",
        encoding="utf-8",
    )
    msgs = wa.parse_file(f, ME)
    assert msgs[-1].text == "5 < 10 hai na"


def test_media_only_message_still_flagged(tmp_path):
    f = tmp_path / "WhatsApp Chat with C.txt"
    f.write_text(
        "2/16/24, 9:52 AM - C: <Media omitted>\n2/16/24, 9:53 AM - Pardeep: acha\n",
        encoding="utf-8",
    )
    msgs = wa.parse_file(f, ME)
    assert msgs[0].is_media and msgs[0].text == "[media]"
