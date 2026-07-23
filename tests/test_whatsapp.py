from datetime import datetime
from pathlib import Path

from src.ingest import db as dbm
from src.ingest import whatsapp as wa

FIXTURE = Path(__file__).parent / "fixtures" / "WhatsApp Chat with Rohan.txt"
ME = ["Pardeep"]


def test_parse_android_export():
    msgs = wa.parse_file(FIXTURE, ME)
    # system message skipped, deleted message skipped -> 6 remain
    assert len(msgs) == 6
    assert all(m.conversation_id == "whatsapp:Rohan" for m in msgs)

    assert msgs[0].speaker == "other" and msgs[0].speaker_name == "Rohan"
    assert msgs[0].timestamp == datetime(2023, 7, 26, 21, 15)  # 9:15 pm -> 21:15

    # multiline folded into one message
    assert msgs[1].speaker == "me"
    assert msgs[1].text == "kuch nahi yaar\nbas timepass"

    # media marker normalized + flagged
    media = [m for m in msgs if m.is_media]
    assert len(media) == 1 and media[0].text == "[media]"

    # dd/mm auto-detected from 13/08 -> August, not month 13
    assert msgs[-1].timestamp == datetime(2023, 8, 13, 23, 30)

    assert sum(1 for m in msgs if m.speaker == "me") == 3


def test_parse_ios_export(tmp_path):
    f = tmp_path / "WhatsApp Chat - Amit.txt"
    f.write_text(
        "[26/07/23, 9:15:42 PM] Amit: hello\n"
        "[26/07/23, 9:16:01 PM] Pardeep: hi bro\n"
        "[26/07/23, 9:16:30 PM] Amit: ‎image omitted\n",
        encoding="utf-8",
    )
    msgs = wa.parse_file(f, ME)
    assert len(msgs) == 3
    assert msgs[0].timestamp == datetime(2023, 7, 26, 21, 15, 42)
    assert msgs[1].speaker == "me"
    assert msgs[2].is_media
    assert msgs[0].conversation_id == "whatsapp:Amit"


def test_am_pm_edges(tmp_path):
    f = tmp_path / "chat.txt"
    f.write_text(
        "26/07/23, 12:05 am - Pardeep: midnight msg\n"
        "26/07/23, 12:30 pm - Pardeep: noon msg\n",
        encoding="utf-8",
    )
    msgs = wa.parse_file(f, ME)
    assert msgs[0].timestamp.hour == 0
    assert msgs[1].timestamp.hour == 12


def test_db_idempotent(tmp_path):
    conn = dbm.connect(tmp_path / "test.db")
    msgs = wa.parse_file(FIXTURE, ME)

    assert dbm.insert_messages(conn, msgs) == 6
    assert dbm.insert_messages(conn, msgs) == 0  # re-insert is a no-op

    sha = dbm.file_sha256(FIXTURE)
    assert not dbm.already_ingested(conn, FIXTURE, sha)
    dbm.record_file(conn, FIXTURE, sha, 6)
    assert dbm.already_ingested(conn, FIXTURE, sha)

    s = dbm.stats(conn)
    assert s["messages"] == 6
    assert s["by_speaker"] == {"me": 3, "other": 3}
    assert s["conversations"] == 1
