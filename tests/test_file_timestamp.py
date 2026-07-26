from datetime import datetime

from src.ingest.transcribe import timestamp_from_name


def test_call_recorder_format():
    # the dominant pattern across the call archive
    assert timestamp_from_name("Roshan 2025-10-14 13-38-46.m4a") == datetime(2025, 10, 14, 13, 38, 46)
    assert timestamp_from_name("Mummy or ਡਿੰਪੀ 2025-02-23 14-49-31.m4a") == datetime(
        2025, 2, 23, 14, 49, 31
    )
    assert timestamp_from_name("+917007567191 2025-02-21 17-53-58.m4a") == datetime(
        2025, 2, 21, 17, 53, 58
    )


def test_older_recorder_underscore_format():
    assert timestamp_from_name("919876075331_2020_09_20_15_34_02_out.mp3") == datetime(
        2020, 9, 20, 15, 34, 2
    )


def test_whatsapp_media_format():
    assert timestamp_from_name("PTT-20230726-WA0001.opus") == datetime(2023, 7, 26)
    assert timestamp_from_name("VID-20251201-WA0027.mp4") == datetime(2025, 12, 1)


def test_phone_recorder_prefix_format():
    assert timestamp_from_name("REC20191026134952.mp3") == datetime(2019, 10, 26, 13, 49, 52)
    assert timestamp_from_name("VID20220816183243.mp4") == datetime(2022, 8, 16, 18, 32, 43)


def test_amr_two_digit_year_format():
    # "Khusraj3-2012161436.amr" -> 2020-12-16 14:36 (yy mm dd hh mm)
    assert timestamp_from_name("Khusraj3-2012161436.amr") == datetime(2020, 12, 16, 14, 36)
    assert timestamp_from_name("Taran-2006101941.amr") == datetime(2020, 6, 10, 19, 41)
    assert timestamp_from_name("+918146518888-2006191749.amr") == datetime(2020, 6, 19, 17, 49)


def test_no_date_returns_none():
    assert timestamp_from_name("Voice 001.m4a") is None
    assert timestamp_from_name("random-clip.mp3") is None


def test_impossible_dates_rejected():
    # digit runs that parse but cannot be real recordings
    assert timestamp_from_name("track-9999999999.mp3") is None
    assert timestamp_from_name("x 2025-13-45 99-99-99.m4a") is None


def test_year_sanity_window():
    # a 1998 stamp is a coincidental digit run, not a phone recording
    assert timestamp_from_name("REC19980101120000.mp3") is None
