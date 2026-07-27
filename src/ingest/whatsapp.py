"""Parser for WhatsApp exported chat .txt files (Android and iOS formats).

Handles: multiline messages, media markers, deleted messages, system messages,
12h/24h clocks, dd/mm vs mm/dd auto-detection, unicode direction marks.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .db import Message

# Matches both:
#   Android: "26/07/23, 9:15 pm - Rohan: text"
#   iOS:     "[26/07/23, 9:15:42 PM] Rohan: text"
_HEADER = re.compile(
    r"^[‎‏﻿]*\[?"
    r"(?P<d1>\d{1,2})[./-](?P<d2>\d{1,2})[./-](?P<yr>\d{2,4})"
    r",?\s+"
    r"(?P<h>\d{1,2}):(?P<mi>\d{2})(?::(?P<s>\d{2}))?"
    r"(?:\s*(?P<ampm>[AaPp])\.?\s?[Mm]\.?)?"
    r"\]?\s*(?:-\s+)?"
    r"(?P<rest>.*)$"
)

_MEDIA_TEXTS = {
    "<media omitted>",
    "null",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "document omitted",
    "contact card omitted",
    "voice message omitted",
}
_DELETED_TEXTS = {
    "this message was deleted",
    "you deleted this message",
    "this message was edited",  # marker-only lines
}
_STRIP_CHARS = "‎‏﻿"


# "<This message was edited>" is appended by WhatsApp, not typed by the person.
# Leaving it in taught the twin to end replies with it — 18.5% of training
# examples carried one. Stripped in _clean because it never carries meaning.
_EDITED_MARKER = re.compile(r"\s*<This message was edited>\s*", re.IGNORECASE)
# Media markers are stripped only AFTER media classification, so that a
# media-only message is still recognised as one rather than becoming empty.
_INLINE_MEDIA = re.compile(r"\s*<(?:Media omitted|attached:[^>]*)>\s*", re.IGNORECASE)


def _clean(text: str) -> str:
    text = text.translate(str.maketrans("", "", _STRIP_CHARS))
    return _EDITED_MARKER.sub(" ", text).strip()


def _is_media(text: str) -> bool:
    t = _clean(text).lower()
    return t in _MEDIA_TEXTS or t.startswith("<attached:")


def _is_deleted(text: str) -> bool:
    return _clean(text).lower() in _DELETED_TEXTS


def conversation_id_from_filename(path: Path) -> str:
    name = path.stem
    for prefix in ("WhatsApp Chat with ", "WhatsApp Chat - ", "Chat with "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return f"whatsapp:{name.strip()}"


def _detect_dayfirst(day_month_pairs: list[tuple[int, int]], fallback: bool) -> bool:
    if any(d1 > 12 for d1, _ in day_month_pairs):
        return True
    if any(d2 > 12 for _, d2 in day_month_pairs):
        return False
    return fallback


def parse_file(
    path: Path,
    me_names: list[str],
    dayfirst_fallback: bool = True,
) -> list[Message]:
    me = {n.strip().lower() for n in me_names}
    conversation_id = conversation_id_from_filename(path)
    raw = path.read_text(encoding="utf-8", errors="replace")

    # First pass: split into (header_match, name, text) entries, folding multiline
    # continuations into the previous entry.
    entries: list[dict] = []
    for line in raw.splitlines():
        m = _HEADER.match(line)
        rest = _clean(m.group("rest")) if m else ""
        if m and ": " in rest:
            name, text = rest.split(": ", 1)
            entries.append({"m": m, "name": _clean(name), "text": text})
        elif m and rest:
            continue  # system message (encryption notice, group events, ...)
        elif entries and line.strip():
            entries[-1]["text"] += "\n" + line
        # bare header with empty rest, or leading junk before first header: drop

    dayfirst = _detect_dayfirst(
        [(int(e["m"].group("d1")), int(e["m"].group("d2"))) for e in entries],
        fallback=dayfirst_fallback,
    )

    messages: list[Message] = []
    for e in entries:
        m = e["m"]
        d1, d2, yr = int(m.group("d1")), int(m.group("d2")), int(m.group("yr"))
        day, month = (d1, d2) if dayfirst else (d2, d1)
        if yr < 100:
            yr += 2000
        hour = int(m.group("h"))
        ampm = (m.group("ampm") or "").lower()
        if ampm == "p" and hour != 12:
            hour += 12
        elif ampm == "a" and hour == 12:
            hour = 0
        try:
            ts = datetime(yr, month, day, hour, int(m.group("mi")), int(m.group("s") or 0))
        except ValueError:
            continue  # impossible date from a misdetected format; skip the row

        text = _clean(e["text"])
        if not text or _is_deleted(text):
            continue
        is_media = _is_media(text)
        messages.append(
            Message(
                source="whatsapp",
                conversation_id=conversation_id,
                timestamp=ts,
                speaker="me" if e["name"].lower() in me else "other",
                speaker_name=e["name"],
                # a message can mix a caption with an attachment marker; keep the
                # caption, drop the marker
                text="[media]" if is_media else _INLINE_MEDIA.sub(" ", text).strip(),
                is_media=is_media,
            )
        )
    return messages
