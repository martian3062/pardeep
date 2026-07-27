"""Read time out of a question.

"2017 june me kya kar raha tha" is mostly a *when*, but embeddings only encode
*what*: asked plainly, it retrieved eight recent chats and nothing from 2017,
and the twin truthfully answered that it did not remember a year it has 2,100
photos from. Dates in a query are therefore parsed and applied as a filter
rather than left to similarity.
"""
from __future__ import annotations

import re
from datetime import datetime

# The archive runs from about 2010; anything outside is a number, not a year.
_MIN_YEAR, _MAX_YEAR = 2008, datetime.now().year + 1

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_MONTH = re.compile(r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b", re.I)

# Hinglish/Punjabi time words that carry a rough era rather than a date
_RELATIVE = {
    "aaj": 0, "today": 0,
    "kal": 1, "yesterday": 1,
    "parso": 2,
    "last week": 7, "pichle hafte": 7, "is hafte": 7, "this week": 7,
    "last month": 31, "pichle mahine": 31, "is mahine": 31, "this month": 31,
    "last year": 365, "pichle saal": 365,
    "bachpan": 4000, "childhood": 4000,  # "when I was a kid" -> the oldest end
}


def _month_end(year: int, month: int) -> datetime:
    return datetime(year + (month // 12), (month % 12) + 1, 1)


def parse_time_range(query: str, now: datetime | None = None) -> tuple[datetime, datetime] | None:
    """Return the window a question is asking about, or None if it names no time."""
    now = now or datetime.now()
    text = query.lower()

    years = [int(y) for y in _YEAR.findall(text) if _MIN_YEAR <= int(y) <= _MAX_YEAR]
    months = [_MONTHS[m.lower()] for m in _MONTH.findall(text)]

    if years:
        lo, hi = min(years), max(years)
        if len(years) == 1 and months:
            # "june 2017" — a single month, widened slightly so a photo dated by
            # its folder's median still lands inside
            month = months[0]
            start = datetime(lo, month, 1)
            return start, _month_end(lo, month)
        return datetime(lo, 1, 1), datetime(hi + 1, 1, 1)

    for phrase, days in _RELATIVE.items():
        if phrase in text:
            if days >= 4000:  # childhood: everything before the recent years
                return datetime(_MIN_YEAR, 1, 1), datetime(now.year - 6, 1, 1)
            span = max(days, 1)
            return (
                datetime.fromtimestamp(now.timestamp() - span * 86400),
                datetime.fromtimestamp(now.timestamp() + 86400),
            )
    return None
