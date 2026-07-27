"""Gather the memories that should go into a reply.

Straight top-k is the wrong shape for a context window. The demo query
"celebration with lots of people" returned two frames of the same warehouse
moment as its top two hits — one slot of information for two slots of budget.
So candidates are over-fetched and then chosen for relevance *and* difference
from what has already been picked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..memory.store import MemoryStore
from .when import parse_time_range

# A conversation chunk can run to thousands of characters. Whole scenes are
# rarely needed to answer a question, and every character spent on one memory is
# one not spent on another.
SNIPPET_CHARS = 700


# "koi photo hai?" is a request for pictures, but a photo caption is one short
# sentence while a chat memory is a long thread that may say "college" ten times,
# so on wording alone the conversation wins and the picture never surfaces.
_WANTS_PHOTO = re.compile(
    r"\b(photo|photos|pic|pics|picture|pictures|image|images|selfie|album)\b"
    r"|फोटो|तस्वीर|ਫੋਟੋ|ਤਸਵੀਰ",
    re.IGNORECASE,
)

# Enough to lift a genuinely relevant photo past a wordier chat, not enough to
# surface an irrelevant one.
PHOTO_INTENT_BOOST = 1.6

# Enough to put every real photograph ahead of a forward on a photo question,
# without hiding forwards from someone who asks about a meme directly.
FORWARD_PENALTY = 0.35


# "meri koi purani photo describe kar" asks for HIS photograph. A forward is by
# definition not his photograph, so it should be excluded rather than merely
# demoted — penalising was not enough, because a meme caption quoting a punchline
# matches a vague request far better than "a green field of tall grass" does.
_OWN_PHOTO = re.compile(
    r"\b(?:meri|mere|mera|apni|apne|apna|khud|my|mine|our)\b", re.IGNORECASE
)


def wants_photos(query: str) -> bool:
    return bool(_WANTS_PHOTO.search(query))


def wants_own_photos(query: str) -> bool:
    """Asking for his own pictures, not for something he once forwarded."""
    return wants_photos(query) and bool(_OWN_PHOTO.search(query))


@dataclass
class Snippet:
    text: str
    kind: str
    timestamp: str
    counterpart: str
    source_ref: str
    score: float

    def render(self) -> str:
        when = self.timestamp[:10]
        head = f"[{self.kind} {when}"
        if self.counterpart:
            head += f" · {self.counterpart}"
        head += "]"
        body = " ".join(self.text.split())
        if len(body) > SNIPPET_CHARS:
            body = body[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
        return f"{head} {body}"


def _cosine(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def _diversify(rows: list[dict], limit: int, redundancy_penalty: float) -> list[dict]:
    """Greedy MMR: take the best, then repeatedly take whatever is most relevant
    while least like everything already taken."""
    if not rows:
        return []
    picked = [rows[0]]
    pool = rows[1:]
    while pool and len(picked) < limit:
        best, best_value = None, None
        for cand in pool:
            cv = cand.get("vector")
            worst_overlap = 0.0
            if cv is not None:
                worst_overlap = max(
                    (_cosine(cv, p["vector"]) for p in picked if p.get("vector") is not None),
                    default=0.0,
                )
            value = cand["score"] - redundancy_penalty * worst_overlap
            if best_value is None or value > best_value:
                best, best_value = cand, value
        picked.append(best)
        pool.remove(best)
    return picked


def gather(
    query: str,
    store: MemoryStore | None = None,
    limit: int = 8,
    guest: bool = False,
    redundancy_penalty: float = 0.45,
) -> list[Snippet]:
    """Retrieve memories for a query, favouring a varied set over a repetitive one."""
    store = store or MemoryStore()
    window = parse_time_range(query)
    trust = "public" if guest else "secret"

    rows = store.search(
        query,
        limit=limit * 4,  # over-fetch so diversification has something to choose from
        max_trust=trust,
        time_range=(window[0].timestamp(), window[1].timestamp()) if window else None,
    )
    # a period he simply has nothing from should fall back to the whole archive
    # rather than answering "I don't remember" when a related memory exists
    if not rows and window:
        rows = store.search(query, limit=limit * 4, max_trust=trust)
    if not rows:
        return []

    if wants_photos(query):
        # A general search returns whatever matches wording, and for a photo
        # question that is mostly forwards and chat threads: filtering them out
        # afterwards left one query with no photographs at all. So photographs
        # are fetched in their own right rather than hoped for.
        own = wants_own_photos(query)
        kinds = ("photo",) if own else ("photo", "forward")
        extra = store.search(
            query,
            limit=limit * 2,
            kinds=kinds,
            max_trust=trust,
            time_range=(window[0].timestamp(), window[1].timestamp()) if window else None,
        )
        seen = {(r.get("timestamp"), r.get("text", "")[:60]) for r in rows}
        rows += [r for r in extra if (r.get("timestamp"), r.get("text", "")[:60]) not in seen]
        if own:
            real = [r for r in rows if r.get("kind") != "forward"]
            if real:  # never empty the answer just to exclude forwards
                rows = real

    if wants_photos(query):
        for r in rows:
            if r.get("kind") == "photo":
                r["score"] *= PHOTO_INTENT_BOOST
            elif r.get("kind") == "forward":
                # Asked for "a photo of mine", a forwarded joke is the wrong
                # answer even though its text-dense caption matches well. It
                # stays reachable — he did share it — but never ahead of a real
                # photograph.
                r["score"] *= FORWARD_PENALTY
        rows.sort(key=lambda r: r["score"], reverse=True)
    chosen = _diversify(rows, limit, redundancy_penalty)
    return [
        Snippet(
            text=r["text"],
            kind=r["kind"],
            timestamp=r.get("timestamp", ""),
            counterpart=r.get("counterpart", ""),
            source_ref=r.get("conversation_id", ""),
            score=r["score"],
        )
        for r in chosen
    ]
