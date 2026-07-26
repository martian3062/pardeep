"""Shared Anthropic client + corpus sampling for identity extraction.

Privacy: only sampled snippets are sent to the API, never the raw archive, and
never audio or photos. See the hybrid privacy model in README.
"""
from __future__ import annotations

import os
import random
import sqlite3

from dotenv import load_dotenv

from ..config import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")


def client():
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY missing from .env")
    return anthropic.Anthropic(api_key=key)


def ask(prompt: str, model: str, max_tokens: int = 8000, system: str | None = None) -> str:
    resp = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system or "You analyse a person's own messages to model who they are.",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def sample_my_messages(
    conn: sqlite3.Connection, n: int = 400, min_words: int = 3, seed: int = 7
) -> list[dict]:
    """Spread the sample across sources and conversations so one chatty contact
    doesn't define the persona."""
    rows = [
        dict(r)
        for r in conn.execute(
            """SELECT source, conversation_id, timestamp, text FROM messages
               WHERE speaker = 'me' AND text != '[media]'
               ORDER BY timestamp"""
        )
        if len(r["text"].split()) >= min_words
    ]
    by_conv: dict[str, list[dict]] = {}
    for r in rows:
        by_conv.setdefault(r["conversation_id"], []).append(r)

    rng = random.Random(seed)
    out: list[dict] = []
    convs = list(by_conv)
    rng.shuffle(convs)
    per_conv = max(1, n // max(1, len(convs)))
    for conv in convs:
        msgs = by_conv[conv]
        out.extend(rng.sample(msgs, min(per_conv, len(msgs))))
    rng.shuffle(out)
    return out[:n]


def sample_exchanges(
    conn: sqlite3.Connection, n_conversations: int = 25, turns: int = 14, seed: int = 7
) -> list[tuple[str, list[dict]]]:
    """Whole conversation slices — needed to see how he ADAPTS per person."""
    convs = [
        r[0]
        for r in conn.execute(
            """SELECT conversation_id FROM messages GROUP BY conversation_id
               HAVING SUM(speaker = 'me') >= 4 ORDER BY COUNT(*) DESC"""
        )
    ]
    rng = random.Random(seed)
    picked = convs[: n_conversations * 2]
    rng.shuffle(picked)
    out = []
    for conv in picked[:n_conversations]:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT speaker, text FROM messages WHERE conversation_id = ? ORDER BY timestamp LIMIT ?",
                (conv, turns),
            )
        ]
        if rows:
            out.append((conv, rows))
    return out


def format_messages(msgs: list[dict], with_meta: bool = True) -> str:
    lines = []
    for m in msgs:
        if with_meta:
            lines.append(f"[{m['timestamp'][:10]} · {m['source']}] {m['text']}")
        else:
            lines.append(m["text"])
    return "\n".join(lines)


def format_exchanges(exchanges: list[tuple[str, list[dict]]]) -> str:
    blocks = []
    for conv, rows in exchanges:
        who = conv.split(":", 1)[-1]
        turns = "\n".join(
            f"  {'PARDEEP' if r['speaker'] == 'me' else 'THEM'}: {r['text']}" for r in rows
        )
        blocks.append(f"--- conversation with {who} ---\n{turns}")
    return "\n\n".join(blocks)
