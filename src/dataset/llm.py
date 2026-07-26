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
    convs = list(by_conv)
    rng.shuffle(convs)
    # Round-robin instead of a fixed per-conversation quota: hundreds of call
    # fragments hold one message each, so a flat quota starves the sample long
    # before reaching n. A per-conversation cap still stops one long chat from
    # defining the persona on its own.
    cap = max(5, int(n * 0.15))
    pools = {c: rng.sample(by_conv[c], min(cap, len(by_conv[c]))) for c in convs}
    out: list[dict] = []
    while len(out) < n and any(pools.values()):
        for conv in convs:
            if pools[conv]:
                out.append(pools[conv].pop())
                if len(out) >= n:
                    break
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


def format_messages(
    msgs: list[dict], with_meta: bool = True, max_tokens: int = 120_000
) -> str:
    """Render samples for a prompt, stopping before the context limit.

    Gurmukhi and Devanagari cost far more tokens per character than Latin text,
    so budget on an estimate rather than a message count: 800 mixed-script
    messages overflowed a 200k window.
    """
    lines: list[str] = []
    budget = max_tokens
    for m in msgs:
        line = f"[{m['timestamp'][:10]} · {m['source']}] {m['text']}" if with_meta else m["text"]
        cost = _estimate_tokens(line)
        if cost > budget:
            break
        budget -= cost
        lines.append(line)
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Rough upper bound: ~1 token per char for Indic scripts, ~1 per 3.5 for Latin."""
    indic = sum(1 for ch in text if "ऀ" <= ch <= "ൿ")
    return int(indic + (len(text) - indic) / 3.5) + 2


def format_exchanges(exchanges: list[tuple[str, list[dict]]]) -> str:
    blocks = []
    for conv, rows in exchanges:
        who = conv.split(":", 1)[-1]
        turns = "\n".join(
            f"  {'PARDEEP' if r['speaker'] == 'me' else 'THEM'}: {r['text']}" for r in rows
        )
        blocks.append(f"--- conversation with {who} ---\n{turns}")
    return "\n\n".join(blocks)
