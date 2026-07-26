"""Turn the unified message store into supervised fine-tuning examples.

One example = the conversation leading up to a turn Pardeep took, plus what he
actually said. Training on these teaches the model to answer as him, in his
languages, with his rhythm — the core of the "talks like me" layer.

Loss is applied only to his turns; the caller's words are context, so the
trainer must mask everything except the final assistant message.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ..config import REPO_ROOT


def load_relationship_map() -> dict[str, str]:
    """contact -> "Roshan (close friend)" labels, from the Mind Model.

    Pardeep speaks very differently to his mother, his closest friends and work
    callers. Without naming the counterpart, every register averages into one
    flat voice; with it, the twin can adapt the way he actually does.
    """
    path = REPO_ROOT / "src" / "twin" / "mind_model.json"
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        people = doc.get("relationships", {}).get("people", [])
    except (json.JSONDecodeError, AttributeError):
        return {}
    out: dict[str, str] = {}
    for p in people:
        contact = str(p.get("contact", "")).strip()
        rel = str(p.get("likely_relationship", "")).strip().rstrip(".")
        if not contact:
            continue
        # the model sometimes returns "A / B" for the same person
        for alias in (a.strip() for a in contact.split("/")):
            if alias:
                out[alias.lower()] = f"{alias} ({rel})" if rel else alias
    return out


def counterpart_label(conversation_id: str, relationships: dict[str, str]) -> str:
    """Human label for who the conversation is with."""
    raw = conversation_id.split(":", 1)[-1].strip()
    # call files are named "<contact> <date> <time>"; strip the timestamp
    name = re.sub(r"\s*\d{4}-\d{2}-\d{2}[\s\d:-]*$", "", raw).strip()
    # phone contact names carry sort-order junk: ".Jaya", "-...Pachi"
    name = name.strip(" .-_") or raw
    known = relationships.get(name.lower())
    if known:
        return known
    if re.fullmatch(r"[+\d][\d\s-]{6,}", name):
        return "an unsaved number"
    return name

# Whisper leftovers and content that teaches nothing about how he talks
_NOISE = re.compile(r"^\W*$|^\[media\]$", re.IGNORECASE)


@dataclass
class BuildStats:
    conversations: int = 0
    examples: int = 0
    named_contacts: int = 0
    incremental_from: str = ""
    replay_included: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def is_low_quality(text: str, min_words: int, max_words: int) -> str | None:
    """Return a drop-reason, or None if the text is usable as a target."""
    if _NOISE.match(text):
        return "empty_or_media"
    words = text.split()
    if len(words) < min_words:
        return "too_short"
    if len(words) > max_words:
        return "too_long"
    # survived cleaning but still mostly one repeated token
    if len(words) >= 6 and len(set(words)) <= max(2, len(words) // 4):
        return "repetitive"
    return None


_MEDIA_TOKEN = re.compile(r"\[media\]\s*", re.IGNORECASE)


def strip_media_markers(text: str) -> str:
    """Remove the placeholder the WhatsApp parser writes for photos/videos.

    Media-only messages are already rejected as targets, but merging a run of
    consecutive messages splices the placeholder into an otherwise real reply —
    and 9.4% of targets ended up containing it, which taught the twin to answer
    literally "[media]". It carries no linguistic signal, so drop it entirely.
    """
    return _MEDIA_TOKEN.sub("", text).strip()


def merge_consecutive(turns: list[tuple]) -> list[tuple]:
    """Collapse runs by the same speaker into one turn (people send several
    messages in a row; the model should learn the whole reply, not fragments).

    Accepts (speaker, text) or (speaker, text, timestamp); a merged turn keeps
    the timestamp of its last message, which is what dates the example.
    """
    merged: list[tuple] = []
    for turn in turns:
        speaker, text = turn[0], turn[1]
        ts = turn[2] if len(turn) > 2 else ""
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, merged[-1][1] + "\n" + text, ts or merged[-1][2])
        else:
            merged.append((speaker, text, ts))
    return merged


def split_new_and_replay(
    examples: list[dict], since: str, replay_ratio: float = 0.3, seed: int = 17
) -> list[dict]:
    """Incremental training set: everything after `since`, plus a replay sample
    of older examples.

    Fine-tuning only on the newest month makes the model drift toward it and
    forget how he wrote before — catastrophic forgetting. Mixing in a slice of
    the older corpus keeps the established voice stable while the new data
    updates it, and costs minutes instead of a full retrain.
    """
    new = [e for e in examples if e.get("last_ts", "") >= since]
    old = [e for e in examples if e.get("last_ts", "") < since]
    if not new:
        return []
    rng = random.Random(seed)
    n_replay = min(len(old), int(len(new) * replay_ratio / max(1e-9, 1 - replay_ratio)))
    mixed = new + rng.sample(old, n_replay)
    rng.shuffle(mixed)
    return mixed


def load_conversations(
    conn: sqlite3.Connection, exclude_sources: tuple[str, ...] = ()
) -> dict[str, list[tuple[str, str]]]:
    """Load the corpus grouped by conversation.

    `exclude_sources` keeps low-signal material out of training while leaving it
    in the store. Video audio is excluded by default: the archive's videos are
    overwhelmingly forwarded entertainment (song lyrics, Korean drama, dubbed TV
    dialogue) and sung/dubbed speech scored high enough against the voice
    fingerprint to be mislabelled as his.
    """
    sql = "SELECT conversation_id, speaker, text, timestamp FROM messages"
    params: list[str] = []
    if exclude_sources:
        sql += f" WHERE source NOT IN ({','.join('?' * len(exclude_sources))})"
        params = list(exclude_sources)
    sql += " ORDER BY conversation_id, timestamp"

    convs: dict[str, list[tuple[str, str, str]]] = {}
    for row in conn.execute(sql, params):
        convs.setdefault(row["conversation_id"], []).append(
            (row["speaker"], row["text"], row["timestamp"])
        )
    return convs


def build_examples(
    convs: dict[str, list[tuple[str, str]]],
    context_turns: int = 6,
    min_words: int = 2,
    max_words: int = 250,
    max_per_conversation: int = 400,
    system_prompt: str = "",
    relationships: dict[str, str] | None = None,
) -> tuple[list[dict], BuildStats]:
    stats = BuildStats(conversations=len(convs))
    examples: list[dict] = []
    seen: set[str] = set()
    relationships = relationships or {}

    for conv_id, raw_turns in convs.items():
        turns = merge_consecutive(raw_turns)
        who = counterpart_label(conv_id, relationships)
        from_conv = 0
        for i, turn in enumerate(turns):
            speaker, text = turn[0], turn[1]
            if speaker != "me" or i == 0:
                continue  # need at least one preceding turn to respond to
            text = strip_media_markers(text)
            reason = is_low_quality(text, min_words, max_words)
            if reason:
                stats.drop(reason)
                continue
            context = turns[max(0, i - context_turns) : i]
            if not any(t[0] == "other" for t in context):
                stats.drop("no_counterpart_context")
                continue
            # dedupe on (immediate prompt, reply)
            key = hashlib.sha1((context[-1][1] + "\x1f" + text).encode()).hexdigest()
            if key in seen:
                stats.drop("duplicate")
                continue
            seen.add(key)

            preamble = f"You are talking to {who}." if who else ""
            system = "\n\n".join(x for x in (system_prompt, preamble) if x)
            messages = [{"role": "system", "content": system}] if system else []
            for ctx in context:
                messages.append(
                    {"role": "user" if ctx[0] == "other" else "assistant", "content": ctx[1]}
                )
            messages.append({"role": "assistant", "content": text})
            examples.append(
                {
                    "conversation_id": conv_id,
                    "messages": messages,
                    # dates the example so incremental runs can select what is new
                    "last_ts": turn[2] if len(turn) > 2 else "",
                }
            )
            from_conv += 1
            if from_conv >= max_per_conversation:
                stats.drop("conversation_cap")
                break

    stats.examples = len(examples)
    return examples, stats


def write_splits(
    examples: list[dict], out_dir: Path, eval_fraction: float = 0.05, seed: int = 17
) -> tuple[int, int]:
    """Split by CONVERSATION (sharing one across train/eval leaks context), but
    budget by EXAMPLE COUNT: conversation sizes are wildly uneven here — a single
    long WhatsApp chat holds half the corpus, and picking conversations blindly
    put it in eval and starved training."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for e in examples:
        counts[e["conversation_id"]] = counts.get(e["conversation_id"], 0) + 1

    target = len(examples) * eval_fraction
    conv_ids = sorted(counts)
    rng = random.Random(seed)
    rng.shuffle(conv_ids)

    eval_convs: set[str] = set()
    used = 0
    for conv in conv_ids:
        if used >= target:
            break
        if counts[conv] > target:  # never let one giant conversation swallow eval
            continue
        eval_convs.add(conv)
        used += counts[conv]

    counts = [0, 0]
    for name, keep in (("sft_train.jsonl", False), ("sft_eval.jsonl", True)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for e in examples:
                if (e["conversation_id"] in eval_convs) is keep:
                    f.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")
                    counts[1 if keep else 0] += 1
    return counts[0], counts[1]


def build(
    conn: sqlite3.Connection, cfg: dict, since: str = "", replay_ratio: float = 0.3
) -> tuple[BuildStats, int, int]:
    dcfg = cfg.get("dataset", {})
    # The persona card belongs at inference time, steering models that were never
    # fine-tuned (Claude/GPT). Embedding it in every training example made each
    # sample ~1800 tokens of mostly repeated boilerplate around ~300 tokens of
    # actual conversation — 6x the training cost to teach the model to recite a
    # card it is already learning to embody from the data itself.
    persona_path = REPO_ROOT / "src" / "twin" / "persona.md"
    system_prompt = (
        persona_path.read_text(encoding="utf-8")
        if (dcfg.get("include_persona") and persona_path.exists())
        else ""
    )

    convs = load_conversations(conn, tuple(dcfg.get("exclude_sources", ["video"])))
    relationships = load_relationship_map() if dcfg.get("person_aware", True) else {}
    examples, stats = build_examples(
        convs,
        context_turns=dcfg.get("context_turns", 6),
        min_words=dcfg.get("min_words", 2),
        max_words=dcfg.get("max_words", 250),
        max_per_conversation=dcfg.get("max_per_conversation", 400),
        system_prompt=system_prompt,
        relationships=relationships,
    )
    stats.named_contacts = len(relationships)
    if since:
        before = len(examples)
        examples = split_new_and_replay(examples, since, replay_ratio)
        stats.incremental_from = since
        stats.replay_included = len(examples) - sum(
            1 for e in examples if e.get("last_ts", "") >= since
        )
        stats.examples = len(examples)
        if not examples:
            raise SystemExit(f"no examples newer than {since} (corpus has {before})")

    n_train, n_eval = write_splits(
        examples, REPO_ROOT / "data" / "datasets", dcfg.get("eval_fraction", 0.05)
    )
    return stats, n_train, n_eval
