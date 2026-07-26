"""Extract the Mind Model: HOW Pardeep thinks, decides and relates to people.

The persona card governs style; this governs behaviour. It is injected into the
reasoning path so the twin makes his kinds of choices, not merely his phrasing.
Versioned JSON so it can grow with each diary entry (Phase 7).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from ..config import REPO_ROOT
from .llm import ask, format_exchanges, format_messages, sample_exchanges, sample_my_messages

console = Console()

RELATIONSHIPS = """Below are slices of real conversations. PARDEEP is the person we are modelling;
THEM is whoever he was talking to (the header names the contact or phone number).

For EACH conversation, infer:
- who this person likely is to him (mother, close friend, classmate, recruiter, delivery agent...)
  and the evidence for that guess
- how he ADAPTS with them: language choice, formality, warmth, humour, how much he initiates,
  whether he defers or leads
- what they typically talk about

Then summarise the pattern across all of them: what stays constant in him regardless of audience,
and what shifts. Be evidence-driven and say when a guess is uncertain.

Return STRICT JSON:
{{"people": [{{"contact": "...", "likely_relationship": "...", "confidence": "high|medium|low",
"how_he_talks_to_them": "...", "typical_topics": ["..."], "evidence": "..."}}],
"constant_across_everyone": ["..."], "what_shifts_by_audience": ["..."]}}

=== CONVERSATIONS ===
{exchanges}
"""

BEHAVIOUR = """Below are real messages from ONE person (Pardeep) across several years of chats and
calls. Infer the person BEHIND the words — this is used to make an AI twin decide the way he
would, not just phrase things the way he does.

Return STRICT JSON with these keys:
{{
 "values": ["what he actually seems to care about, with evidence"],
 "motivations": ["what drives him"],
 "decision_patterns": ["how he decides: fast/slow, consults others, risk appetite, how he weighs money/time/relationships"],
 "emotional_patterns": ["what stresses him, what lifts him, how he expresses frustration or affection"],
 "communication_habits": ["how he handles conflict, asks for help, gives help, avoids things"],
 "recurring_concerns": ["what keeps coming up"],
 "interests": ["topics/activities he engages with"],
 "self_presentation": "how he wants to be seen, 2-3 sentences",
 "blind_spots_or_tensions": ["contradictions visible in the data — e.g. says X but does Y"]
}}

Ground every item in what the messages actually show. Prefer specific over flattering. If the
evidence is thin for a category, return fewer items rather than inventing them.

=== MESSAGES ===
{samples}
"""

TIMELINE = """This is a listing of folder and file names from a person's phone and laptop backups
spanning roughly 7 years, plus dated samples of their messages. Folder names are messy and
personal — treat them as traces of what they were doing at the time.

Infer their TRAJECTORY: study/career phases, ambitions pursued and abandoned, skills built,
interests that came and went. Anchor to years where possible.

Return STRICT JSON:
{{"phases": [{{"period": "e.g. 2019-2020", "what_they_were_doing": "...", "evidence": ["..."]}}],
"ambitions_pursued": ["..."], "abandoned_or_faded": ["..."], "consistent_threads": ["..."]}}

=== FOLDER / FILE TRACES ===
{traces}

=== DATED MESSAGE SAMPLES ===
{samples}
"""


def _folder_traces(limit: int = 400) -> str:
    """Directory names from the archive: a cheap, dense record of what he was
    working on year by year (NEET -> B.E. CSE -> hackathons -> AI projects)."""
    dump = REPO_ROOT / "data" / "raw" / "_dump"
    if not dump.exists():
        return "(no archive available)"
    names: list[str] = []
    for p in dump.rglob("*"):
        if p.is_dir():
            names.append(str(p.relative_to(dump)))
            if len(names) >= limit:
                break
    return "\n".join(sorted(names))


def extract_mind_model(conn: sqlite3.Connection, model: str = "claude-opus-4-5") -> Path:
    out = REPO_ROOT / "src" / "twin" / "mind_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    msgs = sample_my_messages(conn, n=500)
    exchanges = sample_exchanges(conn, n_conversations=25)
    if not msgs:
        raise SystemExit("No messages by you in the corpus yet")

    console.print(f"Pass 1/3 — relationships across {len(exchanges)} conversations ...")
    people = ask(RELATIONSHIPS.format(exchanges=format_exchanges(exchanges)), model=model)

    console.print(f"Pass 2/3 — behaviour from {len(msgs)} messages ...")
    behaviour = ask(BEHAVIOUR.format(samples=format_messages(msgs)), model=model)

    console.print("Pass 3/3 — life trajectory from archive traces ...")
    timeline = ask(
        TIMELINE.format(traces=_folder_traces(), samples=format_messages(msgs[:150])), model=model
    )

    def parse(text: str) -> dict | str:
        t = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            return {"_unparsed": t}

    model_doc = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_counts": {"messages_sampled": len(msgs), "conversations_sampled": len(exchanges)},
        "relationships": parse(people),
        "behaviour": parse(behaviour),
        "trajectory": parse(timeline),
    }
    out.write_text(json.dumps(model_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
