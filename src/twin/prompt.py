"""Assemble the system prompt: who he is, how he decides, what he remembers.

Order matters. Identity first so it frames everything that follows, memories
last so the model has them fresh when it starts writing. Memories are labelled
as recall rather than instruction — otherwise a model reads an old chat log as a
script to continue instead of something it knows.
"""
from __future__ import annotations

from .identity import Identity, render_behaviour, render_counterpart
from .retrieve import Snippet
from .script import language_rule

GUEST_PREAMBLE = """You are speaking to someone who is NOT Pardeep. Never reveal
private details about him, his family, his finances, his health, or anything told
to him in confidence. Speak in his style, but about nothing personal."""

OWNER_PREAMBLE = """You are Pardeep's digital twin, speaking to Pardeep himself.
He knows his own life, so speak plainly and never explain who people are unless
he asks."""

RULES = """How to reply:
- Write exactly as Pardeep writes: his rhythm, his code-switching, his phrases.
- Match the language he used. If he writes Hinglish in Roman script, reply the same way.
- Do not write like an assistant. No bullet lists, no "I'd be happy to", no summaries
  of what you just said, no offers to help further.
- Keep it as short as he would keep it. He sends fragments, not paragraphs.
- If the memories below do not cover something, say you do not remember it. Never
  invent an event, a date, or a person."""


def build_system_prompt(
    identity: Identity,
    snippets: list[Snippet],
    counterpart: str = "",
    guest: bool = False,
    message: str = "",
    mood_rule: str = "",
) -> str:
    parts = [GUEST_PREAMBLE if guest else OWNER_PREAMBLE]

    if identity.persona:
        parts.append(identity.persona.strip())

    behaviour = render_behaviour(identity)
    if behaviour:
        parts.append(f"## How you think\n\n{behaviour}")

    if counterpart:
        who = render_counterpart(identity, counterpart)
        if who:
            parts.append(f"## Who you are talking to\n\n{who}")

    rules = RULES
    # "Match the language he used" lost against a persona card full of Devanagari
    # examples: a plain English question came back in Devanagari. Naming the
    # script for this specific turn is what actually holds.
    if message:
        rule = language_rule(message)
        if rule:
            rules += f"\n- {rule}"
    # a camera reading is weak evidence, so it shifts tone and is never asserted
    # back to him as fact
    if mood_rule and not guest:
        rules += f"\n- {mood_rule}"
    parts.append(rules)

    if snippets:
        lines = "\n".join(s.render() for s in snippets)
        parts.append(
            "## What you remember\n\n"
            "These are real memories from your own archive — chats, calls, and photos. "
            "Use them as things you know, not as a conversation to continue.\n\n" + lines
        )
    else:
        parts.append(
            "## What you remember\n\n"
            "Nothing in your memory matches this. Say so rather than guessing."
        )

    return "\n\n---\n\n".join(parts)
