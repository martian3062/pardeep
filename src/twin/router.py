"""Choose what generates the reply.

The hybrid privacy rule: raw archives never leave this machine. What a cloud
model sees is the persona card plus the handful of snippets retrieval already
chose — never the database, never a whole conversation, never a photo.

Backends in order of preference:
  local   — a fine-tuned twin served by Ollama. His actual voice, fully offline.
  claude  — strong reasoning, snippets only. Works with no GPU at all.
  openai  — same role as claude, used when it is the key that exists.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import model_id

# Resolved from config.yaml `models:` so a retired provider id is a one-line edit
# rather than a code change. Read once at import — these do not change per turn.
CLAUDE_MODEL = model_id("claude")
OPENAI_MODEL = model_id("openai")
OLLAMA_URL = model_id("ollama_url")

# Ollama on this machine holds general-purpose models (granite, gemma, a coder
# model) that know nothing about him. Serving one as "the twin" would produce
# fluent replies in the wrong voice, which is worse than no local backend at all,
# so a local model counts only when its name marks it as the fine-tune.
TWIN_MARKERS = ("twin", "pardeep", "self", "sarvam-m-ft", "me-too")


@dataclass
class Reply:
    text: str
    backend: str
    model: str


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def available_backends() -> list[str]:
    _load_env()
    found = []
    if twin_model():
        found.append("local")
    if os.getenv("ANTHROPIC_API_KEY"):
        found.append("claude")
    if os.getenv("OPENAI_API_KEY"):
        found.append("openai")
    return found


def _ollama_models() -> list[str]:
    try:
        import requests

        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def twin_model() -> str | None:
    """The locally served fine-tune, if one has been installed."""
    for name in _ollama_models():
        if any(marker in name.lower() for marker in TWIN_MARKERS):
            return name
    return None


def _claude(system: str, messages: list[dict], max_tokens: int, temperature: float) -> Reply:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    # temperature is rejected by the Claude 5 models; style comes from the
    # persona card and the retrieved memories, not from sampling noise
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return Reply(text=text.strip(), backend="claude", model=CLAUDE_MODEL)


def _openai(system: str, messages: list[dict], max_tokens: int, temperature: float) -> Reply:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, *messages],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return Reply(text=resp.choices[0].message.content.strip(), backend="openai", model=OPENAI_MODEL)


def _local(system: str, messages: list[dict], max_tokens: int, temperature: float) -> Reply:
    import requests

    model = twin_model()
    if model is None:
        raise RuntimeError(
            "no local twin model in Ollama — the models present are general-purpose "
            "and would answer in the wrong voice"
        )
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=180,
    )
    r.raise_for_status()
    return Reply(text=r.json()["message"]["content"].strip(), backend="local", model=model)


def generate(
    system: str,
    messages: list[dict],
    backend: str = "auto",
    max_tokens: int = 400,
    temperature: float = 0.85,  # he is not a deterministic person
) -> Reply:
    _load_env()
    order = available_backends() if backend == "auto" else [backend]
    if not order:
        raise RuntimeError(
            "No backend available: start Ollama with a twin model, or set "
            "ANTHROPIC_API_KEY / OPENAI_API_KEY in .env"
        )
    errors = []
    for name in order:
        fn = {"local": _local, "claude": _claude, "openai": _openai}.get(name)
        if fn is None:
            errors.append(f"{name}: unknown backend")
            continue
        try:
            return fn(system, messages, max_tokens, temperature)
        except Exception as exc:  # try the next backend rather than failing the turn
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("all backends failed — " + " | ".join(errors))
