"""Load config.yaml from the repo root."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def db_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return REPO_ROOT / cfg["paths"]["db"]


# Defaults match what was hardcoded before config.yaml grew a `models:` block, so
# an older config.yaml keeps working instead of raising KeyError mid-turn.
_MODEL_DEFAULTS = {
    "claude": "claude-sonnet-5",
    "openai": "gpt-4.1",
    "extract": "claude-opus-4-5",
    "tts": "eleven_v3",
    "ollama_url": "http://127.0.0.1:11434",
}


def model_id(name: str, cfg: dict | None = None) -> str:
    """Look up a provider model id by role (claude | openai | extract | tts)."""
    cfg = cfg or load_config()
    return (cfg.get("models") or {}).get(name) or _MODEL_DEFAULTS[name]
