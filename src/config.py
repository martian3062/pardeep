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
