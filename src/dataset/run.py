"""Phase 2 CLI: build training data and extract identity from the corpus.

    python -m src.dataset.run sft        # build SFT train/eval JSONL
    python -m src.dataset.run persona    # extract persona card (Claude)
    python -m src.dataset.run mind       # extract Mind Model (Claude)
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ..config import REPO_ROOT, db_path, load_config
from ..ingest import db as dbm

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def sft():
    """Build supervised fine-tuning examples from the message store."""
    from . import build_sft

    cfg = load_config()
    conn = dbm.connect(db_path(cfg))
    stats, n_train, n_eval = build_sft.build(conn, cfg)

    table = Table(title="SFT dataset")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("conversations", str(stats.conversations))
    table.add_row("[bold]training examples[/bold]", f"[bold]{n_train}[/bold]")
    table.add_row("eval examples", str(n_eval))
    for reason, n in sorted(stats.dropped.items(), key=lambda kv: -kv[1]):
        table.add_row(f"  dropped: {reason}", str(n))
    console.print(table)
    console.print(f"[dim]-> {REPO_ROOT / 'data' / 'datasets'}[/dim]")


@app.command()
def persona(
    model: str = typer.Option("claude-opus-4-5", help="Anthropic model for extraction"),
    samples: int = typer.Option(400, help="How many of your messages to analyse"),
):
    """Extract the persona card (tone, slang, habits) from your own messages."""
    from . import persona as p

    cfg = load_config()
    conn = dbm.connect(db_path(cfg))
    out = p.extract_persona(conn, model=model, n_samples=samples)
    console.print(f"[bold green]Persona card -> {out}[/bold green]")


@app.command()
def mind(
    model: str = typer.Option("claude-opus-4-5", help="Anthropic model for extraction"),
):
    """Extract the Mind Model: values, decisions, per-person styles, interests."""
    from . import mind_model as mm

    cfg = load_config()
    conn = dbm.connect(db_path(cfg))
    out = mm.extract_mind_model(conn, model=model)
    console.print(f"[bold green]Mind Model -> {out}[/bold green]")


if __name__ == "__main__":
    app()
