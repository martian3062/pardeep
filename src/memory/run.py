"""Phase 4 CLI — the twin's memory.

    python -m src.memory.run index          # build memories from the corpus
    python -m src.memory.run ask "what do I talk to Roshan about?"
    python -m src.memory.run stats
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ..config import db_path, load_config
from ..ingest import db as dbm

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def index(
    reset: bool = typer.Option(False, "--reset", help="Drop existing memories first"),
    facts_only: bool = typer.Option(False, help="Only re-index the Mind Model facts"),
):
    """Index conversations + Mind Model into the local vector store."""
    from .build import build_episodic, build_facts
    from .store import TABLE, MemoryStore

    cfg = load_config()
    conn = dbm.connect(db_path(cfg))
    store = MemoryStore()
    if reset and TABLE in store.db.table_names():
        store.db.drop_table(TABLE)
        store._table = None
        console.print("[yellow]dropped existing memories[/yellow]")

    total = 0
    if not facts_only:
        total += build_episodic(
            conn, store, tuple(cfg.get("dataset", {}).get("exclude_sources", ["video"]))
        )
    total += build_facts(store)
    store.ensure_fts_index(rebuild=True)  # hybrid search needs it to cover the new rows
    console.print(f"[bold green]indexed {total} memories (store now holds {store.count()})[/bold green]")


@app.command()
def ask(
    query: str = typer.Argument(..., help="What do you want the twin to recall?"),
    limit: int = typer.Option(6),
    kind: str = typer.Option("", help="episodic | fact"),
    guest: bool = typer.Option(False, help="Retrieve as a guest would (public only)"),
):
    """Search memory the way the twin will at conversation time."""
    from .store import MemoryStore

    store = MemoryStore()
    rows = store.search(
        query,
        limit=limit,
        kinds=(kind,) if kind else (),
        max_trust="public" if guest else "secret",
    )
    if not rows:
        console.print("[yellow]nothing found (index built? guest mode hides private memories)[/yellow]")
        return
    for r in rows:
        head = f"[{r['kind']}] {r['timestamp'][:10]}"
        if r.get("counterpart"):
            head += f" · {r['counterpart']}"
        console.print(f"\n[bold]{head}[/bold]  [dim]score {r['score']:.3f}[/dim]")
        console.print(r["text"][:600])
        # photo memories carry their file path, so a recalled picture can be opened
        if r["kind"] == "photo" and r.get("conversation_id"):
            console.print(f"[dim]{r['conversation_id']}[/dim]")


@app.command()
def stats():
    """What is in memory."""
    from .store import MemoryStore

    store = MemoryStore()
    table = store._open()
    if table is None:
        console.print("[yellow]no memories yet — run `index`[/yellow]")
        return
    df = table.to_pandas()
    t = Table(title=f"memory store ({len(df)} rows)")
    t.add_column("dimension")
    t.add_column("breakdown")
    for col in ("kind", "source", "trust"):
        counts = df[col].value_counts().to_dict()
        t.add_row(col, ", ".join(f"{k}={v}" for k, v in counts.items()))
    t.add_row("date range", f"{df['timestamp'].min()[:10]} → {df['timestamp'].max()[:10]}")
    console.print(t)


if __name__ == "__main__":
    app()
