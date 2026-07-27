"""CLI for Phase 4b: turn photos into dated memories.

    python -m src.vision.run scan          # inventory + dates, no GPU
    python -m src.vision.run caption       # local VLM captions (laptop GPU)
    python -m src.vision.run index         # push captions into LanceDB
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import photos as ph

console = Console()

DEFAULT_ROOTS = [Path("data/raw/_dump"), Path("D:/iqoo"), Path("D:/iqooz6data")]
DB_PATH = Path("data/processed/photos.db")


def cmd_scan(args) -> None:
    roots = [Path(r) for r in args.roots] if args.roots else DEFAULT_ROOTS
    console.print(f"[cyan]Scanning {len(roots)} roots...[/cyan]")
    found = ph.scan(roots)
    console.print(f"Found [bold]{len(found)}[/bold] images")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        ph.save(conn, found)

    ranks = Counter(p.rank for p in found)
    table = Table(title="Provenance")
    table.add_column("rank")
    table.add_column("meaning")
    table.add_column("count", justify="right")
    for rank, meaning in [
        (3, "his camera / images he sent"),
        (2, "events, albums"),
        (1, "received WhatsApp, unknown"),
        (0, "Facebook cache, screenshots"),
    ]:
        table.add_row(str(rank), meaning, str(ranks.get(rank, 0)))
    console.print(table)

    sources = Counter(p.date_source for p in found)
    console.print("\nDate source: " + "  ".join(f"{s}:{c}" for s, c in sources.most_common()))

    trusted = [p for p in found if p.date_trusted]
    console.print(f"Trustworthy dates: [bold]{len(trusted)}[/bold]/{len(found)}")
    if trusted:
        years = Counter(p.taken_at.year for p in trusted)
        console.print("By year: " + "  ".join(f"{y}:{c}" for y, c in sorted(years.items())))

    usable = [p for p in found if p.usable]
    console.print(f"\n[green]Worth captioning: {len(usable)}[/green] (rank >= 2)")


def cmd_caption(args) -> None:
    from .caption import run as caption_run

    caption_run(
        DB_PATH,
        limit=args.limit,
        min_rank=args.min_rank,
        four_bit=not args.fp16,
    )


def cmd_dates(args) -> None:
    """Fill in dates for photos whose only timestamp was the archive copy date."""
    with sqlite3.connect(DB_PATH) as conn:
        if args.redo:
            restored = ph.reset_imputed(conn)
            console.print(f"[yellow]Reset {restored} previously imputed dates to mtime[/yellow]")
        n = ph.impute_dates(conn, min_siblings=args.min_siblings)
        console.print(f"[green]Imputed {n} dates[/green] from directory medians")
        for row in conn.execute(
            "SELECT date_source, count(*) FROM photos GROUP BY 1 ORDER BY 2 DESC"
        ):
            console.print(f"  {row[0]}: {row[1]}")


def cmd_index(args) -> None:
    from .index_photos import run as index_run

    index_run(DB_PATH)


def cmd_show(args) -> None:
    """Eyeball captions — the only real check that the model understood a photo."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """SELECT taken_at, date_source, folder_hint, caption, path
               FROM photos WHERE caption IS NOT NULL
               ORDER BY RANDOM() LIMIT ?""",
            (args.n,),
        ).fetchall()
    for taken, src, hint, caption, path in rows:
        when = (taken or "?")[:10] + ("" if src in ("exif", "filename") else " (untrusted)")
        console.print(f"\n[cyan]{when}[/cyan]  [dim]{hint}  {Path(path).name}[/dim]")
        console.print(f"  {caption}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.vision.run")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="inventory photos and extract dates")
    p_scan.add_argument("--roots", nargs="*", help="override default photo roots")
    p_scan.set_defaults(func=cmd_scan)

    p_cap = sub.add_parser("caption", help="caption photos with the local VLM")
    p_cap.add_argument("--limit", type=int, help="stop after N photos")
    p_cap.add_argument("--min-rank", type=int, default=2, help="provenance floor")
    p_cap.add_argument("--fp16", action="store_true", help="skip 4-bit quantization")
    p_cap.set_defaults(func=cmd_caption)

    p_dates = sub.add_parser("dates", help="impute folder-median dates for mtime-only photos")
    p_dates.add_argument("--min-siblings", type=int, default=3)
    p_dates.add_argument(
        "--redo", action="store_true", help="re-read mtimes and impute again from scratch"
    )
    p_dates.set_defaults(func=cmd_dates)

    p_index = sub.add_parser("index", help="index captioned photos into LanceDB")
    p_index.set_defaults(func=cmd_index)

    p_show = sub.add_parser("show", help="print random captions to spot-check")
    p_show.add_argument("-n", type=int, default=10)
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
