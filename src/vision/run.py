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
        if args.redate:
            n = ph.redate(conn)
            console.print(f"[yellow]Recomputed {n} dates from the files[/yellow]")
        if args.redo:
            restored = ph.reset_imputed(conn)
            console.print(f"[yellow]Reset {restored} previously imputed dates to mtime[/yellow]")
        n = ph.impute_dates(conn, min_siblings=args.min_siblings)
        console.print(f"[green]Imputed {n} dates[/green] from directory medians")
        for row in conn.execute(
            "SELECT date_source, count(*) FROM photos GROUP BY 1 ORDER BY 2 DESC"
        ):
            console.print(f"  {row[0]}: {row[1]}")


def cmd_polish(args) -> None:
    """Text-only caption cleanup, and requeue the ones that were cut off."""
    from .caption import clean_stored_captions, truncated_captions

    with sqlite3.connect(DB_PATH) as conn:
        n = clean_stored_captions(conn)
        console.print(f"[green]Cleaned {n} captions[/green] of absence boilerplate")
        cut = truncated_captions(conn)
        console.print(f"{len(cut)} captions were truncated by the old token cap")
        if cut and args.requeue:
            conn.executemany(
                "UPDATE photos SET caption = NULL, captioned_at = NULL WHERE id = ?",
                [(i,) for i in cut],
            )
            conn.commit()
            console.print(f"[yellow]Requeued {len(cut)} for re-captioning[/yellow]")


def cmd_dedupe(args) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        n = ph.find_duplicates(conn, min_rank=args.min_rank)
        remaining = len(ph.pending_captions(conn, min_rank=args.min_rank))
        console.print(f"[green]Marked {n} duplicate copies[/green]; {remaining} left to caption")


def cmd_faces(args) -> None:
    """Detect faces across the photo archive — locally, nothing uploaded."""
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

    from . import faces as fc

    conn = sqlite3.connect(DB_PATH)
    todo = fc.pending(conn, min_rank=args.min_rank)
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        console.print("[green]every photo has been scanned for faces[/green]")
        return

    console.print(f"scanning [bold]{len(todo)}[/bold] photos for faces…")
    app = fc.analyser()
    with_faces = 0
    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("faces", total=len(todo))
        for photo_id, path in todo:
            found = fc.detect(app, Path(path))
            fc.save_faces(conn, photo_id, found)
            with_faces += bool(found)
            progress.advance(task)

    total = conn.execute("SELECT count(*) FROM photo_faces").fetchone()[0]
    console.print(f"[green]{with_faces} photos contain faces[/green]; {total} faces stored")


def cmd_people(args) -> None:
    """Group the detected faces into recurring people."""
    from . import faces as fc

    conn = sqlite3.connect(DB_PATH)
    fc.ensure_schema(conn)
    embeddings = fc.load_embeddings(conn)
    if not embeddings:
        console.print("[yellow]no faces yet — run `faces` first[/yellow]")
        return
    assignment = fc.cluster(embeddings, threshold=args.threshold)
    conn.executemany(
        "UPDATE photo_faces SET person = ? WHERE id = ?",
        [(f"person_{grp:02d}", fid) for fid, grp in assignment.items()],
    )
    conn.commit()

    rows = conn.execute(
        """SELECT person, count(*) n, count(DISTINCT photo_id) photos,
                  min(substr(p.taken_at,1,10)) first, max(substr(p.taken_at,1,10)) last
           FROM photo_faces f JOIN photos p ON p.id = f.photo_id
           GROUP BY person ORDER BY n DESC LIMIT ?""",
        (args.show,),
    ).fetchall()
    table = Table(title=f"recurring people ({len(set(assignment.values()))} groups)")
    table.add_column("person")
    table.add_column("faces", justify="right")
    table.add_column("photos", justify="right")
    table.add_column("seen between")
    for person, n, photos, first, last in rows:
        table.add_row(person, str(n), str(photos), f"{first} → {last}")
    console.print(table)
    console.print("[dim]person_00 appears most — in a personal archive that is usually you[/dim]")


def cmd_index(args) -> None:
    from .index_photos import run as index_run

    index_run(DB_PATH, cpu=args.cpu)


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
    p_dates.add_argument(
        "--redate", action="store_true", help="recompute all dates from the files, keeping captions"
    )
    p_dates.set_defaults(func=cmd_dates)

    p_polish = sub.add_parser("polish", help="strip absence boilerplate; find truncated captions")
    p_polish.add_argument(
        "--requeue", action="store_true", help="clear truncated captions so they are redone"
    )
    p_polish.set_defaults(func=cmd_polish)

    p_dedupe = sub.add_parser("dedupe", help="mark byte-identical copies so each is captioned once")
    p_dedupe.add_argument("--min-rank", type=int, default=2)
    p_dedupe.set_defaults(func=cmd_dedupe)

    p_faces = sub.add_parser("faces", help="detect faces in the archive (local ONNX)")
    p_faces.add_argument("--limit", type=int)
    p_faces.add_argument("--min-rank", type=int, default=2)
    p_faces.set_defaults(func=cmd_faces)

    p_people = sub.add_parser("people", help="group faces into recurring people")
    p_people.add_argument("--threshold", type=float, default=0.42)
    p_people.add_argument("--show", type=int, default=12)
    p_people.set_defaults(func=cmd_people)

    p_index = sub.add_parser("index", help="index captioned photos into LanceDB")
    p_index.add_argument(
        "--cpu", action="store_true", help="embed on the CPU (use while captioning holds the GPU)"
    )
    p_index.set_defaults(func=cmd_index)

    p_show = sub.add_parser("show", help="print random captions to spot-check")
    p_show.add_argument("-n", type=int, default=10)
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
