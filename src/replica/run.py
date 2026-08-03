"""The replica pipeline, end to end.

    python -m src.replica.run prep       # gate + crop + caption the 20-30 photos
    python -m src.replica.run baseline   # what "him" scores as himself
    python -m src.replica.run verify DIR # score a folder of generated images
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def cmd_prep(args) -> None:
    from . import dataset as ds

    src = Path(args.src)
    photos = sorted(p for p in src.iterdir() if p.is_file()) if src.exists() else []
    if not photos:
        console.print(f"[yellow]no photos in {src} — drop 20-30 pictures there first[/yellow]")
        return

    console.print(f"curating [bold]{len(photos)}[/bold] photos…")
    cur = ds.curate(src)
    ds.caption_all(cur, use_vlm=not args.no_vlm)
    ds.export(cur)

    table = Table(title="reference photos")
    table.add_column("photo")
    table.add_column("face px", justify="right")
    table.add_column("sharp", justify="right")
    table.add_column("id sim", justify="right")
    table.add_column("verdict")
    for p in cur.photos:
        verdict = "[green]ok[/green]" if p.accepted else f"[red]{p.reason}[/red]"
        if p.warning:
            verdict += f" [yellow]({p.warning})[/yellow]"
        table.add_row(p.path.name[:34], str(p.face_px), f"{p.sharpness:.0f}",
                      f"{p.identity_sim:.2f}", verdict)
    console.print(table)

    n = len(cur.accepted)
    console.print(f"\n[bold]{n} accepted[/bold], {len(cur.rejected)} rejected")
    if n < 15:
        console.print("[yellow]under 15 usable photos — likeness will suffer; add more[/yellow]")
    else:
        archive = ds.pack()
        console.print(f"[green]bundle ready:[/green] {archive}  (scp this to the GPU box)")


def cmd_baseline(args) -> None:
    from . import verify as vf

    cal = vf.calibrate(vf.load_ground_truth())
    console.print("his own photos, each vs the rest of him (leave-one-out):")
    console.print(f"  median {cal.median:.3f} · p25 {cal.p25:.3f} · floor(p5) {cal.floor:.3f}")
    console.print("[dim]a generated battery passes when its median ≥ p25[/dim]")


def cmd_verify(args) -> None:
    from . import verify as vf

    report = vf.score_dir(Path(args.dir))
    table = Table(title="generated vs him")
    table.add_column("image")
    table.add_column("similarity", justify="right")
    table.add_column("faces", justify="right")
    table.add_column("note")
    for s in report.scores:
        style = "green" if s.passes else "red"
        table.add_row(s.file[:38], f"[{style}]{s.similarity:.3f}[/{style}]", str(s.faces), s.note)
    console.print(table)
    console.print(
        f"\nbattery median [bold]{report.generated_median:.3f}[/bold] "
        f"vs his real-photo p25 {report.calibration_p25:.3f}"
    )
    verdict = report.verdict
    colour = "green" if verdict == "pass" else "red"
    console.print(f"[{colour} bold]{verdict.upper()}[/{colour} bold]")
    out = vf.save(report)
    sheet = vf.contact_sheet(Path(args.dir), report)
    console.print(f"[dim]report: {out}[/dim]")
    console.print(
        f"[bold yellow]open {sheet} and judge the dastar by eye — "
        "the score cannot see headwear[/bold yellow]"
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="src.replica.run")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prep", help="gate, crop and caption the reference photos")
    prep.add_argument("--src", default="data/raw/face_ref")
    prep.add_argument("--no-vlm", action="store_true", help="template captions only")
    prep.set_defaults(func=cmd_prep)

    sub.add_parser("baseline", help="his real-photo similarity distribution").set_defaults(
        func=cmd_baseline
    )

    ver = sub.add_parser("verify", help="score generated images against him")
    ver.add_argument("dir")
    ver.set_defaults(func=cmd_verify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
