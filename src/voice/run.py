"""Phase 5 CLI — the voice twin.

    python -m src.voice.run refs      # build clean reference clips of his voice
    python -m src.voice.run inspect   # what was selected and how confident
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from . import reference as ref

console = Console()


def cmd_refs(args) -> None:
    if not ref.VOICE_EMBED.exists():
        console.print(f"[red]no enrolled voice at {ref.VOICE_EMBED}[/red]")
        return
    me = np.load(ref.VOICE_EMBED)

    if args.source == "calls":
        console.print("scanning diarized call segments where he speaks…")
        clips = ref.from_call_segments(me)
    else:
        files = sorted(p for p in ref.REFERENCE_DIR.iterdir() if p.is_file())
        console.print(f"scanning [bold]{len(files)}[/bold] voice notes…")
        clips = ref.find_clips(files, me, window_sec=args.window, hop_sec=args.hop)
    if not clips:
        console.print("[yellow]no segment cleared the similarity bar[/yellow]")
        return

    console.print(f"found [bold]{len(clips)}[/bold] matching windows across "
                  f"{len({c.path for c in clips})} files")
    written = ref.export(clips, keep=args.keep)
    console.print(f"[green]wrote {len(written)} reference clips[/green] to {ref.OUT_DIR}")
    cmd_inspect(args)


def cmd_inspect(args) -> None:
    manifest = ref.OUT_DIR / "manifest.json"
    if not manifest.exists():
        console.print("[yellow]no reference clips yet — run `refs`[/yellow]")
        return
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    table = Table(title="voice clone reference")
    table.add_column("clip")
    table.add_column("source")
    table.add_column("span", justify="right")
    table.add_column("similarity", justify="right")
    for r in rows:
        table.add_row(
            r["file"],
            Path(r["source"]).name,
            f"{r['start']:.1f}–{r['end']:.1f}s",
            f"{r['similarity']:.3f}",
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.voice.run")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_refs = sub.add_parser("refs", help="extract clean single-speaker reference clips")
    p_refs.add_argument("--window", type=float, default=8.0, help="clip length in seconds")
    p_refs.add_argument("--hop", type=float, default=4.0)
    p_refs.add_argument("--keep", type=int, default=6)
    p_refs.add_argument(
        "--source",
        choices=["calls", "notes"],
        default="calls",
        help="calls = diarized segments where he speaks (32.7h); notes = the 12 set-aside clips",
    )
    p_refs.set_defaults(func=cmd_refs)

    sub.add_parser("inspect", help="show the selected clips").set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
