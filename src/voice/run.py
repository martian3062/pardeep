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


def cmd_clone(args) -> None:
    from . import tts

    clips = sorted(ref.OUT_DIR.glob("*.wav"))
    if not clips:
        console.print("[red]no reference clips — run `refs` first[/red]")
        return
    existing = tts.saved_voice()
    if existing and not args.replace:
        console.print(f"[yellow]already cloned: {existing.name} ({existing.voice_id})[/yellow]")
        console.print("  pass --replace to make a new one")
        return
    total = sum(c.stat().st_size for c in clips)
    console.print(
        f"uploading [bold]{len(clips)}[/bold] clips ({total/1e6:.1f} MB) to ElevenLabs — "
        "this is the one thing that leaves the machine"
    )
    voice = tts.clone(clips)
    console.print(f"[green]cloned:[/green] {voice.name}  id={voice.voice_id}")


def cmd_say(args) -> None:
    from . import tts

    engine = tts.ElevenLabs()
    out = tts.OUT_DIR / f"{args.name}.mp3"
    engine.say(args.text, out)
    console.print(f"[green]wrote[/green] {out}  [dim]({engine.model_id})[/dim]")


def cmd_forget(args) -> None:
    from . import tts

    voice = tts.saved_voice()
    if not voice:
        console.print("no cloned voice stored")
        return
    ok = tts.delete(voice.voice_id)
    console.print("[green]deleted from ElevenLabs[/green]" if ok else "[red]delete failed[/red]")


def cmd_devices(args) -> None:
    from .listen import list_devices

    console.print(list_devices())


def cmd_talk(args) -> None:
    from .loop import VoiceTwin

    VoiceTwin(device=args.device, speak=not args.mute).run(counterpart=args.as_person)


def cmd_hear(args) -> None:
    """Mic + transcription only — check the ears before wiring the mouth."""
    from .listen import Ears

    ears = Ears(device=args.device)
    console.print("[dim]say something…[/dim]")
    utt = ears.listen()
    if utt is None:
        console.print("[yellow]heard nothing[/yellow]")
        return
    text, lang = ears.transcribe(utt)
    console.print(f"[dim]{utt.seconds:.1f}s, detected {lang}[/dim]")
    console.print(f"[bold]{text or '(nothing intelligible)'}[/bold]")


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

    p_clone = sub.add_parser("clone", help="create the cloned voice (uploads the clips)")
    p_clone.add_argument("--replace", action="store_true")
    p_clone.set_defaults(func=cmd_clone)

    p_say = sub.add_parser("say", help="speak a line in his voice")
    p_say.add_argument("text")
    p_say.add_argument("--name", default="sample")
    p_say.set_defaults(func=cmd_say)

    sub.add_parser("forget", help="delete the clone from ElevenLabs").set_defaults(func=cmd_forget)

    sub.add_parser("devices", help="list audio devices").set_defaults(func=cmd_devices)

    p_hear = sub.add_parser("hear", help="mic + local transcription only")
    p_hear.add_argument("--device", type=int, default=None)
    p_hear.set_defaults(func=cmd_hear)

    p_talk = sub.add_parser("talk", help="full voice conversation with the twin")
    p_talk.add_argument("--device", type=int, default=None, help="input device index")
    p_talk.add_argument("--as-person", default="", help="who the twin is talking to")
    p_talk.add_argument("--mute", action="store_true", help="text replies, no speech")
    p_talk.set_defaults(func=cmd_talk)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
