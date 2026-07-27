"""Phase 7 CLI — the daily diary and the feedback loop.

    python -m src.diary.run write "aaj lab me poora din gaya, deploy fail hua"
    python -m src.diary.run speak            # say it instead
    python -m src.diary.run index            # push new entries into memory
    python -m src.diary.run correct          # tell the twin how you'd have said it
    python -m src.diary.run digest           # summarise the week
    python -m src.diary.run export           # write a DPO batch
    python -m src.diary.run stats
"""
from __future__ import annotations

import argparse
from datetime import date

from rich.console import Console
from rich.panel import Panel

from . import capture, learn
from . import store as st

console = Console()


def _twin_reacts(text: str) -> None:
    """The point of a diary that talks back: it responds as him, in his voice."""
    from ..twin.chat import Twin

    reply = Twin().reply(f"Aaj ka mera diary entry: {text}")
    console.print(Panel(reply.text, border_style="cyan", title="twin"))


def cmd_write(args) -> None:
    conn = st.connect()
    entry_id = capture.capture_text(conn, args.text, day=args.day)
    console.print(f"[green]saved entry {entry_id}[/green]")
    added = capture.index_new(conn)
    console.print(f"[dim]indexed {added} into memory[/dim]")
    if not args.quiet:
        _twin_reacts(args.text)


def cmd_speak(args) -> None:
    conn = st.connect()
    console.print("[dim]speak your entry, then pause…[/dim]")
    got = capture.capture_voice(conn, device=args.device)
    if got is None:
        console.print("[yellow]nothing captured[/yellow]")
        return
    entry_id, text = got
    console.print(f"[green]saved entry {entry_id}[/green]: {text}")
    added = capture.index_new(conn)
    console.print(f"[dim]indexed {added} into memory[/dim]")
    if not args.quiet:
        _twin_reacts(text)


def cmd_index(args) -> None:
    conn = st.connect()
    console.print(f"[green]indexed {capture.index_new(conn)}[/green] new entries")


def cmd_correct(args) -> None:
    """Log a preference pair: what it said, and what he'd have said."""
    conn = st.connect()
    prompt = args.prompt or console.input("[bold]what was asked >[/bold] ").strip()
    rejected = args.said or console.input("[bold]what the twin said >[/bold] ").strip()
    chosen = args.instead or console.input("[bold]how you'd say it >[/bold] ").strip()
    if not (prompt and rejected):
        console.print("[yellow]need at least the question and the twin's answer[/yellow]")
        return
    fb_id = st.add_feedback(
        conn,
        st.Feedback(
            prompt=prompt,
            rejected=rejected,
            chosen=chosen,
            verdict="down" if chosen else args.verdict,
            counterpart=args.as_person,
        ),
    )
    console.print(f"[green]logged correction {fb_id}[/green]")
    if not chosen:
        console.print("[dim]no rewrite given — this counts, but cannot train on its own[/dim]")


def cmd_digest(args) -> None:
    conn = st.connect()
    text = learn.build_digest(conn, date.fromisoformat(args.day) if args.day else None)
    if text is None:
        console.print("[yellow]no entries this week[/yellow]")
        return
    console.print(Panel(text, title=f"week {learn.week_of(date.today())}", border_style="green"))


def cmd_export(args) -> None:
    conn = st.connect()
    path, skipped = learn.export_pairs(conn)
    if path is None:
        console.print(f"[yellow]no usable pairs[/yellow] ({skipped} too similar or empty)")
        return
    n = sum(1 for _ in path.open(encoding="utf-8"))
    console.print(f"[green]wrote {n} preference pairs[/green] to {path}")
    if skipped:
        console.print(f"[dim]{skipped} skipped as too similar to train on[/dim]")


def cmd_stats(args) -> None:
    conn = st.connect()
    for k, v in st.stats(conn).items():
        console.print(f"  {k:<14} {v}")


def main() -> None:
    p = argparse.ArgumentParser(prog="src.diary.run")
    p.add_argument("--quiet", action="store_true", help="skip the twin's reaction")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="type an entry")
    w.add_argument("text")
    w.add_argument("--day", help="YYYY-MM-DD, defaults to today")
    w.set_defaults(func=cmd_write)

    s = sub.add_parser("speak", help="say an entry (local transcription)")
    s.add_argument("--device", type=int, default=None)
    s.set_defaults(func=cmd_speak)

    sub.add_parser("index", help="push new entries into memory").set_defaults(func=cmd_index)

    c = sub.add_parser("correct", help="log how you'd have said it")
    c.add_argument("--prompt", default="")
    c.add_argument("--said", default="")
    c.add_argument("--instead", default="")
    c.add_argument("--verdict", default="down", choices=["up", "down"])
    c.add_argument("--as-person", default="")
    c.set_defaults(func=cmd_correct)

    d = sub.add_parser("digest", help="summarise the week")
    d.add_argument("--day", help="any day in the week, YYYY-MM-DD")
    d.set_defaults(func=cmd_digest)

    sub.add_parser("export", help="write a DPO batch").set_defaults(func=cmd_export)
    sub.add_parser("stats", help="what the diary holds").set_defaults(func=cmd_stats)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
