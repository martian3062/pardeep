"""Talk to the twin from the terminal.

    python -m src.twin.run chat
    python -m src.twin.run ask "kal kya kar raha tha 2017 me?"
    python -m src.twin.run backends
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

console = Console()


def cmd_backends(args) -> None:
    from .router import available_backends

    found = available_backends()
    console.print("available backends: " + (", ".join(found) if found else "[red]none[/red]"))
    if not found:
        console.print("  start Ollama with a twin model, or set ANTHROPIC_API_KEY in .env")


def _show(reply, show_memories: bool) -> None:
    console.print(Panel(reply.text, border_style="cyan"))
    console.print(f"[dim]via {reply.backend}/{reply.model} · {len(reply.memories)} memories[/dim]")
    if show_memories:
        for m in reply.memories:
            console.print(f"  [dim]{m.render()[:160]}[/dim]")


def cmd_ask(args) -> None:
    from .chat import Twin

    twin = Twin(guest=args.guest)
    _show(twin.reply(args.message, counterpart=args.as_person, backend=args.backend), args.memories)


def cmd_chat(args) -> None:
    from .chat import Twin

    twin = Twin(guest=args.guest)
    console.print("[dim]talking to your twin — Ctrl-C to leave[/dim]")
    while True:
        try:
            message = console.input("[bold]you >[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        if not message:
            continue
        try:
            _show(
                twin.reply(message, counterpart=args.as_person, backend=args.backend),
                args.memories,
            )
        except Exception as exc:
            console.print(f"[red]{exc}[/red]")


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.twin.run")
    parser.add_argument("--backend", default="auto", help="auto | local | claude | openai")
    parser.add_argument("--guest", action="store_true", help="reply as if to someone else")
    parser.add_argument("--as-person", default="", help="who the twin is talking to")
    parser.add_argument("--memories", action="store_true", help="show what was recalled")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="one question")
    p_ask.add_argument("message")
    p_ask.set_defaults(func=cmd_ask)

    sub.add_parser("chat", help="interactive").set_defaults(func=cmd_chat)
    sub.add_parser("backends", help="what can generate replies").set_defaults(func=cmd_backends)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
