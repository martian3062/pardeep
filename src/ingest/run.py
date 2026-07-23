"""Ingest CLI. Idempotent: re-running skips files already ingested (by sha256).

    uv run python -m src.ingest.run whatsapp data\\raw\\whatsapp
    uv run python -m src.ingest.run audio data\\raw\\voice_notes --kind voice_note
    uv run python -m src.ingest.run audio data\\raw\\calls --kind call
    uv run python -m src.ingest.run stats
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..config import db_path, load_config
from . import db as dbm
from . import whatsapp as wa

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def whatsapp(
    path: Path = typer.Argument(..., help="Directory with (or single) exported .txt files"),
    me: list[str] = typer.Option(None, "--me", help="Override me_names from config.yaml"),
):
    """Parse WhatsApp chat exports into the message DB."""
    cfg = load_config()
    me_names = me or cfg["me_names"]
    dayfirst = cfg.get("whatsapp", {}).get("dayfirst", True)
    conn = dbm.connect(db_path(cfg))

    files = [path] if path.is_file() else sorted(path.glob("*.txt"))
    if not files:
        console.print(f"[red]No .txt files found in {path}[/red]")
        raise typer.Exit(1)

    total_new = 0
    for f in files:
        sha = dbm.file_sha256(f)
        if dbm.already_ingested(conn, f, sha):
            console.print(f"  [dim]{f.name}: unchanged, skipped[/dim]")
            continue
        messages = wa.parse_file(f, me_names, dayfirst_fallback=dayfirst)
        n = dbm.insert_messages(conn, messages)
        dbm.record_file(conn, f, sha, n)
        me_count = sum(1 for m in messages if m.speaker == "me")
        console.print(
            f"  {f.name}: {len(messages)} parsed, {n} new "
            f"({me_count} from me, {len(messages) - me_count} from others)"
        )
        if messages and me_count == 0:
            console.print(
                "    [yellow]WARNING: 0 messages attributed to you — "
                "check me_names in config.yaml matches your name in this export[/yellow]"
            )
        total_new += n
    console.print(f"[bold green]Done: {total_new} new messages[/bold green]")


@app.command()
def audio(
    path: Path = typer.Argument(..., help="Directory with audio files"),
    kind: str = typer.Option("voice_note", help="voice_note | call"),
    model: str = typer.Option(None, help="Whisper model override (default from config.yaml)"),
):
    """Transcribe voice notes or call recordings into the DB."""
    from . import transcribe as tr  # heavy import, only when needed

    cfg = load_config()
    model_size = model or cfg["transcription"]["model"]
    language = cfg["transcription"].get("language")
    conn = dbm.connect(db_path(cfg))

    if kind == "voice_note":
        files, items = tr.ingest_voice_notes(conn, path, model_size, language)
        console.print(f"[bold green]Done: {files} files, {items} voice-note messages[/bold green]")
    elif kind == "call":
        files, items = tr.ingest_calls(conn, path, model_size, language)
        console.print(f"[bold green]Done: {files} files, {items} call segments[/bold green]")
    else:
        console.print(f"[red]Unknown kind: {kind} (use voice_note | call)[/red]")
        raise typer.Exit(1)


@app.command()
def enroll(
    path: Path = typer.Argument(..., help="Directory with YOUR voice notes"),
):
    """Build your voice fingerprint (for speaker ID) + TTS reference clips."""
    from . import diarize as dz

    dz.enroll(path, load_config())


@app.command()
def diarize():
    """Label pending call segments me/other and promote them into messages."""
    from . import diarize as dz

    cfg = load_config()
    conn = dbm.connect(db_path(cfg))
    files, promoted = dz.diarize_calls(conn, cfg)
    console.print(
        f"[bold green]Done: {files} calls labeled, {promoted} messages promoted[/bold green]"
    )


@app.command()
def stats():
    """Show what's in the message DB."""
    conn = dbm.connect(db_path())
    s = dbm.stats(conn)
    table = Table(title="messages.db")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("total messages", str(s["messages"]))
    for src, n in s["by_source"].items():
        table.add_row(f"  source: {src}", str(n))
    for spk, n in s["by_speaker"].items():
        table.add_row(f"  speaker: {spk}", str(n))
    table.add_row("conversations", str(s["conversations"]))
    table.add_row("call segments (unlabeled)", str(s["call_segments"]))
    table.add_row("files ingested", str(s["files_ingested"]))
    console.print(table)


if __name__ == "__main__":
    app()
