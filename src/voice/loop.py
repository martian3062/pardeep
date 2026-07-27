"""Speak to the twin, hear it answer in his voice.

    you speak → silero-VAD → Whisper → memory + persona → reply → his cloned voice

Only the last hop leaves the machine. The recording is transcribed locally and
discarded; what goes out is the text of the reply.
"""
from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from ..twin.chat import Twin
from .listen import Ears, play
from .tts import ElevenLabs

console = Console()


@dataclass
class Exchange:
    heard: str
    language: str
    said: str
    memories: int


class VoiceTwin:
    def __init__(self, device: int | None = None, speak: bool = True):
        self.ears = Ears(device=device)
        self.twin = Twin()
        self.mouth = ElevenLabs() if speak else None

    def turn(self, counterpart: str = "") -> Exchange | None:
        utt = self.ears.listen()
        if utt is None:
            return None
        text, lang = self.ears.transcribe(utt)
        if not text:
            return None  # noise, not speech — say nothing rather than guess

        reply = self.twin.reply(text, counterpart=counterpart)
        if self.mouth is not None:
            play(self.mouth.pcm(reply.text))
        return Exchange(
            heard=text, language=lang, said=reply.text, memories=len(reply.memories)
        )

    def run(self, counterpart: str = "") -> None:
        console.print("[dim]listening — speak, then pause. Ctrl-C to stop.[/dim]")
        while True:
            try:
                exchange = self.turn(counterpart)
            except KeyboardInterrupt:
                console.print("\n[dim]bye[/dim]")
                return
            except Exception as exc:
                console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
                continue
            if exchange is None:
                continue
            console.print(f"[bold]you[/bold] [dim]({exchange.language})[/dim]: {exchange.heard}")
            console.print(f"[cyan]twin[/cyan]: {exchange.said}")
            console.print(f"[dim]{exchange.memories} memories[/dim]\n")
