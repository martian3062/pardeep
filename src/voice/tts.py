"""Speak in his voice.

The privacy line for this module, stated plainly: everything else in this project
stays on the machine, and this is the one exception. Creating the clone uploads
about a minute of him speaking to ElevenLabs, where it is stored as a voice
model; synthesis then uploads the text to be spoken. The text is generated
locally from local memories, so what crosses is his voice timbre and the words
the twin chose to say — never the archive, the photos, or the chats.

That trade buys the only engine with real Punjabi support. Every local candidate
either has no Punjabi at all (GPT-SoVITS, XTTS-v2, Chatterbox) or has it
unevaluated (IndicF5). Engines sit behind one interface so a local model can
replace this later without touching the caller.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

API = "https://api.elevenlabs.io/v1"
VOICE_DIR = Path("data/processed/voice_clone")
VOICE_ID_FILE = VOICE_DIR / "voice_id.json"
OUT_DIR = Path("data/processed/voice_out")

# eleven_v3 is the only model in the account that lists Punjabi as well as Hindi.
MODEL_ID = "eleven_v3"
FALLBACK_MODEL_ID = "eleven_multilingual_v2"


class Speaker(Protocol):
    def say(self, text: str, out_path: Path) -> Path: ...


@dataclass
class Voice:
    voice_id: str
    name: str


def _key() -> str:
    load_dotenv()
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY missing from .env")
    return key


def _multipart(fields: dict[str, str], files: list[Path]) -> tuple[bytes, str]:
    """Build a multipart body without pulling in `requests` just for this."""
    boundary = "----pardeepself" + os.urandom(8).hex()
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for path in files:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
            f'filename="{path.name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        )
        parts.append(path.read_bytes())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def clone(clips: list[Path], name: str = "Pardeep") -> Voice:
    """Create the cloned voice. This is the step that uploads his audio."""
    if not clips:
        raise RuntimeError("no reference clips — run `python -m src.voice.run refs` first")
    body, content_type = _multipart(
        {
            "name": name,
            "description": "Personal digital twin voice, Punjabi/Hindi/English code-switching.",
        },
        clips,
    )
    req = urllib.request.Request(
        f"{API}/voices/add",
        data=body,
        headers={"xi-api-key": _key(), "Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    voice = Voice(voice_id=data["voice_id"], name=name)
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    VOICE_ID_FILE.write_text(json.dumps(voice.__dict__, indent=2), encoding="utf-8")
    return voice


def saved_voice() -> Voice | None:
    if not VOICE_ID_FILE.exists():
        return None
    return Voice(**json.loads(VOICE_ID_FILE.read_text(encoding="utf-8")))


def delete(voice_id: str) -> bool:
    """Remove the clone from their servers. The trade is reversible."""
    req = urllib.request.Request(
        f"{API}/voices/{voice_id}", headers={"xi-api-key": _key()}, method="DELETE"
    )
    try:
        urllib.request.urlopen(req, timeout=60)
        VOICE_ID_FILE.unlink(missing_ok=True)
        return True
    except urllib.error.HTTPError:
        return False


class ElevenLabs:
    def __init__(self, voice: Voice | None = None, model_id: str = MODEL_ID):
        self.voice = voice or saved_voice()
        if self.voice is None:
            raise RuntimeError("no cloned voice yet — run `python -m src.voice.run clone`")
        self.model_id = model_id

    def pcm(self, text: str, sample_rate: int = 24000) -> bytes:
        """Raw 16-bit PCM, so live playback needs no mp3 decoder.

        Decoding mp3 in-process would mean PyAV, and importing torch first
        replaces the FFmpeg DLLs it binds to — the same conflict the ingest
        loader works around with a subprocess. Asking for PCM avoids it.
        """
        return self._request(text, accept="audio/pcm", output_format=f"pcm_{sample_rate}")

    def say(self, text: str, out_path: Path) -> Path:
        audio = self._request(text, accept="audio/mpeg", output_format=None)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio)
        return out_path

    def _request(self, text: str, accept: str, output_format: str | None) -> bytes:
        payload = {
            "text": text,
            "model_id": self.model_id,
            # his speech is fragmented and fast; a lower stability keeps that
            # rather than smoothing it into a newsreader
            "voice_settings": {"stability": 0.35, "similarity_boost": 0.85, "style": 0.4},
        }
        url = f"{API}/text-to-speech/{self.voice.voice_id}"
        if output_format:
            url += f"?output_format={output_format}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "xi-api-key": _key(),
                "Content-Type": "application/json",
                "Accept": accept,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if self.model_id != FALLBACK_MODEL_ID:
                # v3 is newer and may be gated on some accounts; multilingual_v2
                # still covers Hindi, just not Punjabi
                self.model_id = FALLBACK_MODEL_ID
                return self._request(text, accept, output_format)
            raise RuntimeError(f"speech failed: {exc.code} {detail}") from exc
