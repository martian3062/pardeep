"""Microphone → speech → text, entirely on this machine.

Nothing here leaves the PC: silero-VAD decides when he has started and stopped
talking, and faster-whisper transcribes locally. Only the resulting text ever
reaches a cloud model, and only if the router sends it there.

Turn-taking is the whole problem. Fixed-length recording either cuts him off
mid-sentence or leaves dead air after it; the pauses inside his speech are also
long, because he talks in bursts with gaps. So a turn ends on sustained silence,
not on the first quiet frame.
"""
from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np

from .numbers import recover_years

SAMPLE_RATE = 16000  # what both silero-VAD and Whisper want
FRAME_MS = 32
FRAME = SAMPLE_RATE * FRAME_MS // 1000

# He pauses mid-thought constantly — "तो फिर उसके बाद... वो ही... मतलब..." — so a
# short silence must not be read as the end of his turn.
SILENCE_TO_END_MS = 900
MIN_SPEECH_MS = 400
MAX_TURN_SEC = 30.0
SPEECH_PROB = 0.5


@dataclass
class Utterance:
    audio: np.ndarray
    seconds: float


class Ears:
    def __init__(self, device: int | None = None, model_size: str = "large-v3"):
        self.device = device
        self._vad = None
        self._whisper = None
        self.model_size = model_size

    @property
    def vad(self):
        if self._vad is None:
            import torch

            model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
            self._vad = model
        return self._vad

    @property
    def whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel

            try:
                self._whisper = WhisperModel(
                    self.model_size, device="cuda", compute_type="int8_float16"
                )
            except Exception:
                self._whisper = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._whisper

    def _speech_prob(self, frame: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            return float(self.vad(torch.from_numpy(frame), SAMPLE_RATE).item())

    def listen(self) -> Utterance | None:
        """Block until he speaks, then until he stops. None if interrupted."""
        import sounddevice as sd

        blocks: queue.Queue = queue.Queue()

        def on_audio(indata, _frames, _time, _status):
            blocks.put(indata[:, 0].copy())

        collected: list[np.ndarray] = []
        speaking = False
        silence_ms = 0
        speech_ms = 0

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME,
            device=self.device,
            callback=on_audio,
        ):
            while True:
                try:
                    frame = blocks.get(timeout=1.0)
                except queue.Empty:
                    continue
                if frame.shape[0] != FRAME:  # silero needs an exact window
                    continue
                is_speech = self._speech_prob(frame) >= SPEECH_PROB

                if is_speech:
                    speaking = True
                    silence_ms = 0
                    speech_ms += FRAME_MS
                elif speaking:
                    silence_ms += FRAME_MS

                if speaking:
                    collected.append(frame)

                if speaking and silence_ms >= SILENCE_TO_END_MS:
                    if speech_ms >= MIN_SPEECH_MS:
                        break
                    # a cough or a door: discard and keep waiting
                    collected, speaking, silence_ms, speech_ms = [], False, 0, 0
                if collected and len(collected) * FRAME_MS / 1000 > MAX_TURN_SEC:
                    break

        if not collected:
            return None
        audio = np.concatenate(collected)
        return Utterance(audio=audio, seconds=len(audio) / SAMPLE_RATE)

    def transcribe(self, utt: Utterance) -> tuple[str, str]:
        """Return (text, language). Empty text means nothing intelligible."""
        segments, info = self.whisper.transcribe(
            utt.audio,
            beam_size=5,
            vad_filter=True,
            # the same guards the archive pass needed: without them Whisper
            # invents text over noise and loops on it
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        # Whisper writes spoken years as words, which the date filter cannot see.
        # Without this, "2017 june me kya kar raha tha" asked aloud reaches
        # retrieval with no year in it at all.
        return recover_years(text), info.language


def play(pcm: bytes, sample_rate: int = 24000) -> None:
    """Play raw 16-bit PCM through the default output."""
    import sounddevice as sd

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio, sample_rate)
    sd.wait()


def list_devices() -> str:
    import sounddevice as sd

    return str(sd.query_devices())
