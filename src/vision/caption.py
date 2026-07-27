"""Local VLM captioning — photos never leave the PC.

A 4B vision model in 4-bit fits the 6GB laptop GPU alongside nothing else, so
captioning runs while the VM does other work. The prompt asks for a plain factual
description rather than a poetic one: these captions are retrieval keys for
"show me the wedding pics", not prose.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

console = Console()

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

PROMPT = (
    "Describe this personal photo in one or two plain sentences. "
    "Say what is happening, what the place looks like, how many people are visible, "
    "and any readable text or occasion. Do not guess names. Be factual, not poetic. "
    "Describe only what you can see: never mention the file, the folder, the camera, "
    "or how the photo was taken, and do not say what is absent."
)

# Folders named for the device rather than the occasion say nothing a caption
# should repeat. Passing "camera" as a hint made the model end almost every
# caption with "as suggested by the 'camera' folder name" — thousands of
# memories sharing one phrase, which blunts retrieval.
_USELESS_HINTS = {
    "camera",
    "camera roll",
    "dcim",
    "sent",
    "photo",
    "photos",
    "images",
    "pictures",
    "100nikon",
    "new folder",
    "whatsapp images",
}

# Long side; the vision encoder cost grows with pixel count and 896px is enough
# to read a signboard while staying inside 6GB.
MAX_SIDE = 896

# 90 truncated 10% of captions mid-sentence, and always the richest ones: a
# banner read as "रेल डिस्ट्रिब्यूटर कार्यकारी…" costs several tokens per word, so
# exactly the captions worth having were the ones cut off.
MAX_NEW_TOKENS = 140

# The model keeps reporting what a photo does NOT contain. 247 sentences across
# 931 captions were pure absence — "No other people or readable text are
# visible." alone appeared 75 times — which adds nothing to a memory and makes
# unrelated photos embed alike, blunting retrieval.
_ABSENCE = re.compile(
    r"""(?:^|(?<=[.!?])\s*)          # sentence start
        (?:
            (?:There\s+(?:is|are)\s+)?no\s+(?:other\s+|readable\s+|discernible\s+)?
            (?:people|person|text|labels?|writing|signage|occasion|words?)\b[^.!?]*
          | only\s+one\s+person\s+is\s+(?:visible|present)\b[^.!?]*
          | no\s+\w+\s+or\s+\w+\s+(?:is|are)\s+(?:visible|present|discernible)\b[^.!?]*
        )
        [.!?]\s*""",
    re.IGNORECASE | re.VERBOSE,
)


def strip_absence(text: str) -> str:
    """Drop sentences that only say what is not in the photo."""
    cleaned = _ABSENCE.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # if a caption was nothing but absence, keep the original rather than nothing
    return cleaned if len(cleaned) >= 30 else text.strip()


def _load_image(path: Path):
    from PIL import Image, ImageOps

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # honour camera rotation
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        scale = MAX_SIDE / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return img


class Captioner:
    def __init__(self, model_id: str = MODEL_ID, four_bit: bool = True):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.torch = torch
        # the checkpoint is bf16-native; matching it avoids a needless cast and
        # the overflow risk fp16 carries on vision activations
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant = None
        if four_bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        # No automatic fallback to a second model. A blanket except here treats
        # a VRAM error or a dropped connection as "model unavailable" and starts
        # a fresh multi-GB download of an uncached model — which happened twice
        # over an already-saturated link before this was removed. Fail loudly and
        # let the download be driven deliberately.
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quant,
            dtype=dtype,
            device_map="cuda:0",
        )
        self.model_id = model_id
        self.model.eval()
        precision = "4-bit nf4" if four_bit else str(dtype).replace("torch.", "")
        console.print(f"[green]{self.model_id} loaded ({precision})[/green]")

    def caption(self, path: Path, hint: str = "") -> str | None:
        """Caption one photo, or None if the file is unreadable or OOM survives."""
        try:
            image = _load_image(path)
        except Exception:
            return None

        prompt = PROMPT
        if hint and hint.strip().lower() not in _USELESS_HINTS:
            prompt += (
                f'\nContext, for your understanding only — do not mention it in the caption: '
                f'this photo was filed under "{hint}".'
            )

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            with self.torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=1.05,
                )
            trimmed = out[0][inputs["input_ids"].shape[1] :]
            text = self.processor.decode(trimmed, skip_special_tokens=True).strip()
            return strip_absence(text) if text else None
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                self.torch.cuda.empty_cache()
            return None


def truncated_captions(conn: sqlite3.Connection) -> list[int]:
    """Captions that stopped mid-sentence against the old token cap."""
    return [
        row[0]
        for row in conn.execute(
            "SELECT id, caption FROM photos WHERE caption IS NOT NULL AND dupe_of IS NULL"
        )
        if not row[1].rstrip().endswith((".", "!", "?", '"', "”"))
    ]


def clean_stored_captions(conn: sqlite3.Connection) -> int:
    """Apply strip_absence to captions already written, without touching the GPU."""
    rows = conn.execute(
        "SELECT id, caption FROM photos WHERE caption IS NOT NULL"
    ).fetchall()
    changed = 0
    for photo_id, caption in rows:
        cleaned = strip_absence(caption)
        if cleaned != caption:
            conn.execute("UPDATE photos SET caption = ? WHERE id = ?", (cleaned, photo_id))
            changed += 1
    conn.commit()
    return changed


def run(
    db_path: Path,
    limit: int | None = None,
    min_rank: int = 2,
    four_bit: bool = True,
) -> int:
    from . import photos as ph

    conn = sqlite3.connect(db_path)
    pending = ph.pending_captions(conn, min_rank=min_rank)
    if limit:
        pending = pending[:limit]
    if not pending:
        console.print("[green]Nothing left to caption.[/green]")
        return 0

    console.print(f"Captioning [bold]{len(pending)}[/bold] photos (rank >= {min_rank})")
    cap = Captioner(four_bit=four_bit)

    done = failed = 0
    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("captioning", total=len(pending))
        for photo_id, path_str in pending:
            path = Path(path_str)
            hint = conn.execute(
                "SELECT folder_hint FROM photos WHERE id = ?", (photo_id,)
            ).fetchone()[0]
            text = cap.caption(path, hint or "")
            if text:
                conn.execute(
                    "UPDATE photos SET caption = ?, captioned_at = ? WHERE id = ?",
                    (text, datetime.now().isoformat(timespec="seconds"), photo_id),
                )
                done += 1
            else:
                failed += 1
            # commit as we go: captioning thousands of photos is long enough that
            # a crash three hours in must not lose the work
            if (done + failed) % 25 == 0:
                conn.commit()
            progress.advance(task)

    conn.commit()
    conn.close()
    console.print(f"[green]Captioned {done}[/green], skipped {failed}")
    return done
