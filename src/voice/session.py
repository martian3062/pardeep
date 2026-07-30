"""A recording session: the twin asks, he talks, one clip per answer.

Reading a script produces read speech — flatter prosody, careful diction, none of
the mid-sentence restarts the persona card describes. So the session is a
conversation instead. Questions come from his own archive, so he is recalling
real things rather than performing, and the answers arrive at conversational
length without being asked for.

Two things the session must guarantee, because the archive already failed at
both: enough total speech, and enough BANDWIDTH. Every call recording in the
corpus is a 48kHz container holding roughly 1.6kHz of real audio, which is why
32 hours of it cannot train a voice. Each clip is measured on arrival and
rejected while he is still sitting there, not after the session ends.
"""
from __future__ import annotations

import json
import random
import sqlite3
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

CLIP_DIR = Path("data/processed/voice_session")
DB_PATH = Path("data/processed/voice_session.db")

TARGET_MINUTES = 45.0  # 30 is usable, 60 is good
MIN_CLIP_SEC = 3.0
MAX_CLIP_SEC = 90.0

# What a voice model needs and the archive did not have.
MIN_BANDWIDTH_HZ = 6000.0
MIN_RMS = 0.01  # too far from the mic
MAX_CLIP_RATIO = 0.005  # more than 0.5% of samples at full scale = clipping

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    question_id TEXT NOT NULL,
    question TEXT NOT NULL,
    lang TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    seconds REAL NOT NULL,
    sample_rate INTEGER NOT NULL,
    bandwidth_hz REAL,
    rms REAL,
    clip_ratio REAL,
    accepted INTEGER NOT NULL,
    reason TEXT,
    transcript TEXT
);
CREATE INDEX IF NOT EXISTS idx_clips_accepted ON clips(accepted);
"""


# ---------------------------------------------------------------- questions

# Each question targets a language and a register. A voice model only reproduces
# what it heard, so a session of calm Hindi produces a twin that can only be
# calmly Hindi.
QUESTION_BANK: list[dict] = [
    # --- Hindi / Hinglish, storytelling: long, relaxed, natural ---
    {"lang": "hi", "register": "story", "text": "Aaj ka din kaisa gaya? Poora bata, subah se leke abhi tak."},
    {"lang": "hi", "register": "story", "text": "Koi ek trip yaad hai jo sabse acha raha ho? Kahan gaye the, kaun kaun tha?"},
    {"lang": "hi", "register": "story", "text": "College ka pehla din kaisa tha? Kya kya hua?"},
    {"lang": "hi", "register": "story", "text": "{friend} ke saath ka koi funny kissa suna."},
    {"lang": "hi", "register": "story", "text": "Sabse yaadgar birthday kaunsa tha? Kya kiya tha us din?"},
    {"lang": "hi", "register": "story", "text": "Bachpan ka ghar kaisa tha? Describe kar de poora."},
    {"lang": "hi", "register": "story", "text": "Koi hackathon ya competition ka experience bata, shuru se end tak."},
    {"lang": "hi", "register": "story", "text": "Aaj tak ka sabse lamba safar kaunsa tha? Kaise gaya?"},
    # --- Hindi, explaining: measured, slightly formal ---
    {"lang": "hi", "register": "explain", "text": "Apna project kisi ko samjha jo technical nahi hai. Kya banaya hai aur kyu."},
    {"lang": "hi", "register": "explain", "text": "AI aur machine learning kya hai, apne shabdon me samjha."},
    {"lang": "hi", "register": "explain", "text": "Deploy karna kya hota hai? Ek non-tech dost ko samjha."},
    {"lang": "hi", "register": "explain", "text": "Apna typical din ka routine bata, ek ek cheez."},
    {"lang": "hi", "register": "explain", "text": "Placement ka process kaise chalta hai? Poora batao."},
    # --- Hindi, annoyed / venting: the register calls need most ---
    {"lang": "hi", "register": "annoyed", "text": "Kis cheez pe sabse zyada gussa aata hai? Khul ke bol."},
    {"lang": "hi", "register": "annoyed", "text": "College admin ka koi aisa kaam jo bilkul bekar laga ho? Vent kar."},
    {"lang": "hi", "register": "annoyed", "text": "Koi banda jo bol ke muker gaya ho — kya hua tha?"},
    {"lang": "hi", "register": "annoyed", "text": "Deploy fail hone pe kaisa lagta hai? Bol jo mann me aata hai."},
    # --- Hindi, excited / happy ---
    {"lang": "hi", "register": "happy", "text": "Koi aisi cheez jo hui ho aur tu bohot khush hua ho — bata."},
    {"lang": "hi", "register": "happy", "text": "Agar kal 10 lakh mil jaye to kya karega? Detail me bata."},
    {"lang": "hi", "register": "happy", "text": "Apna dream job ya dream project kya hai? Excited ho ke bata."},
    # --- Punjabi ---
    {"lang": "pa", "register": "story", "text": "ਯਾਰ ਪਿਛਲੇ ਹਫ਼ਤੇ ਕੀ ਕੀ ਹੋਇਆ? ਸਾਰਾ ਦੱਸ।"},
    {"lang": "pa", "register": "story", "text": "ਘਰ ਦੀ ਕੋਈ ਗੱਲ ਸੁਣਾ — ਪਿੰਡ, ਰਿਸ਼ਤੇਦਾਰ, ਕੋਈ ਫੰਕਸ਼ਨ।"},
    {"lang": "pa", "register": "story", "text": "ਸਕੂਲ ਦੇ ਦਿਨਾਂ ਦੀ ਕੋਈ ਗੱਲ ਦੱਸ।"},
    {"lang": "pa", "register": "annoyed", "text": "ਕਿਹੜੀ ਗੱਲ ਤੇ ਸਭ ਤੋਂ ਵੱਧ ਖਿਝ ਆਉਂਦੀ ਹੈ? ਖੁੱਲ੍ਹ ਕੇ ਬੋਲ।"},
    {"lang": "pa", "register": "explain", "text": "ਆਪਣਾ ਕੰਮ ਕੀ ਹੈ, ਪੰਜਾਬੀ ਵਿੱਚ ਸਮਝਾ।"},
    {"lang": "pa", "register": "happy", "text": "ਕੋਈ ਚੰਗੀ ਖ਼ਬਰ ਜੋ ਹਾਲ ਵਿੱਚ ਮਿਲੀ ਹੋਵੇ — ਦੱਸ।"},
    {"lang": "pa", "register": "story", "text": "{friend} ਨਾਲ ਪਿਛਲੀ ਵਾਰ ਕੀ ਗੱਲ ਹੋਈ ਸੀ?"},
    # --- English ---
    {"lang": "en", "register": "explain", "text": "Explain what you're building right now, in English, like a job interview."},
    {"lang": "en", "register": "explain", "text": "Walk me through your resume out loud. Education, projects, skills."},
    {"lang": "en", "register": "story", "text": "Tell me about a time something went badly wrong in a project."},
    {"lang": "en", "register": "explain", "text": "What are you actually good at, and what are you weak at? Be honest."},
    {"lang": "en", "register": "story", "text": "Describe your city to someone who has never been to India."},
    {"lang": "en", "register": "happy", "text": "What's the most exciting thing happening in AI right now, in your view?"},
    # --- deliberate code-switching, which no scripted corpus covers ---
    {"lang": "mixed", "register": "explain", "text": "Apna project English aur Hindi mix kar ke samjha — jaise dost ko samjhata hai."},
    {"lang": "mixed", "register": "story", "text": "Kal kya kiya? Jaise WhatsApp pe likhta hai waise hi bol, mix kar ke."},
    {"lang": "mixed", "register": "annoyed", "text": "Koi technical problem jo phasa raha ho — gussa mila ke bata, jo language aaye."},
    {"lang": "mixed", "register": "story", "text": "{friend} ko phone kar ke kya bologa agar koi badi khabar ho? Bol ke dikha."},
    # --- short, fast turns: chat-length utterances the long ones will not give ---
    {"lang": "mixed", "register": "short", "text": "Teen chhoti cheezein bol jo aaj karni hain."},
    {"lang": "mixed", "register": "short", "text": "Haan ya na me jawab de: chai ya coffee, subah ya raat, akela ya group? Har ek pe ek line."},
    {"lang": "hi", "register": "short", "text": "Apne pasand ke 5 khane gino, ek ek line me."},
    {"lang": "en", "register": "short", "text": "Count from one to twenty, normally."},
    {"lang": "hi", "register": "short", "text": "Ek se bees tak ginti bol, Hindi me."},
    # --- numbers and spellings, which TTS models get wrong without examples ---
    {"lang": "mixed", "register": "short", "text": "Apna phone number bol — koi bhi 10 digit, asli nahi. Aur ek date, aur ek time."},
    {"lang": "en", "register": "short", "text": "Spell out your name letter by letter, then say it normally."},
]

# Registers a session should not finish without.
REQUIRED_REGISTERS = ("story", "explain", "annoyed", "happy", "short")
REQUIRED_LANGS = ("hi", "pa", "en", "mixed")


def _friends(limit: int = 8) -> list[str]:
    """Real names from the Mind Model, so questions are about actual people."""
    try:
        from ..twin.identity import load

        people = load().people
        names = [str(p.get("name", "")).split("(")[0].strip() for p in people]
        return [n for n in names if n][:limit]
    except Exception:
        return []


def build_questions(seed: int | None = None) -> list[dict]:
    """The session plan: shuffled, but with language and register coverage first."""
    rng = random.Random(seed)
    friends = _friends() or ["apne dost"]

    questions = []
    for i, q in enumerate(QUESTION_BANK):
        text = q["text"].replace("{friend}", rng.choice(friends))
        questions.append(
            {"id": f"q{i:03d}", "text": text, "lang": q["lang"], "register": q["register"]}
        )

    # one of each required combination up front, then everything else shuffled,
    # so a session that ends early still covers the range
    head: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for lang in REQUIRED_LANGS:
        for reg in REQUIRED_REGISTERS:
            match = next(
                (q for q in questions if q["lang"] == lang and q["register"] == reg and q["id"] not in {h["id"] for h in head}),
                None,
            )
            if match:
                head.append(match)
                seen.add((lang, reg))
    tail = [q for q in questions if q["id"] not in {h["id"] for h in head}]
    rng.shuffle(tail)
    return head + tail


# ------------------------------------------------------------------ quality


@dataclass
class Quality:
    seconds: float
    sample_rate: int
    bandwidth_hz: float
    rms: float
    clip_ratio: float

    @property
    def problem(self) -> str:
        if self.seconds < MIN_CLIP_SEC:
            return f"too short ({self.seconds:.1f}s) — say a bit more"
        if self.seconds > MAX_CLIP_SEC:
            return f"very long ({self.seconds:.0f}s) — fine, but shorter answers are easier to use"
        if self.rms < MIN_RMS:
            return "too quiet — move closer to the mic"
        if self.clip_ratio > MAX_CLIP_RATIO:
            return "clipping — too loud, back off or lower input gain"
        if self.bandwidth_hz < MIN_BANDWIDTH_HZ:
            return (
                f"only {self.bandwidth_hz:.0f}Hz of real audio — this is the flaw that made "
                "32 hours of call recordings unusable. Check the mic is not a phone/bluetooth headset."
            )
        return ""

    @property
    def ok(self) -> bool:
        return not self.problem or self.seconds > MAX_CLIP_SEC  # long is a warning, not a reject


def measure(path: Path) -> Quality:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
        channels = w.getnchannels()

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    full = float(np.iinfo(dtype).max)
    x = x / full

    seconds = len(x) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    clip_ratio = float(np.mean(np.abs(x) > 0.98)) if x.size else 0.0
    return Quality(
        seconds=seconds,
        sample_rate=sr,
        bandwidth_hz=bandwidth(x, sr),
        rms=rms,
        clip_ratio=clip_ratio,
    )


def bandwidth(x: np.ndarray, sr: int, drop_db: float = 35.0) -> float:
    """Highest frequency still carrying energy, not what the container claims."""
    n = 1 << 13
    if x.size < n or not sr:
        return 0.0
    win = np.hanning(n)
    spectra = []
    for start in range(0, x.size - n, n):
        spectra.append(np.abs(np.fft.rfft(x[start : start + n] * win)))
        if len(spectra) >= 40:
            break
    if not spectra:
        return 0.0
    db = 20 * np.log10(np.mean(spectra, axis=0) + 1e-12)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    above = np.where(db > db.max() - drop_db)[0]
    return float(freqs[above[-1]]) if above.size else 0.0


# -------------------------------------------------------------------- store


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def save_clip(
    conn: sqlite3.Connection,
    audio: bytes,
    question: dict,
) -> tuple[int, Quality, str]:
    """Write the wav, measure it, record the verdict. Returns (id, quality, problem)."""
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = CLIP_DIR / f"{question['id']}_{stamp}.wav"
    path.write_bytes(audio)

    q = measure(path)
    problem = q.problem
    accepted = q.ok
    if not accepted:
        path.rename(path.with_suffix(".rejected.wav"))
        path = path.with_suffix(".rejected.wav")

    cur = conn.execute(
        """INSERT INTO clips (created_at, question_id, question, lang, path, seconds,
                              sample_rate, bandwidth_hz, rms, clip_ratio, accepted, reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            question["id"],
            question["text"],
            question["lang"],
            str(path),
            round(q.seconds, 2),
            q.sample_rate,
            round(q.bandwidth_hz, 1),
            round(q.rms, 4),
            round(q.clip_ratio, 5),
            int(accepted),
            problem or None,
        ),
    )
    conn.commit()
    return int(cur.lastrowid), q, problem


def progress(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT lang, seconds FROM clips WHERE accepted = 1"
    ).fetchall()
    total = sum(s for _, s in rows)
    per_lang: dict[str, float] = {}
    for lang, s in rows:
        per_lang[lang] = per_lang.get(lang, 0.0) + s
    rejected = conn.execute("SELECT count(*) FROM clips WHERE accepted = 0").fetchone()[0]
    return {
        "clips": len(rows),
        "minutes": round(total / 60, 1),
        "target_minutes": TARGET_MINUTES,
        "pct": round(min(100.0, 100 * total / 60 / TARGET_MINUTES), 1),
        "per_language_minutes": {k: round(v / 60, 1) for k, v in per_lang.items()},
        "rejected": int(rejected),
    }


def answered_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT question_id FROM clips WHERE accepted = 1")}


def export_manifest(conn: sqlite3.Connection, out: Path | None = None) -> Path:
    """A training manifest: path, transcript slot, language, duration."""
    out = out or CLIP_DIR / "manifest.json"
    rows = conn.execute(
        """SELECT path, question, lang, seconds, transcript FROM clips
           WHERE accepted = 1 ORDER BY id"""
    ).fetchall()
    out.write_text(
        json.dumps(
            [
                {
                    "audio": p,
                    "prompt": q,
                    "lang": lang,
                    "seconds": sec,
                    "transcript": tr or "",
                }
                for p, q, lang, sec, tr in rows
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out
