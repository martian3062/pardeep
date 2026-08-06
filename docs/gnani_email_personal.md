# Email — Gnani.ai, personal Indic voice

Personal project only. No VECTRA, nothing commercial, no EU data subjects.

**Correction from the earlier draft:** Prisma and "2.5" are real — `gnani-ai/gnani` on
Hugging Face ships **Gnani Prisma v2.5** (STT) and **Gnani Timbre v2.0** (TTS), Apache-2.0,
API-only. The earlier draft said they didn't exist. They do; that question is now removed
and replaced with the two that actually matter.

---

## Their published stack

| model | what it is | relevance here |
| --- | --- | --- |
| **Prisma v2.5** | STT — 10 Indic languages + Hinglish code-switching, 14M hours of telephonic audio | replaces Whisper, which loses spoken years |
| **Timbre v2.0** | TTS — 10+ Indic languages; 4 stock voices (Pranav, Kaveri, Shubhra, Deepak) | the cloning question — stock voices are documented, custom cloning is not |
| **Vachana** | unified STT+TTS pipeline wrapper | convenience layer over both |
| **Warp** | speech-to-speech | the live avatar loop later |
| **Inya VoiceOS** | 5B voice-to-voice, research preview; 14B announced | same, at foundation-model level |
| **Aion**, **Evon** | language models | not needed — text layer is already Sarvam-M |

Two things the Hugging Face card does **not** say, and both are load-bearing:
**Punjabi is not in the listed language set**, and **custom voice cloning is not
documented for Timbre** — only four stock voices. Press coverage claims both. That
contradiction is the first thing to resolve.

---

## The email

**Subject:** Evaluation credits — Prisma v2.5 and Timbre v2.0 for a personal Punjabi voice project

Hi <name>,

I'm building a personal digital twin — trained on my own chats and calls, speaking in a
cloned version of my voice across Hindi, Punjabi and English, code-switching mid-sentence
the way I actually talk. Personal use, not a product.

**I already went Indian where it counted.** The text layer is a fine-tuned **Sarvam-M 24B**.
I benchmarked it against Qwen3-4B and Gemma-4-E4B and it held Hinglish and Punjabi
code-switching where the Western models flattened it. The voice layer is the missing half,
and that's why I'm writing.

**Punjabi is where I'm stuck.** I surveyed the open Indic TTS field and disqualified all of
it: GPT-SoVITS (no Punjabi at all), XTTS-v2 (Hindi broken in its own tokenizer, Coqui shut
down), Chatterbox (poor Hindi, open issue), IndicF5 (needs a Roman→Devanagari front-end
built on fairseq, archived March 2026). I'm on ElevenLabs today only because it was the
single engine I could find supporting Punjabi at all — not a comfortable place to be.

**What I'd like to evaluate**, in priority order:

1. **Timbre v2.0** — Punjabi output and custom voice cloning
2. **Prisma v2.5** — Hinglish and Punjabi transcription, especially spoken numerals
3. **Warp / Inya VoiceOS** — later, for a real-time conversational loop

**Credits requested:**

| model | amount | what it covers |
| --- | --- | --- |
| Timbre v2.0 | **300 min synthesis + 2 voice clones** | ~1,200 utterances across 4 scripts and 5 speaking registers, with reruns |
| Prisma v2.5 | **2,000 min transcription** | a 553-call personal archive plus a re-run of already-labelled audio for a like-for-like WER comparison |
| Warp | **120 min** | latency and turn-taking probe, if available |

I don't know how your credits meter — please map that to your units. If it works well,
ongoing personal usage is small: roughly 60–90 minutes a month.

**What you get back.** I measure rather than judge by ear — that's how those four engines
got eliminated, each for a specific reason I can point to. You'd get speaker-similarity
scores against my real reference clips, per-script round-trip intelligibility, and a WER
comparison against Whisper large-v3 on Hinglish and Punjabi. Published honestly either way,
and you're welcome to review the methodology first and tell me if it's unfair to your models.
**Punjabi and code-switched results are barely reported publicly** — that's where your
Indic advantage should be strongest.

**Four questions:**

1. Does **Timbre v2.0 support Punjabi (Gurmukhi)** output? The Hugging Face card lists 10+
   Indic languages but doesn't name Punjabi; press coverage does. Which is right?
2. Does Timbre support **custom voice cloning**, or only the four stock voices (Pranav,
   Kaveri, Shubhra, Deepak) documented on the card?
3. Minimum reference-audio **duration and bandwidth** for a clone? I have 32 hours of my own
   archived speech that turned out unusable — 48 kHz files carrying about 1.6 kHz of real
   audio once the mobile network codec had finished with it. I'd rather know your floor than
   rediscover it.
4. Is **Roman-script Hinglish** handled natively, or does it need transliteration first?
   That's exactly where IndicF5 died for me.

One more: someone mentioned a model called **Vidma** — I can't match it to your public
material. What is it?

Thanks,
Pardeep Singh

---

## Notes before sending

**Fill in:** recipient name and one line on how you know them.

**The two questions that decide everything** are 1 and 2. If Timbre has no Punjabi or no
custom cloning, the core project doesn't move and only the Prisma/STT half is worth
credits. Ask them first; everything else is detail.

**API-only matters.** Despite the Apache-2.0 licence, `gnani-ai/gnani` ships a client,
not weights — inference is hosted and needs `GNANI_API_KEY`. Fine for evaluation. Worth
knowing before building anything on it that you wanted to run locally.

**Left out deliberately:** VECTRA, Eraya, anything commercial, and the data-residency
question. No EU data subjects here, so India-only processing raises nothing — which is
exactly what makes this a clean, small first ask. Keep the enterprise conversation
separate and have it *after* you have working evidence from this track.
