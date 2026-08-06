# Email — Gnani.ai, personal Indic voice (short version)

Personal project only. No VECTRA, nothing commercial, no EU data subjects — so their
India-only processing raises nothing, which is what makes this a clean small ask.

---

## The email

**Subject:** Evaluation credits — Vachana TTS for Punjabi voice cloning

Hi <name>,

I'm building a personal digital twin — trained on my own chats and calls, speaks in a
cloned version of my voice, in Hindi, Punjabi and English, code-switching mid-sentence
the way I actually talk. Personal use, not a product.

**I already chose Indian models where it mattered.** The text layer is a fine-tuned
**Sarvam-M 24B** — I benchmarked it against Qwen3-4B and Gemma-4-E4B and it was
clearly better on Hinglish and Punjabi code-switching, which the Western models flatten.

**The voice layer is where I'm stuck.** I surveyed the open Indic TTS field and
disqualified all of it: GPT-SoVITS (no Punjabi at all), XTTS-v2 (Hindi broken in its
own tokenizer, Coqui shut down), Chatterbox (poor Hindi, open issue), IndicF5 (needs a
Roman→Devanagari front-end built on fairseq, archived March 2026). I'm on ElevenLabs
today only because it was the single engine I could find that supports Punjabi at all.

**Vachana TTS** looks like the first Indic-native alternative with real cloning — and
**Inya VoiceOS** is interesting to me later for the live conversational loop, since
voice-to-voice removes the STT→TTS hops entirely.

**What I'd like to test:** Punjabi (Gurmukhi) cloning quality, Roman-script Hinglish
input without transliteration, mid-clause code-switching, and streaming latency.

**Credits needed: ~300 minutes of synthesis and 2 voice clones.** That covers about
1,200 evaluation utterances across four scripts and several speaking registers, with
room to rerun. Please map it to your units. If it works well I'd want ongoing access —
my actual usage after that is small, maybe 60–90 minutes a month.

**In return**, I measure properly rather than by ear — that's how those four engines got
eliminated. You'd get speaker-similarity scores against my real reference clips and
round-trip intelligibility per script, published honestly either way. Punjabi and
code-switched results are barely reported publicly.

Three questions:

- Does Vachana TTS cover **Punjabi (Gurmukhi)** for cloning, or is Indic mainly Hindi
  and the southern languages?
- Roman-script Hinglish — native, or does it need transliteration first?
- Minimum reference-audio **duration and bandwidth** for a clone? I have 32 hours of my
  own archived speech that turned out unusable — 48 kHz files carrying ~1.6 kHz of real
  audio once the mobile codec was done with it. I'd rather know your floor than
  rediscover it.

Also, a **"Prisma"** model and something numbered **"2.5"** came up in conversation but
I can't match them to your public material — what do those refer to?

Thanks,
Pardeep Singh

---

## Notes

### The credit number, justified

| item | amount | basis |
| --- | --- | --- |
| TTS synthesis | 300 min | ~1,200 utterances × ~15 s: 4 scripts × 5 registers × ~60 lines, ×2 for reruns |
| voice clones | 2 | one from studio-clean audio, one from a phone recording — tests their bandwidth floor |
| ongoing (if it passes) | 60–90 min/month | actual personal usage; stated so they can size a real account, not just a trial |

**Deliberately left out:** VECTRA, Eraya, anything commercial, and the data-residency
question. Keep the enterprise conversation separate and have it *after* you have working
evidence from this track.

**Not verified, so asked rather than claimed:** no "Prisma" and no "2.5" appear in
Gnani's public material. Their published stack is Inya VoiceOS (5B voice-to-voice),
Vachana STT and Vachana TTS, with a 14B model announced.

**If you want the STT side too:** 553 unprocessed call recordings ≈ 1,700 min, where
Whisper currently loses spoken years ("2017" → "अगा तत्रा"). Left out to keep the ask
small — a bigger number goes to a committee, this one gets approved by one person.
