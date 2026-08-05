# Email draft — Gnani.ai credit request

Two versions. The short one is the one to send; the long one is if they ask for detail
or if you are forwarding to someone who has not had the earlier conversation.

The short version is built on one principle: **vendors approve narrow, specific asks
far more readily than broad ones**. It leads with what they get, states exactly what is
needed, and puts the data-residency question in the first exchange rather than letting
it surface at legal review three weeks in.

---

## Version 1 — the one to send

**Subject:** Evaluation credits — benchmarking Inya VoiceOS and Vachana against our production baselines

Hi <name>,

Following up on our conversation. I'd like to formally request evaluation credits, and
I've scoped it tightly so it's easy to say yes to.

**Context.** I run two systems that are already built and measured:

- **Eraya** — an AI voice qualification agent in production for VECTRA International BV
  (Brussels), EU AI Act Article 50 compliant, 621 tests. It runs a Retell + Claude
  cascade today.
- A **personal Indic voice project** — Punjabi, Hindi and code-switched Hinglish, with a
  measurement harness that has already disqualified GPT-SoVITS, XTTS-v2, Chatterbox and
  IndicF5 on the record, each for a specific documented reason.

**Why Gnani.** Inya VoiceOS being genuinely voice-to-voice removes two hops from our
latency budget that a cascade cannot. And Vachana TTS is the first Indic cloning model
I've found with a credible Punjabi story — ElevenLabs is currently the *only* engine,
open or closed, that we found supporting Punjabi at all, which is not a comfortable
place to be.

**What I'm asking for.** Roughly **5,600 minutes (~93 hours) of audio** across the
stack, plus API access and docs:

| | |
|---|---|
| Inya VoiceOS | ~2,100 min — 500 pilot calls + 200 benchmark calls |
| Vachana TTS | ~500 min synthesis + 1 voice clone |
| Vachana STT | ~1,700 min — a 553-call archived corpus |
| retries / re-benchmarks | +30% |

I don't know how your credits meter, so please map that to your units and tell me if the
shape is wrong.

**What you get.** A written benchmark report placing your models against Whisper
large-v3, ElevenLabs eleven_v3 and the open Indic TTS field, using the same harness that
produced the disqualifications above — published honestly either way. If the voice-agent
track passes, a named EU reference deployment.

**One thing to resolve first.** Your material says Inya VoiceOS is built, trained and
deployed entirely within India, with full data sovereignty. For the personal Indic work
that's fine. For VECTRA — a Belgian entity processing EU callers' voice — India-only
processing raises a transfer question I'd need answered before a pilot. So:

1. Is there an EU or on-premise deployment path for Inya VoiceOS?
2. If not, can we start with the Indic track only (no EU data subjects), which is
   unblocked either way?

A few quick technical questions while you're there:

- Does Vachana TTS support **Punjabi (Gurmukhi)** for cloning, and Roman-script Hinglish
  input natively? (Transliteration front-ends are exactly where IndicF5 died for us —
  its dependency chain runs through fairseq, archived in March 2026.)
- Minimum reference-audio duration **and bandwidth** for a voice clone? I ask because I
  have 32 hours of archived speech that turned out unusable: 48 kHz containers carrying
  ~1.6 kHz of real audio after the mobile codec. I'd rather know your floor than
  rediscover it.
- Which model did the "2.5" reference point to? Public material names Inya VoiceOS (5B),
  Vachana STT/TTS and a 14B roadmap — I want to make sure I'm benchmarking the right one.
- Is there a model named **Prisma** in your stack? It came up in conversation but I
  can't find it publicly.

Happy to sign an NDA, and happy to share the benchmark methodology in advance so you
can tell me if it's unfair to your models before we run it.

Best,
Pardeep Singh
Zoho Systems · VECTRA International BV

---

## Version 2 — the longer one (only if they ask)

Same content, plus:

- The full evaluation plan with pass/fail gates per track (PRD §4)
- The complete disqualification table for the open Indic field, with reasons
- Our measured baselines: Whisper WER behaviour on spoken numerals, the ~1.6 kHz
  bandwidth finding, the ElevenLabs round-trip results across four scripts
- The Article 50 architecture for Eraya, if the EU reference deployment interests them

Attach `gnani_evaluation_prd.md` rather than pasting it.

---

## Notes before you send

**Fill in first**
- Recipient name and how you know them
- Whether an NDA already exists
- Confirm the call-volume assumptions (500 pilot / 200 benchmark / 553 archived) match
  what you actually intend — the credit ask is derived from them, so if they are wrong
  the number is wrong

**Two things I could not verify, and did not invent**
- **"Prisma"** — no such Gnani model appears in public material. Asked as a question
  rather than assumed.
- **"2.5"** — no Gnani model numbered 2.5 found. Also asked. If it turns out to be a
  different vendor entirely, the PRD's comparison tables still stand; only the vendor
  name changes.

**Why the residency question is in the first email**
It is the item most likely to kill the VECTRA track, and it is cheaper to learn now
than after a pilot is scoped and approved internally. Putting it early also signals
that the evaluation is serious, which tends to help rather than hurt a credits ask.
