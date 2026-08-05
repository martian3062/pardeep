# PRD — Gnani.ai evaluation and credit request

**Author** Pardeep Singh · Zoho Systems, VECTRA International BV
**Date** 3 August 2026
**Status** Draft for vendor discussion — credits not yet requested
**Vendor** Gnani.ai (Bangalore) — Inya VoiceOS · Vachana STT · Vachana TTS

---

## 1 · Why this evaluation

Two shipped systems both hit the same wall, and Gnani.ai is the first vendor whose
stack addresses it directly.

**Eraya** (VECTRA's AI voice qualification agent) is built, deployed, measured, and
carries 621 passing tests. It runs a classic cascade: Retell handles media, our
websocket owns every word, Claude writes the replies. The cascade costs latency at
each hop, and it speaks European languages well but has no Indic depth.

**Pardeep_Self** (personal digital twin, 200 tests) needs a voice model for
**Punjabi**. An adversarial research pass in August 2026 evaluated every open
option and disqualified all of them:

| engine | why it failed |
| --- | --- |
| GPT-SoVITS | no Hindi or Punjabi in any version |
| XTTS-v2 | Hindi broken in its own tokenizer; Coqui shut down |
| Chatterbox | poor Hindi, confirmed open issue |
| IndicF5 | needs a Roman→Devanagari front-end built on **fairseq, archived March 2026** |

The fallback in production today is ElevenLabs `eleven_v3` — the *only* engine found,
open or closed, that lists Punjabi at all. It works, but the voice model lives on a US
vendor's servers and is billed per character.

Separately, a measurement that reframes the whole problem: **32 hours of archived call
audio proved untrainable**. The recordings are 48 kHz containers carrying ~1.6 kHz of
real bandwidth — the mobile network codec discarded everything above it before the
recorder saw it. Volume was never the constraint; bandwidth was.

## 2 · What Gnani.ai brings

| model | claim | which problem it addresses |
| --- | --- | --- |
| **Inya VoiceOS** | 5B **voice-to-voice**, native speech in and out — no STT/TTS layers | Removes two hops from Eraya's latency budget; preserves tone and pauses that a cascade discards |
| **Vachana TTS** | Indic TTS **with voice cloning** | The Punjabi gap no open engine fills |
| **Vachana STT** | Indic speech-to-text | Whisper currently mangles spoken years — "2017" came back as *"अगा तत्रा"*, so date-filtered retrieval silently failed |

Launched at India AI Impact Summit 2026 under the IndiaAI Mission; a 14B successor is
announced.

## 3 · What we have already benchmarked

This is not a greenfield evaluation. Both systems carry measured baselines, so Gnani's
models can be scored against numbers rather than impressions.

### 3.1 Text/reasoning models (already selected, in production)

| model | outcome |
| --- | --- |
| **Sarvam-M 24B** | **chosen** — QLoRA fine-tuned on 5,435 examples of the owner's own messages; best checkpoint at epoch 1.43 |
| Sarvam-1 2B, Qwen3-4B, Gemma-4-E4B | evaluated as laptop-class alternatives, not selected |
| Claude (Haiku/Sonnet) | production reasoning path in both systems |

### 3.2 Image/identity models (evaluated August 2026, 6-agent adversarial pass)

| model | verdict |
| --- | --- |
| **Qwen-Image-2512** | **selected** — Apache-2.0, full bf16 weights, human-realism focus |
| Ideogram 4 | marginally better likeness, but gated FP8-only quants under a non-commercial licence |
| FLUX.2-klein | **disqualified** — observed lightening a South Asian subject's skin |
| FLUX.2-dev | did not complete; ~90 GB |
| Krea 2, Z-Image | ranked below the above on likeness |

### 3.3 Speech models (current state)

| component | in use | measured limitation |
| --- | --- | --- |
| STT | faster-whisper large-v3 | spoken numerals lost in Hindi/Punjabi; 3,846 owner segments transcribed across hi/pa/en/ur |
| TTS | ElevenLabs `eleven_v3` | works incl. Punjabi; cloud-hosted, per-character billing |
| Diarization | pyannote 3.1 | 32.7 h of owner speech isolated — **unusable for training at ~1.6 kHz bandwidth** |

### 3.4 What is deliberately **not** yet measured

Gnani's models. That is the point of this request — and the benchmark below is
designed so their numbers land in the same table as everything above.

## 4 · Evaluation plan

Three tracks, each with a pass/fail gate defined *before* credits are spent.

### Track A — Inya VoiceOS vs the current Eraya cascade

Replay a fixed set of recorded qualification calls through both paths.

| metric | how measured | gate |
| --- | --- | --- |
| time to first audio | server-side timestamps, p50 and p95 | beats the current cascade |
| turn-taking quality | interruption handling, barge-in | no regression |
| qualification accuracy | six mandatory Blueprint fields correctly filled | ≥ current accuracy |
| language coverage | English, Dutch, Hindi | usable on all three |

### Track B — Vachana TTS for Punjabi voice cloning

| metric | how measured | gate |
| --- | --- | --- |
| intelligibility | synthesize → transcribe → compare to source text | round-trips correctly in Punjabi (Gurmukhi), Hindi, English, **Roman-script Hinglish** |
| speaker similarity | speaker-embedding cosine vs real reference clips | within the owner's own photo-to-photo/clip-to-clip distribution |
| code-switching | mid-clause language switches, e.g. *"reschedule वाला भी crash मार देगे"* | no artefacts at the switch |
| latency | time to first audio, streaming | suitable for conversational use |

Roman-script Hinglish is called out because it is the case that disqualified IndicF5:
handling it natively removes an entire dead dependency chain.

### Track C — Vachana STT vs faster-whisper

Fixed set of already-transcribed owner segments, scored against corrected references.

| metric | gate |
| --- | --- |
| WER on Hindi / Punjabi / code-switched speech | beats large-v3 |
| **spoken numerals and dates** | "2017" survives as a year — the specific failure that breaks date-filtered retrieval |
| narrowband robustness | graceful on ~1.6 kHz telephone audio |

## 5 · Credit request

Sizing is derived from real workloads, with assumptions stated so Gnani can correct
them. We do not know Gnani's credit unit — the ask is expressed in **minutes of audio
processed** and we would like it mapped to their metering.

| track | workload | basis |
| --- | --- | --- |
| **A — Inya VoiceOS** | **≈ 2,100 min** | 500 pilot calls × ~3 min, plus 200 benchmark calls × ~3 min |
| **B — Vachana TTS** | **≈ 500 min synth + 1 voice clone** | ~2,000 evaluation utterances × ~15 s |
| **C — Vachana STT** | **≈ 1,700 min** | 553 unprocessed archived calls × ~3 min |
| retries, failed runs, re-benchmarks | **+30%** | first-run recipes rarely survive contact |
| **Total** | **≈ 5,600 min ( ~93 h ) of audio across the stack** | |

Plus API access to Inya VoiceOS (research preview or GA) and Vachana STT/TTS, and
documentation sufficient to self-serve.

## 6 · What Gnani.ai gets

- A **written benchmark report** placing their models against Whisper large-v3,
  ElevenLabs `eleven_v3`, Sarvam-M and the open Indic TTS field — using the same
  measurement harness that has already disqualified four engines on the record
- A **named EU reference deployment** if Track A passes: VECTRA International BV,
  Brussels — an EU AI Act Article 50-compliant voice agent already in production
- Findings on **Punjabi + code-switched** performance, which is under-reported
  publicly and is where their Indic advantage should be strongest
- Honest published results either way

## 7 · Risks and open questions

**Data residency is the commercial blocker to resolve first.** Inya VoiceOS is
described as *"built, trained, and deployed entirely within India — complete data
sovereignty."* For the personal twin that is neutral. For **VECTRA International BV, a
Belgian entity handling EU callers' voice data**, India-only processing is a GDPR
transfer question and Eraya's Article 50 compliance is structural, not a setting. We
need to know before any pilot:

1. Is there an **EU or on-premise deployment option** for Inya VoiceOS?
2. If not, what is the lawful transfer basis for EU caller audio?
3. Is Track B (personal, Indic, no EU subjects) separable so it can proceed regardless?

Other open items:

| question | why it matters |
| --- | --- |
| Which model did the "2.5" reference denote? | Needs confirming — public material names Inya VoiceOS (5B), Vachana STT/TTS, and a 14B roadmap |
| Is there a model named **Prisma**? | Not found in public material; may be internal or a misheard name |
| Punjabi (Gurmukhi) support in Vachana TTS — confirmed? | Track B's entire premise |
| Roman-script Hinglish input — native or does it need transliteration? | Determines whether the IndicF5 dead-end recurs |
| Voice-clone reference audio: minimum duration and **minimum bandwidth**? | 32 h of ~1.6 kHz audio was already proved useless; we would rather know their floor than discover it |
| Streaming latency and self-hosting availability | Real-time avatar work needs both |

## 8 · Decision

Proceed to a credits conversation with Gnani.ai, with **Track B (Indic TTS/STT,
personal, no EU data subjects) as the entry point** because it is unblocked by the
residency question, and Track A gated on a satisfactory answer to §7.

**Internal approvals for Track A:** Shelly (SVP Sales & Marketing) as approver,
Aeshwarya Das (Enterprise & Systems Manager) as technical counterpart.
