**Subject:** Evaluation credits — Timbre v2.0 and Prisma v2.5 for a personal Punjabi voice project

---

Hi <name>,

I came across your Hugging Face release of Gnani Prisma v2.5 and Timbre v2.0, and I'd
like to request evaluation credits for a personal project. It's a small ask, and I think
the results would be useful to you as well.

I've built a personal digital twin — trained on my own chats and call recordings, it
writes in my style and speaks in a cloned version of my voice. It works across Hindi,
Punjabi and English, and it code-switches mid-sentence the way I actually talk. This is
personal use, not a product.

I already chose an Indian model where it mattered most. The text layer is a fine-tuned
Sarvam-M 24B — I benchmarked it against Qwen3-4B and Gemma-4-E4B, and it held Hinglish
and Punjabi code-switching where the others flattened it into plain Hindi. The voice
layer is the missing half.

Punjabi is where I ran out of options. I surveyed the open Indic TTS field and
disqualified all of it: GPT-SoVITS has no Punjabi in any version; XTTS-v2's Hindi is
broken in its own tokenizer and Coqui has shut down; Chatterbox has a confirmed open
issue on Hindi quality; and IndicF5 needs a Roman-to-Devanagari front-end built on
fairseq, which was archived in March 2026. I'm on a US vendor today only because it was
the single engine I could find that supports Punjabi at all.

What I'd like to evaluate, in order of priority:

1. **Timbre v2.0** — Punjabi output quality and custom voice cloning
2. **Prisma v2.5** — Hinglish and Punjabi transcription, particularly spoken numerals
3. **Warp or Inya VoiceOS** — later, for a real-time conversational loop

The credits I'd need:

| model | amount | what it covers |
| --- | --- | --- |
| Timbre v2.0 | 300 minutes of synthesis, 2 voice clones | ~1,200 evaluation utterances across four scripts and five speaking registers, with room to rerun |
| Prisma v2.5 | 2,000 minutes of transcription | a 553-call personal archive, plus re-running audio I've already labelled for a like-for-like accuracy comparison |
| Warp | 120 minutes | a latency and turn-taking probe, if it's available to try |

I don't know how your credits meter, so please map that to your units and tell me if the
shape is wrong. If it works well, my ongoing usage is small — roughly 60 to 90 minutes a
month.

In return, I measure things properly rather than judging by ear; that's how the four
engines above were eliminated, each for a specific reason I can point to. You'd get
speaker-similarity scores against my real reference clips, round-trip intelligibility per
script, and a word-error-rate comparison against Whisper large-v3 on Hinglish and
Punjabi. I'll share the results honestly either way, and you're welcome to review the
methodology first and tell me if it's unfair to your models. Punjabi and code-switched
performance are barely reported publicly, and that's where your Indic advantage should be
strongest.

Four questions before I start:

1. Does Timbre v2.0 support **Punjabi (Gurmukhi)** output? Your model card lists 10+ Indic
   languages but doesn't name Punjabi, while press coverage does — I'd like to know which
   is right before I build an evaluation around it.
2. Does Timbre support **custom voice cloning**, or only the four stock voices documented
   on the card?
3. What's the minimum **duration and audio bandwidth** for a usable voice clone? I ask
   because I have 32 hours of my own archived speech that turned out unusable — 48 kHz
   files carrying only about 1.6 kHz of real audio, once the mobile network codec had
   finished with it. I'd rather know your floor than rediscover it.
4. Is **Roman-script Hinglish** handled natively, or does it need transliteration first?
   That's precisely where IndicF5 became unworkable for me.

One last thing — someone mentioned a model called **Vidma**, which I can't match to
anything public. What is it?

Happy to sign an NDA if that helps.

Thanks,

**Pardeep Singh**
<email> · <phone>
