**Subject:** Evaluation credits — Timbre v2.0 and Prisma v2.5 for a personal Punjabi voice project

---

Hi <name>,

I saw your Hugging Face release of Gnani Prisma v2.5 and Timbre v2.0, and I'd like to
request evaluation credits for a personal project.

I've built a personal digital twin — trained on my own chats and call recordings, it
writes in my style and speaks in a cloned version of my voice, across Hindi, Punjabi and
English, code-switching mid-sentence the way I actually talk. Personal use, not a product.

I already went Indian where it mattered most: the text layer is a fine-tuned **Sarvam-M
24B**, which beat Qwen3-4B and Gemma-4-E4B on Hinglish and Punjabi code-switching where
the others flattened it. The voice layer is the missing half.

Punjabi is where I ran out of options. GPT-SoVITS has none in any version; XTTS-v2's
Hindi is broken in its own tokenizer and Coqui has shut down; Chatterbox has an open
issue on Hindi quality; IndicF5 depends on fairseq, archived in March 2026. I'm on a US
vendor today only because it was the one engine I could find that supports Punjabi at all.

I'd like to test **Timbre v2.0** for Punjabi output and voice cloning, **Prisma v2.5** for
Hinglish and Punjabi transcription, and later **Warp** for a real-time conversational loop.

**Credits requested** — sized with headroom, since my corpus keeps growing and I'd rather
not come back every few weeks with another small request:

| model | credits |
| --- | --- |
| Timbre v2.0 | 600 minutes of synthesis + 4 voice clones |
| Prisma v2.5 | 4,000 minutes of transcription |
| Warp | 250 minutes |

That's roughly double my immediate need, which is deliberate — it covers the archive I
have now plus the recordings I'm still adding. Ongoing usage after evaluation should
settle around 150 minutes a month. Please map these to your units and tell me if the
shape is wrong.

In return, I measure properly rather than judging by ear — that's how those four engines
were eliminated, each for a documented reason. You'd get speaker-similarity scores against
my real reference clips, per-script round-trip intelligibility, and a word-error-rate
comparison against Whisper large-v3 on Hinglish and Punjabi. Published honestly either
way. Punjabi and code-switched performance are barely reported publicly, which is where
your Indic advantage should show best.

Three questions:

1. Does **Timbre v2.0 support Punjabi (Gurmukhi)**, and is Roman-script Hinglish handled
   natively or does it need transliteration first? Your model card lists 10+ Indic
   languages without naming Punjabi, while press coverage does.
2. Does Timbre support **custom voice cloning**, or only the four stock voices on the card?
3. What's the minimum **duration and audio bandwidth** for a usable clone? I have 32 hours
   of my own archived speech that proved unusable — 48 kHz files carrying only ~1.6 kHz of
   real audio once the mobile network codec was done with it. I'd rather know your floor
   than rediscover it.

Also — someone mentioned a model called **Vidma**, which I can't match to anything public.
What is it?

Happy to sign an NDA.

Thanks,

**Pardeep Singh**
<email> · <phone>
