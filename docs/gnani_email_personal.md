# Email — Gnani.ai, personal Indic voice evaluation only

Scope: **personal project only.** No VECTRA, no EU data subjects, no commercial
deployment — which is deliberate, because it removes the data-residency question
entirely and makes this a small, easy yes for them.

The ask is kept deliberately modest (~5 hours of synthesis). A tight, specific request
gets approved by a junior person; a large one goes to a committee.

---

## The email

**Subject:** Evaluation credits — testing Vachana TTS for Punjabi voice cloning

Hi <name>,

I'm building a personal voice project and I'd like to test Vachana TTS against it. It's
a small ask, and I think the results would be useful to you as well as to me.

**What I'm doing.** I've built a personal digital twin trained on my own archives — it
writes in my style and speaks in a cloned version of my voice. Personal use, not a
product. It works in Hindi, English and **Punjabi**, and it code-switches mid-sentence
the way I actually talk.

**Why I'm writing.** Punjabi is where every option runs out. I did a proper survey of
the open field earlier this year and disqualified all of it:

| engine | why it failed |
|---|---|
| GPT-SoVITS | no Hindi or Punjabi in any version |
| XTTS-v2 | Hindi broken in its own tokenizer; Coqui shut down |
| Chatterbox | poor Hindi — confirmed open issue |
| IndicF5 | needs a Roman→Devanagari front-end built on fairseq, **archived March 2026** |

I'm on ElevenLabs today because it was the *only* engine I could find, open or closed,
that supports Punjabi at all. That's not a comfortable position, and Vachana looks like
the first Indic-native alternative with a real cloning story.

**What I'd like to test.** Four things, in order of how much they matter to me:

1. **Punjabi (Gurmukhi) cloning quality** — does my voice survive in Punjabi, not just Hindi
2. **Roman-script Hinglish input** — "chal thik hai bro, kal milte hain fir" rendered as
   speech without a transliteration step. This is exactly where IndicF5 died for me.
3. **Mid-clause code-switching** — "reschedule वाला भी crash मार देगे" — no artefacts at
   the switch point
4. **Streaming latency** — time to first audio, for conversational use

**What I'm asking for.** Small: **~300 minutes of synthesis and 1–2 voice clones.**
That's enough for roughly 1,200 evaluation utterances across four scripts and several
speaking registers, with room to redo runs. I don't know how your credits meter, so
please map that to your units. If it passes the tests I'd want to talk about ongoing
access.

**What you get back.** I measure things properly rather than by ear — that's how the
four engines above got disqualified, each for a specific reason I can point to. You'd
get:

- Speaker-similarity scores against my real reference clips, not impressions
- Round-trip intelligibility per script: Gurmukhi, Devanagari, Latin, Roman-Hinglish
- Honest findings either way, and you're welcome to see the methodology first and tell
  me if it's unfair to your models

**Punjabi and code-switched results are barely reported publicly**, and that's where
your Indic advantage should be strongest — so the data should be worth having.

Three technical questions I'd want answered whatever happens:

- Does Vachana TTS support **Punjabi (Gurmukhi)** for cloning specifically, or is Indic
  coverage mainly Hindi and the southern languages?
- Is Roman-script Hinglish handled natively, or does it need transliteration first?
- Minimum reference-audio **duration and bandwidth** for a clone? I ask because I have
  32 hours of my own archived speech that turned out unusable — 48 kHz files carrying
  about 1.6 kHz of real audio once the mobile network codec had finished with it. I'd
  rather know your floor up front than rediscover that.

Also, two names came up in conversation that I can't place against your public material
— a **"Prisma"** model and something numbered **"2.5"**. Could you tell me what those
refer to, so I benchmark the right thing?

Thanks,
Pardeep Singh

---

## Before you send

**Fill in:** recipient name, and one line on how you know them / where you met.

**Deliberately left out:** VECTRA, Eraya, anything commercial, and the data-residency
question. This is a personal project with no EU data subjects, so India-only processing
raises nothing — which is exactly what makes it a clean first ask. Keep the enterprise
conversation separate; if it comes up, it should come *after* you have working evidence
from this track.

**Two things I could not verify and did not invent:** no "Prisma" model and no "2.5"
appear anywhere in Gnani's public material — their published stack is Inya VoiceOS
(5B voice-to-voice), Vachana STT and Vachana TTS, with a 14B model announced. Both are
asked as questions.

**If you'd rather ask for more:** the STT side is also personal and Indic — 553
unprocessed call recordings, ~1,700 minutes, where Whisper currently loses spoken years
("2017" comes back as "अगा तत्रा"). I left it out to keep the ask small. Say the word
and I'll add a paragraph.
