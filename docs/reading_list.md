# Reading list — papers that match what we built (August 2026)

Curated against the actual components of this project, not by citation count. Each entry
says which of *our* problems it speaks to, because a paper that is excellent in general
is still noise if it does not touch the thing that is broken here.

**Caveat:** these were found by search in August 2026 and selected on abstract and title
relevance. I have not read all of them end to end — the annotations say what each one
*claims* to address and why it maps to our work, not that its method is proven.

---

## Start here — the three closest to our open problems

### 1. LASE: Language-Adversarial Speaker Encoding for Indic Cross-Script Identity Preservation
[`arXiv:2605.00777`](https://arxiv.org/html/2605.00777v1)

**Read this first.** It is the only paper found that addresses our exact, unusual
problem: keeping *one speaker's identity* stable across **different scripts** for Indic
languages. Our voice twin speaks Gurmukhi, Devanagari, Latin and Roman-Hinglish, and the
open field has essentially nothing published on whether a cloned voice survives a script
change. This is also the methodology to bring to the Gnani evaluation — it is what
"does my voice survive in Punjabi, not just Hindi" looks like as a measurement rather
than an impression.

### 2. Learning User-Aware Recall: Personalized Retrieval in Long-Term Conversational Memory
[`arXiv:2607.00017`](https://arxiv.org/pdf/2607.00017)

Directly the problem we got wrong twice. Our recency weighting first buried every old
memory (multiplying similarity by `0.5^(age/540d)` scaled a 2017 memory to 0.014 of
itself), then over-corrected so a flat bonus dominated similarity entirely. This paper is
about learning *when* to prefer what, rather than hand-tuning a decay constant — which is
exactly the thing our two failures suggest should not be hand-tuned.

### 3. Rethinking Memory Mechanisms of Foundation Agents in the Second Half: A Survey
[`arXiv:2602.06052`](https://arxiv.org/pdf/2602.06052)

The map of the territory. Read it after the two above so you can place our design —
LanceDB episodic scenes, Mind Model facts, trust tiers, hybrid dense+BM25 retrieval —
against the taxonomy, and see which options we never considered.

---

## Memory and personalisation — our Phases 2, 4 and 6

| paper | why it matters here |
| --- | --- |
| [PersonaTree: Structured Lifecycle Memory for Person Understanding](https://arxiv.org/abs/2606.04780) `2606.04780` | Explicit **evidence → claim** paths. Our Mind Model asserts things about him with no traceable support; this is the shape that would let a claim be audited back to the messages that produced it. |
| [User as Code: Executable Memory for Personalized Agents](https://arxiv.org/pdf/2606.16707) `2606.16707` | Memory as executable rather than retrievable text — a genuinely different answer to what our persona card and Mind Model are trying to do. |
| [Memory Beyond Recall: Dual-Process Cognitive Memory for Self-Evolving Agents](https://arxiv.org/pdf/2606.09483) `2606.09483` | Fast/slow split. Relevant to the diary loop: what should be written immediately vs consolidated weekly. |
| [MemBench: Comprehensive Evaluation of Agent Memory](https://arxiv.org/pdf/2506.21605) `2506.21605` | We have no memory benchmark at all — our evaluation is 20 hand-written questions. This is what a real one looks like. |
| [PersonaAgent: Bridging Memory and Action](https://arxiv.org/html/2506.06254v2) `2506.06254` | Persona driving *action*, not just style. Our Phase 14 "own intentions" is this, unbuilt. |
| [Awesome-Personalized-LLMs](https://github.com/VanillaCreamer/Awesome-Personalized-LLMs) | Maintained list — the place to check for new work rather than re-searching. |

---

## Avatars and real-time presence — our Phases 10–14

| paper | why it matters here |
| --- | --- |
| [EmbodiedHead: Real-Time Listening and Speaking Avatar](https://arxiv.org/pdf/2604.17211) `2604.17211` | **Listening**, not only speaking. Our plan has idle blinks and backchannels as an afterthought; this treats the listening state as a first-class problem, which is what makes a video call feel like a conversation. |
| [Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation](https://arxiv.org/pdf/2601.00664) `2601.00664` | Interactive rather than render-then-play. The closest match to the live-call architecture in Phase 12. |
| [StreamAvatar: Streaming Diffusion for Real-Time Interactive Avatars](https://arxiv.org/pdf/2512.22065) `2512.22065` | Streaming diffusion — how to get frames out before the whole clip exists, which is our entire latency argument. |
| [LLIA: Low-Latency Interactive Avatars](https://arxiv.org/pdf/2506.05806) `2506.05806` | Single reference image + audio, with LCM and a discriminator for few-step sampling. Nearest to the "one good portrait" path we settled on after the VECTRA mesh experiment. |
| [talking-face-arxiv-daily](https://github.com/liutaocode/talking-face-arxiv-daily) | Updated daily. Watch this instead of searching. |

---

## Voice — our Phase 5 and the Gnani evaluation

| paper | why it matters here |
| --- | --- |
| [Domain-Specific Evaluation of TTS: A Multi-Metric Benchmarking Study](https://arxiv.org/abs/2608.02235) `2608.02235` | August 2026. MUSHRA, ABX, speaker-similarity scoring, and it includes Indic-Parler-TTS. **This is the harness to copy for the Gnani benchmark** — and it makes the offer in that email concrete rather than vague. |
| [Synthetic Speech, Real Signal: Paralinguistic Preservation via Voice Cloning](https://arxiv.org/abs/2607.22304) `2607.22304` | Whether *emotion and prosody* survive cloning, not just words. Our recording session deliberately captures five registers; this is how to check they came through. |
| [Praxy Voice: Commercial-Class Indic TTS from a Frozen Non-Indic Base](https://arxiv.org/pdf/2604.25441) `2604.25441` | A path to Indic TTS without Indic training data — the fallback if Gnani's Punjabi answer is no and we are back to building it ourselves. |
| [BnTTS: Few-Shot Speaker Adaptation in Low-Resource Settings](https://arxiv.org/pdf/2502.05729) `2502.05729` | Bengali, few-shot. Closest published analogue to "45 minutes of one speaker in an Indic language". |
| [Improving Code-Switching ASR with TTS Data Augmentation](https://arxiv.org/abs/2601.00935) `2601.00935` | The inverse of our problem, usefully: synthesising code-switched speech to fix recognition. Our Whisper loses spoken years; this suggests generating the data to fix it. |

---

## Suggested order, if you only have a few evenings

1. **LASE** — our most specific unsolved problem, and nobody else is writing about it
2. **Domain-Specific TTS Evaluation** — makes the Gnani benchmark real
3. **User-Aware Recall** — the retrieval mistake, properly
4. **EmbodiedHead** — before writing any more of Phase 12
5. **Memory Mechanisms survey** — to see what we never considered

## What is missing from the literature, as far as I can tell

Worth knowing, because it means parts of this project have no map to follow:

- **Turban / dastar handling in identity models.** Zero published reports found, across
  several searches. The replica dataset is the experiment.
- **Punjabi voice cloning quality.** Barely reported — which is precisely the leverage in
  the Gnani email.
- **~1.6 kHz effective bandwidth in 48 kHz containers.** The measurement that made 32
  hours of archive audio unusable is not something the TTS literature warns about; it is
  assumed clean data.
