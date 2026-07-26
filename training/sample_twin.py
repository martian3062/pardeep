"""Generate replies from the fine-tuned twin so its voice can be judged by ear.

Runs on the VM (the 24B needs the GPU). Compares adapters if several are given:

    python sample_twin.py --adapters adapters/sarvam24b adapters/sarvam24b/preserved-epoch2.4
"""
from __future__ import annotations

import argparse

import unsloth  # noqa: F401  — must precede transformers/trl
from unsloth import FastLanguageModel

# Prompts chosen to probe different registers the Mind Model found:
# family, closest friends, work/formal, and an unknown caller.
PROMPTS = [
    ("Mummy Airtel (mother)", "beta khana kha liya? kaisa chal raha hai sab?"),
    ("Roshan (very close friend)", "oye kal kya plan hai? movie chalein?"),
    ("Karan Cu (close college friend)", "bhai placement ka form bhara tu?"),
    ("a recruiter", "Hello, I'm calling from the hackathon team. Are you available this weekend?"),
    ("an unsaved number", "hello, kaun bol raha hai?"),
]


def build_prompt(persona: str, who: str, msg: str) -> list[dict]:
    """Must mirror training exactly: Sarvam has no system role, so the persona
    and the counterpart line are folded into the first user turn."""
    preamble = "\n\n".join(x for x in (persona, f"You are talking to {who}.") if x)
    return [{"role": "user", "content": f"{preamble}\n\n{msg}" if preamble else msg}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sarvamai/sarvam-m")
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--persona", default="persona.md")
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram", type=int, default=4)
    # One sample per prompt at temperature 0.8 is not evidence: two adapters that
    # differ by 18 training examples produced visibly different answers purely
    # from sampling noise. Draw several and judge the distribution.
    ap.add_argument("--samples", type=int, default=3, help="generations per prompt")
    args = ap.parse_args()

    from pathlib import Path

    persona = Path(args.persona).read_text(encoding="utf-8").strip() if Path(args.persona).exists() else ""
    print(f"persona card: {len(persona)} chars" if persona else "WARNING: no persona card")

    for adapter in args.adapters:
        print(f"\n{'=' * 70}\nADAPTER: {adapter}\n{'=' * 70}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter, max_seq_length=2048, load_in_4bit=True, dtype=None
        )
        FastLanguageModel.for_inference(model)
        for who, msg in PROMPTS:
            messages = build_prompt(persona, who, msg)
            # Sarvam-M is a hybrid reasoning model: left on, it emits a
            # chain-of-thought monologue instead of replying in his voice.
            try:
                ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=False,
                )
            except TypeError:
                ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )
            ids = ids.to("cuda")
            out = model.generate(
                input_ids=ids,
                attention_mask=(ids != tokenizer.pad_token_id).long(),
                max_new_tokens=args.max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=args.samples,
                # Both checkpoints looped on long English ("I don't have a
                # choice" x15) — the model knows what to say, the decoder gets
                # stuck. Training data is fragmentary call speech, which makes
                # short repeated phrases very likely.
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram,
                pad_token_id=tokenizer.eos_token_id,
            )
            print(f"\n[{who}]\n  THEM: {msg}")
            for i in range(out.shape[0]):
                reply = tokenizer.decode(
                    out[i][ids.shape[-1] :], skip_special_tokens=True
                ).strip()
                print(f"  TWIN {i + 1}: {reply}")
        del model
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
