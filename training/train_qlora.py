"""QLoRA fine-tune of the twin. Runs ON the GPU VM, not the laptop.

    python train_qlora.py --model qwen4b    # local twin  (~2.4GB GGUF)
    python train_qlora.py --model sarvam24b # big twin, served from the VM

Loss is applied to Pardeep's turns only: the other person's words are context.
Training on both would teach the model to imitate everyone in his contact list.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Unsloth MUST be imported before transformers/trl/peft: it patches them at
# import time. Imported later, the patches land only partially — which is how
# TRL kept spawning a 12-process dataset map that then died pickling torch's
# dynamo config.
import unsloth  # noqa: E402,F401  (import order is deliberate)
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

# Model presets. instruction/response markers must match each chat template —
# they are how train_on_responses_only finds the spans to keep in the loss.
PRESETS = {
    "qwen4b": {
        "name": "unsloth/Qwen3-4B-Instruct-2507",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "chat_template": "qwen-3",
        "max_seq_length": 4096,
        "batch_size": 2,
        "grad_accum": 8,
    },
    "qwen1_7b": {
        "name": "unsloth/Qwen3-1.7B",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "chat_template": "qwen-3",
        "max_seq_length": 4096,
        "batch_size": 4,
        "grad_accum": 4,
    },
    "sarvam24b": {
        "name": "sarvamai/sarvam-m",
        "instruction_part": "[INST]",
        "response_part": "[/INST]",
        "chat_template": "mistral",
        "max_seq_length": 2048,  # 24B on a 24GB card: keep activations small
        # batch 1 x accum 16 trains the same tokens per optimizer step as 2 x 8
        # at the same speed, but with a much smaller activation peak — the VM
        # shares its GPU with another service that reclaims several GB.
        "batch_size": 1,
        "grad_accum": 16,
        # Mistral-family templates reject non-alternating turns and have no
        # system role: a conversation window that opens with his own line, or a
        # persona system prompt, would abort templating.
        "strict_alternation": True,
        "supports_system": False,
    },
}


def normalize_messages(messages: list[dict], preset: dict) -> list[dict] | None:
    """Shape one example to what the model's chat template accepts.

    Returns None if nothing trainable survives (must end on his turn).
    """
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    msgs = [dict(m) for m in messages if m["role"] != "system"]

    if preset.get("strict_alternation"):
        while msgs and msgs[0]["role"] == "assistant":
            msgs.pop(0)  # template must open on the other person
        merged: list[dict] = []
        for m in msgs:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] += "\n" + m["content"]
            else:
                merged.append(m)
        msgs = merged

    if not msgs or msgs[-1]["role"] != "assistant" or len(msgs) < 2:
        return None

    if system:
        if preset.get("supports_system", True):
            msgs.insert(0, {"role": "system", "content": system})
        else:
            msgs[0]["content"] = f"{system}\n\n{msgs[0]['content']}"
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PRESETS), default="qwen4b")
    ap.add_argument("--data", default="datasets/sft_train.jsonl")
    ap.add_argument("--eval-data", default="datasets/sft_eval.jsonl")
    ap.add_argument("--out", default=None, help="adapter output dir")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--export-gguf", action="store_true", help="also write Q4_K_M GGUF")
    ap.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    ap.add_argument(
        "--eval-in-loop",
        action="store_true",
        help="evaluate during training (memory-hungry on large-vocab models)",
    )
    args = ap.parse_args()

    preset = PRESETS[args.model]
    out_dir = Path(args.out or f"adapters/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    import datasets
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    # datasets fingerprints every .map() by dill-pickling the mapping function.
    # After unsloth patches torch, that reaches torch._dynamo's ConfigModuleInstance,
    # which is unpicklable, and the run dies before the first step. We do not need
    # the on-disk cache for an 832-example dataset, so disable it and fall back to a
    # random fingerprint when hashing fails.
    datasets.disable_caching()
    from datasets.fingerprint import Hasher

    _orig_hash = Hasher.hash

    def _safe_hash(value):  # noqa: ANN001
        try:
            return _orig_hash(value)
        except Exception:
            return uuid.uuid4().hex

    Hasher.hash = staticmethod(_safe_hash)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=preset["name"],
        max_seq_length=preset["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=17,
    )

    def load(path: str):
        rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        texts, skipped = [], 0
        for r in rows:
            msgs = normalize_messages(r["messages"], preset)
            if msgs is None:
                skipped += 1
                continue
            texts.append(
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            )
        if skipped:
            print(f"  {Path(path).name}: skipped {skipped} examples incompatible with template")
        return Dataset.from_dict({"text": texts})

    train_ds = load(args.data)
    eval_ds = load(args.eval_data) if Path(args.eval_data).exists() else None
    print(f"train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")

    # TRL renames config fields between releases (max_seq_length -> max_length,
    # evaluation_strategy -> eval_strategy, tokenizer -> processing_class).
    # Filter to what this installed version actually accepts.
    import inspect

    cfg_fields = set(getattr(SFTConfig, "__dataclass_fields__", {})) or set(
        inspect.signature(SFTConfig.__init__).parameters
    )
    cfg_kwargs = {
        "output_dir": str(out_dir / "checkpoints"),
        "dataset_text_field": "text",
        "per_device_train_batch_size": preset["batch_size"],
        "gradient_accumulation_steps": preset["grad_accum"],
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "logging_steps": 5,
        # Checkpoint on a step cadence, not per epoch: an epoch-end crash used to
        # destroy 45 minutes of training because saving happens after evaluation.
        "save_strategy": "steps",
        "save_steps": 25,
        "save_total_limit": 2,
        "seed": 17,
        "report_to": "none",
        "dataset_num_proc": 1,  # multiprocess map re-triggers the pickling failure
        # In-loop eval gathers full logits and casts them to fp32 — with a 130k
        # vocab that is a multi-GB spike that OOMed the run. Score the held-out
        # set after training instead.
        "per_device_eval_batch_size": 1,
        "prediction_loss_only": True,
    }
    cfg_kwargs["max_seq_length" if "max_seq_length" in cfg_fields else "max_length"] = preset[
        "max_seq_length"
    ]
    eval_key = "eval_strategy" if "eval_strategy" in cfg_fields else "evaluation_strategy"
    cfg_kwargs[eval_key] = "epoch" if (eval_ds and args.eval_in_loop) else "no"
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in cfg_fields}

    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    tok_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**cfg_kwargs),
        **{tok_key: tokenizer},
    )
    # mask everything except his replies
    trainer = train_on_responses_only(
        trainer,
        instruction_part=preset["instruction_part"],
        response_part=preset["response_part"],
    )

    ckpt_dir = out_dir / "checkpoints"
    resume = args.resume and any(ckpt_dir.glob("checkpoint-*"))
    if resume:
        print(f"resuming from latest checkpoint in {ckpt_dir}")
    stats = trainer.train(resume_from_checkpoint=resume or None)
    print(f"train_loss={stats.training_loss:.4f}")

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"adapter -> {out_dir}")

    if args.export_gguf:
        gguf_dir = out_dir / "gguf"
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method="q4_k_m")
        print(f"gguf -> {gguf_dir}")


if __name__ == "__main__":
    main()
