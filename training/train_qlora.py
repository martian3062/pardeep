"""QLoRA fine-tune of the twin. Runs ON the GPU VM, not the laptop.

    python train_qlora.py --model qwen4b    # local twin  (~2.4GB GGUF)
    python train_qlora.py --model sarvam24b # big twin, served from the VM

Loss is applied to Pardeep's turns only: the other person's words are context.
Training on both would teach the model to imitate everyone in his contact list.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        "batch_size": 1,
        "grad_accum": 16,
    },
}


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
    args = ap.parse_args()

    preset = PRESETS[args.model]
    out_dir = Path(args.out or f"adapters/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

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
        texts = [
            tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
            for r in rows
        ]
        return Dataset.from_dict({"text": texts})

    train_ds = load(args.data)
    eval_ds = load(args.eval_data) if Path(args.eval_data).exists() else None
    print(f"train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir=str(out_dir / "checkpoints"),
            dataset_text_field="text",
            max_seq_length=preset["max_seq_length"],
            per_device_train_batch_size=preset["batch_size"],
            gradient_accumulation_steps=preset["grad_accum"],
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.01,
            logging_steps=5,
            eval_strategy="epoch" if eval_ds else "no",
            save_strategy="epoch",
            save_total_limit=1,
            seed=17,
            report_to="none",
        ),
    )
    # mask everything except his replies
    trainer = train_on_responses_only(
        trainer,
        instruction_part=preset["instruction_part"],
        response_part=preset["response_part"],
    )

    stats = trainer.train()
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
