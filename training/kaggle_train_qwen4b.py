"""Train the laptop twin (Qwen3-4B) on Kaggle's free GPU.

Kaggle gives 30 GPU-hours/week at no cost, which is plenty for a 4B QLoRA and
leaves the GCP L4 free for the 24B. Paste this into a single Kaggle notebook
cell with the accelerator set to GPU T4.

Setup (once):
  1. Kaggle -> Datasets -> New Dataset -> upload data/datasets/sft_train.jsonl
     and sft_eval.jsonl (~7 MB). Name it "pardeep-self-sft", keep it PRIVATE.
  2. New Notebook -> Add Input -> that dataset -> Settings -> Accelerator: GPU T4.
  3. Paste this file into a cell and run.
  4. When it finishes, download /kaggle/working/qwen4b_adapter.zip.

Then locally: unzip into data/models/qwen4b/ and convert to GGUF for Ollama.
"""

# ---------------------------------------------------------------- install
import subprocess
import sys

# Installing unsloth normally drags in its own torch, which lacks kernels for the
# GPU Kaggle allocates. Installing with --no-deps avoids that but leaves unsloth's
# real dependencies missing, so the import fails before logging even starts.
# A constraints file gives both: full dependency resolution, with torch pinned to
# the working build already on the image.
import torch as _t  # noqa: E402

_pin = f"/tmp/constraints.txt"
with open(_pin, "w") as f:
    f.write(f"torch=={_t.__version__.split('+')[0]}\n")
    f.write(f"torchvision\ntorchaudio\n")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", _pin,
     "unsloth", "unsloth_zoo", "trl", "peft", "bitsandbytes", "accelerate",
     "datasets", "setuptools", "hf_transfer"],
    check=False,
)
# NOTE: do NOT upgrade transformers here. Kaggle ships 5.0.0, which cannot parse
# the `gemma4` architecture, but upgrading it in isolation breaks the preinstalled
# stack so badly the kernel dies before any logging starts (empty log, no cause).
# Train an architecture this image supports instead — see MODELS above.

# unsloth must be imported before transformers/trl so its patches land fully
import os  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one T4 is enough for a 4B

import json  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import unsloth  # noqa: E402,F401
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

import datasets  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

# fingerprinting a mapped dataset dill-pickles torch's dynamo config and dies
datasets.disable_caching()
from datasets.fingerprint import Hasher  # noqa: E402

_orig_hash = Hasher.hash
Hasher.hash = staticmethod(lambda v: _orig_hash(v) if _safe(v) else uuid.uuid4().hex)


def _safe(value) -> bool:
    try:
        _orig_hash(value)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- config
# Two candidates for the laptop twin, trained head to head on his own data.
# A cross-lingual audit found Qwen loses accuracy when switching to Indian
# languages while Gemma stays robust, which matters more here than English
# benchmarks — so decide by comparing Punjabi/Hindi output, not by reputation.
MODELS = {
    "qwen4b": {
        "name": "unsloth/Qwen3-4B-Instruct-2507",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "gemma4e4b": {
        "name": "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
        # gemma4 needs a transformers newer than Kaggle's image ships; if the
        # upgrade above is still too old, train the previous generation instead
        # of losing the run — Gemma 3 is strong on Indic too.
        "fallback_model": "unsloth/gemma-3-4b-it",
    },
    "gemma3-4b": {
        "name": "unsloth/gemma-3-4b-it",
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
    },
    # Purpose-built for 10 Indic languages including Punjabi. Only 2B, so it is
    # weaker in general reasoning, but its tokenizer spends far fewer tokens per
    # Gurmukhi word than the global models — which is what this corpus is full of.
    "sarvam1": {
        "name": "sarvamai/sarvam-1",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "fallback_template": "chatml",  # base model may ship no chat template
    },
    # newer revision of the same 2B Indic model; both are LlamaForCausalLM, so
    # they load on Kaggle's transformers where gemma4 could not
    "sarvam1v05": {
        "name": "sarvamai/sarvam-1-v0.5",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "fallback_template": "chatml",
    },
}
MODEL_KEY = os.environ.get("TWIN_BASE_MODEL", "qwen4b")
PRESET = MODELS[MODEL_KEY]

DATA_DIR = next(Path("/kaggle/input").glob("*"), Path("."))
OUT = Path(f"/kaggle/working/{MODEL_KEY}")
MODEL = PRESET["name"]
# Kaggle's T4 is 15GB and Turing (no bfloat16). Examples average ~260 tokens, so
# a 1024 window still packs several per sequence while halving the activation
# peak versus 2048 — the previous run died on this box without leaving a log.
MAX_SEQ = 1024
EPOCHS = 2
RANK = 16

import torch  # noqa: E402

# Kaggle's API returns files from /kaggle/working but NOT the console log, so two
# failed runs came back with no diagnosis at all. Tee everything into the output
# directory instead, where kernels_output can actually retrieve it.
_LOG = open("/kaggle/working/train.log", "w", encoding="utf-8", buffering=1)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def _log_exception(exc_type, exc, tb):
    import traceback

    traceback.print_exception(exc_type, exc, tb, file=_LOG)
    _LOG.flush()
    traceback.print_exception(exc_type, exc, tb, file=sys.__stderr__)


sys.excepthook = _log_exception

print("=" * 60)
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu: {p.name} | {p.total_memory / 1e9:.1f} GB | capability {p.major}.{p.minor}")
    # Requesting a T4 is only a preference — Kaggle still hands out Tesla P100s
    # (Pascal, sm_60), and current PyTorch wheels carry no sm_60 kernels, so the
    # run dies at the first CUDA op. Fail fast with an actionable message rather
    # than burning a GPU hour on a machine that cannot execute anything.
    if (p.major, p.minor) < (7, 0):
        raise SystemExit(
            f"Unusable GPU: {p.name} (compute {p.major}.{p.minor}). Modern PyTorch "
            "ships no kernels below sm_70. Re-run the kernel — Kaggle reassigns "
            "the accelerator each time and a T4 (7.5) works."
        )
else:
    raise SystemExit("No GPU attached — set Accelerator to GPU T4 in notebook settings")
print("input dir:", DATA_DIR, "|", [f.name for f in DATA_DIR.glob("*")])
print("=" * 60, flush=True)

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None
    )
except (ValueError, ImportError, RuntimeError) as exc:
    fallback = PRESET.get("fallback_model")
    if not fallback:
        raise
    # a brand-new architecture the installed transformers cannot parse is worth
    # degrading for: an older sibling trained on his data still beats no model
    print(f"\n!! {MODEL} unavailable ({str(exc)[:140]})")
    print(f"!! falling back to {fallback}\n", flush=True)
    MODEL = fallback
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None
    )
print(f"trained base: {MODEL}", flush=True)
model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    lora_alpha=RANK,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=17,
)


if PRESET.get("fallback_template") and not getattr(tokenizer, "chat_template", None):
    # base models ship no chat template; give it ChatML so the response markers
    # that train_on_responses_only looks for actually appear in the text
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(tokenizer, chat_template=PRESET["fallback_template"])
    print(f"applied fallback chat template: {PRESET['fallback_template']}")


def load(name: str):
    path = DATA_DIR / name
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [
        tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        for r in rows
    ]
    print(f"{name}: {len(texts)} examples")
    return Dataset.from_dict({"text": texts})


train_ds = load("sft_train.jsonl")
eval_ds = load("sft_eval.jsonl") if (DATA_DIR / "sft_eval.jsonl").exists() else None

cfg_fields = set(getattr(SFTConfig, "__dataclass_fields__", {}))
kwargs = {
    "output_dir": "/kaggle/working/checkpoints",
    "dataset_text_field": "text",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "fp16": not torch.cuda.is_bf16_supported(),  # T4 is Turing: fp16 only
    "bf16": torch.cuda.is_bf16_supported(),
    "num_train_epochs": EPOCHS,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "logging_steps": 5,
    "save_strategy": "steps",
    "save_steps": 50,
    "save_total_limit": 1,
    "seed": 17,
    "report_to": "none",
    "dataset_num_proc": 1,
    "packing": True,  # chat turns are short; pack them or waste the window
    "prediction_loss_only": True,
}
kwargs["max_seq_length" if "max_seq_length" in cfg_fields else "max_length"] = MAX_SEQ
kwargs = {k: v for k, v in kwargs.items() if k in cfg_fields}

trainer = SFTTrainer(
    model=model, train_dataset=train_ds, args=SFTConfig(**kwargs), processing_class=tokenizer
)
# loss on his replies only — never on the other person's words
trainer = train_on_responses_only(
    trainer,
    instruction_part=PRESET["instruction_part"],
    response_part=PRESET["response_part"],
)

stats = trainer.train()
print("train_loss:", stats.training_loss)

OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(OUT))
tokenizer.save_pretrained(str(OUT))

# GGUF here would need llama.cpp built inside the session; convert locally instead
subprocess.run(["zip", "-r", "/kaggle/working/qwen4b_adapter.zip", str(OUT)], check=False)
print("done -> download /kaggle/working/qwen4b_adapter.zip")
