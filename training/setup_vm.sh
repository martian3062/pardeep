#!/usr/bin/env bash
# One-time setup of the training VM (Ubuntu + NVIDIA L4). Idempotent.
set -euo pipefail

WORK="$HOME/pardeep_self"
mkdir -p "$WORK"/{datasets,adapters}
cd "$WORK"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Unsloth pins its own torch/triton build; let it resolve the CUDA stack.
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --quiet "unsloth[cu124-torch260]" "trl>=0.12" "datasets>=3.0" \
    "transformers>=4.46" "accelerate" "bitsandbytes" "peft" "huggingface_hub" \
    setuptools wheel  # triton compiles kernels via setuptools at import time

# torchao arrives transitively but calls torch.utils._pytree.register_constant,
# which only exists in torch>=2.7 — unsloth pins 2.6, so importing it breaks
# transformers -> peft -> trl. Quantization here is bitsandbytes, so drop it.
uv pip uninstall --python .venv/bin/python torchao 2>/dev/null || true

python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
EOF
echo "VM ready: $WORK"
