#!/usr/bin/env bash
# One-time setup on the rented GPU box (Linux, 24GB+).
#
# Base model: Qwen/Qwen-Image-2512 — chosen over Ideogram 4 and FLUX after an
# adversarial research pass (Aug 2026):
#   - Apache-2.0 with full bf16 weights. Ideogram 4 scores likeness marginally
#     higher in the one published head-to-head, but ships ONLY gated FP8/NF4
#     quants under a non-commercial licence — an identity welded to it is a
#     dead end the day this fronts anything monetised.
#   - "Enhanced Human Realism" is its headline feature, and FLUX.2-klein was
#     reported LIGHTENING a South Asian subject's skin in that same comparison.
#   - Qwen's open line is frozen at 2512 (2.0/3.0 are API-only), so the
#     training ecosystem around it is stable rather than churning.
#
# Trainer: kohya-ss/musubi-tuner. NOT ai-toolkit — its Qwen-2512 path has
# documented breakage (issues #740 training-loop failure, #506 OOM regression);
# do not spend rental hours rediscovering this.
set -euo pipefail

# musubi self-describes as experimental — pin, don't track main.
MUSUBI_COMMIT="${MUSUBI_COMMIT:-main}"  # replace with the latest release SHA at run time

git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
git checkout "$MUSUBI_COMMIT"

python -m venv .venv && source .venv/bin/activate
pip install -e . --quiet
pip install "huggingface_hub[cli]" --quiet

# DiT + VAE + text encoder (Qwen2.5-VL). ~40GB of downloads; the box needs disk.
hf download Qwen/Qwen-Image-2512 --local-dir models/qwen-image-2512

echo
echo "Setup done. Now:"
echo "  1. scp replica_bundle.zip here and unzip to dataset/"
echo "  2. bash train.sh"
