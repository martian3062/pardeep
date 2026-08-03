#!/usr/bin/env bash
# Identity LoRA on Qwen-Image-2512 — the 24GB recipe.
#
# Numbers and their sources, so future-me does not re-derive them:
#   dim 32 / alpha 16      identity LoRAs need capacity for a face, not a style;
#                          32 is the community consensus for exact likeness
#   240 epochs MAX         the one published 2512-specific ablation tested
#                          120/180/240 and picked 240 — but darker skin tones
#                          have a documented overtraining failure mode
#                          ("slide into caricature"), so 240 is a CEILING and
#                          the checkpoint harness picks the real stopping point
#   save every 25 epochs   the ArcFace harness scores every checkpoint; the best
#                          one wins, exactly like the eval-tracked text twin
#   fp8 + blocks_to_swap   measured ~35GB without swap; swap 16 blocks brings a
#                          24GB card into range at some speed cost. On a 40GB+
#                          box, drop --blocks_to_swap entirely.
set -euo pipefail
source .venv/bin/activate

DATASET="${DATASET:-dataset}"        # unzipped replica bundle: PNGs + .txt captions
OUT="${OUT:-output/pxrdp_lora}"

# musubi pre-caches latents and text-encoder outputs; both passes are mandatory
python src/musubi_tuner/cache_latents.py \
  --dataset_config dataset.toml --vae models/qwen-image-2512/vae/

python src/musubi_tuner/cache_text_encoder_outputs.py \
  --dataset_config dataset.toml --text_encoder models/qwen-image-2512/text_encoder/

python src/musubi_tuner/qwen_image_train_network.py \
  --dit models/qwen-image-2512/transformer/ \
  --vae models/qwen-image-2512/vae/ \
  --text_encoder models/qwen-image-2512/text_encoder/ \
  --dataset_config dataset.toml \
  --network_module networks.lora_qwen_image \
  --network_dim 32 --network_alpha 16 \
  --optimizer_type adamw8bit --learning_rate 1e-4 \
  --max_train_epochs 240 --save_every_n_epochs 25 \
  --fp8_base --fp8_scaled \
  --blocks_to_swap 16 \
  --output_dir "$OUT" --output_name pxrdp \
  --seed 42

echo
echo "Checkpoints in $OUT. Next: bash sample.sh to generate the battery per"
echo "checkpoint, scp the batteries home, and score them with:"
echo "  python -m src.replica.run verify <battery-dir>"
