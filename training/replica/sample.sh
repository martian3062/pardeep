#!/usr/bin/env bash
# Generate the battery for every saved checkpoint, 3 seeds per prompt.
#
# The battery is fixed and versioned (battery.txt) so checkpoint scores are
# comparable — the same discipline as the text twin's fixed sampling prompts.
set -euo pipefail
source .venv/bin/activate

OUT="${OUT:-output/pxrdp_lora}"
TRIGGER="pxrdp man"

for ckpt in "$OUT"/pxrdp*.safetensors; do
  name=$(basename "$ckpt" .safetensors)
  mkdir -p "batteries/$name"
  i=0
  while IFS= read -r prompt; do
    [[ "$prompt" =~ ^#.*$ || -z "$prompt" ]] && continue
    p="${prompt//\{T\}/$TRIGGER}"
    for seed in 7 77 777; do
      python src/musubi_tuner/qwen_image_generate_image.py \
        --dit models/qwen-image-2512/transformer/ \
        --vae models/qwen-image-2512/vae/ \
        --text_encoder models/qwen-image-2512/text_encoder/ \
        --lora_weight "$ckpt" --lora_multiplier 1.0 \
        --prompt "$p" --image_size 1024 1024 \
        --seed "$seed" --fp8 \
        --save_path "batteries/$name/p${i}_s${seed}.png"
    done
    i=$((i+1))
  done < battery.txt
done

echo "Batteries done. scp -r batteries/ home and run the verifier on each."
