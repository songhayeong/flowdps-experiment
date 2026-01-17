#!/usr/bin/env bash
set -euo pipefail

TASKS=(
  "sr_avgpool:12"
  "sr_bicubic:12"
  "deblur_gauss:9"
  "deblur_motion:61"
  "inpainting:12"
  "colorization:12"
  "cs_walshhadamard:1"
  "cs_blockbased:4"
  "deblur_uni:9"
  "deblur_aniso:9"
  "deblur_nonuniform:33"
)
METHODS=(flowdps flowchef psld)
IMAGE_DIR="samples/imagenet"

shopt -s nullglob
IMAGE_FILES=(
  "$IMAGE_DIR"/*.png
  "$IMAGE_DIR"/*.jpg
  "$IMAGE_DIR"/*.jpeg
)
shopt -u nullglob

if [[ ${#IMAGE_FILES[@]} -eq 0 ]]; then
  echo "No images found in $IMAGE_DIR (png/jpg/jpeg)."
  exit 1
fi

for tm in "${TASKS[@]}"; do
  task="${tm%%:*}"
  scale="${tm##*:}"
  if [[ "$task" == "inpainting" ]] && [[ ! -f "exp/inp_masks/mask_768_half.npy" ]]; then
    echo "Skip inpainting: missing exp/inp_masks/mask_768_half.npy"
    continue
  fi
  if [[ "$task" == "deblur_nonuniform" ]]; then
    if [[ ! -f "functions/nonuniform/kernels/000001.npy" ]] || [[ ! -f "functions/nonuniform/masks/000001.npy" ]]; then
      echo "Skip deblur_nonuniform: missing nonuniform kernels/masks"
      continue
    fi
  fi
  for method in "${METHODS[@]}"; do
    for img_path in "${IMAGE_FILES[@]}"; do
      python solve.py \
        --img_size 768 \
        --img_path "$img_path" \
        --prompt "a high quality photo of an object" \
        --task "$task" \
        --deg_scale "$scale" \
        --method "$method" \
        --efficient_memory \
        --workdir "workdir/imagenet_${method}_${task}"
    done
  done
done
