#!/bin/bash
set -e
echo "Training LoRA adapter..."
#!/usr/bin/env bash
set -euo pipefail
python -m autolyrics.training.train \
  --base_model openai/whisper-small \
  --train_csv data/manifests/train.csv \
  --val_csv   data/manifests/val.csv \
  --output_dir outputs/checkpoints/lora_run1 \
  --epochs 5 \
  --lr 1e-3 \
  --per_device_bs 8 \
  --grad_accum 2 \
  --lora_r 32 \
  --lora_alpha 64 \
  --target_modules q_proj v_proj \
  --run_name "ws-small_lora-r32_qv_lr1e-3"