#!/bin/bash

MODEL_PATH="./model/llama-3.2-3b-mlx"
ADAPTER_PATH="./model/adapters"

echo "🚀 Fine-tuning Dioula-AI (round 2)"
echo "======================================="

mkdir -p "$ADAPTER_PATH"

mlx_lm.lora \
  --model "$MODEL_PATH" \
  --train \
  --data "." \
  --adapter-path "$ADAPTER_PATH" \
  --batch-size 2 \
  --num-layers 16 \
  --learning-rate 1e-5 \
  --iters 3000 \
  --steps-per-eval 200 \
  --save-every 500 \
  --max-seq-length 512

echo "✅ Fine-tuning terminé !"
