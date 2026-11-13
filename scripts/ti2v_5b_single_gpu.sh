#!/usr/bin/env bash
# TI2V-5B 单卡文生视频。直接 python，不建进程组。
# 覆盖路径：CKPT_DIR=... OUT_DIR=... PYTHON=... bash scripts/ti2v_5b_single_gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/data2/zhenghao/miniforge3/envs/wan22-ti2v5b/bin/python}"
CKPT_DIR="${CKPT_DIR:-/data2/zhenghao/wan22/Wan2.2-TI2V-5B}"
OUT_DIR="${OUT_DIR:-/data2/zhenghao/wan22/outputs}"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON" generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir "$CKPT_DIR" \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --base_seed 42 \
  --save_file "$OUT_DIR/ti2v5b_t2v_1280x704.mp4" \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
