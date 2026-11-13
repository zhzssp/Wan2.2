#!/usr/bin/env bash
# TI2V-5B 八卡图生视频：FSDP 切参数，Ulysses 切序列。
# 去噪可以跑完；rank 0 的 VAE 解码在 32GB 卡上会 OOM，默认不写 mp4。
# 覆盖路径：CKPT_DIR=... OUT_DIR=... MASTER_PORT=... bash scripts/ti2v_5b_8gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TORCHRUN="${TORCHRUN:-/data2/zhenghao/miniforge3/envs/wan22-ti2v5b/bin/torchrun}"
CKPT_DIR="${CKPT_DIR:-/data2/zhenghao/wan22/Wan2.2-TI2V-5B}"
OUT_DIR="${OUT_DIR:-/data2/zhenghao/wan22/outputs}"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

"$TORCHRUN" --nproc_per_node=8 --master_port="${MASTER_PORT:-29521}" generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir "$CKPT_DIR" \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --base_seed 42 \
  --save_file "$OUT_DIR/ti2v5b_i2v_8gpu.mp4" \
  --image examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
