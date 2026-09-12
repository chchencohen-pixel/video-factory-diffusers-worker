#!/usr/bin/env bash
# Bootstrap for the one-off MiniMax-H3 feasibility pod. Runs as the pod's start command:
#   bash -c "curl -sL https://raw.githubusercontent.com/chchencohen-pixel/video-factory-diffusers-worker/main/experiments/h3_pod_test.sh | bash"
# Expects env: SCENES_JSON, LOG_UPLOAD_URL, optional H3_FRAMES/H3_STEPS/H3_W/H3_H.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf
mkdir -p /workspace/hf
echo "[boot] $(date -u +%H:%M:%S) installing deps"
# v2: newer torch so the flash-attention-3 hub kernels have a matching build (torch>=2.9), ~3x faster attention on Hopper.
if [ "${H3_UPGRADE_TORCH:-1}" = "1" ]; then
  pip install -q --upgrade "torch==2.10.*" "torchvision" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1 || true
  python -c "import torch; print('[boot] torch', torch.__version__)"
fi
pip install -q --upgrade "diffusers>=0.40.0" "transformers>=4.57.0,<5" accelerate av imageio imageio-ffmpeg hf_transfer "huggingface_hub>=0.34" requests kernels 2>&1 | tail -2 || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | head -2 || true
df -h /workspace | tail -1 || true
curl -sL https://raw.githubusercontent.com/chchencohen-pixel/video-factory-diffusers-worker/main/experiments/h3_pod_test.py -o /workspace/h3_pod_test.py
echo "[boot] $(date -u +%H:%M:%S) starting test"
python /workspace/h3_pod_test.py
echo "[boot] $(date -u +%H:%M:%S) finished; idling until terminated"
sleep infinity
