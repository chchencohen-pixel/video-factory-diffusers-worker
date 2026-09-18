#!/usr/bin/env bash
# Bootstrap for the one-off MiniMax-H3 feasibility pod (v4). Runs as the pod's start command.
# Uploads a progress report to LOG_UPLOAD_URL after EVERY phase so a stall is diagnosable.
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf
mkdir -p /workspace/hf
PIPLOG=/workspace/pip.log; : > "$PIPLOG"
report() { echo "[boot] $(date -u +%H:%M:%S) $1" | tee -a "$PIPLOG"; if [ -n "${LOG_UPLOAD_URL:-}" ]; then curl -s -o /dev/null -m 60 -X PUT -H "content-type: text/plain" --data-binary @"$PIPLOG" "$LOG_UPLOAD_URL" || true; fi; }
# HARD COST GUARD (independent of the operator's PC): terminate this pod via the API after MAX_MIN minutes.
if [ -n "${RUNPOD_API_KEY_GUARD:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  ( sleep $(( ${MAX_MIN:-50} * 60 )); curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY_GUARD" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" ) &
fi
report "hello from pod: $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1) | RAM $(free -g | awk '/Mem/{print $2}')G | python $(python --version 2>&1)"
python -m pip install --upgrade "diffusers>=0.40.0" "transformers>=5.0" "huggingface_hub>=1.23,<2" accelerate av imageio imageio-ffmpeg hf_transfer requests kernels >> "$PIPLOG" 2>&1 && report "base pip ok" || { report "base pip FAILED (pinned set); trying resolver-free fallback"; python -m pip install --upgrade "diffusers>=0.40.0" accelerate av imageio imageio-ffmpeg hf_transfer requests kernels >> "$PIPLOG" 2>&1 && python -m pip install --upgrade "transformers" >> "$PIPLOG" 2>&1 && report "fallback pip ok" || report "fallback pip FAILED"; }
if [ "${H3_UPGRADE_TORCH:-1}" = "1" ]; then
  python -m pip install --upgrade "torch==2.10.*" torchvision --index-url https://download.pytorch.org/whl/cu128 >> "$PIPLOG" 2>&1 && report "torch upgrade ok" || report "torch upgrade FAILED (keeping current torch)"
fi
python - >> "$PIPLOG" 2>&1 <<'PY'
import importlib
for m in ("torch", "diffusers", "transformers", "kernels", "requests"):
    try:
        mod = importlib.import_module(m); print("[env]", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print("[env] IMPORT FAIL", m, e)
try:
    import torch; print("[env] cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
except Exception as e:
    print("[env] cuda check failed", e)
PY
report "env check done"
curl -sL -m 60 "https://raw.githubusercontent.com/chchencohen-pixel/video-factory-diffusers-worker/main/experiments/h3_pod_test.py?v=$(date +%s)" -o /workspace/h3_pod_test.py
python -c "import ast; ast.parse(open('/workspace/h3_pod_test.py').read()); print('syntax ok')" >> "$PIPLOG" 2>&1 && report "test script fetched" || report "test script BROKEN"
if [ "${H3_DRY:-0}" = "1" ]; then report "H3_DRY=1: stopping after environment check"; sleep infinity; fi
report "starting test"
PIP_LOG_PATH="$PIPLOG" python /workspace/h3_pod_test.py >> "$PIPLOG" 2>&1
report "test finished"
sleep infinity
