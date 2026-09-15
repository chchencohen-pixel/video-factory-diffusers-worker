#!/usr/bin/env bash
# Bootstrap for the one-off MiniMax-H3 feasibility pod (v3). Runs as the pod's start command.
# Expects env: SCENES_JSON, LOG_UPLOAD_URL, optional H3_FRAMES/H3_STEPS/H3_W/H3_H/H3_UPGRADE_TORCH.
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf
mkdir -p /workspace/hf
PIPLOG=/workspace/pip.log; : > "$PIPLOG"
echo "[boot] $(date -u +%H:%M:%S) installing deps" | tee -a "$PIPLOG"
# 1) the proven base set first (worked in v1 on torch 2.8)
python -m pip install --upgrade "diffusers>=0.40.0" "transformers>=4.57.0,<5" accelerate av imageio imageio-ffmpeg hf_transfer "huggingface_hub>=0.34" requests kernels >> "$PIPLOG" 2>&1 || echo "[boot] base pip FAILED" | tee -a "$PIPLOG"
# 2) optional torch upgrade for the flash-attention-3 hub kernels; failure keeps torch 2.8 (default attention)
if [ "${H3_UPGRADE_TORCH:-1}" = "1" ]; then
  python -m pip install --upgrade "torch==2.10.*" torchvision --index-url https://download.pytorch.org/whl/cu128 >> "$PIPLOG" 2>&1 || echo "[boot] torch upgrade FAILED (keeping current torch)" | tee -a "$PIPLOG"
fi
python - <<'PY' 2>&1 | tee -a "$PIPLOG"
import importlib
for m in ("torch", "diffusers", "transformers", "kernels"):
    try:
        mod = importlib.import_module(m); print("[boot]", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print("[boot] IMPORT FAIL", m, e)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | tee -a "$PIPLOG"
free -g | head -2 | tee -a "$PIPLOG"
curl -sL "https://raw.githubusercontent.com/chchencohen-pixel/video-factory-diffusers-worker/main/experiments/h3_pod_test.py?v=$(date +%s)" -o /workspace/h3_pod_test.py
python -c "import ast,sys; ast.parse(open('/workspace/h3_pod_test.py').read()); print('[boot] test script syntax ok')" 2>&1 | tee -a "$PIPLOG"
# Early boot report: upload the pip/boot log NOW so a crash before the test still tells us what happened.
if [ -n "${LOG_UPLOAD_URL:-}" ]; then curl -s -o /dev/null -w "[boot] early log upload HTTP %{http_code}
" -X PUT -H "content-type: text/plain" --data-binary @"$PIPLOG" "$LOG_UPLOAD_URL" | tee -a "$PIPLOG"; fi
if [ "${H3_DRY:-0}" = "1" ]; then echo "[boot] H3_DRY=1: stopping after environment check" | tee -a "$PIPLOG"; curl -s -o /dev/null -X PUT -H "content-type: text/plain" --data-binary @"$PIPLOG" "$LOG_UPLOAD_URL"; sleep infinity; fi
echo "[boot] $(date -u +%H:%M:%S) starting test" | tee -a "$PIPLOG"
PIP_LOG_PATH="$PIPLOG" python /workspace/h3_pod_test.py 2>&1 | tee -a "$PIPLOG"
# Final upload of the boot log too (the test uploads its own log; this catches crashes before that).
curl -s -o /dev/null -X PUT -H "content-type: text/plain" --data-binary @"$PIPLOG" "$LOG_UPLOAD_URL"
echo "[boot] $(date -u +%H:%M:%S) finished; idling until terminated"
sleep infinity
