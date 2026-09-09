"""Runpod Serverless handler v3: diffusers-based Wan 2.2 family (base or distilled).

Why a second worker: the A/B plan needs several models (FastWan distilled
3-step, Wan 2.2 A14B, ...) that ship in diffusers format, and Runpod's REST API
cannot configure Model Caching for a new endpoint. So this worker downloads the
model itself with huggingface_hub into the container disk at boot (cached for
the worker's lifetime) and keeps the pipeline resident on the GPU.

Same job contract as the v2 worker: health / generate, presigned upload URL,
signed I2V image URL, no secrets in the image. Output: H.264 MP4 at 24 fps.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import runpod

MODEL_ID = os.getenv("MODEL_ID", "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers")
MODEL_CACHE = Path(os.getenv("MODEL_CACHE", "/tmp/models"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "3"))
DEFAULT_GUIDANCE = float(os.getenv("DEFAULT_GUIDANCE", "1.0"))  # distilled models run without CFG
FLOW_SHIFT = float(os.getenv("FLOW_SHIFT", "5.0"))
PRELOAD = os.getenv("WAN_PRELOAD", "true").lower() == "true"
SUPPORTS_I2V = os.getenv("SUPPORTS_I2V", "false").lower() == "true"
SIZES = {(1280, 704), (704, 1280), (832, 480), (480, 832)}
FPS = 24

_pipes: dict[str, Any] = {}
_lock = threading.Lock()
_load_seconds: float | None = None
_download_seconds: float | None = None
_model_error: str | None = None


def fail(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def snapshot_path() -> Path:
    """Download (once) and return the local snapshot directory for MODEL_ID."""
    global _download_seconds
    from huggingface_hub import snapshot_download

    started = time.time()
    path = snapshot_download(MODEL_ID, cache_dir=str(MODEL_CACHE), token=os.getenv("HF_TOKEN") or None)
    _download_seconds = round(time.time() - started, 1)
    return Path(path)


def get_pipe(kind: str):
    """kind: 't2v' or 'i2v'. Pipelines share the model files; loaded once each."""
    global _load_seconds, _model_error
    if kind in _pipes:
        return _pipes[kind]
    with _lock:
        if kind in _pipes:
            return _pipes[kind]
        started = time.time()
        try:
            import torch
            from diffusers import WanImageToVideoPipeline, WanPipeline

            local = snapshot_path()
            cls = WanImageToVideoPipeline if kind == "i2v" else WanPipeline
            pipe = cls.from_pretrained(str(local), torch_dtype=torch.bfloat16)
            if hasattr(pipe.scheduler, "config") and "shift" in getattr(pipe.scheduler, "config", {}):
                pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, shift=FLOW_SHIFT)
            pipe.to("cuda")
            _pipes[kind] = pipe
            _model_error = None
        except Exception as error:  # noqa: BLE001
            _model_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            _load_seconds = round(time.time() - started, 1)
        return _pipes[kind]


def get_image(url: str, directory: Path) -> Path:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "image/png").split(";", 1)[0]
    suffix = ".jpg" if content_type == "image/jpeg" else ".png"
    path = directory / f"source{suffix}"
    path.write_bytes(response.content)
    return path


def validate_frames(frames: int) -> None:
    if not 49 <= frames <= 121:
        raise ValueError("frames must be between 49 and 121")
    if (frames - 1) % 4 != 0:
        raise ValueError("frames must satisfy Wan's 4n+1 requirement")


def health_report() -> dict[str, Any]:
    return {
        "ok": True,
        "worker": "wan-diffusers",
        "worker_version": 3,
        "model_id": MODEL_ID,
        "model_cache_ready": any(MODEL_CACHE.glob("models--*")) if MODEL_CACHE.exists() else False,
        "model_loaded": "t2v" in _pipes,
        "model_download_seconds": _download_seconds,
        "model_load_seconds": _load_seconds,
        "model_error": _model_error,
        "default_steps": DEFAULT_STEPS,
        "default_guidance": DEFAULT_GUIDANCE,
        "supports_i2v": SUPPORTS_I2V,
        "generation_not_run": True,
    }


def generate(job_input: dict[str, Any]) -> dict[str, Any]:
    if job_input.get("action") == "health":
        return health_report()

    mode = job_input.get("mode")
    if mode not in {"t2v", "i2v"}:
        return fail("mode must be t2v or i2v")
    if mode == "i2v" and not SUPPORTS_I2V:
        return fail("this worker/model does not support i2v (continuity); use the Wan 2.2 base worker")
    prompt = str(job_input.get("prompt", "")).strip()
    if not prompt:
        return fail("prompt is required")
    output_upload_url = str(job_input.get("output_upload_url", "")).strip()
    if not output_upload_url:
        return fail("output_upload_url is required")
    output_key = str(job_input.get("output_key", "")).strip()
    if not output_key:
        return fail("output_key is required")

    seed = int(job_input.get("seed", 123))
    steps = int(job_input.get("steps", DEFAULT_STEPS))
    frames = int(job_input.get("frames", 49))
    guidance = float(job_input.get("guidance_scale", DEFAULT_GUIDANCE))
    negative_prompt = str(job_input.get("negative_prompt", "") or "")
    width = int(job_input.get("width", 1280))
    height = int(job_input.get("height", 704))
    try:
        validate_frames(frames)
    except ValueError as error:
        return fail(str(error))
    if not 1 <= steps <= 50:
        return fail("steps must be between 1 and 50")
    if (width, height) not in SIZES:
        return fail(f"unsupported size {width}x{height}")

    try:
        pipe = get_pipe(mode)
    except Exception as error:  # noqa: BLE001
        return fail(f"model load failed: {error}")

    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="video-factory-") as temp_dir:
        temp = Path(temp_dir)
        kwargs: dict[str, Any] = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        if mode == "i2v":
            image_url = str(job_input.get("image_url", "")).strip()
            if not image_url:
                return fail("image_url is required for i2v")
            kwargs["image"] = Image.open(get_image(image_url, temp)).convert("RGB").resize((width, height))

        started = time.time()
        result = pipe(**kwargs)
        video_frames = result.frames[0]
        generation_seconds = round(time.time() - started, 1)

        output_path = temp / f"{uuid.uuid4().hex}.mp4"
        export_to_video(video_frames, str(output_path), fps=FPS)
        if not output_path.exists() or output_path.stat().st_size == 0:
            return fail("pipeline did not produce an MP4 output")

        with output_path.open("rb") as stream:
            upload = requests.put(output_upload_url, data=stream, headers={"content-type": "video/mp4"}, timeout=300)
        upload.raise_for_status()
        return {
            "ok": True,
            "mode": mode,
            "model_id": MODEL_ID,
            "resolution": f"{width}x{height}",
            "frames": frames,
            "steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "generation_seconds": generation_seconds,
            "model_load_seconds": _load_seconds,
            "model_download_seconds": _download_seconds,
            "output_key": output_key,
        }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return generate(job.get("input", {}))
    except requests.RequestException as error:
        return fail(f"asset transfer failed: {error}")
    except Exception as error:  # noqa: BLE001
        return fail(f"unexpected worker error: {error}")


if PRELOAD:
    try:
        get_pipe("t2v")
    except Exception as error:  # noqa: BLE001
        print(f"[worker] model preload failed: {error}", file=sys.stderr, flush=True)

runpod.serverless.start({"handler": handler})
