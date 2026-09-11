"""One-off feasibility test of open-weights MiniMax-H3 on a single 80GB GPU (Runpod pod).

Generates a few 9:16 clips (video only, audio discarded) with the official diffusers
modular pipeline, uploads each MP4 to a presigned URL, and writes a timing log.
Config comes from env: SCENES_JSON = [{"name","prompt","upload_url"}], LOG_UPLOAD_URL,
H3_FRAMES (default 124 = 5 s), H3_STEPS (default 40), H3_W/H3_H (default 768x1344).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import requests

LOG: list[str] = []


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.append(line)


def upload(url: str, path: str, content_type: str) -> int:
    with open(path, "rb") as f:
        r = requests.put(url, data=f, headers={"content-type": content_type}, timeout=600)
    return r.status_code


def main() -> None:
    scenes = json.loads(os.environ["SCENES_JSON"])
    log_url = os.environ.get("LOG_UPLOAD_URL", "")
    frames = int(os.environ.get("H3_FRAMES", "124"))
    steps = int(os.environ.get("H3_STEPS", "40"))
    width = int(os.environ.get("H3_W", "768"))
    height = int(os.environ.get("H3_H", "1344"))
    t0 = time.time()
    try:
        import torch
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.utils.export_utils import encode_video

        log(f"torch {torch.__version__} cuda {torch.cuda.is_available()} gpu {torch.cuda.get_device_name(0)}")
        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", components_manager=manager)
        log("downloading + loading components (t2va)…")
        pipe.load_components(workflow="t2va", dtype=torch.bfloat16)
        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
        try:
            pipe.transformer.set_attention_backend("_flash_3_hub")
            log("attention backend: flash3 hub")
        except Exception as e:  # noqa: BLE001
            log(f"flash3 backend unavailable ({e}); default attention")
        log(f"components ready after {time.time() - t0:.0f}s")

        for scene in scenes:
            name = scene["name"]
            t1 = time.time()
            log(f"generate {name}: {width}x{height} frames={frames} steps={steps}")
            results = pipe(
                prompt=scene["prompt"],
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=steps,
                generator=torch.Generator().manual_seed(int(scene.get("seed", 42))),
                output=["videos"],
            )
            gen_s = time.time() - t1
            out = f"/tmp/{name}.mp4"
            encode_video(results["videos"][0], fps=24, output_path=out)
            size = os.path.getsize(out)
            code = upload(scene["upload_url"], out, "video/mp4")
            log(f"done {name}: gen {gen_s:.0f}s, {size} bytes, upload HTTP {code}")
            del results
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        log("ERROR " + traceback.format_exc()[-1500:])
    finally:
        log(f"total {time.time() - t0:.0f}s")
        if log_url:
            with open("/tmp/h3_test.log", "w") as f:
                f.write("\n".join(LOG))
            code = upload(log_url, "/tmp/h3_test.log", "text/plain")
            print("log upload", code, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
