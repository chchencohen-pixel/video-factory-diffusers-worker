"""Static checks; these tests never download weights or invoke a GPU."""

from pathlib import Path

ROOT = Path(__file__).parent


def test_worker_uses_presigned_urls_and_no_application_secret() -> None:
    source = (ROOT / "handler.py").read_text()
    assert "output_upload_url is required" in source
    assert "output_key is required" in source
    assert '"output_key": output_key' in source
    assert "image_url is required for i2v" in source
    assert "requests.put(" in source
    assert "RUNPOD_API_KEY" not in source
    assert "AWS_SECRET_ACCESS_KEY" not in source


def test_worker_is_diffusers_based_and_resident() -> None:
    source = (ROOT / "handler.py").read_text()
    assert "WanPipeline" in source and "WanImageToVideoPipeline" in source
    assert "snapshot_download(" in source
    assert "def get_pipe" in source and "_lock" in source
    assert "export_to_video(" in source
    assert "validate_frames" in source
    assert 'runpod.serverless.start({"handler": handler})' in source


def test_dockerfile_pins_runtime_and_defaults() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert dockerfile.startswith("FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime")
    assert "diffusers" in dockerfile and "huggingface_hub" in dockerfile
    assert "MODEL_ID=FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers" in dockerfile
    assert "DEFAULT_STEPS=3" in dockerfile
    assert "ffmpeg" in dockerfile
