FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
    MODEL_ID=FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
    MODEL_CACHE=/tmp/models \
    DEFAULT_STEPS=3 \
    DEFAULT_GUIDANCE=1.0 \
    SUPPORTS_I2V=false \
    WAN_PRELOAD=true

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "diffusers>=0.36.0" "transformers>=4.51.0" accelerate==1.10.1 ftfy==6.3.1 \
       "imageio[ffmpeg]==2.37.0" imageio-ffmpeg==0.6.0 huggingface_hub==0.34.4 hf_transfer==0.1.9 \
       runpod==1.7.10 requests==2.32.3 pillow==11.3.0 sentencepiece==0.2.0 protobuf==5.29.5

COPY handler.py /opt/worker/handler.py
WORKDIR /opt/worker

# The model is downloaded at boot into MODEL_CACHE (container disk, >= 50GB recommended)
# because Runpod's REST API cannot attach Model Caching to a new endpoint.
CMD ["python", "handler.py"]
