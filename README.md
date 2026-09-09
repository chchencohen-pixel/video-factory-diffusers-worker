# Video Factory — diffusers worker (v3)

Generic Runpod Serverless worker for Wan 2.2-family models in diffusers format. Default model:
`FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` (DMD-distilled, 3 steps, no CFG) — the cost lever for 720p.
Swap `MODEL_ID` (and `DEFAULT_STEPS`/`DEFAULT_GUIDANCE`/`SUPPORTS_I2V`) on the template to run the base
`Wan-AI/Wan2.2-TI2V-5B-Diffusers` or other candidates for the A/B.

The model is downloaded at boot into the container disk (no Network Volume, no Model Caching needed);
the pipeline stays resident. Same job contract as the v2 worker.

`python run_contract_tests.py` — static checks only.
