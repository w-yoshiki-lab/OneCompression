"""
Qwen3-14B: GPTQ 4-bit (mse+actorder+QEP) quantization and HF inference

Environment: NVIDIA A10G (23 GB VRAM), 30 GB RAM
  - FP16 load (~28 GB) uses device_map="auto" (GPU + CPU offload)
  - After quantization (~8 GB) fits fully in VRAM

Storage layout:
  - /data            : Instance Store (419 GB, ephemeral — lost on VM stop)
  - / (EBS)          : 25 GB root, only ~2.8 GB free — too small for this model
  SAVE_DIR points to /data; model is pushed to HF Hub before the session ends.

  !! IMPORTANT: Push to HF Hub (step 6) BEFORE stopping the VM !!

Prerequisites:
  sudo mkfs -t xfs /dev/nvme1n1 && sudo mkdir -p /data
  sudo mount /dev/nvme1n1 /data && sudo chown ec2-user:ec2-user /data
  export HF_TOKEN=<your_huggingface_token>

Copyright 2025-2026 Fujitsu Ltd.
"""

import gc
import os

# Redirect HF cache to /data (Instance Store, 419 GB) before any HF imports
os.environ.setdefault("HF_HOME", "/data/.cache/huggingface")

import torch

from onecomp import (
    GPTQ,
    CalibrationConfig,
    ModelConfig,
    Runner,
    load_quantized_model,
    push_to_hub,
    setup_logger,
)

setup_logger()

MODEL_ID = "Qwen/Qwen3-8B"
# /data = Instance Store (419 GB); ephemeral but large enough for this model
SAVE_DIR = "/data/Qwen3-8B-gptq-4bit-qep"
# Set HF_HUB_REPO to your "<username>/<repo-name>" to enable auto-upload
HF_HUB_REPO = os.environ.get("HF_HUB_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── 0. Pre-flight check ───────────────────────────────────────
if not os.path.ismount("/data"):
    raise RuntimeError(
        "/data is not mounted. Run:\n"
        "  sudo mkfs -t xfs /dev/nvme1n1 && sudo mkdir -p /data\n"
        "  sudo mount /dev/nvme1n1 /data && sudo chown ec2-user:ec2-user /data"
    )
if not HF_HUB_REPO:
    print(
        "WARNING: HF_HUB_REPO is not set. The quantized model will NOT be uploaded.\n"
        "  Data in /data (Instance Store) is LOST when the VM is stopped.\n"
        "  Set:  export HF_HUB_REPO=<username>/<repo>  before running."
    )

# ── 1. Quantize ───────────────────────────────────────────────
# Qwen3-8B (~16 GB FP16) fits entirely in VRAM; no CPU offload needed
model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")

# mse + actorder + QEP = highest accuracy GPTQ configuration
gptq = GPTQ(
    wbits=4,
    groupsize=128,
    mse=True,
    actorder=True,
)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    qep=False,  # QEP loads a second CPU copy (~16 GB RAM) which exceeds 30 GB
    calibration_config=CalibrationConfig(
        num_calibration_samples=512,
        max_length=1024,  # reduced from 2048 to halve activation tensor size
        batch_size=8,     # reduced from 32 to limit Hessian float64 allocation
    ),
)

runner.run()

# ── 2. Evaluate perplexity ────────────────────────────────────
original_ppl, _, quantized_ppl = runner.calculate_perplexity(
    original_model=True, dequantized_model=False, quantized_model=True
)
print(f"Original PPL : {original_ppl:.4f}")
print(f"Quantized PPL: {quantized_ppl:.4f}")

# ── 3. Save ───────────────────────────────────────────────────
runner.save_quantized_model(SAVE_DIR)
print(f"\nSaved to: {SAVE_DIR}")

# ── 4. Upload to HF Hub BEFORE stopping the VM ───────────────
if HF_HUB_REPO:
    print(f"\nUploading to Hugging Face Hub: {HF_HUB_REPO} ...")
    url = push_to_hub(SAVE_DIR, HF_HUB_REPO, private=True, token=HF_TOKEN or None)
    print(f"Uploaded: {url}")
else:
    print("\nSkipping HF Hub upload (HF_HUB_REPO not set).")
    print("!! Stop the VM only after uploading — /data is ephemeral !!")

# ── 5. Free GPU memory ────────────────────────────────────────
del runner
gc.collect()
torch.cuda.empty_cache()

# ── 6. Load and infer (HF generate, no vLLM required) ─────────
model, tokenizer = load_quantized_model(SAVE_DIR)
print(f"Loaded: {type(model).__name__} on {next(model.parameters()).device}")

prompts = [
    "ポストトレーニング量子化を一文で説明してください:",
    "富士通の量子化技術の特徴は",
    "The key advantage of GPTQ quantization is",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nPrompt   : {prompt}")
    print(f"Generated: {generated}")
