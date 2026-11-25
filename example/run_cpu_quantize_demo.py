"""
CPU quantization demo: GPTQ 4-bit on Qwen3-0.6B (no GPU required)

Copyright 2025-2026 Fujitsu Ltd.
"""

import os

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner, setup_logger

setup_logger()

model_config = ModelConfig(model_id="Qwen/Qwen3-0.6B", dtype="bfloat16", device="cpu")

gptq = GPTQ(wbits=4, groupsize=128)

runner = Runner(
    model_config=model_config,
    quantizer=gptq,
    calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=32),
    qep=False,
)

runner.run()

SAVE_DIR = "./qwen3-0.6b-gptq4"
runner.save_quantized_model(SAVE_DIR)
print(f"Quantized model saved to: {SAVE_DIR}")

# Perplexity on a handful of samples only, to keep the demo fast on CPU.
original_ppl, _, quantized_ppl = runner.calculate_perplexity(
    original_model=True, dequantized_model=False, quantized_model=True, max_samples=5
)
print(f"Original model perplexity : {original_ppl}")
print(f"Quantized model perplexity: {quantized_ppl}")
