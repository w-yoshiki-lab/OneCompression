"""
CPU inference demo using a saved GPTQ model.

Copyright 2025-2026 Fujitsu Ltd.
"""

import torch

from onecomp import load_quantized_model, setup_logger

setup_logger()

MODEL_DIR = "./qwen3-0.6b-gptq4"
PROMPT = "量子化されたLLMの利点を簡潔に説明してください。"

model, tokenizer = load_quantized_model(MODEL_DIR)
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt").to(next(model.parameters()).device)
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))