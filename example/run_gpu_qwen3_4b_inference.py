"""Run Qwen3-4B on a CUDA GPU with Hugging Face Transformers."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-4B"
MODEL_CACHE_DIR = "./models/qwen3-4b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="量子化されたLLMの利点を簡潔に説明してください。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    messages = [{"role": "user", "content": arguments.prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=arguments.max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip())


if __name__ == "__main__":
    main()