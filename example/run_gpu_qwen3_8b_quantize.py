"""Quantize Qwen3-8B with GPTQ 4-bit on CUDA and verify the saved model."""

import argparse
import gc
import subprocess
from pathlib import Path

import torch

from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner, load_quantized_model, setup_logger


MODEL_PATH = Path(
    "models/qwen3-8b/models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
SAVE_DIR = Path("qwen3-8b-gptq4")
PROMPT = "LLMの重みを4ビットに量子化する目的を一文で説明してください。"


def gpu_memory_mib():
    torch.cuda.synchronize()
    allocated_mib = torch.cuda.memory_allocated() / 1024**2
    used_mib = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )
    return allocated_mib, used_mib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--num-calibration-samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")
    if not arguments.model_path.is_dir():
        raise FileNotFoundError(f"Model snapshot not found: {arguments.model_path}")
    if arguments.save_dir.exists():
        raise FileExistsError(f"Save directory already exists: {arguments.save_dir}")

    setup_logger()
    model_config = ModelConfig(
        path=str(arguments.model_path), dtype="bfloat16", device="cuda:0"
    )
    runner = Runner(
        model_config=model_config,
        quantizer=GPTQ(wbits=4, groupsize=128, mse=True, actorder=True),
        calibration_config=CalibrationConfig(
            max_length=arguments.max_length,
            num_calibration_samples=arguments.num_calibration_samples,
            batch_size=1,
        ),
        qep=False,
    )
    runner.run()
    runner.save_quantized_model(str(arguments.save_dir))

    del runner
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = load_quantized_model(str(arguments.save_dir), device_map="cuda:0")
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    answer = tokenizer.decode(output_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    allocated_mib, used_mib = gpu_memory_mib()
    model_size_mib = sum(path.stat().st_size for path in arguments.save_dir.rglob("*") if path.is_file())

    print(f"Saved GPTQ 4-bit model: {arguments.save_dir}")
    print(f"Saved model size: {model_size_mib / 1024**2:.1f} MiB")
    print(f"PyTorch allocated VRAM: {allocated_mib:.0f} MiB")
    print(f"GPU used VRAM: {used_mib} MiB")
    print(f"Generated answer: {answer.strip()}")


if __name__ == "__main__":
    main()