"""Compare BF16 and GPTQ 4-bit Qwen3-8B on the existing 15 questions."""

import argparse
import gc
import json
import subprocess
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from onecomp import load_quantized_model
from run_gpu_qwen3_4b_questions import COMPLEX_QUESTIONS, PHILOSOPHY_QUESTIONS, QUESTIONS


BF16_MODEL_PATH = Path(
    "models/qwen3-8b/models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
GPTQ_MODEL_PATH = Path("qwen3-8b-gptq4")
OUTPUT_PATH = Path("docs/getting-started/qwen3-8b-gptq4-comparison-results.md")
QUESTION_SETS = (
    ("基本", QUESTIONS, 48),
    ("複雑", COMPLEX_QUESTIONS, 256),
    ("哲学", PHILOSOPHY_QUESTIONS, 384),
)


def storage_gib(model_path):
    return sum(path.stat().st_size for path in model_path.glob("*.safetensors")) / 1024**3


def gpu_used_mib():
    return int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )


def load_progress(progress_path):
    if not progress_path.exists():
        return {"models": {}}
    return json.loads(progress_path.read_text(encoding="utf-8"))


def save_progress(progress_path, progress):
    temporary_path = progress_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(progress_path)


def generate(model, tokenizer, question, max_new_tokens):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    started_at = perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed_seconds = perf_counter() - started_at
    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return answer, elapsed_seconds, generated_ids.shape[1]


def load_bf16(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
    )
    return model, tokenizer


def run_model(label, loader, progress, progress_path):
    torch.cuda.empty_cache()
    print(f"{label}: loading model", flush=True)
    model, tokenizer = loader()
    torch.cuda.synchronize()
    allocated_mib = torch.cuda.memory_allocated() / 1024**2
    used_mib = gpu_used_mib()
    model_progress = progress["models"].setdefault(label, {"results": []})
    model_progress["allocated_mib"] = allocated_mib
    model_progress["used_mib"] = used_mib
    completed_questions = {
        (result["set_name"], result["question_index"])
        for result in model_progress["results"]
    }
    for set_name, questions, max_new_tokens in QUESTION_SETS:
        for question_index, question in enumerate(questions, start=1):
            if (set_name, question_index) in completed_questions:
                print(f"{label} {set_name} {question_index}/5: already completed", flush=True)
                continue
            answer, elapsed_seconds, generated_tokens = generate(
                model, tokenizer, question, max_new_tokens
            )
            model_progress["results"].append(
                {
                    "set_name": set_name,
                    "question_index": question_index,
                    "question": question,
                    "max_new_tokens": max_new_tokens,
                    "answer": answer,
                    "elapsed_seconds": elapsed_seconds,
                    "generated_tokens": generated_tokens,
                }
            )
            save_progress(progress_path, progress)
            print(
                f"{label} {set_name} {question_index}/5: "
                f"{elapsed_seconds:.2f}s, {generated_tokens} tokens",
                flush=True,
            )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return model_progress["results"], allocated_mib, used_mib


def format_answer(answer):
    return ["````text", answer, "````"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-model-path", type=Path, default=BF16_MODEL_PATH)
    parser.add_argument("--gptq-model-path", type=Path, default=GPTQ_MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--progress-path", type=Path)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard the saved progress and run every question again.",
    )
    arguments = parser.parse_args()
    progress_path = arguments.progress_path or arguments.output_path.with_suffix(".progress.json")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")
    for path in (arguments.bf16_model_path, arguments.gptq_model_path):
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {path}")

    if arguments.restart and progress_path.exists():
        progress_path.unlink()
    progress = load_progress(progress_path)

    saved_bf16 = progress["models"].get("BF16", {})
    if len(saved_bf16.get("results", [])) == 15:
        print("BF16: all 15 questions already completed; skipping model load", flush=True)
        bf16_results = saved_bf16["results"]
        bf16_allocated_mib = saved_bf16["allocated_mib"]
        bf16_used_mib = saved_bf16["used_mib"]
    else:
        bf16_results, bf16_allocated_mib, bf16_used_mib = run_model(
            "BF16", lambda: load_bf16(arguments.bf16_model_path), progress, progress_path
        )
    gptq_results, gptq_allocated_mib, gptq_used_mib = run_model(
        "GPTQ 4-bit",
        lambda: load_quantized_model(str(arguments.gptq_model_path), device_map="cuda:0"),
        progress,
        progress_path,
    )

    if len(bf16_results) != 15 or len(gptq_results) != 15:
        raise RuntimeError(f"Incomplete progress retained in {progress_path}")

    bf16_storage_gib = storage_gib(arguments.bf16_model_path)
    gptq_storage_gib = storage_gib(arguments.gptq_model_path)
    storage_reduction = (1 - gptq_storage_gib / bf16_storage_gib) * 100
    vram_reduction = (1 - gptq_used_mib / bf16_used_mib) * 100
    total_bf16_seconds = sum(result["elapsed_seconds"] for result in bf16_results)
    total_gptq_seconds = sum(result["elapsed_seconds"] for result in gptq_results)
    total_bf16_tokens = sum(result["generated_tokens"] for result in bf16_results)
    total_gptq_tokens = sum(result["generated_tokens"] for result in gptq_results)

    lines = [
        "# Qwen3-8B BF16とGPTQ 4-bitの15問比較",
        "",
        "NVIDIA L4上で、同じQwen3-8Bに過去の基本・複雑・哲学の15問を実行しました。",
        "各モデルは別々にロードし、greedy decoding、thinking無効で測定しています。",
        "",
        "## 条件",
        "",
        "| 項目 | 値 |",
        "| --- | --- |",
        "| GPU | NVIDIA L4 |",
        f"| PyTorch | `{torch.__version__}` |",
        "| BF16モデル | `Qwen/Qwen3-8B` |",
        "| GPTQモデル | 4-bit、group size 128、MSE・actorder有効 |",
        "| 基本質問 | 最大48トークン |",
        "| 複雑質問 | 最大256トークン |",
        "| 哲学質問 | 最大384トークン |",
        "",
        "## 量子化の効果",
        "",
        "| 指標 | BF16 | GPTQ 4-bit | 効果 |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| 保存済みSafetensors重み | {bf16_storage_gib:.2f} GiB | "
            f"{gptq_storage_gib:.2f} GiB | {storage_reduction:.1f}%削減 |"
        ),
        (
            f"| ロード直後のPyTorch VRAM | {bf16_allocated_mib:.0f} MiB | "
            f"{gptq_allocated_mib:.0f} MiB | "
            f"{(1 - gptq_allocated_mib / bf16_allocated_mib) * 100:.1f}%削減 |"
        ),
        (
            f"| ロード直後のGPU使用VRAM | {bf16_used_mib} MiB | {gptq_used_mib} MiB | "
            f"{vram_reduction:.1f}%削減 |"
        ),
        (
            f"| 15問の合計生成時間 | {total_bf16_seconds:.2f}秒 | {total_gptq_seconds:.2f}秒 | "
            f"{total_gptq_seconds / total_bf16_seconds:.2f}倍 |"
        ),
        (
            f"| 生成速度 | {total_bf16_tokens / total_bf16_seconds:.2f} tokens/s | "
            f"{total_gptq_tokens / total_gptq_seconds:.2f} tokens/s | "
            f"{(total_gptq_tokens / total_gptq_seconds) / (total_bf16_tokens / total_bf16_seconds):.2f}倍 |"
        ),
        "",
        "生成時間と生成トークン数は、モデル出力の違いにより完全には同一ではありません。",
        "回答品質は以下の各出力を比較して評価してください。",
    ]

    for index, (bf16_result, gptq_result) in enumerate(zip(bf16_results, gptq_results), start=1):
        lines.extend(
            [
                "",
                f"## 質問 {index}: {bf16_result['set_name']} {bf16_result['question_index']}",
                "",
                f"**質問**: {bf16_result['question']}",
                "",
                "| 指標 | BF16 | GPTQ 4-bit |",
                "| --- | ---: | ---: |",
                (
                    f"| 生成時間 | {bf16_result['elapsed_seconds']:.2f}秒 | "
                    f"{gptq_result['elapsed_seconds']:.2f}秒 |"
                ),
                (
                    f"| 生成トークン数 | {bf16_result['generated_tokens']} | "
                    f"{gptq_result['generated_tokens']} |"
                ),
                "",
                "**BF16出力**:",
                "",
                *format_answer(bf16_result["answer"]),
                "",
                "**GPTQ 4-bit出力**:",
                "",
                *format_answer(gptq_result["answer"]),
            ]
        )

    arguments.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved comparison to {arguments.output_path}")


if __name__ == "__main__":
    main()