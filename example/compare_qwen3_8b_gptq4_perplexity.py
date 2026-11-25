"""Measure BF16 and GPTQ 4-bit perplexity on a small fixed Japanese corpus."""

import argparse
import gc
import math
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from onecomp import load_quantized_model


BF16_MODEL_PATH = Path(
    "models/qwen3-8b/models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
GPTQ_MODEL_PATH = Path("qwen3-8b-gptq4")
OUTPUT_PATH = Path("docs/getting-started/qwen3-8b-gptq4-perplexity.md")
EVALUATION_TEXTS = (
    "日本の首都は東京です。東京は日本の政治、経済、文化の中心地として知られています。",
    "量子化は、ニューラルネットワークの重みを低いビット数で表現し、保存容量とメモリ使用量を削減する技術です。",
    "障害対応では、影響範囲、緊急度、復旧の見込みを確認し、利用者への影響が大きい問題から優先して対処します。",
)


def load_bf16(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
    )
    return model, tokenizer


def evaluate_perplexity(model, tokenizer):
    total_negative_log_likelihood = 0.0
    total_predicted_tokens = 0
    sample_results = []

    for index, text in enumerate(EVALUATION_TEXTS, start=1):
        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda:0")
        predicted_tokens = input_ids.shape[1] - 1
        if predicted_tokens <= 0:
            raise ValueError("Each evaluation text must contain at least two tokens.")

        started_at = perf_counter()
        with torch.inference_mode():
            loss = model(input_ids=input_ids, labels=input_ids).loss
        torch.cuda.synchronize()
        elapsed_seconds = perf_counter() - started_at
        negative_log_likelihood = loss.float().item() * predicted_tokens
        total_negative_log_likelihood += negative_log_likelihood
        total_predicted_tokens += predicted_tokens
        sample_results.append(
            {
                "index": index,
                "text": text,
                "predicted_tokens": predicted_tokens,
                "loss": loss.float().item(),
                "perplexity": math.exp(loss.float().item()),
                "elapsed_seconds": elapsed_seconds,
            }
        )
        print(
            f"sample {index}/{len(EVALUATION_TEXTS)}: "
            f"loss={loss.float().item():.4f}, ppl={math.exp(loss.float().item()):.4f}, "
            f"{elapsed_seconds:.2f}s",
            flush=True,
        )

    mean_loss = total_negative_log_likelihood / total_predicted_tokens
    return {
        "samples": sample_results,
        "predicted_tokens": total_predicted_tokens,
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
    }


def run_model(label, loader):
    torch.cuda.empty_cache()
    print(f"{label}: loading model", flush=True)
    model, tokenizer = loader()
    results = evaluate_perplexity(model, tokenizer)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-model-path", type=Path, default=BF16_MODEL_PATH)
    parser.add_argument("--gptq-model-path", type=Path, default=GPTQ_MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")
    for path in (arguments.bf16_model_path, arguments.gptq_model_path):
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {path}")

    bf16_results = run_model("BF16", lambda: load_bf16(arguments.bf16_model_path))
    gptq_results = run_model(
        "GPTQ 4-bit",
        lambda: load_quantized_model(str(arguments.gptq_model_path), device_map="cuda:0"),
    )
    ppl_ratio = gptq_results["perplexity"] / bf16_results["perplexity"]
    ppl_increase = (ppl_ratio - 1) * 100

    lines = [
        "# Qwen3-8B BF16とGPTQ 4-bitのPerplexity比較",
        "",
        "NVIDIA L4上で、固定した短い日本語3文に対する次トークン予測損失を測定した。",
        "各モデルで同じ評価文をtokenizeし、予測対象トークン数で重み付けした平均損失からPPLを算出している。",
        "",
        "$$",
        "\\mathrm{PPL} = \\exp\\left(-\\frac{1}{N}\\sum_{i=1}^{N}\\log p(x_i \\mid x_{<i})\\right)",
        "$$",
        "",
        "## 条件",
        "",
        "| 項目 | 値 |",
        "| --- | --- |",
        "| GPU | NVIDIA L4 |",
        f"| PyTorch | `{torch.__version__}` |",
        "| BF16モデル | `Qwen/Qwen3-8B` |",
        "| GPTQモデル | 4-bit、group size 128、MSE・actorder有効 |",
        f"| 評価文数 | {len(EVALUATION_TEXTS)} |",
        "| 評価文 | 固定した短い日本語文、chat templateなし |",
        "",
        "## 結果",
        "",
        "| 指標 | BF16 | GPTQ 4-bit | 差 |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| 評価トークン数 | {bf16_results['predicted_tokens']} | "
            f"{gptq_results['predicted_tokens']} | - |"
        ),
        (
            f"| 平均NLL loss | {bf16_results['mean_loss']:.6f} | "
            f"{gptq_results['mean_loss']:.6f} | "
            f"{gptq_results['mean_loss'] - bf16_results['mean_loss']:+.6f} |"
        ),
        (
            f"| Perplexity | {bf16_results['perplexity']:.4f} | "
            f"{gptq_results['perplexity']:.4f} | {ppl_ratio:.4f}倍 ({ppl_increase:+.2f}%) |"
        ),
        "",
        "PPLは低いほど、評価文の次トークンを高い確率で予測できていることを示す。",
        "この3文は動作確認用の小規模コーパスであり、一般的な日本語能力全体を代表する評価ではない。",
    ]

    for bf16_sample, gptq_sample in zip(bf16_results["samples"], gptq_results["samples"]):
        lines.extend(
            [
                "",
                f"## 評価文 {bf16_sample['index']}",
                "",
                bf16_sample["text"],
                "",
                "| 指標 | BF16 | GPTQ 4-bit |",
                "| --- | ---: | ---: |",
                (
                    f"| 予測トークン数 | {bf16_sample['predicted_tokens']} | "
                    f"{gptq_sample['predicted_tokens']} |"
                ),
                f"| NLL loss | {bf16_sample['loss']:.6f} | {gptq_sample['loss']:.6f} |",
                (
                    f"| Perplexity | {bf16_sample['perplexity']:.4f} | "
                    f"{gptq_sample['perplexity']:.4f} |"
                ),
            ]
        )

    arguments.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved perplexity comparison to {arguments.output_path}")


if __name__ == "__main__":
    main()