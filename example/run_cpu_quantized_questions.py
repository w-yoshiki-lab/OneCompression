"""Run comparison prompts against the saved CPU GGUF model."""

from pathlib import Path
from time import perf_counter

from llama_cpp import Llama

MODEL_PATH = "./qwen3-0.6b-gptq4/qwen3-0.6b-gptq4.gguf"
RESULT_PATH = Path("./docs/getting-started/qwen3-0.6b-gptq4-inference-results.md")
QUESTIONS = [
    "日本の首都はどこですか。答えだけを日本語で出力してください。",
    "LLMの重みを4ビットに量子化する目的を一文で説明してください。",
    "12 x 13 の計算結果だけを答えてください。",
    "次の文章を20字以内で要約してください: LLMの量子化は重みを低ビットで表現してメモリ使用量を減らす技術です。",
    "Pythonでリストの要素数を返す組み込み関数を答えてください。",
]

model = Llama(MODEL_PATH, n_ctx=2048, n_threads=2, verbose=False)
lines = [
    "# Qwen3-0.6B GPTQ 4-bit 推論結果",
    "",
    "保存済みGGUFモデルをCPUの2スレッドで実行した結果です。",
    "",
    "| 項目 | 値 |",
    "| --- | --- |",
    "| モデル | `Qwen3-0.6B` |",
    "| 量子化 | GPTQ 4-bit、groupsize 128 |",
    "| 推論形式 | llama.cpp GGUF |",
    "| CPUスレッド数 | 2 |",
]

for index, question in enumerate(QUESTIONS, start=1):
    started = perf_counter()
    response = model.create_chat_completion(
        messages=[{"role": "user", "content": question}],
        max_tokens=48,
        temperature=0.0,
    )
    answer = response["choices"][0]["message"]["content"]
    elapsed = perf_counter() - started
    lines.extend(
        [
            "",
            f"## 質問 {index}",
            "",
            f"**質問**: {question}",
            "",
            f"**所要時間**: {elapsed:.2f}秒",
            "",
            "**出力**:",
            "",
            "````text",
            answer.strip(),
            "````",
        ]
    )

RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Saved results to {RESULT_PATH}")