"""Run Qwen3 models on CUDA and save the results as Markdown."""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-4B"
MODEL_CACHE_DIR = "./models/qwen3-4b"
OUTPUT_PATH = Path("docs/getting-started/qwen3-4b-gpu-inference-results.md")
QUESTIONS = [
    "日本の首都はどこですか。答えだけを日本語で出力してください。",
    "LLMの重みを4ビットに量子化する目的を一文で説明してください。",
    "12 x 13 の計算結果だけを答えてください。",
    "次の文章を20字以内で要約してください: LLMの量子化は重みを低ビットで表現してメモリ使用量を減らす技術です。",
    "Pythonでリストの要素数を返す組み込み関数を答えてください。",
]
COMPLEX_QUESTIONS = [
    "月額料金がBasic 2,000円、Pro 6,000円、Enterprise 20,000円のSaaSがある。契約数は順に120、45、10件である。翌月にBasicの10%が解約し、Proが5件増えた場合、月次売上はいくら増減するか。現在売上、翌月売上、増減を各1行で示してください。",
    "次の注文を税込10%で集計し、品目ごとの小計、税抜合計、消費税、税込合計をJSONだけで出力してください。コードフェンスや説明文は使わないでください: コーヒー350円を2個、サンドイッチ680円を1個、サラダ420円を3個。",
    "Pythonで、辞書のリストから顧客ごとの購入合計を返す関数を書いてください。各要素はcustomer_idとamountを持ち、空リストでは空の辞書を返し、amountが負ならValueErrorを送出してください。型ヒントと使用例を含め、コードだけを25行以内で出力してください。",
    "障害対応の優先順位を考えてください。Webサービスで、A: 一部ユーザーの検索が遅い、B: 全ユーザーが決済できない、C: 管理画面のCSV出力が失敗する、という報告が同時に来た。優先順位と各項目の最初の確認手順を、B/A/Cの順に各1行、合計3行で示してください。",
    "次の主張を評価してください: 『モデルを4-bit量子化すれば、常に推論が高速になり、精度低下も起きない』。正しい点を1行、誤りを2行、実運用前の測定項目を3行で示してください。各行は簡潔にしてください。",
]
PHILOSOPHY_QUESTIONS = [
    "人の同一性は、記憶、身体、性格、他者との関係のうち何によって最も強く支えられるでしょうか。立場を一つ選び、反対の立場への反論も含めて300字以内で述べてください。",
    "『幸福を最大化すること』と『他者の権利を侵害しないこと』が衝突する具体例を一つ挙げてください。次に功利主義の判断を1行、義務論の判断を1行、相違点を1行で示してください。",
    "自由意志が完全には存在しないとしても、責任を問う制度は正当化できるでしょうか。賛成理由を1行、反対理由を1行、結論を1行で示してください。",
    "AIが人間と区別できない会話をし、苦痛を報告した場合、それを道徳的に配慮すべきでしょうか。判断基準を3行、結論を1行で示してください。意識の有無を断定せず、判断に必要な証拠も述べてください。",
    "『人生に客観的な意味はない』という見解は、虚無主義に直結するでしょうか。実存主義の観点から、簡潔に説明してください。",
]


def generate(model, tokenizer, question, max_new_tokens):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    started_at = perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed_seconds = perf_counter() - started_at
    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return answer, elapsed_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-cache-dir", default=MODEL_CACHE_DIR)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--question-set", choices=("basic", "complex", "philosophy"), default="basic"
    )
    arguments = parser.parse_args()

    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required.")

    device = "cuda:0" if arguments.device == "cuda" else "cpu"
    question_sets = {
        "basic": QUESTIONS,
        "complex": COMPLEX_QUESTIONS,
        "philosophy": PHILOSOPHY_QUESTIONS,
    }
    questions = question_sets[arguments.question_set]

    tokenizer = AutoTokenizer.from_pretrained(arguments.model_id, cache_dir=arguments.model_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model_id,
        cache_dir=arguments.model_cache_dir,
        dtype=torch.bfloat16,
        device_map=device,
    )

    lines = [
        f"# {arguments.model_id} {arguments.device.upper()}推論結果",
        "",
        f"{device}上で、{arguments.model_id}をBF16で実行した結果です。",
        "モデルのロード時間は含めず、各回答の生成時間を計測しています。",
        "",
        "| 項目 | 値 |",
        "| --- | --- |",
        f"| モデル | `{arguments.model_id}` |",
        "| 推論形式 | Hugging Face Transformers |",
        "| データ型 | BF16 |",
        f"| 実行デバイス | `{device}` |",
        f"| GPU | `{torch.cuda.get_device_name(0)}` |" if arguments.device == "cuda" else "| GPU | 使用しない |",
        f"| PyTorch | `{torch.__version__}` |",
        f"| 生成設定 | greedy decoding、最大{arguments.max_new_tokens}トークン、thinking無効 |",
    ]

    for index, question in enumerate(questions, start=1):
        answer, elapsed_seconds = generate(model, tokenizer, question, arguments.max_new_tokens)
        lines.extend(
            [
                "",
                f"## 質問 {index}",
                "",
                f"**質問**: {question}",
                "",
                f"**生成時間**: {elapsed_seconds:.2f}秒",
                "",
                "**出力**:",
                "",
                "````text",
                answer,
                "````",
            ]
        )

    arguments.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved results to {arguments.output_path}")


if __name__ == "__main__":
    main()