# CPU GPTQ量子化の実行記録

このドキュメントは、2026-09-01に開始したCPUのみを使用するGPTQ量子化の実行記録です。

## VM環境

| 項目 | 値 |
| --- | --- |
| OS | Amazon Linux 2023、Linux 6.12.95-124.187.amzn2023.x86_64 |
| アーキテクチャ | x86_64 |
| CPU | 論理4コア |
| メモリ | 15 GiB RAM、スワップなし |
| ルートディスク | 100 GiB (`/`)、記録時点の空き約45 GiB |
| 一時ファイルシステム | 7.7 GiB (`/tmp`)、記録時点の空き2.0 GiB |
| Python | 3.12.13 |
| OneComp | 1.3.2 |
| PyTorch | 2.13.0+cpu |
| 仮想環境 | `.venv`（リポジトリ配下、永続） |

量子化時に使用した`/tmp/onecomp_venv`は一時環境です。再起動後のGGUF推論には、
リポジトリ配下の`.venv`を使用します。

## 量子化の設定

| 項目 | 値 |
| --- | --- |
| 元モデル | `Qwen/Qwen3-0.6B` |
| 実行デバイス | CPU |
| モデルのデータ型 | bfloat16 |
| 量子化器 | GPTQ |
| 重みのビット幅 | 4-bit |
| グループサイズ | 128 |
| QEP | 無効 |
| キャリブレーションデータセット | AllenAI C4 |
| キャリブレーションサンプル数 | 32 |
| キャリブレーション系列長 | 512トークン |
| 量子化時間 | 196個のLinearレイヤーに約33分 |

実行したスクリプトは [run_cpu_quantize_demo.py](../../example/run_cpu_quantize_demo.py) です。

## 保存済みモデル

量子化済みモデルは次の場所に保存されています。

```text
./qwen3-0.6b-gptq4
```

主要な保存ファイル:

| ファイル | サイズ |
| --- | ---: |
| `model.safetensors` | 541,343,840 bytes (約516 MiB) |
| `qwen3-0.6b-gptq4.gguf` | 約836 MiB |
| `config.json` | 47,823 bytes |
| `tokenizer.json` | 11,422,650 bytes |
| `vocab.json` | 2,776,833 bytes |

保存先はリポジトリ配下です。永続ルートディスクを維持した通常のVM再起動であれば、
モデルも保持されます。

## 再起動後の操作手順

VM再起動後は、リポジトリへ移動して永続仮想環境を有効化します。再インストール、
再量子化、GGUFへの再変換は必要ありません。

```bash
cd /home/ec2-user/OneCompression
source .venv/bin/activate
```

次のコマンドで、保存済みGGUFモデルのロードと短い生成を確認できます。

```bash
python -c "from llama_cpp import Llama; model = Llama(model_path='qwen3-0.6b-gptq4/qwen3-0.6b-gptq4.gguf', n_ctx=512, n_threads=2, verbose=False); result = model.create_completion(prompt='The capital of Japan is', max_tokens=8, temperature=0.0); print(result['choices'][0]['text'].strip())"
```

この環境には`torch 2.13.0+cpu`と`llama-cpp-python 0.3.35`が導入済みです。モデル、
`.venv`、およびリポジトリが残る限り、上記の手順だけでCPU推論を再開できます。

## GPUでQwen3-4B/8Bを実行

`g6.xlarge`に搭載されたNVIDIA L4（23 GiB VRAM）では、現在の空き容量に収まる
`Qwen/Qwen3-4B`をBF16でGPU常駐実行できます。0.6BのCPU量子化モデルより高い回答能力を
持つモデルです。

GPU用の仮想環境とモデルは、再起動後も保持される次の場所にあります。

| 項目 | 保存先 | サイズの目安 |
| --- | --- | ---: |
| GPU仮想環境 | `.venv-gpu` | 約5.4 GiB |
| Qwen3-4Bモデル | `models/qwen3-4b` | 約7.6 GiB |
| Qwen3-8Bモデル | `models/qwen3-8b` | 約16 GiB |

再起動後は、次のコマンドでGPU推論を実行します。初回のモデルダウンロードは完了済みのため、
通常はモデルロード後に応答を生成します。

```bash
cd /home/ec2-user/OneCompression
source .venv-gpu/bin/activate
python example/run_gpu_qwen3_4b_inference.py \
	--prompt "量子化されたLLMの利点を日本語で二文以内に説明してください。"
```

動作確認済みの環境は`torch 2.7.1+cu126`、CUDA有効、BF16有効です。現在のルートディスクには
約45 GiBの空きがあり、`Qwen/Qwen3-8B`のBF16モデルも保存済みです。8Bモデルを5問の
質問セットで実行し、結果をMarkdownへ保存するには次を実行します。

```bash
cd /home/ec2-user/OneCompression
source .venv-gpu/bin/activate
python example/run_gpu_qwen3_4b_questions.py \
	--model-id Qwen/Qwen3-8B \
	--model-cache-dir ./models/qwen3-8b \
	--output-path docs/getting-started/qwen3-8b-gpu-inference-results.md \
	--max-new-tokens 48 \
	--device cuda
```

実測結果は [qwen3-8b-gpu-inference-results.md](qwen3-8b-gpu-inference-results.md) にあります。

## 量子化チェックポイントの推論

保存済みモデルをCPUで読み込み、リポジトリのルートから推論例を実行します。

```bash
source /tmp/onecomp_venv/bin/activate
python example/run_cpu_quantized_inference.py
```

この例は`load_quantized_model()`で`./qwen3-0.6b-gptq4`をロードし、日本語プロンプトに対して
最大32トークンを生成します。ただし、PyTorch GPTQのCPU生成は非常に遅いため、通常のCPU推論には
前節のGGUF経路を使用してください。ソースは
[run_cpu_quantized_inference.py](../../example/run_cpu_quantized_inference.py) です。

## 評価の状態

この実行では、任意のperplexity評価より前にモデルを保存します。現在のスクリプトは
`max_samples=5`で元モデルと量子化モデルを評価します。評価は保存後に実行されるため、
ログに保存完了メッセージが出た時点で、評価の終了を待たず推論に利用できます。

バックグラウンドジョブは次のコマンドで確認できます。

```bash
tail -f /tmp/quantize_demo2.log
```

## ストレージの注意

この実行時点ではルートディスクの空き容量が少ない状態でした。容量を確保するかディスクを
拡張するまで、大きなモデルのダウンロードや追加の量子化実行は開始しないでください。
再起動後も保持するモデルは、`/tmp`ではなく`/home/ec2-user/OneCompression`配下に保存します。