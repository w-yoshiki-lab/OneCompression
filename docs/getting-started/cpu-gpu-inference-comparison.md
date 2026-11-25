# CPU/GPU推論比較

CPUとGPUの差を確認するため、同じ`Qwen/Qwen3-0.6B`、BF16、Hugging Face
Transformers、greedy decoding、最大48トークン、thinking無効の条件で5問を実行しました。
モデルのロード時間は含めていません。

| 項目 | CPU | GPU |
| --- | --- | --- |
| 実行デバイス | CPU | NVIDIA L4 (`cuda:0`) |
| モデル | Qwen3-0.6B | Qwen3-0.6B |
| データ型 | BF16 | BF16 |
| 推論形式 | Transformers | Transformers |
| 合計生成時間 | 16.35秒 | 3.74秒 |
| 平均生成時間 | 3.27秒 | 0.75秒 |

## 速度差

| 質問 | CPU | GPU | GPUによる短縮倍率 |
| --- | ---: | ---: | ---: |
| 日本の首都 | 1.29秒 | 0.44秒 | 2.9倍 |
| LLM重みの4-bit量子化 | 4.72秒 | 1.17秒 | 4.0倍 |
| `12 x 13` | 1.85秒 | 0.29秒 | 6.4倍 |
| 20字以内の要約 | 3.93秒 | 0.68秒 | 5.8倍 |
| Pythonの要素数関数 | 4.56秒 | 1.16秒 | 3.9倍 |

平均では、L4 GPUはCPUより約4.4倍短い生成時間でした。回答の長さによって時間は変動するため、
この値はこの5問・最大48トークン設定での実測値です。

## 回答内容の比較

同じ0.6Bモデルをgreedy decodingで実行したCPU/GPUの出力は同一でした。これはGPUが
同一モデルの回答品質を変えず、生成速度を改善することを示します。0.6Bモデルでは、首都を
`京都`と誤答し、LLM量子化を量子コンピューティングとして誤って説明しました。

GPU上の`Qwen/Qwen3-4B`も同じ質問セット・生成設定で実行しました。4Bでは首都を`東京`、
計算結果を`156`、Pythonの関数を`len()`と応答しました。一方、量子化の説明は48トークン上限で
途中まで、要約は20字以内の制約を満たしていません。このため、4Bは0.6Bより実用的な応答を
返しますが、指示追従や専門的な正確性の評価には追加のベンチマークが必要です。

## 参照する実測結果

- CPU 0.6B BF16: [qwen3-0.6b-transformers-cpu-inference-results.md](qwen3-0.6b-transformers-cpu-inference-results.md)
- GPU 0.6B BF16: [qwen3-0.6b-gpu-inference-results.md](qwen3-0.6b-gpu-inference-results.md)
- GPU 4B BF16: [qwen3-4b-gpu-inference-results.md](qwen3-4b-gpu-inference-results.md)
- CPU 0.6B GPTQ 4-bit GGUF: [qwen3-0.6b-gptq4-inference-results.md](qwen3-0.6b-gptq4-inference-results.md)

CPUのGPTQ 4-bit GGUF結果はモデル形式と量子化方式が異なるため、上記の速度比較には含めていません。
これはCPU上での量子化済みモデルの実利用結果として、別途参照できます。