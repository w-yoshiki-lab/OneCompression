# OneComp GPTQ 4-bit推論カーネルの課題と解決策

## 対象と結論

OneComp 1.3.2で量子化した`Qwen/Qwen3-8B`のGPTQ 4-bitモデルを、NVIDIA L4上で
OneComp独自ローダーと`GPTQLinear`により推論したときの性能課題をまとめる。

保存済みモデルは`bits=4`、`group_size=128`、`sym=True`、`actorder=True`である。
重みとVRAMはおよそ62%削減できた一方、現行の推論経路は4-bit重みを直接行列積に使わず、
レイヤー呼び出しのたびに密な浮動小数点重みへ復元する。このため、同じL4上のBF16推論より
大幅に遅い。

## 実測結果

`g6.2xlarge`（RAM約30 GiB、swapなし）で、greedy decoding、thinking無効、同一の15問を
実行した。BF16とGPTQは別々にロードし、生成トークン数は出力差により完全には同一ではない。

| 指標 | BF16 | OneComp GPTQ 4-bit | GPTQ/BF16 |
| --- | ---: | ---: | ---: |
| 15問の合計生成時間 | 109.75秒 | 4086.13秒 | 37.23倍 |
| 生成速度 | 14.86 tokens/s | 0.40 tokens/s | 0.03倍 |
| 保存済みSafetensors重み | 15.26 GiB | 5.69 GiB | 62.7%削減 |
| ロード直後のPyTorch VRAM | 15623 MiB | 5841 MiB | 62.6%削減 |

RAMを15 GiBから約30 GiBへ増やし、swapなしでも完走できたため、ロード時のCPUメモリ不足は
解消した。しかしGPTQの生成速度は以前の測定とほぼ同じである。従って、ロード時CPUメモリと
推論時GPUカーネルは別の問題であり、後者が現在の速度制約である。

詳細な15問の出力と計測値は
[`qwen3-8b-gptq4-comparison-g6-2xlarge.md`](qwen3-8b-gptq4-comparison-g6-2xlarge.md)を参照する。

## 行列積で必要な処理

Transformerの線形層では、活性化$X$と重み$W$から次を計算する。

$$
Y = XW^T + b
$$

GPTQ 4-bitの$W$は、4-bit整数値$q$とグループごとのscale $s$、zero point $z$で表現する。

$$
W \approx s \times (q - z)
$$

`group_size=128`の場合、入力次元128要素ごとに$s$と$z$を共有する。保存時には$q$が
packed INT4であり、小さいVRAM使用量の理由になる。

## 現行OneCompカーネルの課題

`onecomp/quantizer/gptq/gptq_layer.py`の`GPTQLinear.forward()`におけるGemLite非使用時の
経路は、概ね以下の処理を行う。

1. packed INT4の`qweight`を整数行列へ展開する。
2. packed zero pointの`qzeros`を展開する。
3. グループ単位の`scales`とzero pointを、重みと同じ形状の密行列へ拡張する。
4. INT4値をFP16/BF16の密な重み$W$へ逆量子化する。
5. その一時重みを引数に通常のPyTorch `F.linear()`を実行する。

この処理は線形層ごと、かつ生成中の各decode tokenごとに繰り返される。Qwen3-8Bには多数の
Attention/MLP線形層があるため、INT4の展開、scale/zeroの拡張、逆量子化、dtype変換、
通常の行列積が積み重なる。

主なコストは以下である。

| 課題 | 影響 |
| --- | --- |
| INT4重みを毎forwardで展開 | 量子化済み表現を保持しても、繰り返しの変換コストが生じる |
| scale/zeroを密行列へ拡張 | 大きな一時テンソルとGPUメモリ帯域を消費する |
| 逆量子化重みをメモリ上に実体化 | INT4の帯域削減効果が計算直前に失われる |
| `F.linear()`との分離 | 展開・逆量子化・GEMMの複数カーネル起動と中間テンソルの読み書きが必要になる |
| decodeは小さいバッチになりやすい | 各tokenの固定オーバーヘッドを隠しにくい |

この経路は「INT4モデルを動かす」ための正しい機能経路だが、「INT4で高速なGEMMを行う」経路
ではない。

## BF16が速い理由

L4はBF16 Tensor Coreによる行列積を高速に実行できる。BF16モデルでは、すでにGPU上にある
BF16重みを直接GEMMへ渡せるため、各forwardの重み復元が不要である。

したがって、低ビット量子化そのものは速度向上を保証しない。圧縮重みをそのまま使える
専用カーネルがない場合、BF16 Tensor Coreの直接GEMMが、逆量子化してから行うGPTQより高速に
なることがある。今回の測定はこの条件に該当する。

## `actorder=True`の影響

保存済みモデルの`actorder=True`は量子化精度のため、活性化重要度に基づく列の並べ替えを使う。
一方で現在のOneComp実装では、これは高速パスの適用条件と衝突する。

| 機能 | 必要条件 | 現行checkpoint |
| --- | --- | --- |
| OneComp GemLite自動パス | `actorder=False`、grouped quantization、2/4/8 bit | `actorder=True`のため不使用 |
| vLLM GPTQ Marlin | GPTQ、4/8 bit、`sym=True`、`desc_act=False` | `desc_act=True`のため不適合 |

`mse=True`はキャリブレーション時の量子化誤差最小化に関係する設定であり、推論カーネルを
選択する条件ではない。

## Fused INT4カーネル

Marlinなどのfused INT4 GEMMは、次の処理を単一または密結合したGPUカーネルで行う。

1. packed INT4重みをタイル単位で直接ロードする。
2. レジスタまたはshared memory内でscale/zeroを適用する。
3. 逆量子化された値をグローバルメモリに密な重みとして書き戻さず、直ちに行列積へ使う。
4. Tensor Coreを使って出力タイルを計算する。

概念的には、現行経路とfused経路は次の違いになる。

```text
現行OneComp fallback
INT4 packed -> unpack -> dense INT -> dense FP16/BF16 weight -> F.linear

Fused INT4 GEMM
INT4 packed + scale/zero + activation -> fused dequantization and GEMM -> output
```

fused経路は、密な重みの中間テンソル、余分なGPUメモリ転送、カーネル起動回数を減らす。
実際の速度改善量はモデル、prompt長、batch size、GPU、カーネル版に依存するため、導入後に
同一条件で測定する必要がある。

## 推奨する解決策

### 短期: Marlin互換モデルを別出力として作成する

現在のcheckpointは維持し、速度検証用に別ディレクトリへ再量子化する。推奨設定は次の通りである。

```python
GPTQ(
    wbits=4,
    groupsize=128,
    mse=True,
    actorder=False,
    sym=True,
)
```

これは`bits=4`、対称量子化、activation reorderingなしというMarlinの要件に合わせる設定である。
`actorder=False`は量子化誤差へ影響し得るため、現在のcheckpointと同じ質問セットで品質を比較する。

### 短期: vLLMの`gptq_marlin`で推論する

OneCompリポジトリには、GPTQ設定を判定して`GPTQMarlinLinearMethod`を選択するvLLM pluginがある。
Marlin互換checkpointをvLLMで読み込み、ログまたは設定解決結果が`gptq_marlin`であることを確認する。

OneCompは量子化・エクスポートを担当し、vLLMは高性能な推論実行系を担当する。この構成なら
モデルアーキテクチャを変更せず、Qwen3-8Bを継続利用できる。

vLLMは依存関係が大きいため、現在の`.venv-gpu`へ混在させず、専用仮想環境または公式コンテナで
検証する。

### 中期: OneComp内部にfused backendを持たせる

OneComp単体の`load_quantized_model()`と`GPTQLinear`で高速化を完結させたい場合は、次のいずれかを
実装する必要がある。

| 選択肢 | 内容 | 留意点 |
| --- | --- | --- |
| GemLite経路の利用 | `actorder=False`モデルでGemLiteを有効化する | 実測とcheckpoint互換性の確認が必要 |
| Marlin binding | `GPTQLinear.forward()`をMarlinカーネル呼出へ置き換える | 重みレイアウト変換、CUDA/対応GPUの保守が必要 |
| Triton fused kernel | OneComp用のINT4復元・GEMM融合を実装する | 性能チューニングと広いハードウェア検証が必要 |

単独実装は保守範囲が大きいため、まずvLLM/Marlinで性能要件を満たせるか確認する方針が現実的である。

## 検証計画

1. `actorder=False`、`sym=True`でGPTQ 4-bitモデルを別名で量子化する。
2. 保存済み設定が`desc_act=False`であり、Marlin適格と判定されることを確認する。
3. vLLMでロードし、`gptq_marlin`が選択されたことをログで確認する。
4. BF16、現行OneComp GPTQ、Marlin GPTQに同一15問を実行する。
5. first-token latency、decode tokens/s、合計時間、VRAM、出力品質を比較する。

期待する受け入れ条件は、VRAM削減を維持しつつ、少なくとも現行の0.40 tokens/sを大幅に上回ること
である。BF16を超えるかどうかや最終速度は、実測結果で判断する。

## 関連する別課題

保存済みcheckpointを読む際のCPUメモリピークは、推論カーネルとは独立したローダー設計の課題である。
その原因とmeta構築・逐次ロードによる改善案は
[`qwen3-8b-gptq4-loading-memory-issue.md`](qwen3-8b-gptq4-loading-memory-issue.md)を参照する。