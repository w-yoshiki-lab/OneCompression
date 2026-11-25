# Qwen3-8B GPTQ 4-bitロード時のメモリ問題と改善方針

## 対象

NVIDIA L4を搭載した`g6.xlarge`で、OneComp 1.3.2を使い
`Qwen/Qwen3-8B`をGPTQ 4-bit（group size 128、MSE、actorder）へ量子化し、
保存済みモデルを再ロードして推論したときの事象をまとめる。

## 確認された事象

GPTQモデルの再ロード中にPythonプロセスがOSに強制終了された。カーネルログには
`Out of memory: Killed process ... (python)`が記録され、終了直前のPythonの
匿名RSSは約12.7 GiBだった。

これはGPU VRAM不足ではない。再ロード時に問題となったのはCPU側のメモリであり、
GPUへの配置が始まる前に終了した。

| 項目 | 観測値 |
| --- | ---: |
| ホストRAM | 15 GiB |
| 当初のswap | なし |
| OOM時のPython匿名RSS | 約12.7 GiB |
| GPTQ 4-bitモデルファイル | 5.70 GiB |
| 追加後のswapファイル | 16 GiB |
| EBSルートボリューム | 100 GiB |
| swap作成後のEBS空き容量 | 約17 GiB |

GPTQモデルのファイルサイズが5.70 GiBであっても、ロード時のホストメモリ使用量が
5.70 GiB以内になるわけではない。非量子化のテンソル、Pythonオブジェクト、モデル本体、
一時的なコピーが同時に存在するためである。

## 原因

直接の原因は、15 GiB RAMかつswapなしのホストで、ロード時の必要メモリが物理RAMを
超えたことだった。LinuxのOOM KillerがPythonを終了したため、SSHやVS Code Remoteの
応答も悪化した。

OneCompのローダー実装は、このメモリピークを増幅する要因になっている。現行の
`QuantizedModelLoader.load_quantized_model()`は、次の順序で動作する。

1. `AutoModelForCausalLM.from_config()`で空モデルをCPU上に生成する。
2. `_load_state_dict_from_dir()`で`load_file()`を呼び、全`*.safetensors`のテンソルを
   一つのCPU上の`state_dict`へ読み込む。
3. 全体の`state_dict`を使ってキー変換と量子化レイヤー置換を行う。
4. `model.load_state_dict(..., assign=True)`の後に`dispatch_model()`でGPUへ配置する。

そのため、GPUの空きVRAMがあっても、GPU配置前に「空モデル」と「checkpoint全体」を
CPU上で保持し、メモリピークが発生する。`device_map="auto"`はこの最後の配置に使われるため、
現在の構造ではロード初期のCPUピークを解消しない。

## 現在の対応

`/swapfile-onecompression`として16 GiBのswapを有効化した。これによりRAMが逼迫しても
直ちにOOM Killerがプロセスを終了しにくくなり、GPTQ 4-bitモデルの推論開始を確認できた。

swapはEBS上のディスク領域を仮想メモリとして使う暫定策であり、RAMを増設するものではない。
ディスクI/Oが発生するため、ロードや推論の遅延、SSH応答の低下を招くことがある。

## 推奨する環境対策

最小限の運用対策として、単一の量子化ジョブだけを実行し、十分なswapを確保する。
ただし、安定性と速度を優先する場合は、物理RAMを増やすべきである。

| 優先度 | 対策 | 効果 |
| --- | --- | --- |
| 高 | `g6.2xlarge`（RAM 32 GiB）以上へ変更 | swap依存を減らし、再ロードを安定化 |
| 中 | swapを維持し、EBS空き容量を監視 | OOMによる強制終了を緩和 |
| 中 | 同時に複数のモデルロードをしない | CPUメモリ競合を回避 |
| 低 | ロード完了までVS Codeの重い操作を控える | メモリ逼迫時の対話不能を軽減 |

## OneCompローダーの恒久改善案

目標は、checkpoint全体をCPU RAMに常駐させず、テンソルまたは量子化レイヤー単位でGPUへ
転送することである。推論中のCPUオフロードではなく、ロード中のCPU stagingを小さくする。

1. `accelerate.init_empty_weights()`でモデルを`meta`デバイスに構築する。
   空モデルのパラメータ領域をCPU RAMへ確保しない。
2. `safetensors.safe_open(...).keys()`でキーとファイル位置だけを索引化する。
   テンソル実体を読み込まず、キー変換と量子化レイヤーの対応付けを先に解決する。
3. 量子化レイヤーごとに`qweight`、`scales`、`g_idx`など必要なテンソルだけを読む。
   `GPTQLinear`を構築してGPUへ配置した直後に、そのレイヤーのテンソル参照を破棄する。
4. 埋め込み層やLayerNormなど非量子化テンソルも、一つずつ`set_module_tensor_to_device()`で
   目的のデバイスへ設定する。
5. 全テンソルの配置完了後に、モデル検証、LoRA適用、生成設定の読込を行う。

概念的な実装は次のようになる。

```python
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open

with init_empty_weights(include_buffers=True):
    model = build_model_from_config(config)

key_index = index_checkpoint_keys(model_dir)
prepare_quantized_modules(model, key_index, quant_config)

for target_name, source_file, source_key in resolve_tensor_mapping(key_index):
    with safe_open(source_file, framework="pt", device="cpu") as checkpoint:
        tensor = checkpoint.get_tensor(source_key)
    set_module_tensor_to_device(model, target_name, "cuda:0", value=tensor)
    del tensor
```

この変更では、現在の`_remap_state_dict_keys()`、`_replace_quantized_layers()`、
`_find_layer_state()`を、全テンソル辞書ではなく「キー索引」と「一層分のテンソル辞書」を
扱えるよう分離する必要がある。`load_checkpoint_and_dispatch(..., offload_state_dict=True)`の
設計は参考になるが、OneComp独自の`GPTQLinear`再構築があるため、そのまま置き換えることはできない。

## 受け入れ条件

- 8B GPTQモデルのロード中に、CPUメモリ使用量がcheckpoint全体のサイズに比例して増えない。
- 15 GiB RAM環境でもswapなし、またはごく少量のswapでロードが完了する。
- ロード後の`generate()`結果が現行ローダーと一致する。
- GPTQ、DBF、MDBF、OneBit、およびLoRA sidecarの既存ロードテストが通る。
- `device_map="auto"`、CPUのみ、単一GPUの各配置モードでスモークテストが通る。

## 結論

今回の強制終了のトリガーはホストRAM不足であり、量子化失敗、GPU VRAM不足、NVIDIA
ドライバ不整合、SSH/SSMの接続枯渇ではない。一方でOneCompの全state dict一括ロードは
そのメモリピークを生む実装上の改善対象である。

短期的には32 GiB以上のRAMとswapで運用し、恒久的にはmeta構築と逐次ロードへ移行する。