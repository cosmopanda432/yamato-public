# yamatoLLM ロードマップ

> **目標**: llm-jp-4-8b-base をベースに 3 層独自アーキテクチャ（言語処理 / コード生成 / ガバナンス）を被せた日本語特化 LLM を構築し、llm-jp-eval で同系列の `llm-jp-4-8b-instruct` 以上のスコアを出す。

---

## ベースモデル: llm-jp-4-8b-base

NII LLM-jp 主導の Apache 2.0 ライセンスモデル。標準 Llama アーキテクチャ (`LlamaForCausalLM`) で、Phase 2 以降の SFT/DPO 用データセット (`llm-jp-4-8b-thinking-dpo-data` 等) も公式公開されている。

| パラメータ | 値 |
|---|---|
| architectures | LlamaForCausalLM |
| model_type | llama |
| hidden_size | 4096 |
| num_hidden_layers | 32 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 (GQA) |
| intermediate_size | 14,336 |
| vocab_size | 196,608 |
| rope_theta | 500,000 |
| max_position_embeddings | 65,536 |
| head_dim | 128 |
| hidden_act | silu (SwiGLU) |
| torch_dtype | bfloat16 |

**互換性**: yamatoLLM のカスタム層は `hidden_size=4096` 前提で設計されており、llm-jp-4-8b-base と完全一致。LoRA target (`q_proj`, `v_proj`, `gate_proj`) は Llama 系列も標準命名。

---

## フェーズ概要

```
Phase 1: 国譲り       → llm-jp-4-8b-base を yamatoLLM 骨格にロード（動作確認）
Phase 2: 天孫降臨     → QLoRA SFT（3層ルーティング + yamato 固有能力）
Phase 3: 禊           → 3層分化 SFT
Phase 4: 神武東征     → DPO + 最適化
Phase 5: 評価・公開  → llm-jp-eval ベンチマーク → 公開判断
```

備考: 「国譲り」の意味は「Qwen → yamato」から「llm-jp-4 → yamato」へ拡張。LLM-jp は NII 主導の日本語公共 LLM プロジェクトで、その重みを yamato として継承する構造。

---

## Phase 1: 国譲り — ベース動作確認

**目標**: llm-jp-4-8b-base を yamatoLLM として動かせる状態にする

| タスク | 内容 | スクリプト | 状態 |
|--------|------|----------|------|
| ロード/初期化 | llm-jp-4-8b-base ロード → カスタムヘッド初期化 → テキスト生成確認 | `scripts/kuniumi_init.py` | 作成済、Qwen3-8B で初期動作確認後 device fix 適用済 |
| 量子化テスト | `TensonKorinQuantizer` で INT4 変換 → 推論確認 | `scripts/test_quantization.py` | 作成済（実機未検証） |
| ベースライン計測 | llm-jp-4-8b-base (素) を llm-jp-eval で評価 → 比較基準を取得 | `scripts/eval_baseline.py` | ラッパー作成済（llm-jp-eval 別途要セットアップ） |

**完了条件**: RTX 3060 (12GB) でテキスト生成が動く

### 推奨実行環境

| 項目 | 値 |
|---|---|
| OS | Ubuntu 22.04+ |
| GPU | RTX 3060 12GB 以上 |
| Python | 3.10+ |
| 主要依存 | torch 2.9+ (CUDA 12.x), transformers 5.8+, bitsandbytes 0.49+, accelerate 1.x, peft 0.19+ |

### VRAM 見積（llm-jp-4-8b, vocab=196,608）

| 量子化 | 重み | + KV cache (2K) + 実行時 | 合計 | RTX 3060 12GB |
|---|---|---|---|---|
| FP16/BF16 | ~17.2GB | +1〜2GB | **~19GB** | ❌ |
| INT8 (bnb) | ~8.6GB | +1〜2GB | **~10〜11GB** | △ 厳しい |
| **INT4 (NF4)** ※既定 | **~6.7GB** | +1〜2GB | **~8〜9GB** | ✅ 余裕 |

embed/lm_head が vocab=196K で約 3.2GB を占めるため、Qwen3-8B (vocab=152K) より +1.5GB ほど重い。

### 既知の確認事項

- **device mismatch fix**: `yamato_model.py` の `init_custom_heads()` で custom_heads を backbone と同じ device/compute dtype に揃える処理を追加済（適用済）。
- **llm-jp-eval**: `external/llm-jp-eval/` にクローンしデータセット前処理を済ませてから `eval_baseline.py` を実行する。詳細は `eval_baseline.py` の docstring 参照。

---

## Phase 2: 天孫降臨 — QLoRA SFT

**目標**: llm-jp-4 の日本語能力を維持しつつ yamatoLLM 固有能力（3層ルーティング、信頼度、コード生成）を付与

### データ戦略

| データソース | 件数 | 用途 |
|------------|------|------|
| LLM-jp 公式 SFT data (Apache 2.0) | ~10,000 | 日本語汎用能力の維持 |
| 3層ルーティング合成データ | ~3,000 | chat / codegen / retrieval 分類学習 |
| 言依さし変換データ | ~2,000 | コード生成層への橋渡し |
| 合計 | ~15,000 | 混合比 汎用40% / 固有60% |

**メリット**: llm-jp-4 は既に日本語特化済みのため、汎用日本語データの比率を抑えても能力劣化が起こりにくい。Qwen ベースのときより必要データ量を圧縮できる可能性あり。

### 学習設定

| 項目 | 値 |
|------|-----|
| 手法 | QLoRA (NF4 + double quant) |
| LoRA rank | 64 |
| LoRA target | `q_proj`, `v_proj`, `gate_proj` + カスタムヘッド |
| 実行環境 | RunPod (A100 80GB / H100) |
| 出力 | `yamato_tenson_korin.pt` |

---

## Phase 3: 禊 — 3層分化 SFT

**目標**: 天照・月読・須佐之男の各層を特化

| 層 | 担当 | データ | 件数 |
|----|------|--------|------|
| 天照 (Amaterasu) | 日本語言語処理 | 日本語対話・要約・翻訳 | ~5,000 |
| 月読 (Tsukuyomi) | コード生成 | Julia / Python コード | ~5,000 |
| 須佐之男 (Susanoo) | ガバナンス | 安全性・信頼度判断 | ~2,000 |

**備考**: 憲法十七条ガバナンス層は最小実装に留める

---

## Phase 4: 神武東征 — DPO + 最適化

**目標**: 応答品質の向上と推論速度の最適化

| タスク | 内容 |
|--------|------|
| DPO | 選好データで応答品質を改善。`llm-jp-4-8b-thinking-dpo-data` (公式公開) を流用可能 |
| 天孫降臨量子化 | INT4 最終変換・RTX 3060 推論速度検証 |
| vLLM 対応 | バッチ推論の最適化 |

---

## Phase 5: 評価・公開判断

### ベンチマーク

[llm-jp-eval](https://github.com/llm-jp/llm-jp-eval) を使用。

| ベンチマーク | 内容 |
|------------|------|
| JCommonsenseQA | 日本語常識推論 |
| JNLI | 自然言語推論 |
| JSQuAD | 読解 |
| XL-Sum (ja) | 要約 |
| MT-Bench (ja) | 多段階対話 |

### 比較対象

| モデル | サイズ | 役割 |
|--------|--------|------|
| llm-jp-4-8b-base | 9B | **直接のベースライン（学習前）** |
| llm-jp-4-8b-instruct | 9B | **同系列の SOTA（同 base からの公式 SFT 版）** |
| llm-jp-4-8b-thinking | 9B | 思考モデル（推論重視タスクの比較） |
| Qwen3-8B (参考) | 8B | 別系列ベースラインとして任意 |

### 公開判断基準

- llm-jp-4-8b-instruct **以上**のスコア → weights + コード公開（yamato 独自層の効果あり）
- 同等 → コードのみ公開、追加 SFT / データ拡充で再挑戦
- 以下 → アーキテクチャ前提を見直し（カスタム層の design 改修）

**所感**: ベースが同じ llm-jp-4-8b なので、instruct 版との直接比較が「yamato の3層独自設計が公式 SFT に対して価値を出せるか」をクリーンに示せる構図。

---

## 現在の状況

- [x] モデルアーキテクチャ設計 (`yamato_model.py`, `qwen_adapter.py`)
- [x] INT4 量子化パイプライン (`tenson_korin_quantizer.py`)
- [x] 言語処理層スケルトン (`iwato/`)
- [x] ガバナンス層スケルトン (`kenpou/`)
- [x] Phase 1: スクリプト作成 (`scripts/kuniumi_init.py`, `scripts/test_quantization.py`, `scripts/eval_baseline.py`)
- [x] Phase 1: device mismatch fix (custom_heads → backbone device 移動)
- [x] ベースモデル切り替え (Qwen3.5-9B → Qwen3-8B → llm-jp-4-8b-base)
- [ ] Phase 1: 実機（Ubuntu / RTX 3060 12GB）での動作確認
- [ ] Phase 1: ベースライン llm-jp-eval スコア取得
- [ ] Phase 2: データパイプライン + QLoRA 学習
- [ ] Phase 3: 3層分化 SFT
- [ ] Phase 4: DPO + 最適化
- [ ] Phase 5: ベンチマーク評価
