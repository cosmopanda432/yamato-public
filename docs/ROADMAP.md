# yamatoLLM ロードマップ

> **目標**: Qwen3.5-9B ベースの日本語特化 LLM を構築し、llm-jp-eval で競合モデルと比較可能なスコアを出す

---

## フェーズ概要

```
Phase 1: 国譲り       → Qwen3.5-9B を yamatoLLM 骨格にロード（動作確認）
Phase 2: 天孫降臨     → QLoRA SFT（日本語 + 固有能力）
Phase 3: 禊           → 3層分化 SFT
Phase 4: 神武東征     → DPO + 最適化
Phase 5: 評価・公開  → llm-jp-eval ベンチマーク → 公開判断
```

---

## Phase 1: 国譲り — ベース動作確認

**目標**: Qwen3.5-9B を yamatoLLM として動かせる状態にする

| タスク | 内容 |
|--------|------|
| `scripts/kuniumi_init.py` | Qwen3.5-9B ロード → カスタムヘッド初期化 → テキスト生成確認 |
| 量子化テスト | `TensonKorinQuantizer` で INT4 変換 → RTX 3060 で推論確認 |
| ベースライン計測 | Qwen3.5-9B (素) を llm-jp-eval で評価 → 比較基準を取得 |

**完了条件**: RTX 3060 ���テキスト生成が動く

---

## Phase 2: 天孫降臨 — QLoRA SFT

**目標**: 日本語能力の維持 + yamatoLLM 固有能力の付与

### データ戦略

| データソース | 件数 | 用�� |
|------------|------|------|
| LLM-jp SFT data (Apache 2.0) | ~10,000 | 日本語汎用能力の維持 |
| 3層ルーティング合成データ | ~3,000 | chat / codegen / retrieval 分類 |
| 言依さし変換データ | ~2,000 | コード生成層への橋渡し |
| 合計 | ~15,000 | 混合比 汎用40% / 固有60% |

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

**目標**: 天照・月読・須佐之男の各層を特化させる

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
| DPO | 選好データで応答品質を改��� |
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

| モデル | サイズ | 特徴 |
|--------|--------|------|
| Qwen3.5-9B (base) | 9B | ベースライン |
| LLM-jp-4-8b-instruct | 8B | 日本語特化 SOTA |
| LLM-jp-4-8b-thinking | 8B | 思考モデル |

### 公開判断基準

- LLM-jp-4-8b-instruct と同等以上のスコア → weights + コード公開
- それ以下 → コードのみ公開、学習継続

---

## 現在の状���

- [x] モデルアーキテクチャ設計 (`yamato_model.py`, `qwen_adapter.py`)
- [x] INT4 量子化パイプライン (`tenson_korin_quantizer.py`)
- [x] 言語処理層スケルトン (`iwato/`)
- [x] ガバナンス層スケルトン (`kenpou/`)
- [ ] Phase 1: 国譲りスクリプト
- [ ] Phase 2: データパイプライン + QLoRA 学習
- [ ] Phase 3: 3層分化 SFT
- [ ] Phase 4: DPO + 最適化
- [ ] Phase 5: ベン���マーク評価
