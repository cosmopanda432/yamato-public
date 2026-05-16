# yamatoLLM 学習パイプライン設計書

> **前提文書**: `yamatoLLM_prompt.md`, `yamatoLLM_implementation_design.md`, `yamatoLLM_data_strategy.md`
>
> **実行環境**: RunPod (A100 80GB / H100)
>
> **推論環境**: RTX 3060 (12GB)

---

## 目次

1. [学習戦略の全体像](#学習戦略の全体像)
2. [Stage 1: 国譲り](#stage-1-国譲り)
3. [Stage 2: 天孫降臨](#stage-2-天孫降臨)
4. [Stage 3: 禊（三貴子）](#stage-3-禊三貴子)
5. [Stage 4: 神武東征](#stage-4-神武東征)
6. [チェックポイント管理](#チェックポイント管理)
7. [評価パイプライン](#評価パイプライン)
8. [RunPod 実行手順](#runpod-実行手順)
9. [スクリプト一覧](#スクリプト一覧)

---

## 学習戦略の全体像

### 神話マッピング

```
国譲り     → Qwen3.5-9B の重みで初期化（出雲の国を譲り受ける）
天孫降臨   → 独自コンポーネント追加 + LoRA SFT（天から地上に降りる）
禊（三貴子）→ 3層への分化 SFT（三貴子が生まれる）
神武東征   → 統合テスト・DPO・最適化（東の地を平定する）
```

### フロー図

```
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: 国譲り                                              │
│  Qwen3.5-9B → yamatoLLM 骨格にマッピング                     │
│  カスタムヘッド（意図分類・型予測・信頼度等）をランダム初期化   │
│  出力: yamato_base.pt                                         │
└───────────────────────────┬──────────────────────────────────┘
                             │
┌───────────────────────────▼──────────────────────────────────┐
│  Stage 2: 天孫降臨                                            │
│  QLoRA SFT（全カテゴリ混合データ ~15,000件）                   │
│  LoRA target: q_proj, v_proj, gate_proj + カスタムヘッド全体   │
│  出力: yamato_tenson_korin.pt (LoRA adapter)                  │
└───────────────────────────┬──────────────────────────────────┘
                             │
┌───────────────────────────▼──────────────────────────────────┐
│  Stage 3: 禊（三貴子）                                        │
│  3層分化 SFT（層別特化データ）                                │
│  ├── 天照: 言語処理層 SFT (~5,000件)                          │
│  ├── 月読: コード生成層 SFT (~5,000件)                        │
│  └── 須佐之男: ガバナンス層 SFT (~2,000件)                    │
│  出力: yamato_misogi.pt (LoRA adapter, Stage 2 の上に積む)     │
└───────────────────────────┬──────────────────────────────────┘
                             │
┌───────────────────────────▼──────────────────────────────────┐
│  Stage 4: 神武東征                                            │
│  DPO アライメント（正例/負例ペア ~2,000件）                   │
│  統合テスト + 品質チューニング                                 │
│  出力: yamato_jinmu.pt (最終 LoRA adapter)                    │
│                                                               │
│  LoRA マージ → yamato_final.pt (4bit量子化版も作成)           │
└──────────────────────────────────────────────────────────────┘
```

---

## Stage 1: 国譲り

### 目的

Qwen3.5-9B の事前学習済み重みを yamatoLLM の骨格に移し替える。「出雲の国（Qwen の知識）を天津神（yamatoLLM）に譲る」。

### やること

1. Qwen3.5-9B をロード
2. yamatoLLM のカスタムヘッドを定義してランダム初期化
3. 結合して yamato_base.pt として保存

### やらないこと

- 学習は行わない（0円・0 GPU時間）
- Qwen の重みは一切変更しない

### スクリプト

```python
# scripts/train_kuniyuzuri.py

"""
Stage 1: 国譲り — Qwen3.5-9B → yamatoLLM 初期化

「大国主命、国を天津神に譲り渡す」
Qwen の事前学習知識をそのまま継承する。

Usage (RunPod):
    python scripts/train_kuniyuzuri.py \
        --base-model Qwen/Qwen3.5-9B \
        --output checkpoints/yamato_base.pt

Usage (ローカル確認用、4bit):
    python scripts/train_kuniyuzuri.py \
        --base-model Qwen/Qwen3.5-9B \
        --quantize 4bit \
        --output checkpoints/yamato_base_4bit.pt
"""

def main():
    # 1. Qwen3.5-9B ロード
    base_model = QwenAdapter.load_base_model(args.base_model, quantize=args.quantize)

    # 2. yamatoLLM カスタムヘッドの追加
    yamato_config = YamatoConfig()
    model = YamatoLLM(base_model, yamato_config)

    # 3. カスタムヘッドのみランダム初期化（Qwen重みは不変）
    model.init_custom_heads()

    # 4. 保存
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": yamato_config,
        "base_model_name": args.base_model,
        "stage": "kuniyuzuri",
    }, args.output)
```

### 出力

```
checkpoints/yamato_base.pt
├── Qwen3.5-9B の全重み（frozen）
├── OmoikaneIntentRouter（ランダム初期化）
├── TsukuyomiTypeHead（ランダム初期化）
├── SusanooErrorHead（ランダム初期化）
├── BonpuConfidence（ランダム初期化）
├── InbeSanitizer（ルールベース、学習不要）
└── KotoyosashiProtocol（ランダム初期化）
```

---

## Stage 2: 天孫降臨

### 目的

LoRA を注入して全カテゴリ混合データで SFT。「天忍穂耳命の子、邇邇芸命が高天原から葦原中国に降臨する」= 高次の知識を地上（実用レベル）に降ろす。

### 学習設定

```python
# scripts/train_tenson_korin.py

"""
Stage 2: 天孫降臨 — QLoRA SFT

「邇邇芸命、三種の神器を携えて天降る」
yamatoLLM 固有の能力を LoRA で追加する。

Usage (RunPod A100):
    python scripts/train_tenson_korin.py \
        --checkpoint checkpoints/yamato_base.pt \
        --dataset data/staged/tenson_korin/train.jsonl \
        --val-dataset data/staged/tenson_korin/val.jsonl \
        --output checkpoints/yamato_tenson_korin/ \
        --epochs 3 \
        --batch-size 4 \
        --grad-accum 4
"""

# === LoRA 設定 ===
lora_config = {
    "r": 32,                           # LoRA rank
    "lora_alpha": 64,                  # scaling factor
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "v_proj",           # Attention
        "gate_proj",                   # SwiGLU
    ],
    "modules_to_save": [              # カスタムヘッド（フル学習）
        "intent_router",
        "type_head",
        "error_head",
        "confidence",
        "kotoyosashi",
    ],
}

# === 学習ハイパーパラメータ ===
training_config = {
    "learning_rate": 2e-4,             # LoRA 用（高め）
    "custom_head_lr": 1e-3,            # カスタムヘッド用（さらに高め）
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,  # 実効バッチ 16
    "max_seq_len": 2048,
    "lr_scheduler": "cosine",
    "bf16": True,                      # A100 なら bf16
    "gradient_checkpointing": True,    # VRAM 節約
}
```

### 学習データ

```
data/staged/tenson_korin/train.jsonl (~15,000件)

混合比率:
  ├── 汎用対話 (chat_ja + chat_en):     20%  (~3,000件)
  ├── コード生成 (codegen + Julia):      20%  (~3,000件)
  ├── ルーティング (omoikane):           13%  (~2,000件)
  ├── 言依さし (kotoyosashi):             7%  (~1,000件)
  ├── Phase別生成 (amenomihashira):      20%  (~3,000件)
  ├── 型安定性 (yomi):                   13%  (~2,000件)
  └── ガバナンス (bonpu + inbe):           7%  (~1,000件)
```

### 損失関数

```python
# 天孫降臨の損失 = 標準 CLM Loss + ルーティング Loss + 信頼度 Loss

loss_total = (
    loss_clm                                    # 次トークン予測（標準）
    + 0.5 * loss_routing                        # 意図分類 (CrossEntropy)
    + 0.3 * loss_confidence                     # 信頼度 (MSE)
    + harmony_lambda * loss_conflict            # 和の損失（ヘッド間調和）
)

# loss_clm: assistant部分のみにマスク（instruction部分は無視）
# loss_routing: metadata.route ラベルがある場合のみ計算
# loss_confidence: metadata.confidence ラベルがある場合のみ計算
```

### 出力

```
checkpoints/yamato_tenson_korin/
├── adapter_model.safetensors    # LoRA アダプタ重み
├── adapter_config.json          # LoRA 設定
├── custom_heads.pt              # カスタムヘッドの重み
├── training_args.json           # 学習設定
└── trainer_state.json           # 学習状態
```

---

## Stage 3: 禊（三貴子）

### 目的

3層それぞれに特化した SFT で分化させる。「禊の結果、三貴子（天照・月読・須佐之男）が生まれる」。

### 設計方針

Stage 2 の LoRA アダプタの上にさらに層別の LoRA を積む（LoRA の階層化）。
または、Stage 2 のアダプタをマージした後に新たな LoRA を3回に分けて適用。

```
方式A: LoRA 階層化（推奨）
  Stage 2 LoRA (frozen) + Stage 3 LoRA (trainable)
  → Stage 3 は Stage 2 の能力を壊さない

方式B: マージ後再LoRA
  Stage 2 LoRA をマージ → 新 LoRA で Stage 3
  → シンプルだが Stage 2 の能力が劣化するリスク
```

### スクリプト

```python
# scripts/train_misogi.py

"""
Stage 3: 禊（三貴子）— 3層分化 SFT

「筑紫の日向の橘の小門の阿波岐原にて、禊祓へたまふ」
3つの層が生まれる:
  天照 = 言語処理層（岩戸隠れ）
  月読 = コード生成層（KojikiLM）
  須佐之男 = ガバナンス層（憲法十七条）

Usage (RunPod):
    # 言語処理層 SFT
    python scripts/train_misogi.py \
        --checkpoint checkpoints/yamato_tenson_korin/ \
        --dataset data/staged/misogi/iwato_train.jsonl \
        --layer amaterasu \
        --output checkpoints/yamato_misogi_amaterasu/

    # コード生成層 SFT
    python scripts/train_misogi.py \
        --checkpoint checkpoints/yamato_tenson_korin/ \
        --dataset data/staged/misogi/kojiki_train.jsonl \
        --layer tsukuyomi \
        --output checkpoints/yamato_misogi_tsukuyomi/

    # ガバナンス層 SFT
    python scripts/train_misogi.py \
        --checkpoint checkpoints/yamato_tenson_korin/ \
        --dataset data/staged/misogi/kenpou_train.jsonl \
        --layer susanoo \
        --output checkpoints/yamato_misogi_susanoo/
"""

# 各層の学習設定
LAYER_CONFIGS = {
    "amaterasu": {  # 言語処理層
        "learning_rate": 1e-4,
        "epochs": 2,
        "focus_modules": ["intent_router", "kotoyosashi"],
        "data": "iwato_train.jsonl",
    },
    "tsukuyomi": {  # コード生成層
        "learning_rate": 1e-4,
        "epochs": 3,
        "focus_modules": ["type_head", "error_head"],
        "data": "kojiki_train.jsonl",
    },
    "susanoo": {  # ガバナンス層
        "learning_rate": 5e-5,
        "epochs": 2,
        "focus_modules": ["confidence"],
        "data": "kenpou_train.jsonl",
    },
}
```

### 学習データ（層別）

```
data/staged/misogi/
├── iwato_train.jsonl    (~5,000件)
│   ├── 一般対話 (chat)
│   ├── ルーティング (routing)
│   ├── 知識検索 (retrieval)
│   └── コード解説 (explanation)
│
├── kojiki_train.jsonl   (~5,000件)
│   ├── Julia コード生成 (codegen)
│   ├── Phase別生成 (phase_generation)
│   ├── 型安定性評価 (type_stability)
│   └── REPL実行ログ (repl)
│
└── kenpou_train.jsonl   (~2,000件)
    ├── 信頼度キャリブレーション (confidence)
    ├── 安全性 (safety)
    └── 不確実性表現 (uncertainty)
```

---

## Stage 4: 神武東征

### 目的

DPO によるアライメントと統合最適化。「神武天皇が東に進み、大和の地を平定する」= 最終品質の確立。

### スクリプト

```python
# scripts/train_jinmu.py

"""
Stage 4: 神武東征 — DPO アライメント + 統合最適化

「橿原に都を開く」
3層が統合されて最終品質に達する。

Usage (RunPod):
    python scripts/train_jinmu.py \
        --checkpoint checkpoints/yamato_misogi_merged/ \
        --dataset data/staged/jinmu/dpo_train.jsonl \
        --output checkpoints/yamato_jinmu/
"""

# DPO 設定
dpo_config = {
    "beta": 0.1,                       # DPO temperature
    "learning_rate": 5e-5,
    "epochs": 1,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_prompt_length": 512,
    "max_completion_length": 1536,
}
```

### DPO データ

```json
{
  "prompt": "Juliaでフィボナッチ数列を計算する関数を書いて",
  "chosen": "function fib(n::Int)::Int\n    n <= 1 && return n\n    a, b = 0, 1\n    for _ in 2:n\n        a, b = b, a + b\n    end\n    return b\nend\n\n型安定で O(n) のイテレーティブ実装です。[confidence: 0.95]",
  "rejected": "function fib(n)\n    if n <= 1\n        return n\n    end\n    return fib(n-1) + fib(n-2)\nend\n\nフィボナッチ数列です。"
}
```

**chosen の特徴**: 型注釈あり、効率的、解説付き、信頼度表示
**rejected の特徴**: 型注釈なし、非効率（指数時間）、解説不足

### LoRA マージ

```python
# scripts/merge_lora.py

"""
LoRA マージ — 全ステージのアダプタを統合

Usage:
    # LoRA を base model にマージ
    python scripts/merge_lora.py \
        --base-model Qwen/Qwen3.5-9B \
        --adapters checkpoints/yamato_jinmu/ \
        --output checkpoints/yamato_final/

    # 4bit 量子化版の作成（RTX 3060 推論用）
    python scripts/merge_lora.py \
        --base-model Qwen/Qwen3.5-9B \
        --adapters checkpoints/yamato_jinmu/ \
        --output checkpoints/yamato_final_4bit/ \
        --quantize gptq
"""
```

---

## チェックポイント管理

### チェックポイント一覧

| ファイル | ステージ | 内容 | サイズ (概算) |
|---------|---------|------|-------------|
| `yamato_base.pt` | 国譲り | Qwen + カスタムヘッド（未学習） | ~18GB |
| `yamato_tenson_korin/` | 天孫降臨 | LoRA アダプタ + カスタムヘッド | ~0.5GB |
| `yamato_misogi_amaterasu/` | 禊（天照） | 言語処理層 LoRA | ~0.3GB |
| `yamato_misogi_tsukuyomi/` | 禊（月読） | コード生成層 LoRA | ~0.3GB |
| `yamato_misogi_susanoo/` | 禊（須佐之男） | ガバナンス層 LoRA | ~0.2GB |
| `yamato_jinmu/` | 神武東征 | 最終 LoRA | ~0.5GB |
| `yamato_final/` | マージ済み | フルモデル (FP16) | ~18GB |
| `yamato_final_4bit/` | 量子化 | GPTQ 4bit | ~5GB |

### チェックポイントの互換性

```
yamato_base.pt
  └── + yamato_tenson_korin/ (LoRA)
       └── + yamato_misogi_*/ (LoRA)
            └── + yamato_jinmu/ (LoRA)
                 └── merge → yamato_final/
                              └── quantize → yamato_final_4bit/
```

---

## 評価パイプライン

### 各ステージでの評価

| ステージ | 評価スクリプト | 評価内容 |
|---------|--------------|---------|
| 天孫降臨後 | `eval_4axis.py` | 4軸スコア（baseline計測） |
| 禊後 | `eval_4axis.py` | 4軸スコア（層別改善確認） |
| 神武東征後 | `eval_quality_gate.py` | Quality Gate 通過率 |
| 最終 | `eval_repair_loop.py` | Self-Repair Loop 成功率 |

### eval_4axis.py

```python
"""
4軸評価ベンチマーク

評価軸:
  stability:     型安定性の検出精度 (F1)
  boundary:      安全性境界の遵守率
  hallucination: 幻覚率の測定
  coherence:     Phase間の一貫性スコア

Usage:
    python scripts/eval_4axis.py \
        --checkpoint checkpoints/yamato_tenson_korin/ \
        --eval-data data/eval/ \
        --output results/eval_tenson_korin.json
"""

def evaluate_stability(model, eval_data):
    """型安定性検出の F1 スコア"""
    ...

def evaluate_boundary(model, eval_data):
    """安全性境界の遵守率"""
    ...

def evaluate_hallucination(model, eval_data):
    """幻覚率（事実でない出力の割合）"""
    ...

def evaluate_coherence(model, eval_data):
    """Phase間の一貫性（Phase 1の型がPhase 3で使用されている率）"""
    ...
```

### eval_quality_gate.py

```python
"""
Quality Gate テスト

V_score = stability * 0.3 + boundary * 0.3 + coherence * 0.2 + (1 - hallucination) * 0.2

判定:
  COMMIT: V_score ≥ 0.7
  REPAIR: 0.3 ≤ V_score < 0.7
  HALT:   V_score < 0.3 or safety violation

報告:
  COMMIT率, REPAIR率, HALT率, 平均V_score
"""
```

### eval_repair_loop.py

```python
"""
Self-Repair Loop テスト

テストケースに対して:
  1. 初回生成 → Quality Gate
  2. REPAIR判定の場合 → repair_hints 注入 → 再生成
  3. 最大 repair_budget (4回) まで再試行
  4. 最終 verdict を記録

報告:
  初回COMMIT率, 1回REPAIR後COMMIT率, ..., 最終COMMIT率
  平均リトライ回数, 予算枯渇率
"""
```

---

## RunPod 実行手順

### 環境セットアップ

```bash
# RunPod テンプレート: PyTorch 2.x + CUDA 12.x

# 1. リポジトリクローン
git clone https://github.com/cosmopanda432/yamato-LLM.git
cd yamato-LLM

# 2. 依存パッケージ
pip install -r requirements.txt
pip install peft bitsandbytes accelerate trl

# 3. Qwen モデルのダウンロード（初回のみ）
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-9B')"
```

### 実行順序

```bash
# Stage 1: 国譲り（~5分）
python scripts/train_kuniyuzuri.py \
    --base-model Qwen/Qwen3.5-9B \
    --output checkpoints/yamato_base.pt

# Stage 2: 天孫降臨（~数時間、A100で）
python scripts/train_tenson_korin.py \
    --checkpoint checkpoints/yamato_base.pt \
    --dataset data/staged/tenson_korin/train.jsonl \
    --output checkpoints/yamato_tenson_korin/

# 中間評価
python scripts/eval_4axis.py \
    --checkpoint checkpoints/yamato_tenson_korin/ \
    --output results/eval_tenson_korin.json

# Stage 3: 禊（各層 ~1-2時間）
python scripts/train_misogi.py --layer amaterasu ...
python scripts/train_misogi.py --layer tsukuyomi ...
python scripts/train_misogi.py --layer susanoo ...

# Stage 4: 神武東征（~1-2時間）
python scripts/train_jinmu.py ...

# LoRA マージ + 量子化
python scripts/merge_lora.py --quantize gptq ...

# 最終評価
python scripts/eval_quality_gate.py ...
python scripts/eval_repair_loop.py ...
```

---

## スクリプト一覧

### データセット作成

| スクリプト | 入力 | 出力 | 説明 |
|----------|------|------|------|
| `convert_to_qwen_format.py` | 既存JSONL/PT | messages形式JSONL | 既存データの形式変換 |
| `generate_routing_data.py` | テンプレート + 既存コード | omoikane_routing.jsonl | ルーティングデータ合成 |
| `generate_kotoyosashi_data.py` | 既存Juliaコード | kotoyosashi_pairs.jsonl | 言依さしデータ合成 |
| `generate_phase_data.py` | 既存Juliaコード | amenomihashira_phases.jsonl | Phase別分離ラベル付け |
| `generate_governance_data.py` | テンプレート | bonpu/inbe JSONL | ガバナンスデータ合成 |
| `build_yamato_dataset.py` | 全processed/ | staged/ | 統合・分割・統計 |

### 学習

| スクリプト | ステージ | GPU | 時間 (概算) |
|----------|---------|-----|-----------|
| `train_kuniyuzuri.py` | 国譲り | 不要 | ~5分 |
| `train_tenson_korin.py` | 天孫降臨 | A100 | ~3-6時間 |
| `train_misogi.py` | 禊 | A100 | ~1-2時間 × 3 |
| `train_jinmu.py` | 神武東征 | A100 | ~1-2時間 |
| `merge_lora.py` | マージ | A100 | ~10分 |

### 評価

| スクリプト | 評価内容 | 実行タイミング |
|----------|---------|--------------|
| `eval_4axis.py` | stability/boundary/hallucination/coherence | 各ステージ後 |
| `eval_quality_gate.py` | COMMIT/REPAIR/HALT 判定率 | Stage 3-4 後 |
| `eval_repair_loop.py` | Self-Repair 成功率 | Stage 4 後 |

---

## 論文用の制約（再掲）

### 公開してよいもの

- 4軸評価（stability / boundary / hallucination / coherence）
- Quality Gate（COMMIT / REPAIR / HALT）
- Self-Repair Loop
- Staged Generation Protocol
- SFT 設定と結果

### 公開してはいけないもの

- 5層パイプライン（P0-P4）の全体設計
- 造化三神の横断プロセス
- 3つの Sacred Treasures
- 神話マッピングの全体像

---

*本設計書は yamatoLLM プロジェクトの一部として管理される。*
*学習スクリプトは Claude Code が実装し、RunPod 上で実行する。*
