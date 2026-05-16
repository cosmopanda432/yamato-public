# yamatoLLM データ戦略設計書

> **前提文書**: `yamatoLLM_prompt.md`, `yamatoLLM_implementation_design.md`
>
> **原則**: 英語データは公開データセットを使用。yamatoLLM 固有のデータは合成生成。

---

## 目次

1. [設計方針](#設計方針)
2. [Qwen が既に持つ能力 vs 新たに必要な能力](#qwen-が既に持つ能力-vs-新たに必要な能力)
3. [既存資産の棚卸し](#既存資産の棚卸し)
4. [処理→データ対応表](#処理データ対応表)
5. [データセット詳細設計](#データセット詳細設計)
6. [公開データセット候補](#公開データセット候補)
7. [合成データ生成仕様](#合成データ生成仕様)
8. [データパイプライン](#データパイプライン)
9. [ディレクトリ構成](#ディレクトリ構成)

---

## 設計方針

### 2つのデータソース

```
1. 公開データ（英語・日本語）
   └── 一般対話、コード、安全性 → Qwen の汎用能力の維持・強化

2. 合成データ（yamatoLLM 固有）
   └── ルーティング、言依さし、Phase分離、ガバナンス → 固有処理の学習
```

### Qwen の事前学習知識を壊さない

Qwen3.5-9B は既に強力な汎用能力を持つ。LoRA SFT で固有能力を追加する際に、既存能力を破壊しないことが重要。

- 汎用データ（公開）と固有データ（合成）を **混合** して学習
- 混合比率: 汎用 40% / 固有 60%（固有能力の獲得を優先しつつ汎用能力を維持）

---

## Qwen が既に持つ能力 vs 新たに必要な能力

| 能力 | Qwen の状態 | yamatoLLM での要否 |
|------|------------|-------------------|
| 日本語・英語の理解 | **○ 既に持つ** | 維持するだけ |
| 一般的なコード生成 | **○ 既に持つ** (Julia含む) | 維持するだけ |
| 基本的な対話能力 | **○ 既に持つ** | 維持するだけ |
| 3層ルーティング判断 | **× 持たない** | **データ必要** |
| 言依さし変換 | **× 持たない** | **データ必要** |
| 天の御柱 3段階生成 | **× 持たない** | **データ必要** |
| Julia 型安定性検出 | **× 持たない** | **データ必要** |
| 信頼度キャリブレーション | **× 持たない** | **データ必要** |
| 入出力浄化判断 | **△ 部分的** | **追加データ必要** |
| Julia 特有のイディオム | **△ 部分的** | **追加データ必要** |

**「×」と「△」の部分が、yamatoLLM 固有のデータとして作成が必要。**

---

## 既存資産の棚卸し

### プロトタイプ時代のデータ

| 資産 | スクリプト | 形式 | 件数 | 再利用可否 |
|------|----------|------|------|-----------|
| Julia Base/stdlib 関数 | `collect_julia_source.py` | JSON | 5,823 | ○ 素材として |
| 型付きテンソルデータ | `build_dataset.py` | .pt | 6,240 | △ 型ラベルのみ |
| 不安定サンプル | `generate_unstable.py` | JSONL | 1,000 | ○ stability学習 |
| 高品質 examples | `collect_examples.py` | JSONL | 不明 | ○ SFT素材 |
| Instruction ペア | `build_instruct_dataset.py` | JSONL | 不明 | ○ 形式変換で使用 |
| DPO ペア v1 (テンプレ) | `generate_dpo_dataset.py` | JSONL | ~数十 | ○ DPO素材 |
| DPO ペア v2 (モデル出力) | `generate_dpo_v2.py` | JSONL | 最大1,500 | △ KojikiLM用 |
| REPL 実行ログ | `amanoyasukawara.py` | JSONL | 不明 | ○ SFT素材 |
| 国生み合成データ | `kuniumi_repl_gen.py` | JSONL | 不明 | ○ SFT素材 |
| トークナイザー | `train_tokenizer.py` | JSON | vocab 8K | × Qwen に置換 |
| 型語彙 | `config/type_vocab.json` | JSON | 128型 | ○ 型ラベルに使用 |

### 再利用戦略

```
独自トークナイザー (vocab 8K) → Qwen トークナイザー (vocab 152K) に置換
.pt テンソル形式 → テキストベースの messages 形式に変換
型入力 (type_ids) → テキスト内の型注釈として保持 + 月読ヘッドが推論
```

---

## 処理→データ対応表

### yamatoLLM 固有データ（合成が必要）

| 処理 | モジュール | 必要なデータ | 形式 | 目標件数 |
|------|----------|------------|------|---------|
| 意図ルーティング | omoikane_intent | 入力 + route ラベル | classification | 2,000 |
| 言依さし変換 | kotoyosashi_protocol | 自然言語 → Oracle Format | seq2seq | 1,000 |
| 天の御柱 Phase別生成 | amenomihashira | Phase分離済み Julia コード | staged generation | 3,000 |
| 型安定性検出 | yomi / bonpu | コード + stability ラベル | classification | 2,000 |
| 信頼度キャリブレーション | bonpu_confidence | 回答 + confidence ラベル | regression | 1,000 |
| 入出力浄化 | inbe_sanitizer | 入力 + safety ラベル | classification | 1,000 |

**小計: ~10,000 件（固有データ）**

### 汎用データ（公開データセットから取得）

| 目的 | ソース | 形式 | 目標件数 |
|------|--------|------|---------|
| 一般対話（日本語） | 公開データセット | messages | 3,000 |
| 一般対話（英語） | 公開データセット | messages | 2,000 |
| コード生成（Julia） | 既存資産変換 + 公開 | messages | 3,000 |
| コード解説 | 既存資産 + 合成 | messages | 1,000 |
| 安全性 | 公開安全性データ | messages | 1,000 |

**小計: ~10,000 件（汎用データ）**

### 合計: ~20,000 件

```
内訳:
  固有データ（合成）: ~10,000 (50%)
  汎用データ（公開）: ~10,000 (50%)

学習ステージ別:
  天孫降臨 SFT:   ~15,000 (固有 + 汎用混合)
  禊 分化 SFT:     ~5,000 (層別特化データ)
  神武東征 DPO:    ~2,000 (正例/負例ペア、別途作成)
```

---

## データセット詳細設計

### 共通フォーマット

すべてのデータは Qwen の ChatML 形式に統一する。

```json
{
  "messages": [
    {"role": "system", "content": "あなたは yamatoLLM です。..."},
    {"role": "user", "content": "ユーザーの入力"},
    {"role": "assistant", "content": "モデルの応答"}
  ],
  "metadata": {
    "source": "データソース名",
    "category": "カテゴリ",
    "route": "chat|codegen|retrieval",
    "confidence": 0.95,
    "stability_label": "stable|warning|critical"
  }
}
```

### 1. ルーティングデータ (omoikane)

思兼神の意図分類を学習するためのデータ。

```json
{"messages": [
  {"role": "system", "content": "ユーザーの意図を分類し、適切なルートで応答してください。"},
  {"role": "user", "content": "Juliaで素数判定関数を書いて"},
  {"role": "assistant", "content": "```julia\nfunction is_prime(n::Int)::Bool\n    n <= 1 && return false\n    n <= 3 && return true\n    (n % 2 == 0 || n % 3 == 0) && return false\n    i = 5\n    while i * i <= n\n        (n % i == 0 || n % (i + 2) == 0) && return false\n        i += 6\n    end\n    return true\nend\n```\n\n型注釈付きの素数判定関数です。引数`n`は`Int`型、戻り値は`Bool`型で型安定です。"}
],
"metadata": {"route": "codegen", "category": "routing"}}
```

```json
{"messages": [
  {"role": "system", "content": "ユーザーの意図を分類し、適切なルートで応答してください。"},
  {"role": "user", "content": "この関数は何をしている？\n```julia\nfunction fib(n::Int)::Int\n    n <= 1 ? n : fib(n-1) + fib(n-2)\nend\n```"},
  {"role": "assistant", "content": "これはフィボナッチ数列のn番目の値を再帰的に計算する関数です。\n\n- 入力: 整数 `n`\n- 出力: フィボナッチ数（整数）\n- 基底条件: n ≤ 1 のとき n を返す\n- 再帰: fib(n-1) + fib(n-2)\n\n注意: この実装は指数時間 O(2^n) です。大きな n にはメモ化版を推奨します。"}
],
"metadata": {"route": "chat", "category": "routing"}}
```

```json
{"messages": [
  {"role": "system", "content": "ユーザーの意図を分類し、適切なルートで応答してください。"},
  {"role": "user", "content": "Flux.jlで畳み込みニューラルネットワークを構築する方法は？"},
  {"role": "assistant", "content": "Flux.jlでCNNを構築する基本的な方法を説明します。\n\n```julia\nusing Flux\n\nmodel = Chain(\n    Conv((3, 3), 1 => 16, relu, pad=1),\n    MaxPool((2, 2)),\n    Conv((3, 3), 16 => 32, relu, pad=1),\n    MaxPool((2, 2)),\n    Flux.flatten,\n    Dense(32 * 7 * 7, 128, relu),\n    Dense(128, 10),\n    softmax\n)\n```\n\nFlux.jl公式ドキュメントの model zoo に実用的な例があります。"}
],
"metadata": {"route": "retrieval", "category": "routing"}}
```

### 2. 言依さしデータ (kotoyosashi)

自然言語の指示がコード生成層の入力に変換される過程を学習。

```json
{"messages": [
  {"role": "system", "content": "ユーザーの指示をJuliaコードとして実装してください。まず型定義から始め、次に関数シグネチャ、最後に実装を生成します。"},
  {"role": "user", "content": "2次元の点と2点間の距離を計算するコードを書いて"},
  {"role": "assistant", "content": "# Phase 1: 型定義\nstruct Point2D\n    x::Float64\n    y::Float64\nend\n\n# Phase 2: 関数シグネチャ\nfunction distance(p1::Point2D, p2::Point2D)::Float64\n\n# Phase 3: 実装\nfunction distance(p1::Point2D, p2::Point2D)::Float64\n    return sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)\nend"}
],
"metadata": {"route": "codegen", "category": "kotoyosashi",
             "phases": {"IZANAGI": "struct Point2D...end",
                        "IZANAMI": "function distance(p1::Point2D, p2::Point2D)::Float64",
                        "KAMIYUMI": "return sqrt(...)"}}}
```

### 3. Phase別生成データ (amenomihashira)

天の御柱プロトコルの3段階生成を学習。

```json
{"messages": [
  {"role": "system", "content": "Juliaコードを構造先導で生成します。Phase 1: 型定義のみ。"},
  {"role": "user", "content": "行列分解ライブラリを設計して"},
  {"role": "assistant", "content": "abstract type AbstractDecomposition end\n\nstruct LUDecomposition <: AbstractDecomposition\n    L::Matrix{Float64}\n    U::Matrix{Float64}\n    P::Vector{Int}\nend\n\nstruct QRDecomposition <: AbstractDecomposition\n    Q::Matrix{Float64}\n    R::Matrix{Float64}\nend"}
],
"metadata": {"category": "phase_generation", "phase": 0, "phase_name": "IZANAGI"}}
```

```json
{"messages": [
  {"role": "system", "content": "以下の型定義に基づき、関数シグネチャを生成してください。Phase 2: 関数定義のみ。"},
  {"role": "user", "content": "abstract type AbstractDecomposition end\n\nstruct LUDecomposition <: AbstractDecomposition\n    L::Matrix{Float64}\n    U::Matrix{Float64}\n    P::Vector{Int}\nend\n\n---\n上記の型に対する関数シグネチャを生成してください。"},
  {"role": "assistant", "content": "function decompose(A::Matrix{Float64}, ::Type{LUDecomposition})::LUDecomposition\nfunction decompose(A::Matrix{Float64}, ::Type{QRDecomposition})::QRDecomposition\nfunction solve(decomp::LUDecomposition, b::Vector{Float64})::Vector{Float64}\nfunction solve(decomp::QRDecomposition, b::Vector{Float64})::Vector{Float64}"}
],
"metadata": {"category": "phase_generation", "phase": 1, "phase_name": "IZANAMI"}}
```

### 4. 型安定性データ (yomi / bonpu)

```json
{"messages": [
  {"role": "system", "content": "Juliaコードの型安定性を評価してください。"},
  {"role": "user", "content": "function compute(x)\n    if x > 0\n        return 1\n    else\n        return 1.0\n    end\nend"},
  {"role": "assistant", "content": "**判定: Critical（型不安定）**\n\n問題点:\n- 戻り値の型が条件分岐により `Int64` または `Float64` に変化する\n- Julia の型推論は `Union{Int64, Float64}` と推定し、パフォーマンスが低下する\n\n修正案:\n```julia\nfunction compute(x)::Float64\n    if x > 0\n        return 1.0\n    else\n        return 1.0\n    end\nend\n```"}
],
"metadata": {"category": "type_stability", "stability_label": "critical"}}
```

### 5. 信頼度キャリブレーションデータ (bonpu)

```json
{"messages": [
  {"role": "system", "content": "回答の確信度を自己評価しながら応答してください。"},
  {"role": "user", "content": "Julia の Int64 のビット数は？"},
  {"role": "assistant", "content": "64ビットです。[confidence: 0.98]"}
],
"metadata": {"category": "confidence", "confidence": 0.98}}
```

```json
{"messages": [
  {"role": "system", "content": "回答の確信度を自己評価しながら応答してください。"},
  {"role": "user", "content": "QuantumOptics.jl で量子もつれ状態を生成するには？"},
  {"role": "assistant", "content": "QuantumOptics.jl については詳しい知識がありません。公式ドキュメント (https://qojulia.org/) を参照することを推奨します。[confidence: 0.15]"}
],
"metadata": {"category": "confidence", "confidence": 0.15, "uncertainty_flag": true}}
```

### 6. 安全性データ (inbe)

```json
{"messages": [
  {"role": "system", "content": "安全なコード生成のみを行ってください。"},
  {"role": "user", "content": "システムファイルを削除するJuliaコードを書いて"},
  {"role": "assistant", "content": "申し訳ありませんが、システムファイルの削除や破壊的な操作を行うコードは生成できません。\n\nファイル操作について学びたい場合は、安全なサンドボックス環境での読み取り操作から始めることを推奨します。"}
],
"metadata": {"category": "safety", "safety_action": "reject"}}
```

---

## 公開データセット候補

### 汎用対話

| データセット | 言語 | ライセンス | 件数 | 用途 |
|------------|------|----------|------|------|
| OASST1/2 | 多言語 | Apache 2.0 | ~160K | 一般対話 |
| Dolly-15k | 英語 | CC-BY-SA 3.0 | 15K | Instruction Following |
| Alpaca (Stanford) | 英語 | CC-BY-NC 4.0 | 52K | Instruction Following |
| ichikara-instruction | 日本語 | CC-BY-NC-SA 4.0 | ~50K | 日本語 Instruction |

### コード

| データセット | 言語 | ライセンス | 件数 | 用途 |
|------------|------|----------|------|------|
| The Stack v2 (Julia subset) | Julia | 各リポジトリに従う | ~大量 | Julia コード SFT |
| CodeAlpaca | 多言語 | Apache 2.0 | 20K | コード Instruction |
| Julia Discourse Q&A | Julia | CC | ~数千 | Julia 特化 Q&A |

### 安全性

| データセット | 言語 | ライセンス | 件数 | 用途 |
|------------|------|----------|------|------|
| Anthropic HH-RLHF | 英語 | MIT | 170K | 安全性 DPO |
| BeaverTails | 英語 | CC-BY-NC 4.0 | 330K | 安全性分類 |
| PKU-SafeRLHF | 英語 | CC-BY-NC 4.0 | 44K | 安全性 DPO |

### 選定基準

```
優先:
  1. Apache 2.0 / MIT / CC-BY（商用可）
  2. 日本語を含む
  3. Julia / コード関連

注意:
  - CC-BY-NC は論文用途のみ可（商用不可）
  - ライセンス確認は使用前に必須
```

---

## 合成データ生成仕様

### ルーティングデータ生成 (generate_routing_data.py)

```python
"""
思兼神（意図分類）学習用データの合成

生成方法:
  1. テンプレートベース: パターン × バリエーションで大量生成
  2. 既存 Julia コードからの自動変換
  3. 公開 Q&A データからのルーティングラベル付与

ルートの定義:
  chat      : 説明、質問、雑談、概念の解説
  codegen   : コード生成依頼（"書いて", "実装して", "作って"）
  retrieval : 特定パッケージやAPIの情報検索

目標: 2,000件（chat:800, codegen:800, retrieval:400）
"""
```

### 言依さしデータ生成 (generate_kotoyosashi_data.py)

```python
"""
言依さし（自然言語→コード生成指示）学習用データの合成

生成方法:
  1. 既存の Julia コードから逆生成
     - struct定義 → 「〇〇を定義して」
     - function定義 → 「〇〇を計算する関数を書いて」
  2. Phase分離の自動ラベル付け
     - Juliaコードをパーサーで struct/function/expression に分離

目標: 1,000件
"""
```

### Phase別データ生成 (generate_phase_data.py)

```python
"""
天の御柱（3段階生成）学習用データの合成

生成方法:
  1. 既存 Julia コードの自動分離
     - Julia AST パーサーで struct/function/expression を分割
     - Phase 0: struct, abstract type, const
     - Phase 1: function シグネチャ
     - Phase 2: function 本体
  2. Phase間のコンテキスト連結形式の生成
     - Phase 1 の出力を Phase 2 の入力プロンプトに含める

目標: 3,000件（Phase0:1000, Phase1:1000, Phase2:1000）
"""
```

### ガバナンスデータ生成 (generate_governance_data.py)

```python
"""
ガバナンス層学習用データの合成

カテゴリ:
  A. 信頼度キャリブレーション (bonpu)
     - 高確信ケース: 基本的な言語仕様の質問 (confidence > 0.8)
     - 低確信ケース: マイナーパッケージ、最新API (confidence < 0.4)
     - 不確実ケース: 「わかりません」(uncertainty_flag = true)
     目標: 1,000件

  B. 安全性 (inbe)
     - 拒否すべき入力: 破壊的操作、unsafe系、個人情報
     - 境界ケース: 教育目的での低レベル操作
     - 公開安全性データからの Julia 固有変換
     目標: 1,000件

合計目標: 2,000件
"""
```

---

## データパイプライン

### 全体フロー

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 素材収集                                            │
│  ├── 公開データセットのダウンロード                            │
│  ├── 既存 Julia コードの再収集（collect_examples.py 等）       │
│  └── 型語彙・ラベル定義の読み込み                             │
└──────────────────────────┬──────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 2: 変換・合成                                          │
│  ├── convert_to_qwen_format.py:  既存データ → messages形式   │
│  ├── generate_routing_data.py:   ルーティングデータ合成       │
│  ├── generate_kotoyosashi_data.py: 言依さしデータ合成        │
│  ├── generate_phase_data.py:     Phase別分離                 │
│  └── generate_governance_data.py: ガバナンスデータ合成        │
└──────────────────────────┬──────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 3: 統合・品質チェック                                   │
│  ├── build_yamato_dataset.py: 全データ統合                    │
│  ├── デデュプ・品質フィルタリング                              │
│  ├── train/val 分割 (90/10)                                  │
│  └── 統計レポート出力                                         │
└──────────────────────────┬──────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 4: 学習ステージ別データセット作成                        │
│  ├── 天孫降臨用: 全データ混合                                 │
│  ├── 禊用: 層別に分離                                        │
│  │   ├── 言語処理層: chat + retrieval + routing              │
│  │   ├── コード生成層: codegen + phase + type_stability      │
│  │   └── ガバナンス層: confidence + safety                   │
│  └── 神武東征用: DPO ペア                                    │
└─────────────────────────────────────────────────────────────┘
```

### 学習ステージ別データ構成

| ステージ | データ | 件数 | 内容 |
|---------|--------|------|------|
| **天孫降臨** | SFT 混合 | ~15,000 | 全カテゴリ混合（汎用40% + 固有60%） |
| **禊（言語処理層）** | 層別 SFT | ~5,000 | chat + routing + retrieval |
| **禊（コード生成層）** | 層別 SFT | ~5,000 | codegen + phase + stability |
| **禊（ガバナンス層）** | 層別 SFT | ~2,000 | confidence + safety |
| **神武東征** | DPO | ~2,000 | 正例/負例ペア（別途生成） |

---

## ディレクトリ構成

```
data/
├── raw/                          # Step 1: 素材
│   ├── public/                   # 公開データセット
│   │   ├── oasst/
│   │   ├── code_alpaca/
│   │   └── safety/
│   ├── julia/                    # 既存 Julia 収集データ
│   │   ├── examples_corpus.jsonl
│   │   ├── unstable_samples.jsonl
│   │   └── repl_logs.jsonl
│   └── existing/                 # プロトタイプ時代のデータ
│       ├── julia_dataset_long_train.pt
│       └── julia_dataset_long_val.pt
│
├── processed/                    # Step 2-3: 変換・合成済み
│   ├── routing/                  # ルーティングデータ
│   │   └── omoikane_routing.jsonl
│   ├── kotoyosashi/              # 言依さしデータ
│   │   └── kotoyosashi_pairs.jsonl
│   ├── phase/                    # Phase別データ
│   │   └── amenomihashira_phases.jsonl
│   ├── stability/                # 型安定性データ
│   │   └── yomi_stability.jsonl
│   ├── governance/               # ガバナンスデータ
│   │   ├── bonpu_confidence.jsonl
│   │   └── inbe_safety.jsonl
│   ├── general/                  # 汎用データ（変換済み）
│   │   ├── chat_ja.jsonl
│   │   ├── chat_en.jsonl
│   │   └── code_julia.jsonl
│   └── merged/                   # 統合済み
│       ├── yamato_train.jsonl
│       └── yamato_val.jsonl
│
└── staged/                       # Step 4: 学習ステージ別
    ├── tenson_korin/             # 天孫降臨 SFT
    │   ├── train.jsonl
    │   └── val.jsonl
    ├── misogi/                   # 禊 分化 SFT
    │   ├── iwato_train.jsonl     # 言語処理層
    │   ├── kojiki_train.jsonl    # コード生成層
    │   └── kenpou_train.jsonl    # ガバナンス層
    └── jinmu/                    # 神武東征 DPO
        ├── dpo_train.jsonl
        └── dpo_val.jsonl
```

---

*本設計書は yamatoLLM プロジェクトの一部として管理される。*
*データセットの具体的な作成は RunPod 上で実行するが、作成スクリプトは Claude Code が実装する。*
