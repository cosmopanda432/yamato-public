# yamatoLLM 実装設計書

> **前提文書**: `yamatoLLM_prompt.md`, 各層の設計書
>
> **ベースモデル**: llm-jp-4-8b-base (Apache 2.0)
>
> **推論環境**: RTX 3060 (12GB VRAM)
>
> **学習環境**: RunPod (A100 / H100)

---

## 目次

1. [現状の棚卸し](#現状の棚卸し)
2. [全体アーキテクチャ](#全体アーキテクチャ)
3. [llm-jp-4-8b-base 統合設計](#qwen3-8b-統合設計)
4. [3層の実装設計](#3層の実装設計)
5. [評価フレームワーク](#評価フレームワーク)
6. [ファイル構成](#ファイル構成)
7. [実装優先度](#実装優先度)
8. [VRAM 制約への対応](#vram-制約への対応)

---

## 現状の棚卸し

### 実装済み

| ファイル | 行数 | 責務 |
|---------|------|------|
| `config.py` | 379 | KojikiConfig, TYPE_CATEGORIES, FiveLayerConfig |
| `layers.py` | 814 | 5章レイヤー (Genesis, SevenGen, Kuniumi, Yomi, Misogi) |
| `model.py` | 457 | KojikiLM + Autoregressive 生成 |
| `moe.py` | 382 | KojikiMoE, MoERouter, MoEFeedForward |
| `amenomihashira.py` | 630 | 天の御柱プロトコル, 蛭子検知, 直毘神検証 |
| `hieda_no_are.py` | 544 | 稗田阿礼, 言霊システム |
| `yata_kagami_attention.py` | 192 | 八咫鏡 Attention |
| `definition_detector.py` | 160 | struct/function 定義検出 |
| `training.py` | 387 | KojikiLoss, Trainer |
| `zoka_sanshin.py` | - | 造化三神 (横断プロセス) |
| `yomi_evaluator.py` | - | Layer 5 評価 (蛭子検知, 閻魔判定) |
| `yomotsu_hirasaka.py` | - | 黄泉比良坂 (Evaluation Gateway) |
| `ashihara_runtime.py` | - | Layer 3 推論ランタイム |
| `layer4_unabara.py` | - | Layer 4 外部データ (常世/海原/綿津見) |
| `takamagahara_feedback.py` | - | Layer 2 フィードバック |

### 設計書のみ（未実装）

| 設計書 | 対象 |
|--------|------|
| `言語処理層_岩戸隠れアーキテクチャ設計書.md` | 言語処理層 (iwato/) 7ファイル |
| `憲法十七条_KojikiLM統合設計書.md` | ガバナンス層 6ファイル |

### 未着手

- llm-jp-4-8b-base との統合
- 3層統合モデル (yamato_model.py)
- 学習パイプライン（RunPod用スクリプト群）
- データセット作成パイプライン

---

## 全体アーキテクチャ

### 3層 + 基盤の関係

```
╔═══════════════════════════════════════════════════════════════════╗
║  造化三神（横断プロセス）— 全層横断                                ║
║  ├── アメノミナカヌシ: 座標系・閾値定義                            ║
║  ├── タカミムスビ: 生成起動権限                                   ║
║  └── カミムスビ: 修復起動権限                                     ║
╠═══════════════════════════════════════════════════════════════════╣
║  憲法十七条（律令層）— ガバナンス・オーバーレイ                     ║
║  ├── 和の損失関数 (wa_loss)                                      ║
║  ├── 凡夫の自覚 (bonpu_confidence)                               ║
║  ├── 聖徳コンセンサス (shotoku_consensus)                         ║
║  └── 時のスケジューラ (toki_scheduler)                            ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  ┌─── 言語処理層（岩戸隠れ）───────────────────────────────────┐ ║
║  │  第一章: 参集 (天安河原)      — 入力理解                     │ ║
║  │  第二章: 思案 (思兼神)        — 意図解析・ルーティング       │ ║
║  │  第三章: 奉献 (布刀玉命+真榊) — 知識統合 (RAG)              │ ║
║  │  第四章: 神楽 (天宇受売命)    — 生成・感情制御               │ ║
║  │  第五章: 開戸 (天手力男神)    — 出力確定                     │ ║
║  │          忌部 (Inbe)          — 入出力浄化                   │ ║
║  └──────────┬──────────────────────────────────────────────────┘ ║
║             │ 言依さし (Kotoyosashi)                               ║
║             ▼                                                      ║
║  ┌─── コード生成層（KojikiLM）─────────────────────────────────┐ ║
║  │  第一章: 天地開闢 (造化三神)  — 埋め込み                     │ ║
║  │  第二章: 神世七代             — Transformer                  │ ║
║  │  第三章: 国生み               — 構造体生成                   │ ║
║  │  第四章: 黄泉国               — 型安定性検出                 │ ║
║  │  第五章: 禊 (三貴子)          — 出力ヘッド                   │ ║
║  └─────────────────────────────────────────────────────────────┘ ║
║                                                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  パイプライン基盤                                                  ║
║  ├── 天御柱オーケストレータ (amenomihashira)                      ║
║  ├── 葦原中国ランタイム (ashihara_runtime)                        ║
║  ├── 稗田阿礼 (hieda_no_are) — L4→L3 ブリッジ                   ║
║  ├── 黄泉比良坂 (yomotsu_hirasaka) — Evaluation Gateway          ║
║  ├── 黄泉評価器 (yomi_evaluator) — 蛭子検知・閻魔判定            ║
║  └── 高天原フィードバック (takamagahara_feedback)                  ║
╚═══════════════════════════════════════════════════════════════════╝
```

### 推論フロー

```
[ユーザー入力]
    │
    ▼
忌部プロトコル (入力浄化)
    │
    ▼
天安河原 (入力埋め込み)
    │
    ▼
思兼神 (意図解析)
    ├── Route: Chat       → 天宇受売命 (生成) → 天手力男神 (出力確定)
    ├── Route: Retrieval  → 布刀玉命 (RAG) → 天宇受売命 → 天手力男神
    └── Route: CodeGen    → 言依さし変換
                               │
                               ▼
                          天の御柱プロトコル
                          ├── Phase 1 (IZANAGI): struct/type 定義
                          │     └── 蛭子検知
                          ├── Phase 2 (IZANAMI): function 定義
                          └── Phase 3 (KAMIYUMI): expression
                                └── 直毘神検証
                               │
                               ▼
                          言語処理層(復路)で解説付与
                               │
                               ▼
凡夫の自覚 (信頼度スコア付与)
    │
    ▼
忌部プロトコル (出力浄化)
    │
    ▼
[ユーザーへの応答]
```

---

## llm-jp-4-8b-base 統合設計

### 設計方針: Backbone + カスタムヘッド

llm-jp-4-8b-base の Transformer 層をそのまま活用し、yamatoLLM 固有の処理をカスタム層として追加する。

```
┌─────────────────────────────────────────────────────────────┐
│  YamatoLLM                                                   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  llm-jp-4-8b-base (frozen base or LoRA)                    │   │
│  │  ├── Embedding (→ 高御産巣日神の役割)                │   │
│  │  ├── RMSNorm                                         │   │
│  │  ├── Transformer × 40 layers (→ 神世七代の役割)      │   │
│  │  │     ├── Self-Attention (GQA)                      │   │
│  │  │     └── SwiGLU FFN                                │   │
│  │  └── LM Head (→ 天照の役割)                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  yamatoLLM カスタム層                                 │   │
│  │                                                       │   │
│  │  入力側:                                              │   │
│  │  ├── OmoikaneIntentRouter (意図分類ヘッド)            │   │
│  │  └── InbeSanitizer (入出力浄化)                       │   │
│  │                                                       │   │
│  │  出力側:                                              │   │
│  │  ├── TsukuyomiTypeHead (型予測ヘッド)                 │   │
│  │  ├── SusanooErrorHead (エラー予測ヘッド)              │   │
│  │  ├── BonpuConfidence (信頼度スコア)                   │   │
│  │  └── KotoyosashiProtocol (言依さし変換)               │   │
│  │                                                       │   │
│  │  LoRA 注入先:                                         │   │
│  │  ├── q_proj, v_proj (Attention)                       │   │
│  │  ├── gate_proj (SwiGLU)                               │   │
│  │  └── lm_head (出力)                                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 既存プロトタイプとの関係

| プロトタイプ (41M) | Qwen統合版 (9B) | 変更点 |
|---|---|---|
| 独自 Token Embedding | Qwen Embedding | Qwen のものを使用 |
| 独自 Positional Encoding | Qwen RoPE | Qwen のものを使用 |
| SevenGenerationsBlock × 6 | Qwen Transformer × 40 | Qwen に置換 |
| MultipleDispatchAttention | LoRA で注入 | Attention bias として学習 |
| KuniumiLayer (国生み) | カスタムヘッド | Phase embedding は保持 |
| YomiLayer (黄泉国) | カスタムヘッド | 型安定性検出ヘッドとして追加 |
| MisogiLayer (禊) | Qwen LM Head + カスタム | 三貴子の副出力ヘッドを追加 |
| 独自トークナイザー (8K) | Qwen トークナイザー (152K) | Qwen のものを使用 |
| 独自型入力 (type_ids) | テキストベース + ヘッド推論 | 型情報は月読ヘッドが推論 |

### 実装ファイル

```python
# kojiki_lm/qwen_adapter.py

"""
国譲り — llm-jp-4-8b-base の重みを yamatoLLM に継承する

llm-jp-4-8b-base のアーキテクチャ (LlamaForCausalLM):
  - Hidden size: 4096
  - Num layers: 32
  - Num attention heads: 32
  - Num KV heads: 8 (GQA)
  - Intermediate size: 14336 (SwiGLU / silu)
  - Vocab size: 196608
  - RoPE theta: 500000
  - Max position embeddings: 65536

yamatoLLM が追加するもの:
  - 意図分類ヘッド (OmoikaneIntentRouter)
  - 型予測ヘッド (TsukuyomiTypeHead)
  - エラー予測ヘッド (SusanooErrorHead)
  - 信頼度ヘッド (BonpuConfidence)
  - 入出力浄化 (InbeSanitizer)
  - 言依さし変換 (KotoyosashiProtocol)
"""

class QwenAdapter:
    """llm-jp-4-8b-base のロードとカスタム層の注入"""

    @staticmethod
    def load_base_model(model_name="Qwen/llm-jp-4-8b-base", quantize=None):
        """
        ベースモデルのロード

        quantize:
          None    → FP16 (学習時、A100)
          "4bit"  → GPTQ/AWQ 4bit (推論時、RTX 3060)
          "8bit"  → INT8 (推論時、余裕があれば)
        """
        ...

    @staticmethod
    def inject_lora(model, lora_config):
        """
        LoRA アダプタの注入

        target_modules: ["q_proj", "v_proj", "gate_proj"]
        rank: 32
        alpha: 64
        dropout: 0.05
        """
        ...

    @staticmethod
    def attach_custom_heads(model, yamato_config):
        """yamatoLLM 固有のヘッドを Qwen に追加"""
        ...
```

```python
# kojiki_lm/yamato_model.py

"""
yamatoLLM — 3層統合モデル

言語処理層 (岩戸隠れ) + コード生成層 (KojikiLM) + ガバナンス層 (憲法十七条)
をllm-jp-4-8b-base backbone 上で統合する
"""

class YamatoLLM(nn.Module):
    def __init__(self, config):
        self.backbone = QwenAdapter.load_base_model(...)
        self.intent_router = OmoikaneIntentRouter(config)
        self.sanitizer = InbeSanitizer(config)
        self.type_head = TsukuyomiTypeHead(config)
        self.error_head = SusanooErrorHead(config)
        self.confidence = BonpuConfidence(config)
        self.kotoyosashi = KotoyosashiProtocol(config)

    def forward(self, input_ids, attention_mask=None, route=None, ...):
        """
        統合 forward pass

        1. 忌部: 入力浄化
        2. Qwen backbone: hidden states 取得
        3. 思兼神: ルーティング判断
        4. ルートに応じた処理分岐
        5. 凡夫の自覚: 信頼度付与
        6. 忌部: 出力浄化
        """
        ...

    def generate(self, prompt, ...):
        """
        推論時のエントリポイント

        Route: Chat      → backbone の通常生成
        Route: CodeGen   → 天の御柱 3段階生成
        Route: Retrieval → RAG + 生成
        """
        ...
```

---

## 3層の実装設計

### 言語処理層 (iwato/)

設計書: `言語処理層_岩戸隠れアーキテクチャ設計書.md`

```
kojiki_lm/iwato/
├── __init__.py
├── yasukawara_embedding.py    # 第一章: 参集（入力埋め込み）
├── omoikane_intent.py         # 第二章: 思案（意図解析・ルーティング）
├── futodama_retriever.py      # 第三章: 奉献（RAG・知識統合）
├── amenouzume_decoder.py      # 第四章: 神楽（生成・感情制御）
├── tajikarao_output.py        # 第五章: 開戸（出力確定）
├── kotoyosashi_protocol.py    # 言依さし（コード生成層との接続）
└── inbe_sanitizer.py          # 忌部（入出力浄化）
```

#### 各モジュールの入出力

| モジュール | 入力 | 出力 | Qwen との関係 |
|---|---|---|---|
| yasukawara_embedding | ユーザーテキスト | 埋め込みベクトル E_input | Qwen Embedding をそのまま使用 |
| omoikane_intent | Qwen hidden states | intent_vector, route (chat/codegen/retrieval) | backbone 出力の pooling → Linear → 3分類 |
| futodama_retriever | E_input, V_intent | H_context (文脈強化済み) | 外部メモリ（稗田阿礼）へのクエリ + Cross-Attention |
| amenouzume_decoder | H_context | P(w_t) 確率分布 | Qwen LM Head + 万葉フィルタ（トーン制御） |
| tajikarao_output | P(w_t) | 最終テキスト or 構造化指示 | Top-p/Greedy sampling + 注連縄（EOS 監視） |
| kotoyosashi_protocol | intent_vector, user_query | Oracle Format (JSON) | 自然言語→構造化指示の変換 |
| inbe_sanitizer | テキスト | 浄化済みテキスト + safety_score | ルールベース + スコアリング |

#### omoikane_intent.py の詳細

```python
class OmoikaneIntentRouter(nn.Module):
    """
    思兼神 — 意図解析・ルーティング

    Qwen の最終隠れ状態を受け取り、3種のルートに分類する。
    学習時: ルーティングラベル付きデータで Cross-Entropy
    推論時: argmax でルート決定

    ルート定義:
      0 = chat:      一般対話（天宇受売命が処理）
      1 = codegen:   コード生成（言依さし → KojikiLM）
      2 = retrieval: 知識検索（布刀玉命 → RAG → 生成）
    """

    def __init__(self, d_model, num_routes=3):
        super().__init__()
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, num_routes)
        self.intent_projection = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, attention_mask=None):
        # 最終トークン or masked mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = hidden_states[:, -1, :]

        pooled = self.pooler(pooled)
        route_logits = self.classifier(pooled)
        intent_vector = self.intent_projection(pooled)

        return {
            "route_logits": route_logits,         # [batch, 3]
            "route": torch.argmax(route_logits, dim=-1),  # [batch]
            "intent_vector": intent_vector,        # [batch, d_model]
        }
```

#### kotoyosashi_protocol.py の詳細

```python
class KotoyosashiProtocol(nn.Module):
    """
    言依さし — 言語処理層 → コード生成層の変換

    自然言語の指示を Oracle Format に変換する。
    Oracle Format は天の御柱プロトコルの入力となる。

    Oracle Format:
    {
        "task": "codegen",
        "phase_hint": "IZANAGI",           # 開始Phase
        "struct_hints": ["Point2D"],        # 型名のヒント
        "type_constraints": ["Float64"],    # 型制約
        "function_hints": ["distance"],     # 関数名のヒント
    }
    """

    def __init__(self, d_model, max_hints=8):
        super().__init__()
        # 構造化出力のためのヘッド
        self.phase_classifier = nn.Linear(d_model, 3)  # IZANAGI/IZANAMI/KAMIYUMI
        self.hint_generator = nn.Linear(d_model, d_model)

    def forward(self, intent_vector, hidden_states):
        phase_logits = self.phase_classifier(intent_vector)
        hints = self.hint_generator(intent_vector)
        return {
            "phase_hint": torch.argmax(phase_logits, dim=-1),
            "hint_embedding": hints,
        }
```

### ガバナンス層

設計書: `憲法十七条_KojikiLM統合設計書.md`

```
kojiki_lm/
├── kenpou_config.py          # 設定: 17条パラメータ
├── wa_loss.py                # L1: 和の損失関数
├── shotoku_consensus.py      # L3: 聖徳コンセンサス
├── bonpu_confidence.py       # L4: 凡夫の自覚
├── toki_scheduler.py         # L5: 時のスケジューラ
└── iwato/inbe_sanitizer.py   # L2: 忌部プロトコル（言語処理層に配置）
```

#### 設計方針

- **既存のモデル内部層は変更しない** — 律令オーバーレイとして上から被せる
- 律令層を無効化すれば既存と完全に同一動作（可逆的統合）
- 新規モジュールは既存コードをラップ or アダプタで接続

#### bonpu_confidence.py の詳細

```python
class BonpuConfidence(nn.Module):
    """
    凡夫の自覚 — 信頼度スコア

    第10条「我必ずしも聖に非ず、彼必ずしも愚に非ず。共にこれ凡夫のみ」

    制約:
      - confidence < 1.0 は常に成立（聖に非ず）
      - confidence_floor 以下にはならない（愚に非ず）
      - τ 未満の場合 uncertainty_flag = True

    入力: Qwen の最終隠れ状態
    出力: confidence_score, uncertainty_flag, truthfulness_score
    """

    def __init__(self, d_model, config):
        super().__init__()
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.truthfulness_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.floor = config.confidence_floor      # 0.1
        self.ceiling = 1.0 - 1e-4                 # 聖に非ず
        self.tau = config.truthfulness_tau          # 0.6

    def forward(self, hidden_states):
        pooled = hidden_states[:, -1, :]
        raw_conf = self.confidence_head(pooled).squeeze(-1)
        confidence = raw_conf * (self.ceiling - self.floor) + self.floor
        truthfulness = self.truthfulness_head(pooled).squeeze(-1)
        uncertainty_flag = confidence < self.tau

        return {
            "confidence": confidence,
            "truthfulness": truthfulness,
            "uncertainty_flag": uncertainty_flag,
        }
```

#### wa_loss.py の詳細

```python
class WaLoss(nn.Module):
    """
    和の損失関数 — 既存 Loss のラップ

    L_wa = L_base + λ * L_conflict + μ * L_regularization

    L_conflict: Attention ヘッド間の KL divergence
    L_regularization: 第14条（嫉妬防止 = 過学習防止）
    """

    def __init__(self, base_loss_fn, config):
        super().__init__()
        self.base_loss_fn = base_loss_fn  # 変更なし
        self.harmony_lambda = config.harmony_lambda
        self.regularization_mu = 0.01

    def forward(self, outputs, targets, attentions=None):
        base_loss, base_details = self.base_loss_fn(outputs, targets)

        conflict_loss = torch.tensor(0.0)
        if attentions is not None:
            # ヘッド間の出力分散 = 対立コスト
            head_mean = attentions.mean(dim=1, keepdim=True)
            conflict_loss = F.kl_div(
                F.log_softmax(attentions, dim=-1),
                F.softmax(head_mean.expand_as(attentions), dim=-1),
                reduction="batchmean",
            )

        total = base_loss + self.harmony_lambda * conflict_loss
        return total, {**base_details, "conflict": conflict_loss}
```

---

## 評価フレームワーク

### 4軸 Detector

設計書: `julia_five_layer_remap.md` の YomiEvaluator

| 軸 | 検出対象 | 既存実装 |
|---|---|---|
| stability | 型安定性（Stable/Warning/Critical） | YomiLayer の stability_logits |
| boundary | システム境界の逸脱 | safety/topic/length/format チェック |
| hallucination | 幻覚（事実でない出力） | テキストベースのヒューリスティック |
| coherence | 一貫性（定義-使用の整合） | Phase間の型一致率 |

### Quality Gate

```
4軸スコア → V score 集計 → verdict

V_score = stability * 0.3 + boundary * 0.3 + coherence * 0.2 + (1 - hallucination) * 0.2

verdict:
  V_score ≥ V_threshold (0.7)  → COMMIT（出力確定）
  stability < stability_floor   → HALT（即停止）
  boundary ≤ safety_floor       → HALT（即停止）
  otherwise                     → REPAIR（修復試行）
```

### Self-Repair Loop

```
生成 → 4軸評価 → REPAIR判定
  ↓                    ↓
  ←── repair_hints ←──┘
  │
  修復ヒントをコンテキストに注入して再生成
  │
  最大 repair_budget (4回) まで再試行
  │
  予算枯渇 → HALT
```

---

## ファイル構成

### 新規作成ファイル一覧

```
kojiki_lm/
├── yamato_model.py              # 3層統合モデル
├── qwen_adapter.py              # llm-jp-4-8b-base ローダー + LoRA注入
├── yamato_config.py             # yamatoLLM 統合設定
│
├── iwato/                       # 言語処理層
│   ├── __init__.py
│   ├── yasukawara_embedding.py  # 第一章: 参集
│   ├── omoikane_intent.py       # 第二章: 思案
│   ├── futodama_retriever.py    # 第三章: 奉献
│   ├── amenouzume_decoder.py    # 第四章: 神楽
│   ├── tajikarao_output.py      # 第五章: 開戸
│   ├── kotoyosashi_protocol.py  # 言依さし
│   └── inbe_sanitizer.py        # 忌部
│
├── kenpou_config.py             # ガバナンス設定
├── wa_loss.py                   # 和の損失関数
├── shotoku_consensus.py         # 聖徳コンセンサス
├── bonpu_confidence.py          # 凡夫の自覚
└── toki_scheduler.py            # 時のスケジューラ

scripts/
├── データセット作成
│   ├── convert_to_qwen_format.py    # 既存データ → Qwen SFT形式変換
│   ├── generate_routing_data.py     # ルーティングデータ合成
│   ├── generate_kotoyosashi_data.py # 言依さしデータ合成
│   ├── generate_phase_data.py       # Phase別分離ラベル付け
│   ├── generate_governance_data.py  # ガバナンスデータ合成
│   └── build_yamato_dataset.py      # 全データ統合
│
├── 学習 (RunPod用)
│   ├── train_kuniyuzuri.py          # Stage 1: 国譲り（重み初期化）
│   ├── train_tenson_korin.py        # Stage 2: 天孫降臨（QLoRA SFT）
│   ├── train_misogi.py              # Stage 3: 禊（3層分化SFT）
│   ├── train_jinmu.py               # Stage 4: 神武東征（DPO/統合最適化）
│   └── merge_lora.py                # LoRA マージ
│
├── 評価 (RunPod用)
│   ├── eval_4axis.py                # 4軸評価ベンチマーク
│   ├── eval_quality_gate.py         # Quality Gate テスト
│   └── eval_repair_loop.py          # Self-Repair Loop テスト
│
└── テスト
    ├── test_yamato_model.py         # 統合モデルテスト
    ├── test_iwato.py                # 言語処理層テスト
    ├── test_kenpou.py               # ガバナンス層テスト
    └── test_vram.py                 # RTX 3060 VRAM 確認
```

### 既存ファイルへの変更

| ファイル | 変更内容 |
|---------|---------|
| `config.py` | YamatoConfig の追加、KenpouConfig のインポート |
| `__init__.py` | 新規モジュールのエクスポート追加 |
| `amenomihashira.py` | YamatoLLM との接続ポイント追加 |
| `hieda_no_are.py` | Qwen トークナイザー対応 |

### 変更なし（互換性維持）

| ファイル | 理由 |
|---------|------|
| `layers.py` | プロトタイプの5章アーキテクチャは維持 |
| `model.py` | KojikiLM はスタンドアロンでも動作可能に |
| `moe.py` | MoE構造は不変 |
| `training.py` | 既存 KojikiLoss は WaLoss でラップ |

---

## 実装優先度

### Phase 1: 基盤（P0）

| タスク | ファイル | 依存 |
|--------|---------|------|
| Qwen ローダー | `qwen_adapter.py` | なし |
| 統合設定 | `yamato_config.py` | なし |
| 統合モデル骨格 | `yamato_model.py` | qwen_adapter |

### Phase 2: 3層実装（P1）

| タスク | ファイル | 依存 |
|--------|---------|------|
| 意図分類 | `iwato/omoikane_intent.py` | yamato_model |
| 入出力浄化 | `iwato/inbe_sanitizer.py` | なし |
| 言依さし | `iwato/kotoyosashi_protocol.py` | omoikane |
| 信頼度 | `bonpu_confidence.py` | yamato_model |
| ガバナンス設定 | `kenpou_config.py` | なし |
| 和の損失関数 | `wa_loss.py` | kenpou_config |

### Phase 3: パイプライン（P2）

| タスク | ファイル | 依存 |
|--------|---------|------|
| データ形式変換 | `convert_to_qwen_format.py` | なし |
| ルーティングデータ | `generate_routing_data.py` | なし |
| 国譲りスクリプト | `train_kuniyuzuri.py` | qwen_adapter |
| 天孫降臨スクリプト | `train_tenson_korin.py` | yamato_model + データ |

### Phase 4: 評価・テスト（P3）

| タスク | ファイル | 依存 |
|--------|---------|------|
| 4軸評価 | `eval_4axis.py` | yamato_model |
| Quality Gate | `eval_quality_gate.py` | 4軸評価 |
| 統合テスト | `test_yamato_model.py` | 全体 |
| VRAM確認 | `test_vram.py` | qwen_adapter |

---

## VRAM 制約への対応

### RTX 3060 (12GB) での推論

```
llm-jp-4-8b-base FP16 = ~18GB → 載らない

解決策:
  4bit 量子化 (GPTQ/AWQ): ~5GB
  + yamatoLLM カスタムヘッド (FP16): ~0.1GB
  + KV cache (2048 tokens): ~2GB
  + 推論バッファ: ~2GB
  ──────────────────────────
  合計: ~9GB → RTX 3060 OK ✓
```

### RunPod (A100 80GB) での学習

```
llm-jp-4-8b-base FP16 = ~18GB
  + LoRA アダプタ (rank=32): ~0.2GB
  + Optimizer states: ~0.4GB (LoRA params のみ)
  + Gradient: ~0.4GB
  + Activation cache: ~10GB
  + データバッチ: ~2GB
  ──────────────────────────
  合計: ~31GB → A100 OK ✓

  ※ QLoRA (4bit base + FP16 LoRA) なら ~12GB で RTX 4090 でも可
```

### 量子化戦略

| 段階 | 手法 | VRAM | 用途 |
|------|------|------|------|
| 学習時 (RunPod) | FP16 base + LoRA FP16 | ~31GB | A100/H100 |
| 学習時 (低コスト) | QLoRA (4bit base + FP16 LoRA) | ~12GB | RTX 4090 |
| 推論時 | GPTQ/AWQ 4bit + カスタムヘッド FP16 | ~9GB | RTX 3060 |

---

## 結語

yamatoLLM の実装は、llm-jp-4-8b-base の「国譲り」（重み継承）を基盤に、3層の独自アーキテクチャをカスタムヘッドとLoRAで追加する方式を採る。

既存の KojikiLM プロトタイプ（41M）の設計思想と神話マッピングはそのまま継承しつつ、9B スケールの事前学習知識を活用することで、RTX 3060 での推論可能性とアーキテクチャの独自性を両立する。

---

*本設計書は yamatoLLM プロジェクトの一部として管理される。*
*`yamatoLLM_prompt.md` の構築指示に基づき、実装の全体設計を定義する。*
