# yamatoLLM 構築指示

## 何を作るか
日本神話をアーキテクチャ設計原理とする LLM。3層構成。
ベースモデル: Qwen3.5-9B（Apache 2.0、重み初期化に使用）

## 3層

### コード生成層（KojikiLM / Julia-no-Mikoto）
古事記 → Julia コード生成。設計書: docs/kojiki_llm_architecture.md
- 天地開闢 = 入力埋め込み
- 神世七代 = Transformer ブロック
- 国生み = 構造体生成
- 黄泉国 = 型安定性検出（Malformed Output Detector、4軸評価）
- 禊 = 出力ヘッド

### 言語処理層（岩戸隠れアーキテクチャ）
天岩戸神話 → 自然言語理解・応答。設計書: docs/言語処理層_岩戸隠れアーキテクチャ設計書.md
- 天安河原 = 入力理解
- 思兼神 = 意図解析・ルーティング
- 布刀玉命 + 真榊 = 知識統合 (RAG)
- 天宇受売命 = 生成・感情制御
- 天手力男神 = 出力確定

### ガバナンス層（憲法十七条）
聖徳太子の憲法十七条 → アライメント・制御。設計書: docs/憲法十七条_LLM設計仕様書.md
- 既存層を変更せず上から被せる律令層
- 第1条「和」= 損失関数の全体最適化
- 第10条「凡夫の自覚」= 信頼度スコア
- 第17条「衆とともに論ぜよ」= MoE 動的ルーティング

## 学習戦略（神話マッピング）
1. 国譲り: Qwen3.5-9B の重みで初期化（0円）
2. 天孫降臨: 独自コンポーネント追加 + LoRA SFT（RunPod）
3. 禊（三貴子）: 3層への分化 SFT（RunPod）
4. 神武東征: 統合テスト・最適化

学習は RunPod で実行。RTX 3060 (12GB) で推論可能なこと。

## Claude Code の担当
- yamatoLLM のアーキテクチャコード（kojiki_lm/ 配下）
- ガバナンス層の実装
- 評価フレームワーク（Detector / Quality Gate / Repair Loop）
- 言語処理層の実装
- テスト

## RunPod の担当
- データセット作成（Julia コード、英語 NLU、アライメント）
- QLoRA SFT 実行
- チェックポイント評価
- LoRA マージ

## 論文用の制約
公開してよいもの:
- 4軸評価（stability / boundary / hallucination / coherence）
- Quality Gate（COMMIT / REPAIR / HALT）
- Self-Repair Loop
- Staged Generation Protocol
- SFT 設定と結果

公開してはいけないもの:
- 5層パイプライン（P0-P4）の全体設計
- 造化三神の横断プロセス
- 3つの Sacred Treasures
- 神話マッピングの全体像

## ディレクトリ
既存の yamatoLLM/ 構造に従う。docs/ の設計書を参照して実装。
