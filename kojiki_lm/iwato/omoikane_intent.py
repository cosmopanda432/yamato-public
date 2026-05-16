"""
第二章: 思案 (思兼神) — 意図解析・ルーティング
omoikane_intent.py

天岩戸神話において、思兼神（おもいかねのかみ）は知恵の神として
岩戸を開くための計略を立案した。

本モジュールはその「思案」に対応する。
ユーザ入力の意図を解析し、適切な処理ルートへ振り分ける
インテントルーターである。yamatoLLM の中核モジュールの一つ。

ルート定義:
    0 = chat     (一般対話)     → 天宇受売命が処理 (AmenouzumeDecoder)
    1 = codegen  (コード生成)   → 言依さし (KotoyosashiProtocol) → KojikiLM
    2 = retrieval(知識検索)     → 布刀玉命 (FutodamaRetriever) → RAG → 生成

神話的対応:
    思兼神は「思い」を「兼ねる」神、すなわち多くの思慮を一身に兼ね備え、
    最善の策を導き出す。同様に本モジュールは入力を分析し、
    最適な処理パスを選択する。

入出力:
    Input:  hidden_states [batch, seq_len, d_model] — 埋め込み層からの出力
    Output: dict {
        route_logits:  [batch, num_routes]  — ルート分類ロジット
        route:         [batch]              — 選択されたルート (argmax)
        intent_vector: [batch, d_model]     — 意図ベクトル (下流モジュール用)
    }
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ルート定数
ROUTE_CHAT = 0       # 一般対話 → 天宇受売命
ROUTE_CODEGEN = 1    # コード生成 → 言依さし → KojikiLM
ROUTE_RETRIEVAL = 2  # 知識検索 → 布刀玉命 → RAG


ROUTE_NAMES = {
    ROUTE_CHAT: "chat",
    ROUTE_CODEGEN: "codegen",
    ROUTE_RETRIEVAL: "retrieval",
}


class OmoikaneIntentRouter(nn.Module):
    """思兼神 — 意図解析・ルーティングモジュール

    入力の hidden_states をプーリングし、意図を分類する。
    分類結果に基づき、後続の処理パスを決定する。

    ルーティング:
        route 0 (chat)      → 天宇受売命 (AmenouzumeDecoder) で対話生成
        route 1 (codegen)   → 言依さし (KotoyosashiProtocol) でコード生成層に接続
        route 2 (retrieval) → 布刀玉命 (FutodamaRetriever) で RAG 知識検索

    Args:
        d_model:    隠れ層の次元数 (llm-jp-4-8b hidden_size = 4096)
        num_routes: ルーティング先の数 (デフォルト: 3)
        dropout:    ドロップアウト率
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_routes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_routes = num_routes

        # プーリング層: hidden_states を単一ベクトルに集約
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

        # ルート分類ヘッド
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_routes),
        )

        # 意図ベクトル射影: 下流モジュールへ渡す意図表現を生成
        self.intent_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def _masked_mean_pooling(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """マスク付き平均プーリング

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len] (1 = 有効, 0 = パディング)

        Returns:
            pooled: [batch, d_model]
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        # マスクを拡張して d_model 次元に適用
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [batch, seq_len, 1]
        masked_hidden = hidden_states * mask
        sum_hidden = masked_hidden.sum(dim=1)          # [batch, d_model]
        count = mask.sum(dim=1).clamp(min=1.0)         # [batch, 1]
        return sum_hidden / count

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """意図解析・ルーティング

        Args:
            hidden_states: [batch, seq_len, d_model] 埋め込み層からの出力
            attention_mask: [batch, seq_len] attention mask (任意)

        Returns:
            dict:
                route_logits:  [batch, num_routes] — 各ルートへの分類スコア
                route:         [batch]             — 選択されたルート (argmax)
                route_probs:   [batch, num_routes] — ソフトマックス確率
                intent_vector: [batch, d_model]    — 意図ベクトル
        """
        # Step 1: プーリング — シーケンスを単一表現に集約
        pooled = self._masked_mean_pooling(hidden_states, attention_mask)
        pooled = self.pooler(pooled)  # [batch, d_model]

        # Step 2: ルート分類
        route_logits = self.classifier(pooled)  # [batch, num_routes]
        route_probs = F.softmax(route_logits, dim=-1)
        route = route_logits.argmax(dim=-1)     # [batch]

        # Step 3: 意図ベクトル生成 — 下流モジュールが利用する意図表現
        intent_vector = self.intent_projection(pooled)  # [batch, d_model]

        return {
            "route_logits": route_logits,
            "route": route,
            "route_probs": route_probs,
            "intent_vector": intent_vector,
        }

    def get_route_name(self, route_id: int) -> str:
        """ルート ID から名前を取得する"""
        return ROUTE_NAMES.get(route_id, f"unknown({route_id})")
