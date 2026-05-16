"""
第三章: 奉献 (布刀玉命+真榊) — RAG・知識統合
futodama_retriever.py

天岩戸神話において、布刀玉命（ふとだまのみこと）は真榊（まさかき）に
八尺瓊勾玉（やさかにのまがたま）、八咫鏡（やたのかがみ）、白丹寸手・青丹寸手
（しらにきて・あおにきて）の布帛を掛け、御幣として岩戸の前に捧げた。

本モジュールはその「奉献」に対応する。
外部知識ベース（稗田阿礼メモリ）への問い合わせと、
取得した知識を hidden_states に統合する RAG モジュールである。

構成要素:
    FutodamaRetriever  — クロスアテンションによる知識検索・統合
    MagatamaChaining   — 八尺瓊勾玉: 意味的連鎖の保持・強化

神話的対応:
    布刀玉命が真榊に掛けた勾玉・鏡・布帛は、それぞれ
    知識の連鎖（勾玉）、知識の照合（鏡）、知識の装飾（布帛）に対応する。

入出力:
    Input:  hidden_states [batch, seq_len, d_model] — 現在の表現
            intent_vector [batch, d_model]          — 思兼神からの意図ベクトル
            memory_bank   [num_docs, d_model]       — 外部知識ベース (任意)
    Output: H_context     [batch, seq_len, d_model] — 知識統合済み表現
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MagatamaChaining(nn.Module):
    """八尺瓊勾玉 — 意味的連鎖の保持・強化

    勾玉の形状（循環・連鎖）に倣い、コンテキスト全体にわたる
    意味的連鎖を残差接続で保持・強化する。

    知識統合の前後で意味の一貫性が崩れることを防ぎ、
    元の hidden_states が持つ文脈情報を確実に下流へ伝達する。

    Args:
        d_model: 隠れ層の次元数 (llm-jp-4-8b hidden_size = 4096)
        dropout: ドロップアウト率
    """

    def __init__(self, d_model: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.chain_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, hidden_states: Tensor, residual: Tensor) -> Tensor:
        """意味的連鎖の保持

        Args:
            hidden_states: [batch, seq_len, d_model] 知識統合後の表現
            residual:      [batch, seq_len, d_model] 知識統合前の元表現

        Returns:
            output: [batch, seq_len, d_model] 連鎖強化済み表現
        """
        # 残差接続による元の意味情報の保持
        combined = hidden_states + residual

        # 連鎖変換: 意味的つながりを強化
        chain = self.chain_transform(self.norm(combined))

        # スケーリング付き残差: 過度な変換を防ぐ
        return combined + self.scale * chain


class FutodamaRetriever(nn.Module):
    """布刀玉命 — RAG・知識統合モジュール

    外部知識ベース（稗田阿礼メモリ）に対してクロスアテンションで
    問い合わせを行い、取得した知識をゲート機構で統合する。

    処理の流れ:
        1. intent_vector で問い合わせクエリを生成
        2. memory_bank から top-k の関連知識を検索 (retrieve)
        3. クロスアテンションで hidden_states と検索結果を統合
        4. ゲート機構で統合度を制御
        5. MagatamaChaining で意味的連鎖を保持

    Args:
        d_model:   隠れ層の次元数 (llm-jp-4-8b hidden_size = 4096)
        num_heads: クロスアテンションのヘッド数
        top_k:     検索時の上位 k 件数
        dropout:   ドロップアウト率
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 4,
        top_k: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.top_k = top_k

        # クエリ射影: hidden_states + intent_vector → 検索クエリ
        self.query_proj = nn.Linear(d_model, d_model)
        self.intent_proj = nn.Linear(d_model, d_model)

        # クロスアテンション: hidden_states (Q) × retrieved_knowledge (K, V)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ゲート機構: 検索知識の統合度を制御
        # 入力を [hidden_states, cross_attn_output] の連結とし、
        # シグモイドでブレンド比率を出力する
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # 正規化
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 八尺瓊勾玉: 意味的連鎖の保持
        self.magatama = MagatamaChaining(d_model, dropout)

    def retrieve(
        self,
        query_embedding: Tensor,
        memory_bank: Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """稗田阿礼メモリからの知識検索

        コサイン類似度に基づき、クエリに最も関連する
        上位 k 件のメモリベクトルを返す。

        Args:
            query_embedding: [batch, d_model] 検索クエリ
            memory_bank:     [num_docs, d_model] 外部知識ベース
            top_k:           上位何件を返すか (デフォルト: self.top_k)

        Returns:
            retrieved:  [batch, top_k, d_model] 検索結果ベクトル
            scores:     [batch, top_k] 類似度スコア
        """
        if top_k is None:
            top_k = self.top_k

        # 実際の文書数が top_k より少ない場合に対応
        top_k = min(top_k, memory_bank.size(0))

        # コサイン類似度の計算
        query_norm = F.normalize(query_embedding, p=2, dim=-1)    # [batch, d_model]
        memory_norm = F.normalize(memory_bank, p=2, dim=-1)       # [num_docs, d_model]

        # [batch, num_docs]
        similarity = torch.matmul(query_norm, memory_norm.t())

        # 上位 k 件を取得
        scores, indices = similarity.topk(top_k, dim=-1)  # [batch, top_k]

        # インデックスでメモリから取得
        # indices: [batch, top_k] → 各バッチのインデックスで memory_bank を参照
        retrieved = memory_bank[indices]  # [batch, top_k, d_model]

        return retrieved, scores

    def forward(
        self,
        hidden_states: Tensor,
        intent_vector: Tensor,
        memory_bank: Optional[Tensor] = None,
    ) -> Tensor:
        """知識統合パイプライン

        Args:
            hidden_states: [batch, seq_len, d_model] 現在の表現
            intent_vector: [batch, d_model] 思兼神からの意図ベクトル
            memory_bank:   [num_docs, d_model] 外部知識ベース (None の場合パススルー)

        Returns:
            H_context: [batch, seq_len, d_model] 知識統合済み表現
        """
        residual = hidden_states

        # メモリバンクが無い場合はパススルー (RAG 不要)
        if memory_bank is None:
            return hidden_states

        batch_size = hidden_states.size(0)

        # Step 1: 意図ガイド付きクエリ生成
        # intent_vector を hidden_states の各位置に加算して検索クエリとする
        intent_expanded = intent_vector.unsqueeze(1)  # [batch, 1, d_model]
        intent_bias = self.intent_proj(intent_expanded)
        query = self.query_proj(hidden_states) + intent_bias  # [batch, seq_len, d_model]

        # Step 2: 知識検索 — intent_vector をクエリとして top-k を取得
        query_for_retrieval = self.intent_proj(intent_vector)  # [batch, d_model]
        retrieved, _scores = self.retrieve(
            query_for_retrieval, memory_bank, self.top_k
        )  # [batch, top_k, d_model]

        # Step 3: クロスアテンション — hidden_states が retrieved knowledge に注意
        attn_output, _attn_weights = self.cross_attention(
            query=query,
            key=retrieved,
            value=retrieved,
        )  # [batch, seq_len, d_model]
        attn_output = self.dropout(attn_output)

        # Step 4: ゲート機構 — 検索知識の統合度を適応的に制御
        gate_input = torch.cat([hidden_states, attn_output], dim=-1)
        gate_value = self.gate(gate_input)  # [batch, seq_len, 1]

        # ゲートによるブレンド: gate=0 → 元の hidden_states, gate=1 → 検索知識
        blended = hidden_states + gate_value * attn_output
        blended = self.norm(blended)

        # Step 5: 八尺瓊勾玉 — 意味的連鎖の保持
        output = self.magatama(blended, residual)

        return output
