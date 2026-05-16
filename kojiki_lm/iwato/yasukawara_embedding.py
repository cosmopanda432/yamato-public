"""
第一章: 参集 (天安河原) — 入力埋め込み
yasukawara_embedding.py

天岩戸神話において、天照大御神が岩戸に隠れた後、
八百万の神々が天安河原（あめのやすかわら）に集まり対策を練った。

本モジュールはその「参集」に対応する。
ユーザ入力テキストを受け取り、Qwen バックボーンの埋め込みベクトルとして
言語処理パイプラインに導入する薄いラッパーである。

機能:
    - Qwen backbone の hidden_states をそのまま受け取る
    - 会話履歴 (context_hidden) が存在する場合、射影して統合する
    - Naganakidori: 入力開始位置を attention mask にマークし、
      後続モジュールがユーザ入力の境界を識別できるようにする

入出力:
    Input:  hidden_states [batch, seq_len, d_model] — Qwen backbone からの埋め込み
    Output: E_input       [batch, seq_len, d_model] — コンテキスト強化済み埋め込み
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class YasukawaraEmbedding(nn.Module):
    """天安河原 — 入力埋め込みラッパー

    Qwen backbone が生成した hidden_states を受け取り、
    オプションで会話履歴コンテキストを統合して返す。

    Args:
        d_model: 隠れ層の次元数 (Qwen3.5-9B hidden_size = 3584)
        dropout: コンテキスト射影後のドロップアウト率
    """

    def __init__(self, d_model: int = 3584, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model

        # コンテキスト履歴を現在の hidden_states 空間に射影する
        self.context_projection = nn.Linear(d_model, d_model)
        self.context_gate = nn.Sequential(
            nn.Linear(d_model * 2, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: Tensor,
        context_hidden: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """入力埋め込みの処理

        Args:
            hidden_states:  [batch, seq_len, d_model] Qwen backbone の出力
            context_hidden: [batch, ctx_len, d_model] 過去の会話履歴 (任意)
            attention_mask: [batch, seq_len] attention mask (Naganakidori 用)

        Returns:
            E_input: [batch, seq_len, d_model] コンテキスト強化済み埋め込み
        """
        if context_hidden is not None:
            # 会話履歴を射影し、平均プーリングで単一ベクトルにする
            ctx_proj = self.context_projection(context_hidden)  # [batch, ctx_len, d_model]
            ctx_summary = ctx_proj.mean(dim=1, keepdim=True)    # [batch, 1, d_model]
            ctx_summary = ctx_summary.expand_as(hidden_states)  # [batch, seq_len, d_model]

            # ゲート機構: hidden_states と context の混合比を学習
            gate_input = torch.cat([hidden_states, ctx_summary], dim=-1)
            gate = self.context_gate(gate_input)  # [batch, seq_len, 1]

            hidden_states = hidden_states + gate * self.dropout(ctx_summary)
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def mark_input_start(self, attention_mask: Tensor, input_start_pos: int) -> Tensor:
        """Naganakidori — 入力開始位置マーキング

        attention_mask にユーザ入力の開始位置を記録し、
        後続モジュール（思兼神など）が境界を識別できるようにする。

        Args:
            attention_mask: [batch, seq_len]
            input_start_pos: ユーザ入力が始まるトークン位置

        Returns:
            marked_mask: [batch, seq_len] 入力開始位置に 2 をマークした mask
        """
        marked_mask = attention_mask.clone()
        marked_mask[:, input_start_pos] = 2  # 特殊マーカー値
        return marked_mask
