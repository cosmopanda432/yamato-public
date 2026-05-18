"""
月読 (Tsukuyomi) — per-token TypeScript 型予測ヘッド。

Qwen2.5-Coder backbone の hidden_states を受け取り、
各トークンに対する TS 型ラベル (`config/ts_type_vocab.json`) を予測する。

学習時:
    type_logits  [B, L, V]  に対し、ManyTypes4TypeScript 由来の
    type_labels  [B, L]     で CrossEntropy。

推論時:
    各トークン位置の TS 型ラベルを出力（per-token 型予測）。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TsukuyomiTypeHead(nn.Module):
    """
    per-token 型予測ヘッド。

    Args:
        d_model:         backbone hidden_size (Qwen2.5-Coder-7B = 3584)
        type_vocab_size: ts_type_vocab.json の vocab_size (default 256)
        hidden_dim:      中間層次元
        dropout:         dropout 率
        ignore_index:    無注釈トークンを無視するラベル ID
    """

    def __init__(
        self,
        d_model: int = 3584,
        type_vocab_size: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.d_model = d_model
        self.type_vocab_size = type_vocab_size
        self.ignore_index = ignore_index

        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, type_vocab_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        type_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [B, L, d_model]
            type_labels:   [B, L] (任意。あれば loss を返す)

        Returns:
            dict with:
                type_logits: [B, L, V]
                type_preds:  [B, L]   (argmax)
                type_loss:   scalar   (labels あり時のみ)
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"expected [B, L, d_model], got shape {tuple(hidden_states.shape)}"
            )

        type_logits = self.proj(hidden_states)  # [B, L, V]
        type_preds = type_logits.argmax(dim=-1)  # [B, L]

        out: Dict[str, torch.Tensor] = {
            "type_logits": type_logits,
            "type_preds": type_preds,
        }

        if type_labels is not None:
            if type_labels.shape != hidden_states.shape[:2]:
                raise ValueError(
                    f"type_labels shape {tuple(type_labels.shape)} must equal "
                    f"hidden_states[:2] {tuple(hidden_states.shape[:2])}"
                )
            loss = F.cross_entropy(
                type_logits.reshape(-1, self.type_vocab_size),
                type_labels.reshape(-1),
                ignore_index=self.ignore_index,
            )
            out["type_loss"] = loss

        return out
