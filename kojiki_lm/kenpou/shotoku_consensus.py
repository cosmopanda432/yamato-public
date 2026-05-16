"""
聖徳コンセンサス (Shotoku Consensus)

第17条: 「夫れ事は独り断ずべからず。必ず衆とともに論ずべし」
第4条:  「群卿百寮、礼を以て本とせよ」

MoE ルーティングの拡張として、クエリの重要度（少事/大事）に応じて
エキスパートの合議方式を切り替えるコンセンサスモジュール。

動作:
    少事 (importance < threshold): top-1 エキスパートで即断（高速パス）
    大事 (importance >= threshold): 全エキスパートの加重合議（コンセンサス）

既存の MoE ルーターを変更せず、その出力をラップして使用する。
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kenpou_config import KenpouConfig, DEFAULT_KENPOU_CONFIG


def _get_config_value(config: Any, attr: str, default: Any) -> Any:
    """KenpouConfig または YamatoConfig から属性を安全に取得する。"""
    if hasattr(config, "kenpou") and hasattr(config.kenpou, attr):
        return getattr(config.kenpou, attr)
    if hasattr(config, attr):
        return getattr(config, attr)
    return default


class ShotokuConsensus(nn.Module):
    """
    聖徳コンセンサスモジュール

    第17条: 「独り断ずべからず、衆とともに論ぜよ」

    クエリの重要度を評価し、重要度に応じてエキスパート出力の
    統合方法を切り替える。少事は即断、大事は合議。

    Args:
        d_model: モデルの隠れ層次元数（default: 4096）。
        config: KenpouConfig または kenpou 属性を持つ YamatoConfig。
    """

    def __init__(
        self,
        d_model: int = 4096,
        config: Optional[Any] = None,
    ):
        super().__init__()

        if config is None:
            config = DEFAULT_KENPOU_CONFIG

        self.d_model = d_model
        self.importance_threshold = _get_config_value(
            config, "importance_threshold", 0.5
        )
        self.consensus_method = _get_config_value(
            config, "consensus_method", "weighted_vote"
        )
        self.max_consensus_rounds = _get_config_value(
            config, "max_consensus_rounds", 3
        )

        # 重要度評価ネットワーク
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # コンセンサス用の重み調整ネットワーク
        self.consensus_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query: torch.Tensor,
        expert_outputs: torch.Tensor,
        router_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        コンセンサスに基づくエキスパート出力の統合。

        Args:
            query: 入力クエリ [batch, seq_len, d_model]。
            expert_outputs: エキスパート出力 [batch, num_experts, seq_len, d_model]
                            または [batch, num_experts, d_model]。
            router_weights: ルーターの重み [batch, num_experts]。
                            None の場合、均等重みを使用。

        Returns:
            (output, importance_score, consensus_info) のタプル。
            output: 統合された出力 [batch, seq_len, d_model] or [batch, d_model]。
            importance_score: 重要度スコア [batch, 1]。
            consensus_info: メタ情報の辞書。
        """
        num_experts = expert_outputs.shape[1]

        # デフォルトのルーター重み（均等）
        if router_weights is None:
            router_weights = torch.ones(
                query.shape[0], num_experts,
                device=query.device, dtype=query.dtype,
            ) / num_experts

        # --- 1. 重要度評価 ---
        importance_score = self.assess_importance(query)  # [batch, 1]

        # --- 2. 少事/大事で分岐 ---
        consensus_info: Dict[str, Any] = {
            "importance_score": importance_score.detach(),
            "threshold": self.importance_threshold,
            "num_experts": num_experts,
        }

        # バッチ内で要素ごとに分岐するのは非効率なので、
        # importance に応じた soft blending を行う。
        # importance が低い → top-1 に集中、高い → 合議

        # --- top-1 パス (少事) ---
        top1_output = self._select_top1(expert_outputs, router_weights)

        # --- コンセンサスパス (大事) ---
        consensus_output = self.aggregate(
            expert_outputs, router_weights, method=self.consensus_method
        )

        # --- soft blending ---
        # importance が高いほどコンセンサス出力を重視
        # importance_score: [batch, 1] → broadcast 可能な形に拡張
        blend_weight = (importance_score > self.importance_threshold).float()
        # soft transition 版: sigmoid でスムーズに切り替え
        soft_blend = torch.sigmoid(
            10.0 * (importance_score - self.importance_threshold)
        )

        # 集約後の出力次元に合わせて拡張
        # soft_blend: [batch, 1]
        # top1_output / consensus_output: [batch, seq_len, d_model] or [batch, d_model]
        if top1_output.dim() == 3:
            # [batch, 1] → [batch, 1, 1] → broadcasts to [batch, seq_len, d_model]
            soft_blend_expanded = soft_blend.unsqueeze(-1)
        else:
            # [batch, 1] → broadcasts to [batch, d_model]
            soft_blend_expanded = soft_blend

        output = (1.0 - soft_blend_expanded) * top1_output + soft_blend_expanded * consensus_output

        consensus_info["blend_weight"] = soft_blend.detach()
        consensus_info["path"] = "blended"

        return output, importance_score, consensus_info

    def assess_importance(self, query: torch.Tensor) -> torch.Tensor:
        """
        クエリの重要度を [0, 1] のスコアとして評価する。

        第17条: 大事か少事かを判定する。

        Args:
            query: [batch, seq_len, d_model] または [batch, d_model]。

        Returns:
            importance_score: [batch, 1]。
        """
        if query.dim() == 3:
            # シーケンスの平均プーリング
            pooled = query.mean(dim=1)  # [batch, d_model]
        else:
            pooled = query  # [batch, d_model]

        return self.importance_scorer(pooled)  # [batch, 1]

    def _select_top1(
        self,
        expert_outputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        最も重みの高いエキスパートの出力を選択する（少事の即断）。

        Args:
            expert_outputs: [batch, num_experts, ...] エキスパート出力。
            weights: [batch, num_experts] ルーター重み。

        Returns:
            top-1 エキスパートの出力。
        """
        top1_idx = weights.argmax(dim=1)  # [batch]

        # gather で top-1 を取得
        if expert_outputs.dim() == 4:
            # [batch, num_experts, seq_len, d_model]
            batch_size, _, seq_len, d = expert_outputs.shape
            idx = top1_idx.view(-1, 1, 1, 1).expand(-1, 1, seq_len, d)
            selected = expert_outputs.gather(1, idx).squeeze(1)
        elif expert_outputs.dim() == 3:
            # [batch, num_experts, d_model]
            batch_size, _, d = expert_outputs.shape
            idx = top1_idx.view(-1, 1, 1).expand(-1, 1, d)
            selected = expert_outputs.gather(1, idx).squeeze(1)
        else:
            raise ValueError(
                f"expert_outputs must be 3D or 4D, got {expert_outputs.dim()}D"
            )

        return selected

    def aggregate(
        self,
        expert_outputs: torch.Tensor,
        weights: torch.Tensor,
        method: str = "weighted_vote",
    ) -> torch.Tensor:
        """
        全エキスパートの出力を合議（コンセンサス）で統合する。

        第17条: 「衆とともに論ぜよ」

        Args:
            expert_outputs: [batch, num_experts, ...] エキスパート出力。
            weights: [batch, num_experts] ルーター重み。
            method: 合議方式 ("weighted_vote" | "mean" | "softmax_vote")。

        Returns:
            統合された出力。
        """
        if method == "mean":
            return expert_outputs.mean(dim=1)

        if method == "softmax_vote":
            weights = F.softmax(weights, dim=-1)

        # weighted_vote (default) — 重みで加重平均
        normalized_weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        if expert_outputs.dim() == 4:
            # [batch, num_experts, seq_len, d_model]
            w = normalized_weights.unsqueeze(-1).unsqueeze(-1)
        elif expert_outputs.dim() == 3:
            # [batch, num_experts, d_model]
            w = normalized_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"expert_outputs must be 3D or 4D, got {expert_outputs.dim()}D"
            )

        aggregated = (expert_outputs * w).sum(dim=1)
        return aggregated
