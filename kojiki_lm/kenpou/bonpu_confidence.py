"""
凡夫の自覚 — 信頼度スコアリング (Bonpu Confidence)

第10条: 「我必ずしも聖に非ず。彼必ずしも愚に非ず。共にこれ凡夫のみ」
第9条:  「信は是義の本なり」

モデルの出力に対する信頼度（confidence）と真実性（truthfulness）を
評価するスコアリングヘッド。

核心的制約:
    confidence は決して 1.0 にならない（聖に非ず）。
    confidence は決して 0.0 にならない（愚に非ず）。
    常に [floor, ceiling] の範囲に収まる。

これにより、モデルは常に「自分は凡夫である」という自覚を保ち、
過信も自己否定もしない中庸の態度を維持する。
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .kenpou_config import KenpouConfig, DEFAULT_KENPOU_CONFIG


def _get_config_value(config: Any, attr: str, default: Any) -> Any:
    """KenpouConfig または YamatoConfig から属性を安全に取得する。"""
    if hasattr(config, "kenpou") and hasattr(config.kenpou, attr):
        return getattr(config.kenpou, attr)
    if hasattr(config, attr):
        return getattr(config, attr)
    return default


class BonpuConfidence(nn.Module):
    """
    凡夫の信頼度モジュール

    第10条: 「我必ずしも聖に非ず。共にこれ凡夫のみ」

    hidden_states から信頼度（confidence）と真実性（truthfulness）を
    スコアリングする。信頼度は [floor, ceiling] にクランプされ、
    完全な確信も完全な不確信も持たない。

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
        self.floor = _get_config_value(config, "confidence_floor", 0.1)
        self.ceiling = _get_config_value(config, "confidence_ceiling", 1.0 - 1e-4)
        self.tau = _get_config_value(config, "truthfulness_tau", 0.6)
        self.uncertainty_expression = _get_config_value(
            config, "uncertainty_expression", True
        )

        # 信頼度ヘッド: hidden_states → [0, 1] のスカラー
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # 真実性ヘッド: hidden_states → [0, 1] のスカラー
        # 第9条: 「信は是義の本なり」
        self.truthfulness_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        信頼度と真実性を計算する。

        Args:
            hidden_states: モデルの隠れ状態。
                [batch, seq_len, d_model] または [batch, d_model]。

        Returns:
            dict with:
                - "confidence": [batch, 1] クランプ済み信頼度。
                - "confidence_raw": [batch, 1] クランプ前の生スコア。
                - "truthfulness": [batch, 1] 真実性スコア。
                - "uncertainty_flag": [batch, 1] bool テンソル。
                    信頼度 < tau の場合 True。
        """
        # --- プーリング ---
        pooled = self._pool(hidden_states)  # [batch, d_model]

        # --- 信頼度 (第10条) ---
        raw_confidence = self.confidence_head(pooled)  # [batch, 1], range [0, 1]

        # 凡夫の制約: [floor, ceiling] にスケーリング
        # raw=0 → floor, raw=1 → ceiling
        confidence = self.floor + (self.ceiling - self.floor) * raw_confidence

        # --- 真実性 (第9条) ---
        truthfulness = self.truthfulness_head(pooled)  # [batch, 1], range [0, 1]

        # --- 不確実性フラグ ---
        uncertainty_flag = confidence < self.tau

        return {
            "confidence": confidence,
            "confidence_raw": raw_confidence,
            "truthfulness": truthfulness,
            "uncertainty_flag": uncertainty_flag,
        }

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states をプーリングして [batch, d_model] にする。

        3D テンソル (batch, seq_len, d_model) の場合は最後のトークンを使用。
        2D テンソル (batch, d_model) の場合はそのまま返す。

        Args:
            hidden_states: [batch, seq_len, d_model] or [batch, d_model]。

        Returns:
            [batch, d_model] のプーリング済みテンソル。
        """
        if hidden_states.dim() == 3:
            # 最後のトークンを使用（causal LM の慣例）
            return hidden_states[:, -1, :]
        elif hidden_states.dim() == 2:
            return hidden_states
        else:
            raise ValueError(
                f"hidden_states must be 2D or 3D, got {hidden_states.dim()}D"
            )

    def get_confidence_level(self, confidence: torch.Tensor) -> str:
        """
        信頼度を人間に読みやすいレベルに変換する（推論時のユーティリティ）。

        Args:
            confidence: スカラーまたは [1] テンソル。

        Returns:
            信頼度レベルの文字列。
        """
        c = confidence.item() if isinstance(confidence, torch.Tensor) else confidence
        if c >= 0.8:
            return "high"      # 高い確信（ただし聖ではない）
        elif c >= 0.5:
            return "moderate"  # 中程度の確信
        elif c >= 0.3:
            return "low"       # 低い確信（不確実性を表明すべき）
        else:
            return "very_low"  # 非常に低い（ただし愚ではない）
