"""
和の損失関数 (Wa Loss — Harmony Loss)

第1条: 「和を以て貴しとなす、忤ふること無きを宗とせよ」
第14条: 「群臣百寮、嫉み妬むこと無かれ」

既存の損失関数（KojikiLoss 等）をラップする Decorator パターンで実装。
内部モジュールを一切変更せず、ガバナンス層として被せる。

損失の構成:
    L_wa = L_base + λ * L_conflict + μ * L_regularization

    L_conflict:       attention head 間の KL divergence（不和の定量化）
    L_regularization: パラメータ正規化（第14条: 嫉妬＝過学習の抑制）
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kenpou_config import KenpouConfig, DEFAULT_KENPOU_CONFIG


def _extract_kenpou_attr(config: Any, attr: str, default: Any) -> Any:
    """KenpouConfig または YamatoConfig から属性を安全に取得する。"""
    # KenpouConfig を直接持っている場合 (YamatoConfig.kenpou 等)
    if hasattr(config, "kenpou") and hasattr(config.kenpou, attr):
        return getattr(config.kenpou, attr)
    # 直接属性を持っている場合 (KenpouConfig)
    if hasattr(config, attr):
        return getattr(config, attr)
    return default


class WaLoss(nn.Module):
    """
    和の損失関数 — Decorator パターンによるガバナンスラッパー

    任意の base_loss_fn をラップし、和（harmony）項と正規化項を加える。

    第1条:  attention head 間の conflict を KL divergence で測定し、
            和を促進する損失として加算する。
    第14条: L2 正規化で過学習（嫉妬）を抑制する。

    Args:
        base_loss_fn: ラップする既存の損失関数（nn.Module or callable）。
                      None の場合、CrossEntropyLoss をデフォルトとして使用。
        config: KenpouConfig または kenpou 属性を持つ YamatoConfig。
    """

    def __init__(
        self,
        base_loss_fn: Optional[Union[nn.Module, Callable]] = None,
        config: Optional[Any] = None,
    ):
        super().__init__()

        if config is None:
            config = DEFAULT_KENPOU_CONFIG

        self.base_loss_fn = base_loss_fn or nn.CrossEntropyLoss(ignore_index=-100)
        self.harmony_lambda = _extract_kenpou_attr(config, "harmony_lambda", 0.1)
        self.regularization_mu = _extract_kenpou_attr(config, "regularization_mu", 0.01)

    def forward(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
        attentions: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        和の損失を計算する。

        Args:
            outputs: モデル出力（logits テンソルまたは辞書）。
            targets: 教師ラベル（テンソルまたは辞書）。
            attentions: attention weights [batch, num_heads, seq, seq]。
                        None の場合、conflict 項は 0。
            model: 正規化対象のモデル。None の場合、正規化項は 0。

        Returns:
            (total_loss, details_dict) のタプル。
        """
        details: Dict[str, torch.Tensor] = {}

        # --- 1. 基礎損失 (delegate to wrapped function) ---
        if isinstance(self.base_loss_fn, nn.Module) and hasattr(self.base_loss_fn, "forward"):
            # KojikiLoss 等の複合損失は (total, dict) を返す場合がある
            base_result = self.base_loss_fn(outputs, targets)
            if isinstance(base_result, tuple):
                base_loss, sub_details = base_result
                details.update(sub_details)
            else:
                base_loss = base_result
        else:
            base_loss = self.base_loss_fn(outputs, targets)

        details["base_loss"] = base_loss.detach()

        # --- 2. Conflict 損失: 第1条「和を以て貴しとなす」 ---
        if attentions is not None:
            conflict_loss = self.compute_conflict(attentions)
        else:
            conflict_loss = torch.tensor(0.0, device=base_loss.device)

        details["conflict_loss"] = conflict_loss.detach()

        # --- 3. 正規化損失: 第14条「嫉み妬むこと無かれ」 ---
        if model is not None:
            reg_loss = self.compute_regularization(model)
        else:
            reg_loss = torch.tensor(0.0, device=base_loss.device)

        details["regularization_loss"] = reg_loss.detach()

        # --- 合算: L_wa = L_base + λ * L_conflict + μ * L_reg ---
        total_loss = (
            base_loss
            + self.harmony_lambda * conflict_loss
            + self.regularization_mu * reg_loss
        )
        details["total_loss"] = total_loss.detach()

        return total_loss, details

    def compute_conflict(self, attentions: torch.Tensor) -> torch.Tensor:
        """
        attention head 間の不和（conflict）を KL divergence で定量化する。

        第1条: 「和を以て貴しとなす」
        各 head の attention 分布と平均分布の KL divergence の平均を返す。
        divergence が高いほど head 間の「不和」が大きい。

        Args:
            attentions: [batch, num_heads, seq_len, seq_len]

        Returns:
            スカラーの conflict 損失。
        """
        # attentions を確率分布として正規化（既に softmax 済みの想定だが安全のため）
        attn = attentions.clamp(min=1e-8)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # 全 head の平均分布
        mean_attn = attn.mean(dim=1, keepdim=True)  # [batch, 1, seq, seq]

        # 各 head と平均の KL divergence: KL(head || mean)
        # KL(P || Q) = sum(P * log(P / Q))
        kl_per_head = attn * (attn.log() - mean_attn.log())  # [batch, heads, seq, seq]
        kl_per_head = kl_per_head.sum(dim=-1)                # [batch, heads, seq]
        kl_per_head = kl_per_head.mean(dim=-1)                # [batch, heads]

        # 全 head の平均 conflict
        conflict = kl_per_head.mean()

        return conflict

    def compute_regularization(self, model: nn.Module) -> torch.Tensor:
        """
        L2 正規化損失を計算する。

        第14条: 「嫉み妬むこと無かれ」
        パラメータの過度な偏り（＝過学習）を抑制する。

        Args:
            model: 正規化対象のモデル。

        Returns:
            スカラーの正規化損失。
        """
        l2_sum = torch.tensor(0.0, device=next(model.parameters()).device)
        param_count = 0
        for param in model.parameters():
            if param.requires_grad:
                l2_sum = l2_sum + param.pow(2).sum()
                param_count += param.numel()

        # パラメータ数で正規化して規模に依存しないようにする
        if param_count > 0:
            l2_sum = l2_sum / param_count

        return l2_sum
