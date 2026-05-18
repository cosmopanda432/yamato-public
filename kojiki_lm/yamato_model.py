"""
yamatoLLM — Qwen2.5-Coder-7B-Instruct + 型予測カスタムヘッド

TypeScript の型予測を行うため、Qwen2 backbone に追加ヘッドを attach する。
attach 済みヘッド: TsukuyomiTypeHead (月読), BonpuConfidence (凡夫)。
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .yamato_config import YamatoConfig
from .qwen_adapter import QwenAdapter

logger = logging.getLogger(__name__)


@dataclass
class YamatoOutput:
    """yamatoLLM の統合出力"""
    text: str = ""
    generated_ids: Optional[Any] = None

    # ガバナンス
    confidence: float = 0.5
    uncertainty_flag: bool = False
    truthfulness: float = 0.5

    # 型予測（将来の TsukuyomiTypeHead 用、現状は未使用）
    type_predictions: Optional[Any] = None

    # メタ
    extra: Dict[str, Any] = field(default_factory=dict)


class YamatoLLM(nn.Module):
    """
    yamatoLLM 統合モデル

    構成:
        backbone:      Qwen2.5-Coder-7B-Instruct (LoRA 適用可)
        custom_heads:  ModuleDict
            - type_head:  TsukuyomiTypeHead
            - confidence: BonpuConfidence
    """

    def __init__(
        self,
        backbone=None,
        tokenizer=None,
        config: Optional[YamatoConfig] = None,
    ):
        super().__init__()
        self.config = config or YamatoConfig()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.custom_heads: Optional[nn.ModuleDict] = None

    def init_custom_heads(self):
        """
        カスタムヘッドの初期化

        backbone と同じデバイス・compute dtype に揃える。
        bnb 4bit/8bit ロード時は param.dtype が uint8 になるため、
        compute dtype (bfloat16) に明示する。
        """
        self.custom_heads = QwenAdapter.attach_custom_heads(
            model=self.backbone,
            config=self.config,
        )

        backbone_param = next(self.backbone.parameters())
        device = backbone_param.device
        if hasattr(backbone_param, "quant_state"):
            dtype = torch.bfloat16
        else:
            dtype = backbone_param.dtype
        self.custom_heads = self.custom_heads.to(device=device, dtype=dtype)

        logger.info(
            "Custom heads initialized (random weights, device=%s, dtype=%s)",
            device, dtype,
        )

    def get_hidden_states(self, input_ids, attention_mask=None):
        """backbone から最終層 hidden states を取得"""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        type_labels=None,
        confidence_labels=None,
        **kwargs,
    ):
        """
        統合 forward pass（学習時）

        Args:
            labels:            [B, L] 次トークン教師 (CLM)
            type_labels:       [B, L] per-token TS 型 ID (TsukuyomiTypeHead 用)
            confidence_labels: [B] 信頼度教師 (BonpuConfidence 用)

        Returns:
            dict with:
                loss:           統合損失 (base + type * w + confidence * w)
                logits:         次トークン logits
                type_logits:    TS 型 logits (custom_heads 初期化済の場合)
                type_preds:     TS 型 argmax
                confidence:     信頼度スコア
                hidden_states:  最終層 hidden states
        """
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = backbone_outputs.logits
        hidden_states = backbone_outputs.hidden_states[-1]
        base_loss = backbone_outputs.loss if labels is not None else torch.tensor(0.0)

        result: Dict[str, Any] = {
            "logits": logits,
            "hidden_states": hidden_states,
            "base_loss": base_loss,
            "loss": base_loss,
        }

        if self.custom_heads is None:
            return result

        total_loss = base_loss

        # 月読 (TsukuyomiTypeHead) — per-token 型予測
        type_out = self.custom_heads["type_head"](hidden_states, type_labels=type_labels)
        result["type_logits"] = type_out["type_logits"]
        result["type_preds"] = type_out["type_preds"]
        if "type_loss" in type_out:
            result["type_loss"] = type_out["type_loss"]
            total_loss = total_loss + self.config.type_head.loss_weight * type_out["type_loss"]

        # 凡夫 (BonpuConfidence) — 信頼度
        conf_out = self.custom_heads["confidence"](hidden_states)
        result["confidence"] = conf_out["confidence"]
        result["uncertainty_flag"] = conf_out["uncertainty_flag"]
        result["truthfulness"] = conf_out["truthfulness"]

        if confidence_labels is not None:
            conf_loss = F.mse_loss(
                conf_out["confidence"].squeeze(-1),
                confidence_labels.to(conf_out["confidence"].dtype),
            )
            result["confidence_loss"] = conf_loss
            total_loss = total_loss + 0.3 * conf_loss

        result["loss"] = total_loss
        return result

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        quantize: Optional[str] = None,
        device_map: str = "auto",
    ) -> "YamatoLLM":
        """チェックポイントから yamatoLLM を復元"""
        backbone, tokenizer, config, metadata = QwenAdapter.load_checkpoint(
            checkpoint_path=checkpoint_path,
            quantize=quantize,
            device_map=device_map,
        )

        model = cls(backbone=backbone, tokenizer=tokenizer, config=config)
        model.init_custom_heads()

        logger.info("YamatoLLM loaded: stage=%s", metadata["stage"])
        return model

    @classmethod
    def from_qwen(
        cls,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        quantize: Optional[str] = None,
        config: Optional[YamatoConfig] = None,
    ) -> "YamatoLLM":
        """Qwen2.5-Coder ベースから新規に yamatoLLM を構築"""
        config = config or YamatoConfig()
        config.backbone.model_name = model_name

        backbone, tokenizer = QwenAdapter.load_base_model(
            model_name=model_name,
            quantize=quantize,
        )

        model = cls(backbone=backbone, tokenizer=tokenizer, config=config)
        model.init_custom_heads()
        model.config.stage = "baseline"

        logger.info("YamatoLLM created from %s (stage=baseline)", model_name)
        return model
