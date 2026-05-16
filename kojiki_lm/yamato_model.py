"""
yamatoLLM — Qwen2.5-Coder-7B-Instruct + 型予測カスタムヘッド

TypeScript の型予測とハルシネーション制御を行うため、
Qwen2 backbone に追加ヘッドを attach できる骨格を提供する。

現時点で attach 済みなのは BonpuConfidence のみ。
TsukuyomiTypeHead / HirukoDetector / AmenomihashiraProtocol は
後続フェーズで `custom_heads` に追加していく。
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
            - confidence: BonpuConfidence
            - (将来) type_head: TsukuyomiTypeHead
            - (将来) hiruko_detector: HirukoDetector
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
        confidence_labels=None,
        **kwargs,
    ):
        """
        統合 forward pass（学習時）

        Returns:
            dict with:
                loss: 統合損失（base_loss + 補助損失）
                logits: 次トークン logits
                hidden_states: 最終層 hidden states
                confidence: 信頼度スコア（custom_heads 初期化済の場合）
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

        # 信頼度ヘッド
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
            result["loss"] = base_loss + 0.3 * conf_loss

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> YamatoOutput:
        """
        推論時のエントリポイント

        backbone.generate で素直に生成し、生成後の hidden states から
        信頼度を計算する。
        """
        inference_config = self.config.inference
        max_new_tokens = max_new_tokens or inference_config.max_new_tokens
        temperature = temperature or inference_config.temperature
        top_p = top_p or inference_config.top_p

        output = YamatoOutput()

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        device = next(self.backbone.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        generated_ids = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=inference_config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        new_ids = generated_ids[0, input_ids.shape[1]:]
        output.text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        output.generated_ids = generated_ids

        if self.custom_heads is not None:
            gen_hidden = self.get_hidden_states(generated_ids)
            conf_out = self.custom_heads["confidence"](gen_hidden)
            output.confidence = conf_out["confidence"].item()
            output.uncertainty_flag = bool(conf_out["uncertainty_flag"].item())
            output.truthfulness = conf_out["truthfulness"].item()

        return output

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
