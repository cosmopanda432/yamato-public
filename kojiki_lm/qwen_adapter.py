"""
Qwen2.5-Coder-7B-Instruct を yamatoLLM の backbone として組み込むアダプタ

提供する操作:
  1. load_base_model:    Qwen2 のロード（INT4/INT8/FP オプション）
  2. inject_lora:        LoRA アダプタの注入
  3. attach_custom_heads: yamatoLLM 固有ヘッド（TsukuyomiTypeHead, BonpuConfidence）の追加
  4. load_checkpoint:    保存済みチェックポイントからの復元
"""

import logging
import os
from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .yamato_config import YamatoConfig, LoRAConfig

logger = logging.getLogger(__name__)


class QwenAdapter:
    """Qwen2.5-Coder backbone の組み込みアダプタ"""

    @staticmethod
    def resolve_model_path(model_name_or_path: str) -> str:
        """
        ローカルにダウンロード済みのモデルがあれば優先して返す。

        `models/<basename>/` が存在すればそのパスを、なければ HuggingFace 形式の
        識別子をそのまま返す（オフライン起動を優先）。
        """
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        basename = model_name_or_path.split("/")[-1]
        local = os.path.join("models", basename)
        if os.path.isdir(local):
            logger.info("Using local model snapshot: %s", local)
            return local
        return model_name_or_path

    @staticmethod
    def load_base_model(
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        quantize: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: Optional[Any] = None,
    ):
        """
        Qwen2.5-Coder-7B-Instruct ベースモデルのロード

        Args:
            model_name: HuggingFace モデル名 or ローカルパス
            quantize: 量子化設定
                None   → BF16/FP16（学習時、A100/H100）
                "4bit" → BnB NF4（推論時、RTX 3060）
                "8bit" → INT8（推論時）
            device_map: デバイス配置
            torch_dtype: テンソル型（None で自動選択）

        Returns:
            (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = QwenAdapter.resolve_model_path(model_name)
        logger.info("Loading %s (quantize=%s)", resolved, quantize)

        tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
        }

        if quantize == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantize == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            if torch_dtype is None:
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            load_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(resolved, **load_kwargs)

        logger.info(
            "Loaded %s: %.1fB params",
            resolved,
            sum(p.numel() for p in model.parameters()) / 1e9,
        )

        return model, tokenizer

    @staticmethod
    def inject_lora(model, lora_config: Optional[LoRAConfig] = None):
        """LoRA アダプタの注入（Attention と SwiGLU の一部）"""
        from peft import get_peft_model, LoraConfig, TaskType

        if lora_config is None:
            lora_config = LoRAConfig()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            modules_to_save=lora_config.modules_to_save,
        )

        model = get_peft_model(model, peft_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "LoRA injected: %.1fM trainable / %.1fB total (%.2f%%)",
            trainable / 1e6, total / 1e9, trainable / total * 100,
        )
        return model

    @staticmethod
    def attach_custom_heads(
        model,
        config: Optional[YamatoConfig] = None,
    ) -> "nn.ModuleDict":
        """
        yamatoLLM 固有のカスタムヘッドを ModuleDict として作成して返す。

        現在組み込み:
          - confidence (BonpuConfidence): 信頼度スコア
          - type_head (TsukuyomiTypeHead): per-token TS 型予測
        """
        if config is None:
            config = YamatoConfig()

        d_model = config.d_model

        from .kenpou.bonpu_confidence import BonpuConfidence
        from .yomi.tsukuyomi_type_head import TsukuyomiTypeHead

        heads = nn.ModuleDict({
            "confidence": BonpuConfidence(d_model=d_model),
            "type_head": TsukuyomiTypeHead(
                d_model=d_model,
                type_vocab_size=config.type_head.vocab_size,
                hidden_dim=config.type_head.hidden_dim,
            ),
        })

        total_params = sum(p.numel() for p in heads.parameters())
        logger.info("Custom heads attached: %.2fM params", total_params / 1e6)

        return heads

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        quantize: Optional[str] = None,
        device_map: str = "auto",
    ):
        """yamatoLLM チェックポイントから復元"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        config = checkpoint.get("config", YamatoConfig())
        base_model_name = checkpoint.get("base_model_name", config.backbone.model_name)
        stage = checkpoint.get("stage", "unknown")

        logger.info("Loading checkpoint: stage=%s, base=%s", stage, base_model_name)

        model, tokenizer = QwenAdapter.load_base_model(
            model_name=base_model_name,
            quantize=quantize,
            device_map=device_map,
        )

        if stage != "baseline":
            model = QwenAdapter.inject_lora(model, config.lora)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        metadata = {
            "stage": stage,
            "base_model_name": base_model_name,
        }

        return model, tokenizer, config, metadata
