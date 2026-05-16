"""
国譲り — llm-jp-4-8b-base の重みを yamatoLLM に継承する

「大国主命、国を天津神に譲り渡す」

注: クラス名は履歴経緯で "QwenAdapter" を維持しているが、
現在のデフォルトは llm-jp-4-8b-base (LlamaForCausalLM)。
カスタムヘッドが期待する hidden_size=4096 は両者で一致。

llm-jp-4-8b-base のアーキテクチャ:
  - Architecture: LlamaForCausalLM
  - Hidden size: 4096
  - Num layers: 32
  - Num attention heads: 32 (GQA: 8 KV heads)
  - Intermediate size: 14336 (SwiGLU / silu)
  - Vocab size: 196608
  - RoPE theta: 500000
  - Max position embeddings: 65536

yamatoLLM が追加するもの:
  - 意図分類ヘッド (OmoikaneIntentRouter)
  - 型予測ヘッド (TsukuyomiTypeHead) — julia_no_mikoto から流用
  - エラー予測ヘッド (SusanooErrorHead) — julia_no_mikoto から流用
  - 信頼度ヘッド (BonpuConfidence)
  - 入出力浄化 (InbeSanitizer)
  - 言依さし変換 (KotoyosashiProtocol)
"""

import logging
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
    """
    llm-jp-4-8b のロードとカスタム層の注入

    3つの操作を提供:
    1. load_base_model: Qwen ロード（量子化オプション付き）
    2. inject_lora: LoRA アダプタの注入
    3. attach_custom_heads: yamatoLLM 固有ヘッドの追加
    """

    @staticmethod
    def load_base_model(
        model_name: str = "llm-jp/llm-jp-4-8b-base",
        quantize: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: Optional[Any] = None,
    ):
        """
        llm-jp-4-8b ベースモデルのロード

        Args:
            model_name: HuggingFace モデル名
            quantize: 量子化設定
                None   → FP16（学習時、A100）
                "4bit" → GPTQ/BnB 4bit（推論時、RTX 3060）
                "8bit" → INT8（推論時）
            device_map: デバイス配置
            torch_dtype: テンソル型（None で自動選択）

        Returns:
            model: Qwen モデル
            tokenizer: Qwen トークナイザー
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading {model_name} (quantize={quantize})")

        # トークナイザー
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 量子化設定
        load_kwargs = {
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
            # FP16 / BF16
            if torch_dtype is None:
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            load_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        logger.info(
            f"Loaded {model_name}: "
            f"{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params"
        )

        return model, tokenizer

    @staticmethod
    def inject_lora(model, lora_config: Optional[LoRAConfig] = None):
        """
        LoRA アダプタの注入

        PEFT ライブラリを使用。Qwen の Attention と SwiGLU に LoRA を適用。
        カスタムヘッドは modules_to_save として フル学習対象に設定。

        Args:
            model: Qwen ベースモデル
            lora_config: LoRA 設定（None でデフォルト）

        Returns:
            model: LoRA 注入済みモデル
        """
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
            f"LoRA injected: {trainable / 1e6:.1f}M trainable / "
            f"{total / 1e9:.1f}B total ({trainable / total * 100:.2f}%)"
        )

        return model

    @staticmethod
    def attach_custom_heads(
        model,
        config: Optional[YamatoConfig] = None,
    ) -> Dict[str, nn.Module]:
        """
        yamatoLLM 固有のカスタムヘッドを追加

        Qwen モデルの hidden_states を受け取り、追加の出力を生成する。
        モデル本体には組み込まず、別管理の nn.ModuleDict として返す。
        yamato_model.py で統合される。

        Args:
            model: Qwen ベースモデル（hidden_size 取得用）
            config: yamatoLLM 設定

        Returns:
            heads: カスタムヘッドの辞書
        """
        if config is None:
            config = YamatoConfig()

        d_model = config.d_model

        from .iwato.omoikane_intent import OmoikaneIntentRouter
        from .iwato.kotoyosashi_protocol import KotoyosashiProtocol
        from .iwato.inbe_sanitizer import InbeSanitizer
        from .kenpou.bonpu_confidence import BonpuConfidence

        heads = nn.ModuleDict({
            "intent_router": OmoikaneIntentRouter(
                d_model=d_model,
                num_routes=config.iwato.num_routes,
            ),
            "kotoyosashi": KotoyosashiProtocol(
                d_model=d_model,
            ),
            "confidence": BonpuConfidence(
                d_model=d_model,
                config=config,
            ),
        })

        # InbeSanitizer はルールベース（学習不要）なので ModuleDict 外
        sanitizer = InbeSanitizer(config=config)

        total_params = sum(p.numel() for p in heads.parameters())
        logger.info(f"Custom heads attached: {total_params / 1e6:.2f}M params")

        return heads, sanitizer

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        quantize: Optional[str] = None,
        device_map: str = "auto",
    ):
        """
        yamatoLLM チェックポイントからの復元

        Args:
            checkpoint_path: チェックポイントのパス
            quantize: 量子化設定
            device_map: デバイス配置

        Returns:
            model, tokenizer, config, metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        config = checkpoint.get("config", YamatoConfig())
        base_model_name = checkpoint.get("base_model_name", config.backbone.model_name)
        stage = checkpoint.get("stage", "unknown")

        logger.info(f"Loading checkpoint: stage={stage}, base={base_model_name}")

        # ベースモデルロード
        model, tokenizer = QwenAdapter.load_base_model(
            model_name=base_model_name,
            quantize=quantize,
            device_map=device_map,
        )

        # ステージに応じた復元
        if stage == "kuniyuzuri":
            # 国譲り: LoRA なし、カスタムヘッドのみ
            pass
        else:
            # 天孫降臨以降: LoRA + カスタムヘッド
            model = QwenAdapter.inject_lora(model, config.lora)

        # state_dict の復元
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        metadata = {
            "stage": stage,
            "base_model_name": base_model_name,
        }

        return model, tokenizer, config, metadata
