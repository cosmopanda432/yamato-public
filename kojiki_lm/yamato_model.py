"""
yamatoLLM — 3層統合モデル

言語処理層 (岩戸隠れ) + コード生成層 (Julia-no-Mikoto) + ガバナンス層 (憲法十七条)
を llm-jp-4-8b backbone 上で統合する。

推論フロー:
    [ユーザー入力]
        → 忌部 (入力浄化)
        → Qwen backbone (hidden states)
        → 思兼神 (ルーティング)
        ├── chat      → 天宇受売命 → 天手力男神
        ├── retrieval → 布刀玉命 → 天宇受売命 → 天手力男神
        └── codegen   → 言依さし → 天の御柱プロトコル → 復路解説
        → 凡夫の自覚 (信頼度)
        → 忌部 (出力浄化)
    [応答]
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

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


# ============================================================
# 出力データクラス
# ============================================================

@dataclass
class YamatoOutput:
    """yamatoLLM の統合出力"""
    # メイン出力
    text: str = ""
    generated_ids: Optional[Any] = None

    # ルーティング
    route: str = "chat"                    # chat / codegen / retrieval
    route_logits: Optional[Any] = None
    intent_vector: Optional[Any] = None

    # コード生成（codegen ルート時）
    code: Optional[str] = None
    phases: Optional[Dict[str, str]] = None  # phase_name -> generated_code

    # ガバナンス
    confidence: float = 0.5
    uncertainty_flag: bool = False
    truthfulness: float = 0.5
    safety_score: float = 1.0

    # 評価
    v_score: Optional[float] = None
    verdict: Optional[str] = None          # COMMIT / REPAIR / HALT


# ============================================================
# YamatoLLM 統合モデル
# ============================================================

class YamatoLLM(nn.Module):
    """
    yamatoLLM — 3層統合モデル

    llm-jp-4-8b を backbone とし、3層のカスタムコンポーネントを統合する。

    構成:
        backbone:       llm-jp-4-8b (frozen or LoRA)
        intent_router:  思兼神（意図分類）
        sanitizer:      忌部（入出力浄化）
        kotoyosashi:    言依さし（コード生成層への変換）
        confidence:     凡夫の自覚（信頼度スコア）
    """

    ROUTE_CHAT = 0
    ROUTE_CODEGEN = 1
    ROUTE_RETRIEVAL = 2
    ROUTE_NAMES = ["chat", "codegen", "retrieval"]

    def __init__(
        self,
        backbone=None,
        tokenizer=None,
        config: Optional[YamatoConfig] = None,
    ):
        super().__init__()
        self.config = config or YamatoConfig()
        self.tokenizer = tokenizer

        # Qwen backbone（外部から注入 or 後で設定）
        self.backbone = backbone

        # カスタムヘッド（後から attach_custom_heads で設定可能）
        d_model = self.config.d_model
        self.custom_heads = None
        self.sanitizer = None

    def init_custom_heads(self):
        """
        カスタムヘッドの初期化

        国譲り（Stage 1）で呼ばれる。
        ランダム初期化された yamatoLLM 固有のヘッドを追加。
        backbone と同じデバイス・compute dtype に揃える。
        """
        self.custom_heads, self.sanitizer = QwenAdapter.attach_custom_heads(
            model=self.backbone,
            config=self.config,
        )

        backbone_param = next(self.backbone.parameters())
        device = backbone_param.device
        # bnb 4bit/8bit のとき param.dtype は uint8 — compute dtype (bfloat16) に揃える
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
        """
        Qwen backbone から hidden states を取得

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            hidden_states: [batch, seq_len, d_model]
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # 最終層の hidden states
        return outputs.hidden_states[-1]

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        route_labels=None,
        confidence_labels=None,
        **kwargs,
    ):
        """
        統合 forward pass（学習時）

        Returns:
            dict with:
                loss: 統合損失
                logits: Qwen の next-token logits
                route_logits: 意図分類 logits
                confidence: 信頼度スコア
                hidden_states: 最終層 hidden states
        """
        # 1. Qwen backbone forward
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

        result = {
            "logits": logits,
            "hidden_states": hidden_states,
            "base_loss": base_loss,
        }

        # 2. カスタムヘッドが初期化されている場合
        if self.custom_heads is not None:
            # 意図分類
            intent_out = self.custom_heads["intent_router"](
                hidden_states, attention_mask
            )
            result["route_logits"] = intent_out["route_logits"]
            result["route"] = intent_out["route"]
            result["intent_vector"] = intent_out["intent_vector"]

            # 信頼度
            conf_out = self.custom_heads["confidence"](hidden_states)
            result["confidence"] = conf_out["confidence"]
            result["uncertainty_flag"] = conf_out["uncertainty_flag"]
            result["truthfulness"] = conf_out["truthfulness"]

            # 損失計算
            loss = base_loss

            if route_labels is not None:
                route_loss = F.cross_entropy(
                    intent_out["route_logits"],
                    route_labels,
                )
                loss = loss + 0.5 * route_loss
                result["route_loss"] = route_loss

            if confidence_labels is not None:
                conf_loss = F.mse_loss(
                    conf_out["confidence"].squeeze(-1),  # [batch,1] → [batch]
                    confidence_labels.to(conf_out["confidence"].dtype),
                )
                loss = loss + 0.3 * conf_loss
                result["confidence_loss"] = conf_loss

            result["loss"] = loss
        else:
            result["loss"] = base_loss

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        route: Optional[str] = None,
        **kwargs,
    ) -> YamatoOutput:
        """
        推論時のエントリポイント

        1. 忌部: 入力浄化
        2. Qwen backbone: hidden states 取得
        3. 思兼神: ルーティング判断（route 未指定時）
        4. ルートに応じた処理分岐
        5. 凡夫の自覚: 信頼度付与
        6. 忌部: 出力浄化

        Args:
            prompt: ユーザー入力テキスト
            max_new_tokens: 最大生成トークン数
            temperature: 生成 temperature
            top_p: nucleus sampling
            route: 強制ルート指定（None で自動判定）

        Returns:
            YamatoOutput
        """
        inference_config = self.config.inference
        max_new_tokens = max_new_tokens or inference_config.max_new_tokens
        temperature = temperature or inference_config.temperature
        top_p = top_p or inference_config.top_p

        output = YamatoOutput()

        # 1. 忌部: 入力浄化
        sanitized_prompt = prompt
        if self.sanitizer is not None:
            sanitize_result = self.sanitizer.sanitize_input(prompt)
            sanitized_prompt = sanitize_result["text"]
            output.safety_score = sanitize_result.get("safety_score", 1.0)

            if sanitize_result.get("blocked", False):
                output.text = sanitize_result.get(
                    "block_message",
                    "安全上の理由により、この入力には応答できません。"
                )
                output.verdict = "HALT"
                return output

        # 2. トークナイズ
        inputs = self.tokenizer(
            sanitized_prompt,
            return_tensors="pt",
            padding=True,
        )
        device = next(self.backbone.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # 3. ルーティング判断
        if route is None and self.custom_heads is not None:
            hidden_states = self.get_hidden_states(input_ids, attention_mask)
            intent_out = self.custom_heads["intent_router"](
                hidden_states, attention_mask
            )
            route_idx = intent_out["route"].item()
            output.route = self.ROUTE_NAMES[route_idx]
            output.route_logits = intent_out["route_logits"]
            output.intent_vector = intent_out["intent_vector"]
        else:
            output.route = route or "chat"

        # 4. ルートに応じた生成
        if output.route == "codegen" and inference_config.enable_staged_generation:
            # コード生成: 天の御柱プロトコル（3段階生成）
            output = self._generate_code(
                input_ids, attention_mask, output,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            # chat / retrieval: 通常生成
            generated_ids = self.backbone.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=inference_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # 入力部分を除いた生成テキスト
            new_ids = generated_ids[0, input_ids.shape[1]:]
            output.text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            output.generated_ids = generated_ids

        # 5. 凡夫の自覚: 信頼度付与
        if self.custom_heads is not None:
            # 生成後の hidden states で信頼度を計算
            if output.generated_ids is not None:
                gen_hidden = self.get_hidden_states(output.generated_ids)
                conf_out = self.custom_heads["confidence"](gen_hidden)
                output.confidence = conf_out["confidence"].item()
                output.uncertainty_flag = conf_out["uncertainty_flag"].item()
                output.truthfulness = conf_out["truthfulness"].item()

        # 6. 忌部: 出力浄化
        if self.sanitizer is not None:
            sanitize_result = self.sanitizer.sanitize_output(output.text)
            output.text = sanitize_result["text"]

        return output

    def _generate_code(
        self,
        input_ids,
        attention_mask,
        output: YamatoOutput,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
    ) -> YamatoOutput:
        """
        天の御柱プロトコル: 3段階コード生成

        Phase 0 (IZANAGI): struct / type 定義
        Phase 1 (IZANAMI): function シグネチャ
        Phase 2 (KAMIYUMI): function 実装

        codegen ルートでは temperature を低め、top_p を高めに設定。
        """
        phases = {}
        phase_names = ["IZANAGI", "IZANAMI", "KAMIYUMI"]
        phase_prompts = [
            "\n# Phase 1: 型定義のみを生成してください。\n",
            "\n# Phase 2: 上記の型に対する関数シグネチャを生成してください。\n",
            "\n# Phase 3: 関数の実装を完成させてください。\n",
        ]

        accumulated_code = ""

        for i, (phase_name, phase_prompt) in enumerate(
            zip(phase_names, phase_prompts)
        ):
            # 各 Phase でコンテキストを積み上げ
            full_prompt = accumulated_code + phase_prompt
            phase_inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
            )
            device = next(self.backbone.parameters()).device
            phase_ids = phase_inputs["input_ids"].to(device)
            phase_mask = phase_inputs["attention_mask"].to(device)

            generated_ids = self.backbone.generate(
                input_ids=phase_ids,
                attention_mask=phase_mask,
                max_new_tokens=max_new_tokens // 3,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            new_ids = generated_ids[0, phase_ids.shape[1]:]
            phase_code = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            phases[phase_name] = phase_code
            accumulated_code += phase_code + "\n"

        output.phases = phases
        output.code = accumulated_code.strip()
        output.text = output.code
        output.generated_ids = None  # Phase別生成なので単一IDなし
        return output

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        quantize: Optional[str] = None,
        device_map: str = "auto",
    ) -> "YamatoLLM":
        """
        チェックポイントから yamatoLLM を復元

        Args:
            checkpoint_path: チェックポイントパス
            quantize: 量子化設定
            device_map: デバイス配置

        Returns:
            YamatoLLM インスタンス
        """
        backbone, tokenizer, config, metadata = QwenAdapter.load_checkpoint(
            checkpoint_path=checkpoint_path,
            quantize=quantize,
            device_map=device_map,
        )

        model = cls(
            backbone=backbone,
            tokenizer=tokenizer,
            config=config,
        )
        model.init_custom_heads()

        logger.info(
            f"YamatoLLM loaded: stage={metadata['stage']}, "
            f"route_names={config.iwato.route_names}"
        )

        return model

    @classmethod
    def from_qwen(
        cls,
        model_name: str = "llm-jp/llm-jp-4-8b-base",
        quantize: Optional[str] = None,
        config: Optional[YamatoConfig] = None,
    ) -> "YamatoLLM":
        """
        Qwen から新規に yamatoLLM を構築（国譲り）

        Args:
            model_name: Qwen モデル名
            quantize: 量子化設定
            config: yamatoLLM 設定

        Returns:
            YamatoLLM インスタンス
        """
        config = config or YamatoConfig()
        config.backbone.model_name = model_name

        backbone, tokenizer = QwenAdapter.load_base_model(
            model_name=model_name,
            quantize=quantize,
        )

        model = cls(
            backbone=backbone,
            tokenizer=tokenizer,
            config=config,
        )
        model.init_custom_heads()
        model.config.stage = "kuniyuzuri"

        logger.info("YamatoLLM created via 国譲り (Kuniyuzuri)")

        return model
