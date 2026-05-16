"""
yamatoLLM 統合設定

Qwen2.5-Coder-7B-Instruct backbone に、TypeScript 型予測のカスタムヘッド
（TsukuyomiTypeHead, HirukoDetector, BonpuConfidence など）を載せる
構成の設定を統合管理する。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QwenBackboneConfig:
    """
    Backbone のアーキテクチャパラメータ（Qwen2.5-Coder-7B-Instruct）

    HuggingFace Qwen/Qwen2.5-Coder-7B-Instruct の config.json に対応する。
    Architecture: Qwen2ForCausalLM（GQA / SwiGLU）。
    """
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    local_path: Optional[str] = "models/Qwen2.5-Coder-7B-Instruct"
    hidden_size: int = 3584
    num_layers: int = 28
    num_attention_heads: int = 28
    num_kv_heads: int = 4              # GQA
    intermediate_size: int = 18944     # SwiGLU (silu)
    vocab_size: int = 152064
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768


@dataclass
class LoRAConfig:
    """
    LoRA アダプタ設定

    backbone (Qwen2) は LoRA で適応、追加カスタムヘッドはフル学習。
    """
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "gate_proj",
    ])
    modules_to_save: List[str] = field(default_factory=lambda: [
        "confidence",
        # 実装後に "type_head", "hiruko_detector" を追加する
    ])


@dataclass
class TypeHeadConfig:
    """
    TsukuyomiTypeHead（per-token TypeScript 型予測ヘッド）設定

    実装は後続フェーズ。ここでは語彙サイズと損失重みのみ宣言。
    """
    vocab_path: str = "config/ts_type_vocab.json"
    vocab_size: int = 256              # 200-400 想定、暫定値
    loss_weight: float = 0.3           # SFT 損失に加算する重み
    hidden_dim: int = 512              # 中間層次元


@dataclass
class InferenceConfig:
    """推論設定（RTX 3060 12GB で INT4 ロードを想定）"""
    quantize: Optional[str] = "4bit"   # None / "4bit" / "8bit"
    max_new_tokens: int = 1024
    temperature: float = 0.3           # コード生成向けに低め
    top_p: float = 0.95
    do_sample: bool = True


@dataclass
class YamatoConfig:
    """yamatoLLM 統合設定エントリポイント"""
    backbone: QwenBackboneConfig = field(default_factory=QwenBackboneConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    type_head: TypeHeadConfig = field(default_factory=TypeHeadConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # ステージ名: baseline / sft / dpo
    stage: str = "baseline"

    @property
    def d_model(self) -> int:
        """backbone の hidden_size を返す（カスタムヘッドで共通利用）"""
        return self.backbone.hidden_size


DEFAULT_YAMATO_CONFIG = YamatoConfig()
