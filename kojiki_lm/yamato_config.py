"""
yamatoLLM 統合設定

3層 + Qwen backbone の全設定を統合管理する。
各層の個別設定（KojikiConfig, KenpouConfig）をまとめ、
Qwen3.5-9B 固有のパラメータを追加する。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QwenBackboneConfig:
    """
    Qwen3.5-9B のアーキテクチャパラメータ

    これらは Qwen のモデル仕様に固定され、変更不可。
    """
    model_name: str = "Qwen/Qwen3.5-9B"
    hidden_size: int = 3584
    num_layers: int = 40
    num_attention_heads: int = 28
    num_kv_heads: int = 4          # GQA
    intermediate_size: int = 18944  # SwiGLU
    vocab_size: int = 151936
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768


@dataclass
class LoRAConfig:
    """
    LoRA アダプタ設定

    学習ステージごとに調整可能。
    """
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "gate_proj",
    ])
    modules_to_save: List[str] = field(default_factory=lambda: [
        "intent_router",
        "type_head",
        "error_head",
        "confidence",
        "kotoyosashi",
    ])


@dataclass
class IwatoConfig:
    """
    言語処理層（岩戸隠れ）の設定
    """
    # 思兼神（意図分類）
    num_routes: int = 3                # chat / codegen / retrieval
    route_names: List[str] = field(default_factory=lambda: [
        "chat", "codegen", "retrieval",
    ])

    # 布刀玉命（RAG）
    retriever_top_k: int = 5           # RAG 検索上位件数
    cross_attention_heads: int = 4     # Cross-Attention ヘッド数

    # 天宇受売命（生成）
    manyo_filter_enabled: bool = True  # 万葉フィルタ（トーン制御）

    # 天手力男神（出力確定）
    shimenawa_max_tokens: int = 2048   # 注連縄: 最大出力トークン数
    shimenawa_repeat_penalty: float = 1.2  # 繰り返しペナルティ

    # 忌部（入出力浄化）
    safety_threshold: float = 0.7      # 安全性スコア閾値


@dataclass
class InferenceConfig:
    """
    推論設定（RTX 3060 対応）
    """
    quantize: Optional[str] = "4bit"   # None / "4bit" / "8bit"
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    # 天の御柱プロトコル
    enable_staged_generation: bool = True
    repair_budget: int = 4             # Self-Repair 最大リトライ


@dataclass
class YamatoConfig:
    """
    yamatoLLM 統合設定

    全層の設定を1つにまとめるエントリポイント。
    """
    # Qwen backbone
    backbone: QwenBackboneConfig = field(default_factory=QwenBackboneConfig)

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # 言語処理層
    iwato: IwatoConfig = field(default_factory=IwatoConfig)

    # 推論
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # 4軸評価 Quality Gate
    v_threshold: float = 0.7           # COMMIT 閾値
    stability_floor: float = 0.3       # stability 最低ライン
    safety_floor: float = 0.5          # boundary 最低ライン

    # 学習ステージ名
    stage: str = "kuniyuzuri"          # 現在のステージ

    @property
    def d_model(self) -> int:
        """Qwen の hidden_size を返す（各層で共通）"""
        return self.backbone.hidden_size


# デフォルト設定インスタンス
DEFAULT_YAMATO_CONFIG = YamatoConfig()
