"""
憲法十七条 設定 (Kenpou Configuration)

聖徳太子の憲法十七条の各条文をモデルパラメータとして定量化する。
17条すべてを5つのガバナンス層（L1〜L5）にマッピングし、
各パラメータのデフォルト値を提供する。

参照:
    第1条  「和を以て貴しとなす」          → harmony_lambda
    第2条  「篤く三宝を敬え」              → (reserved for future: spiritual grounding)
    第3条  「詔を承りては必ず謹め」        → (reserved: instruction following)
    第4条  「礼を以て本とせよ」            → (consensus に統合)
    第5条  「饗を絶ち欲を棄てて…」        → (reserved: resource frugality)
    第6条  「悪を懲らし善を勧むる…」      → flattery/deception thresholds
    第7条  「人各任有り」                  → (MoE expert specialization)
    第8条  「群卿百寮、早朝晏退」          → (scheduling に統合)
    第9条  「信は是義の本なり」            → truthfulness_tau
    第10条 「我必ずしも聖に非ず…凡夫」    → confidence floor/ceiling, noise damping
    第11条 「功過を明らかに察して…」      → (reserved: reward shaping)
    第12条 「国司国造、百姓に斂めること…」→ (reserved: resource allocation)
    第13条 「諸の官に任ずる者…」          → (reserved: role assignment)
    第14条 「群臣百寮、嫉み妬む…」        → regularization_mu (anti-overfitting)
    第15条 「私を背きて公に向くは…」      → public_private_filter
    第16条 「民を使うに時を以てする…」    → scheduling parameters
    第17条 「独り断ずべからず…」          → consensus parameters
"""

from dataclasses import dataclass


@dataclass
class KenpouConfig:
    """
    憲法十七条ガバナンスパラメータ

    全17条をL1〜L5の5層にマッピングする設定データクラス。
    各フィールド名は対応する条文の精神を反映する。
    """

    # ================================================================
    # L1: 第1条 — 和 (Harmony) / 第14条 — 正規化
    # ================================================================
    harmony_lambda: float = 0.1       # 和の損失項の重み (conflict term weight)
    regularization_mu: float = 0.01   # 第14条: 嫉妬＝過学習を防ぐ正規化係数

    # ================================================================
    # L2: 第10条前半 — 感情ノイズ減衰 (Emotional Noise Damping)
    #     第6条 — 諂い・詐り検出 (Flattery/Deception Detection)
    # ================================================================
    noise_alpha: float = 0.3          # 怒りノイズ減衰率 (anger)
    noise_beta: float = 0.2           # 嫉妬ノイズ減衰率 (jealousy)
    noise_gamma: float = 0.2          # 恨みノイズ減衰率 (resentment)
    flattery_threshold: float = 0.7   # 第6条: 諂い検出閾値
    deception_threshold: float = 0.5  # 第6条: 詐り検出閾値

    # ================================================================
    # L3: 第17条 — コンセンサス (Consensus)
    # ================================================================
    importance_threshold: float = 0.5  # 少事/大事の分岐閾値
    consensus_method: str = "weighted_vote"  # 合議方式
    max_consensus_rounds: int = 3      # コンセンサス最大ラウンド数

    # ================================================================
    # L4: 第10条後半 — 信頼度 (Confidence / Bonpu Self-Awareness)
    #     第9条 — 信 (Truthfulness)
    # ================================================================
    confidence_floor: float = 0.1      # 最低信頼度 (愚に非ず)
    confidence_ceiling: float = 1.0 - 1e-4  # 最高信頼度 (聖に非ず)
    truthfulness_tau: float = 0.6      # 真実性閾値 (不確実性フラグ)
    uncertainty_expression: bool = True  # 不確実性を表明するか

    # ================================================================
    # L5: 第16条 — スケジューリング (Timing / Toki)
    #     第15条 — 公私の分別
    # ================================================================
    context_sensing_enabled: bool = True   # コンテキスト感知の有効化
    public_private_filter: bool = True     # 第15条: 公私フィルタ
    load_defer_threshold: float = 0.8      # 負荷遅延閾値

    # ================================================================
    # モデル次元 (共通)
    # ================================================================
    d_model: int = 4096                    # llm-jp-4-8b hidden_size

    def validate(self) -> None:
        """設定値の妥当性を検証する。"""
        assert 0.0 <= self.harmony_lambda <= 1.0, \
            f"harmony_lambda must be in [0, 1], got {self.harmony_lambda}"
        assert 0.0 <= self.importance_threshold <= 1.0, \
            f"importance_threshold must be in [0, 1], got {self.importance_threshold}"
        assert 0.0 < self.confidence_floor < self.confidence_ceiling < 1.0, \
            "confidence_floor < confidence_ceiling < 1.0 must hold"
        assert 0.0 < self.truthfulness_tau < 1.0, \
            f"truthfulness_tau must be in (0, 1), got {self.truthfulness_tau}"
        assert 0.0 < self.load_defer_threshold <= 1.0, \
            f"load_defer_threshold must be in (0, 1], got {self.load_defer_threshold}"
        assert self.max_consensus_rounds >= 1, \
            f"max_consensus_rounds must be >= 1, got {self.max_consensus_rounds}"


DEFAULT_KENPOU_CONFIG = KenpouConfig()
