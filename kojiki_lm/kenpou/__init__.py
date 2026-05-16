"""
ガバナンス層 — 憲法十七条（律令層）

聖徳太子の憲法十七条に基づくガバナンス・オーバーレイ。
既存のモデル内部層を変更せず、上から被せる律令層として機能する。

モジュール:
    kenpou_config      — 17条パラメータ設定
    wa_loss            — L1: 和の損失関数（第1条, 第14条）
    shotoku_consensus  — L3: 聖徳コンセンサス（第17条, 第4条）
    bonpu_confidence   — L4: 凡夫の自覚（第10条, 第9条）
    toki_scheduler     — L5: 時のスケジューラ（第16条, 第15条）
    忌部プロトコル     — L2: 入出力浄化（iwato/inbe_sanitizer.py に配置）
"""

from .kenpou_config import KenpouConfig, DEFAULT_KENPOU_CONFIG
from .wa_loss import WaLoss
from .shotoku_consensus import ShotokuConsensus
from .bonpu_confidence import BonpuConfidence
from .toki_scheduler import TokiScheduler

__all__ = [
    "KenpouConfig",
    "DEFAULT_KENPOU_CONFIG",
    "WaLoss",
    "ShotokuConsensus",
    "BonpuConfidence",
    "TokiScheduler",
]
