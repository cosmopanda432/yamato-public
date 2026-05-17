"""
ヒルコ検知 (Hiruko Detector) — 型不安定な生成を検知する。

Julia版の YomiLayer / HirukoValidator に対応するルールベース層。
学習はせず、TsukuyomiTypeHead の `type_preds` を集計するのみ。

判定基準（`ts_type_vocab.json` から ID を解決）:
    - ImplicitAny 率  > implicit_any_threshold   (default 0.30)
    - ExplicitAny 率  > explicit_any_threshold   (default 0.20)
    - ErrorType   率  > error_type_threshold     (default 0.30)
    - unknown     率  > unknown_threshold        (default 0.30, "型推論失敗" 寄り)

AmenomihashiraProtocol が Phase 1 (型定義) の出力をこれにかけ、
失敗時に温度を上げてリトライする。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Union

import torch


@dataclass
class HirukoResult:
    """ヒルコ検知の結果"""
    is_malformed: bool
    reason: str  # 失敗時の主因 ("implicit_any" / "explicit_any" / "error_type" / "unknown" / "")
    implicit_any_rate: float
    explicit_any_rate: float
    error_type_rate: float
    unknown_rate: float
    total_tokens: int
    instability_tokens: int


class HirukoDetector:
    """型不安定な生成を検知するルールベース層"""

    def __init__(
        self,
        vocab_path: Union[str, Path] = "config/ts_type_vocab.json",
        implicit_any_threshold: float = 0.30,
        explicit_any_threshold: float = 0.20,
        error_type_threshold: float = 0.30,
        unknown_threshold: float = 0.30,
        ignore_id: Optional[int] = None,
    ):
        with open(vocab_path) as f:
            vocab = json.load(f)
        type_to_id = vocab["type_to_id"]

        # 必須マーカー
        for name in ["ImplicitAny", "ExplicitAny", "ErrorType"]:
            if name not in type_to_id:
                raise KeyError(f"{name} missing from {vocab_path}")

        self.implicit_any_id: int = type_to_id["ImplicitAny"]
        self.explicit_any_id: int = type_to_id["ExplicitAny"]
        self.error_type_id: int = type_to_id["ErrorType"]
        # unknown は ManyTypes4TS に存在せず force_include で予約された ID
        self.unknown_id: Optional[int] = type_to_id.get("unknown")
        # any も別途追跡 (推論時の "明示的 any" として ExplicitAny と合算可)
        self.any_id: Optional[int] = type_to_id.get("any")

        self.implicit_any_threshold = implicit_any_threshold
        self.explicit_any_threshold = explicit_any_threshold
        self.error_type_threshold = error_type_threshold
        self.unknown_threshold = unknown_threshold

        # UNK (id=0) など、計算から除外したい ID
        self.ignore_ids: Set[int] = set()
        if ignore_id is not None:
            self.ignore_ids.add(ignore_id)

    def detect(
        self,
        type_preds: Union[torch.Tensor, Iterable[int], List[int]],
    ) -> HirukoResult:
        """
        Args:
            type_preds: [L] か [B, L] か iterable of ints

        Returns:
            HirukoResult。複数バッチの場合は flatten してまとめて集計する。
        """
        if isinstance(type_preds, torch.Tensor):
            ids = type_preds.detach().reshape(-1).cpu().tolist()
        else:
            ids = list(type_preds)

        ids = [i for i in ids if i not in self.ignore_ids]
        total = len(ids)
        if total == 0:
            return HirukoResult(
                is_malformed=False, reason="",
                implicit_any_rate=0.0, explicit_any_rate=0.0,
                error_type_rate=0.0, unknown_rate=0.0,
                total_tokens=0, instability_tokens=0,
            )

        n_implicit = sum(1 for i in ids if i == self.implicit_any_id)
        # explicit_any は ExplicitAny 単体、または `any` ID 由来も合算可
        explicit_targets = {self.explicit_any_id}
        if self.any_id is not None:
            explicit_targets.add(self.any_id)
        n_explicit = sum(1 for i in ids if i in explicit_targets)
        n_error = sum(1 for i in ids if i == self.error_type_id)
        n_unknown = (
            sum(1 for i in ids if i == self.unknown_id)
            if self.unknown_id is not None else 0
        )

        rates = {
            "implicit_any": n_implicit / total,
            "explicit_any": n_explicit / total,
            "error_type": n_error / total,
            "unknown": n_unknown / total,
        }
        thresholds = {
            "implicit_any": self.implicit_any_threshold,
            "explicit_any": self.explicit_any_threshold,
            "error_type": self.error_type_threshold,
            "unknown": self.unknown_threshold,
        }

        # 主因を「閾値超過幅が最大」のもので決定
        breaches = {
            k: rates[k] - thresholds[k]
            for k in rates if rates[k] > thresholds[k]
        }
        if breaches:
            reason = max(breaches, key=breaches.get)
            is_malformed = True
        else:
            reason = ""
            is_malformed = False

        instability = n_implicit + n_explicit + n_error + n_unknown

        return HirukoResult(
            is_malformed=is_malformed,
            reason=reason,
            implicit_any_rate=rates["implicit_any"],
            explicit_any_rate=rates["explicit_any"],
            error_type_rate=rates["error_type"],
            unknown_rate=rates["unknown"],
            total_tokens=total,
            instability_tokens=instability,
        )
