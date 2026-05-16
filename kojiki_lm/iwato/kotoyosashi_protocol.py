"""
言依さし (Kotoyosashi) — コード生成層との接続プロトコル
kotoyosashi_protocol.py

古事記において、天照大御神が孫の邇邇芸命に「言依さし」を行い、
葦原中国の統治を委任した。これは神意を具体的な行動指示に変換する行為である。

本モジュールはその「言依さし」に対応する。
言語処理層（岩戸隠れ）で解析された自然言語の意図を、
コード生成層（天の御柱プロトコル）が理解できる構造化された
神託フォーマット (OracleFormat) に変換する。

機能:
    - OracleFormat: コード生成層への構造化入力 (dataclass)
    - KotoyosashiProtocol: 意図ベクトルから OracleFormat を生成し、
      コード生成層が処理可能なプロンプトに変換する

入出力:
    Input:  intent_vector [batch, d_model] — 思兼神が出力した意図ベクトル
            hidden_states [batch, seq_len, d_model] — 最終隠れ状態
    Output: OracleFormat — 構造化された神託フォーマット
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Julia 共通型定義
COMMON_JULIA_TYPES: List[str] = [
    "Int64",
    "Float64",
    "Bool",
    "String",
    "Vector",
    "Matrix",
    "Dict",
    "Nothing",
    "Any",
    "Tuple",
    "Complex",
    "Rational",
    "BigInt",
    "BigFloat",
    "UInt8",
    "AbstractArray",
]

# 天の御柱プロトコル フェーズ名
PHASE_NAMES: List[str] = ["IZANAGI", "IZANAMI", "KAMIYUMI"]


@dataclass
class OracleFormat:
    """神託フォーマット — 天の御柱プロトコルへの入力

    コード生成層が期待する構造化データ。
    言語処理層の解析結果を、生成フェーズの制御情報に変換する。

    Attributes:
        task: タスク種別 ("codegen", "debug", "refactor", "explain")
        phase_hint: 開始フェーズ (0=IZANAGI, 1=IZANAMI, 2=KAMIYUMI)
        struct_hints: 構造体定義のヒント (e.g. ["Point2D(x, y)", "Line(p1, p2)"])
        type_constraints: 型制約のヒント (e.g. ["Float64", "Int64"])
        function_hints: 関数名のヒント (e.g. ["distance", "intersect"])
        hint_embedding: ヒント埋め込みベクトル [d_model]
    """

    task: str = "codegen"
    phase_hint: int = 0
    struct_hints: List[str] = field(default_factory=list)
    type_constraints: List[str] = field(default_factory=list)
    function_hints: List[str] = field(default_factory=list)
    hint_embedding: Optional[Any] = None  # [d_model] tensor

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換する (シリアライズ用)"""
        return {
            "task": self.task,
            "phase_hint": self.phase_hint,
            "phase_name": PHASE_NAMES[self.phase_hint]
            if 0 <= self.phase_hint < len(PHASE_NAMES)
            else "UNKNOWN",
            "struct_hints": self.struct_hints,
            "type_constraints": self.type_constraints,
            "function_hints": self.function_hints,
        }


class KotoyosashiProtocol(nn.Module):
    """言依さし — 言語処理層 → コード生成層の変換

    意図ベクトルと隠れ状態から、コード生成層が処理可能な
    OracleFormat を生成する。

    Args:
        d_model: 隠れ層の次元数 (Qwen3.5-9B hidden_size = 3584)
        max_hints: ヒントとして抽出する最大数
        num_julia_types: Julia 共通型の数
    """

    def __init__(
        self,
        d_model: int = 3584,
        max_hints: int = 8,
        num_julia_types: int = len(COMMON_JULIA_TYPES),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_hints = max_hints

        # フェーズ分類器: 3フェーズ (IZANAGI / IZANAMI / KAMIYUMI)
        self.phase_classifier = nn.Linear(d_model, 3)

        # ヒント埋め込み生成器
        self.hint_generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 構造体検出器: struct 定義が必要か (binary)
        self.struct_detector = nn.Linear(d_model, 1)

        # 型制約予測器: Julia 共通型の使用確率
        self.type_detector = nn.Linear(d_model, num_julia_types)

        # タスク分類器: codegen / debug / refactor / explain
        self.task_classifier = nn.Linear(d_model, 4)
        self.task_names = ["codegen", "debug", "refactor", "explain"]

    def forward(
        self,
        intent_vector: Tensor,
        hidden_states: Tensor,
    ) -> OracleFormat:
        """神託フォーマットを生成する

        Args:
            intent_vector: [batch, d_model] 思兼神が出力した意図ベクトル
            hidden_states: [batch, seq_len, d_model] 最終隠れ状態

        Returns:
            OracleFormat: 構造化された神託フォーマット
                (バッチの先頭要素を使用。推論時は batch=1 を想定)
        """
        # 代表ベクトル: intent_vector を使用 (batch の先頭)
        vec = intent_vector[0]  # [d_model]

        # 1. タスク分類
        task_logits = self.task_classifier(vec)  # [4]
        task_idx = task_logits.argmax(dim=-1).item()
        task = self.task_names[task_idx]

        # 2. 開始フェーズ分類
        phase_logits = self.phase_classifier(vec)  # [3]
        phase_hint = phase_logits.argmax(dim=-1).item()

        # 3. ヒント埋め込み生成
        hint_embedding = self.hint_generator(vec)  # [d_model]

        # 4. 構造体定義の必要性判定
        struct_score = torch.sigmoid(self.struct_detector(vec))  # [1]
        needs_struct = struct_score.item() > 0.5
        struct_hints: List[str] = []
        if needs_struct:
            struct_hints = ["struct"]  # 具体名は上位層で解決

        # 5. 型制約予測
        type_logits = self.type_detector(vec)  # [num_julia_types]
        type_probs = torch.sigmoid(type_logits)
        type_constraints: List[str] = []
        for i, prob in enumerate(type_probs.tolist()):
            if prob > 0.5 and i < len(COMMON_JULIA_TYPES):
                type_constraints.append(COMMON_JULIA_TYPES[i])

        return OracleFormat(
            task=task,
            phase_hint=phase_hint,
            struct_hints=struct_hints,
            type_constraints=type_constraints,
            function_hints=[],  # 関数名ヒントは上位層で付与
            hint_embedding=hint_embedding.detach(),
        )

    def to_prompt(self, oracle: OracleFormat) -> str:
        """OracleFormat を自然言語プロンプトに変換する

        コード生成層が直接利用可能なプロンプト文字列を生成する。

        Args:
            oracle: 神託フォーマット

        Returns:
            prompt: コード生成向けプロンプト文字列
        """
        lines: List[str] = []

        # タスクヘッダ
        phase_name = (
            PHASE_NAMES[oracle.phase_hint]
            if 0 <= oracle.phase_hint < len(PHASE_NAMES)
            else "UNKNOWN"
        )
        lines.append(f"# Task: {oracle.task} (Phase: {phase_name})")

        # 型制約
        if oracle.type_constraints:
            types_str = ", ".join(oracle.type_constraints)
            lines.append(f"# 型定義: {types_str}")

        # 構造体ヒント
        if oracle.struct_hints:
            for hint in oracle.struct_hints:
                lines.append(f"# struct: {hint}")

        # 関数ヒント
        if oracle.function_hints:
            funcs_str = ", ".join(oracle.function_hints)
            lines.append(f"# 関数: {funcs_str}")

        lines.append("")  # 空行で区切り
        return "\n".join(lines)
