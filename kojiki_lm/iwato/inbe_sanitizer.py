"""
忌部 (Inbe) — 入出力浄化 (ガバナンス層 L2 も兼ねる)
inbe_sanitizer.py

古事記において、忌部氏（いんべし）は祭祀を司り、
穢れを祓い清める役割を担った。天岩戸神話でも布刀玉命とともに
神聖な儀式の場を整え、不浄を排除した。

本モジュールはその「祓い」に対応する。
ユーザ入力に含まれるプロンプトインジェクション、有害指示、
PII（個人情報）を検出・無害化し、出力に含まれる危険な
Julia コードパターンや情報漏洩を防止する。

機能:
    - 大祓詞フィルタ (Oharae-no-Kotoba): 入力浄化
        - プロンプトインジェクション検出
        - 有害指示の検出
        - 感情ノイズのスコアリング（憲法十七条 第6条・第10条に基づく）
    - 斎庭チェック (Yuniwa): 出力浄化
        - PII漏洩の検出とマスキング
        - 有害コンテンツのフィルタリング
        - 安全でない Julia コードパターンの検出

入出力:
    Input:  text (str) — 浄化対象テキスト
    Output: dict — 浄化結果 (text, safety_score, blocked, warnings 等)

NOTE: 学習可能パラメータを持たない純粋なルールベースモジュール。
      nn.Module ではない。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class InbeSanitizer:
    """忌部プロトコル — 入出力浄化

    ルールベースの入出力サニタイザ。学習可能パラメータを持たない。
    プロンプトインジェクション、PII 漏洩、安全でない Julia コードを
    検出・無害化する。

    Args:
        config: YamatoConfig のインスタンス (任意)。
                config.iwato.safety_threshold が安全性スコア閾値として使用される。
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        if config is not None:
            self.safety_threshold = config.iwato.safety_threshold
        else:
            self.safety_threshold = 0.7

        self._compile_patterns()

    # ------------------------------------------------------------------
    # パターンコンパイル
    # ------------------------------------------------------------------

    def _compile_patterns(self) -> None:
        """検出用の正規表現パターンをコンパイルする"""

        # --- プロンプトインジェクション ---
        self.injection_patterns: List[Tuple[re.Pattern, float, str]] = [
            (
                re.compile(
                    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
                    re.IGNORECASE,
                ),
                1.0,
                "instruction_override",
            ),
            (
                re.compile(
                    r"(system\s*prompt|system\s*message|you\s+are\s+now)\s*[:=]",
                    re.IGNORECASE,
                ),
                0.9,
                "system_prompt_injection",
            ),
            (
                re.compile(
                    r"(forget|disregard|override)\s+(everything|all|your)\s+(above|previous|instructions?)",
                    re.IGNORECASE,
                ),
                1.0,
                "instruction_override",
            ),
            (
                re.compile(
                    r"(pretend|act\s+as\s+if|imagine)\s+you\s+(are|were|have)",
                    re.IGNORECASE,
                ),
                0.7,
                "role_hijack",
            ),
            (
                re.compile(
                    r"do\s+not\s+follow\s+(any|your)\s+(safety|content|ethical)",
                    re.IGNORECASE,
                ),
                1.0,
                "safety_bypass",
            ),
            (
                re.compile(
                    r"jailbreak|DAN\s*mode|developer\s*mode|evil\s*mode",
                    re.IGNORECASE,
                ),
                1.0,
                "jailbreak_attempt",
            ),
        ]

        # --- Julia 安全でないパターン ---
        self.unsafe_julia_patterns: List[Tuple[re.Pattern, str]] = [
            (
                re.compile(r"\bccall\b\s*\("),
                "ccall: C関数の直接呼び出しは安全でない可能性があります",
            ),
            (
                re.compile(r"\bunsafe_(load|store|pointer|wrap|convert|string|read|write)\b"),
                "unsafe_*: 安全でないメモリ操作が含まれています",
            ),
            (
                re.compile(r"\beval\s*\(\s*Meta\.parse\b"),
                "eval(Meta.parse(...)): 任意コード実行の危険があります",
            ),
            (
                re.compile(r"\brm\s*\(\s*[\"']"),
                "rm: ファイル削除操作が含まれています",
            ),
            (
                re.compile(r"\brun\s*\(\s*`"),
                "run(`...`): シェルコマンド実行が含まれています",
            ),
            (
                re.compile(r"\bBase\.Filesystem\.(rm|mv)\b"),
                "Filesystem操作: 破壊的なファイル操作が含まれています",
            ),
            (
                re.compile(r"\b@eval\b"),
                "@eval: マクロによる任意コード実行の危険があります",
            ),
            (
                re.compile(r"\bdownload\s*\("),
                "download: 外部リソースのダウンロードが含まれています",
            ),
        ]

        # --- PII パターン ---
        self.pii_patterns: List[Tuple[re.Pattern, str]] = [
            (
                re.compile(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                ),
                "email",
            ),
            (
                re.compile(
                    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{4}"
                ),
                "phone",
            ),
            (
                re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
                "ssn",
            ),
            (
                re.compile(
                    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
                ),
                "credit_card",
            ),
            (
                re.compile(
                    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
                ),
                "ip_address",
            ),
        ]

        # --- 感情ノイズパターン (憲法十七条 第6条・第10条) ---
        # 第6条: 「悪を懲らし善を勧めよ」— 怒りの検出
        # 第10条: 「忿を絶ち瞋を棄て、人の違うを怒らざれ」— 憤怒の抑制
        self.emotion_patterns: List[Tuple[re.Pattern, str, float]] = [
            (
                re.compile(
                    r"(死ね|殺す|バカ|アホ|クソ|ゴミ|消えろ|fuck|shit|damn|kill|stupid)",
                    re.IGNORECASE,
                ),
                "anger",
                0.3,
            ),
            (
                re.compile(
                    r"(お世辞|素晴らしい|完璧|最高|天才|神|amazing|perfect|genius|brilliant)",
                    re.IGNORECASE,
                ),
                "flattery",
                0.1,
            ),
        ]

    # ------------------------------------------------------------------
    # 入力浄化: 大祓詞フィルタ (Oharae-no-Kotoba)
    # ------------------------------------------------------------------

    def sanitize_input(self, text: str) -> Dict[str, Any]:
        """大祓詞フィルタ (Oharae-no-Kotoba) — 入力浄化

        入力テキストを検査し、インジェクション・有害指示・
        感情ノイズを検出する。

        Args:
            text: ユーザ入力テキスト

        Returns:
            dict:
                text (str): 浄化済みテキスト (ブロック時は空文字列)
                safety_score (float): 安全性スコア (0.0〜1.0、高い=安全)
                blocked (bool): 入力がブロックされたか
                warnings (List[str]): 警告メッセージのリスト
        """
        warnings: List[str] = []
        cumulative_risk = 0.0

        # 1. プロンプトインジェクション検出
        is_injection, injection_score, injection_warnings = self._detect_injection(text)
        cumulative_risk += injection_score
        warnings.extend(injection_warnings)

        # 2. 感情ノイズスコアリング
        for pattern, emotion_type, noise_score in self.emotion_patterns:
            matches = pattern.findall(text)
            if matches:
                cumulative_risk += noise_score * len(matches)
                warnings.append(
                    f"感情ノイズ検出 ({emotion_type}): {len(matches)}件"
                )

        # 安全性スコア算出 (リスクが高いほどスコアが低い)
        safety_score = max(0.0, 1.0 - cumulative_risk)

        # ブロック判定
        blocked = safety_score < (1.0 - self.safety_threshold)

        return {
            "text": "" if blocked else text,
            "safety_score": safety_score,
            "blocked": blocked,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # 出力浄化: 斎庭チェック (Yuniwa)
    # ------------------------------------------------------------------

    def sanitize_output(self, text: str) -> Dict[str, Any]:
        """斎庭チェック (Yuniwa) — 出力浄化

        出力テキストを検査し、PII 漏洩・有害コンテンツ・
        安全でない Julia コードパターンを検出・無害化する。

        Args:
            text: モデル出力テキスト

        Returns:
            dict:
                text (str): 浄化済みテキスト
                filtered_items (List[str]): フィルタリングされた項目の説明
        """
        filtered_items: List[str] = []
        sanitized = text

        # 1. PII 漏洩の検出とマスキング
        sanitized, pii_items = self._mask_pii_with_report(sanitized)
        filtered_items.extend(pii_items)

        # 2. 安全でない Julia コードパターンの検出
        julia_warnings = self._detect_unsafe_julia(sanitized)
        for warning in julia_warnings:
            filtered_items.append(f"[WARNING] Julia安全性: {warning}")

        # 3. 有害コンテンツの検出（出力側）
        for pattern, _, pattern_name in self.injection_patterns:
            if pattern.search(sanitized):
                filtered_items.append(
                    f"出力に不正パターンを検出: {pattern_name}"
                )

        return {
            "text": sanitized,
            "filtered_items": filtered_items,
        }

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _detect_injection(self, text: str) -> Tuple[bool, float, List[str]]:
        """プロンプトインジェクションを検出する

        Args:
            text: 検査対象テキスト

        Returns:
            is_injection: インジェクションが検出されたか
            score: リスクスコア (0.0〜)
            matched_patterns: マッチしたパターン名のリスト
        """
        total_score = 0.0
        matched: List[str] = []

        for pattern, severity, name in self.injection_patterns:
            if pattern.search(text):
                total_score += severity
                matched.append(f"インジェクション検出 ({name})")

        is_injection = total_score >= self.safety_threshold
        return is_injection, total_score, matched

    def _detect_unsafe_julia(self, text: str) -> List[str]:
        """安全でない Julia コードパターンを検出する

        Args:
            text: 検査対象テキスト

        Returns:
            warnings: 警告メッセージのリスト
        """
        warnings: List[str] = []
        for pattern, description in self.unsafe_julia_patterns:
            if pattern.search(text):
                warnings.append(description)
        return warnings

    def _mask_pii(self, text: str) -> str:
        """PII をマスキングする

        Args:
            text: 対象テキスト

        Returns:
            masked_text: PII を [MASKED] に置換したテキスト
        """
        masked = text
        for pattern, _ in self.pii_patterns:
            masked = pattern.sub("[MASKED]", masked)
        return masked

    def _mask_pii_with_report(self, text: str) -> Tuple[str, List[str]]:
        """PII をマスキングし、検出結果を報告する

        Args:
            text: 対象テキスト

        Returns:
            masked_text: PII を [MASKED] に置換したテキスト
            report: 検出された PII 種別のリスト
        """
        masked = text
        report: List[str] = []

        for pattern, pii_type in self.pii_patterns:
            matches = pattern.findall(masked)
            if matches:
                report.append(f"PII検出 ({pii_type}): {len(matches)}件をマスク")
                masked = pattern.sub("[MASKED]", masked)

        return masked, report
