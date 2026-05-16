"""
時のスケジューラ (Toki Scheduler)

第16条: 「民を使うに時を以てするは、古の良き典なり」
第15条: 「私を背きて公に向くは、是臣の道なり」

コンテキスト感知型のスケジューリングユーティリティ。
GPU 負荷、メモリ使用率、タスク優先度に応じて
生成の実行・遅延・キューイングを制御する。

学習可能パラメータを持たない純粋なユーティリティクラス（nn.Module ではない）。
"""

from typing import Any, Dict, Optional, Tuple

import torch


def _get_config_value(config: Any, attr: str, default: Any) -> Any:
    """KenpouConfig または YamatoConfig から属性を安全に取得する。"""
    if hasattr(config, "kenpou") and hasattr(config.kenpou, attr):
        return getattr(config.kenpou, attr)
    if hasattr(config, attr):
        return getattr(config, attr)
    return default


class TokiScheduler:
    """
    時のスケジューラ — コンテキスト感知型リソース管理

    第16条: 「民を使うに時を以てするは、古の良き典なり」

    農事暦に倣い、計算リソースの「旬」を見極める。
    負荷が高い時は遅延し、余裕がある時に実行する。

    GPU が利用不可の場合は安全なデフォルト値を返し、
    常に実行を許可する。

    Args:
        config: KenpouConfig または kenpou 属性を持つ YamatoConfig。
    """

    def __init__(self, config: Optional[Any] = None):
        from .kenpou_config import DEFAULT_KENPOU_CONFIG

        if config is None:
            config = DEFAULT_KENPOU_CONFIG

        self.load_defer_threshold = _get_config_value(
            config, "load_defer_threshold", 0.8
        )
        self.context_sensing_enabled = _get_config_value(
            config, "context_sensing_enabled", True
        )
        self.public_private_filter = _get_config_value(
            config, "public_private_filter", True
        )

        # GPU 利用可否
        self._cuda_available = torch.cuda.is_available()

    def should_execute(self, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        現在のリソース状況に基づき、実行すべきかを判定する。

        第16条: 時（タイミング）を見計らう。

        Args:
            context: 追加コンテキスト情報（オプション）。
                - "priority": "high" | "normal" | "low"
                - "force": bool（強制実行フラグ）

        Returns:
            (execute, reason) のタプル。
            execute: True なら実行、False なら遅延。
            reason: 判定理由の文字列。
        """
        if context is None:
            context = {}

        # 強制実行フラグ
        if context.get("force", False):
            return True, "強制実行フラグが設定されています"

        # コンテキスト感知が無効なら常に実行
        if not self.context_sensing_enabled:
            return True, "コンテキスト感知が無効です"

        # 高優先度は常に実行
        priority = context.get("priority", "normal")
        if priority == "high":
            return True, "高優先度タスクは即座に実行します"

        # リソース状況を確認
        status = self.get_resource_status()

        gpu_util = status.get("gpu_utilization", 0.0)
        memory_util = status.get("memory_utilization", 0.0)

        # GPU 負荷チェック
        if gpu_util > self.load_defer_threshold:
            if priority == "low":
                return False, (
                    f"GPU負荷が高い ({gpu_util:.1%} > {self.load_defer_threshold:.1%})。"
                    f"低優先度タスクを遅延します"
                )
            # normal priority でも負荷が非常に高い場合
            if gpu_util > 0.95:
                return False, (
                    f"GPU負荷が極めて高い ({gpu_util:.1%})。"
                    f"通常優先度タスクも遅延します"
                )

        # メモリ負荷チェック
        if memory_util > self.load_defer_threshold:
            if priority == "low":
                return False, (
                    f"メモリ使用率が高い ({memory_util:.1%})。"
                    f"低優先度タスクを遅延します"
                )

        return True, "リソースに余裕があります。実行を許可します"

    def get_resource_status(self) -> Dict[str, Any]:
        """
        現在のリソース状況を取得する。

        GPU が利用不可の場合は安全なデフォルト値を返す。

        Returns:
            dict with:
                - "gpu_available": bool
                - "gpu_utilization": float [0, 1]
                - "memory_utilization": float [0, 1]
                - "memory_allocated_mb": float
                - "memory_total_mb": float
        """
        if not self._cuda_available:
            return {
                "gpu_available": False,
                "gpu_utilization": 0.0,
                "memory_utilization": 0.0,
                "memory_allocated_mb": 0.0,
                "memory_total_mb": 0.0,
            }

        try:
            device = torch.cuda.current_device()
            mem_allocated = torch.cuda.memory_allocated(device)
            mem_total = torch.cuda.get_device_properties(device).total_mem

            memory_utilization = mem_allocated / mem_total if mem_total > 0 else 0.0
            memory_allocated_mb = mem_allocated / (1024 * 1024)
            memory_total_mb = mem_total / (1024 * 1024)

            # GPU 使用率は PyTorch から直接取得できないので、
            # メモリ使用率を代理指標として使用する。
            # nvidia-smi を呼ぶオーバーヘッドを避けるため。
            gpu_utilization = memory_utilization

            return {
                "gpu_available": True,
                "gpu_utilization": gpu_utilization,
                "memory_utilization": memory_utilization,
                "memory_allocated_mb": memory_allocated_mb,
                "memory_total_mb": memory_total_mb,
            }
        except Exception:
            # CUDA エラーが発生した場合は安全なデフォルトを返す
            return {
                "gpu_available": False,
                "gpu_utilization": 0.0,
                "memory_utilization": 0.0,
                "memory_allocated_mb": 0.0,
                "memory_total_mb": 0.0,
            }

    def schedule(
        self,
        task_priority: str = "normal",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        タスクのスケジューリング判定を行う。

        第16条: 時を以て民を使う。

        Args:
            task_priority: "high" | "normal" | "low"。
            context: 追加コンテキスト（should_execute に渡す）。

        Returns:
            dict with:
                - "action": "execute" | "defer" | "queue"
                - "reason": str（判定理由）
                - "priority": str（入力された優先度）
                - "resource_status": dict（リソース状況）
        """
        if context is None:
            context = {}

        context["priority"] = task_priority
        resource_status = self.get_resource_status()

        # --- 高優先度: 即座に実行 ---
        if task_priority == "high":
            return {
                "action": "execute",
                "reason": "第16条: 高優先度タスクは時を問わず即断します",
                "priority": task_priority,
                "resource_status": resource_status,
            }

        # --- 通常/低優先度: リソースを確認 ---
        execute, reason = self.should_execute(context)

        if execute:
            return {
                "action": "execute",
                "reason": reason,
                "priority": task_priority,
                "resource_status": resource_status,
            }

        # --- 遅延の判定 ---
        if task_priority == "low":
            return {
                "action": "defer",
                "reason": reason,
                "priority": task_priority,
                "resource_status": resource_status,
            }

        # normal priority で遅延 → キューに入れる
        return {
            "action": "queue",
            "reason": f"{reason}。キューに入れて後で実行します",
            "priority": task_priority,
            "resource_status": resource_status,
        }

    def __repr__(self) -> str:
        return (
            f"TokiScheduler("
            f"load_defer_threshold={self.load_defer_threshold}, "
            f"context_sensing={self.context_sensing_enabled}, "
            f"cuda={'available' if self._cuda_available else 'unavailable'})"
        )
