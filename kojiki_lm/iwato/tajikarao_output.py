"""
第五章: 開戸 (天手力男神) — 出力確定
tajikarao_output.py

天岩戸神話において、天宇受売命の舞により天照大御神が岩戸を
わずかに開いた瞬間、天手力男神（たぢからおのかみ）が岩戸を
一気に引き開け、天照大御神を外へ導いた。

本モジュールはその「開戸」に対応する。
生成された logits から最終的なトークンを確定し、
注連縄（しめなわ）による生成終了条件の監視を行う。

機能:
    - ShimenawaStopper: 生成終了条件の監視（EOS・最大長・繰返し）
    - TajikraoOutput: logits に繰返しペナルティとゲートを適用し、
      top-p (nucleus) サンプリングで最終トークンを選択する

入出力:
    Input:  logits        [batch, vocab_size] — デコーダ出力の生トークン確率
            hidden_states [batch, seq_len, d_model] — 最終隠れ状態
    Output: final_logits  [batch, vocab_size] — ペナルティ・ゲート適用済み logits
            output_confidence [batch, 1] — 出力確信度スコア
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ShimenawaStopper:
    """注連縄 — 生成終了条件の監視

    生成ループにおいて、以下の3条件を監視する:
        1. EOS トークンの出現
        2. 最大トークン数の到達
        3. 過度な繰り返し（同一トークンの連続出現）

    繰り返しペナルティにより、既出トークンの再選択確率を低下させる。
    """

    def __init__(
        self,
        max_tokens: int = 2048,
        repeat_penalty: float = 1.2,
        eos_token_id: int = 151643,
        repeat_window: int = 64,
    ) -> None:
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.eos_token_id = eos_token_id
        self.repeat_window = repeat_window
        self.generated_tokens: List[int] = []

    def reset(self) -> None:
        """生成状態をリセットする"""
        self.generated_tokens = []

    def check(
        self, token_id: int, generated_so_far: List[int]
    ) -> Tuple[bool, Optional[str]]:
        """生成終了条件を判定する

        Args:
            token_id: 今回生成されたトークンID
            generated_so_far: これまでに生成されたトークンIDのリスト

        Returns:
            should_stop: 生成を停止すべきか
            reason: 停止理由 (停止しない場合は None)
        """
        # 1. EOS トークン
        if token_id == self.eos_token_id:
            return True, "eos_token"

        # 2. 最大長到達
        if len(generated_so_far) >= self.max_tokens:
            return True, "max_tokens"

        # 3. 過度な繰り返し検出（直近 window 内で同一トークンが半数以上）
        window = generated_so_far[-self.repeat_window :]
        if len(window) >= self.repeat_window:
            from collections import Counter

            counts = Counter(window)
            most_common_count = counts.most_common(1)[0][1]
            if most_common_count > self.repeat_window // 2:
                return True, "excessive_repetition"

        return False, None

    def apply_repeat_penalty(
        self, logits: Tensor, generated_ids: List[int]
    ) -> Tensor:
        """既出トークンの logits にペナルティを適用する

        Args:
            logits: [batch, vocab_size] 生の logits
            generated_ids: これまでに生成されたトークンIDのリスト

        Returns:
            penalized_logits: [batch, vocab_size] ペナルティ適用済み logits
        """
        if not generated_ids:
            return logits

        penalized = logits.clone()
        # 直近 window 内のトークンにのみペナルティ適用
        recent_ids = list(set(generated_ids[-self.repeat_window :]))
        token_indices = torch.tensor(recent_ids, dtype=torch.long, device=logits.device)

        for b in range(penalized.size(0)):
            scores = penalized[b, token_indices]
            # 正の logits は割り算、負の logits は掛け算でペナルティ
            scores = torch.where(
                scores > 0,
                scores / self.repeat_penalty,
                scores * self.repeat_penalty,
            )
            penalized[b, token_indices] = scores

        return penalized


class TajikraoOutput(nn.Module):
    """天手力男神 — 出力確定

    生成された logits に対して繰り返しペナルティと出力ゲートを適用し、
    最終的なトークン選択を行う。

    Args:
        d_model: 隠れ層の次元数 (Qwen3.5-9B hidden_size = 3584)
        max_tokens: 注連縄の最大出力トークン数
        repeat_penalty: 繰り返しペナルティ係数
    """

    def __init__(
        self,
        d_model: int = 3584,
        max_tokens: int = 2048,
        repeat_penalty: float = 1.2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.shimenawa = ShimenawaStopper(
            max_tokens=max_tokens, repeat_penalty=repeat_penalty
        )
        # 出力ゲート: hidden_states から確信度スコアを算出
        self.output_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        logits: Tensor,
        hidden_states: Tensor,
        generated_ids: Optional[List[int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """出力確定の処理

        Args:
            logits:        [batch, vocab_size] デコーダ出力の生 logits
            hidden_states: [batch, seq_len, d_model] 最終隠れ状態
            generated_ids: これまでに生成されたトークンIDのリスト

        Returns:
            final_logits:      [batch, vocab_size] ペナルティ・ゲート適用済み logits
            output_confidence: [batch, 1] 出力確信度スコア (0〜1)
        """
        # 1. 注連縄による繰り返しペナルティ
        if generated_ids is not None:
            logits = self.shimenawa.apply_repeat_penalty(logits, generated_ids)

        # 2. 出力ゲート: 最終トークン位置の hidden_states から確信度を算出
        last_hidden = hidden_states[:, -1, :]  # [batch, d_model]
        output_confidence = self.output_gate(last_hidden)  # [batch, 1]

        # 3. 確信度で logits をスケーリング（低確信度 → よりフラットな分布）
        final_logits = logits * output_confidence

        return final_logits, output_confidence

    def sample(
        self,
        logits: Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tensor:
        """Top-p (nucleus) サンプリング

        Args:
            logits: [batch, vocab_size] logits
            temperature: 温度パラメータ (低い → 決定的、高い → 多様)
            top_p: 累積確率の閾値 (nucleus sampling)

        Returns:
            selected_token: [batch] 選択されたトークンID
        """
        # 温度スケーリング
        logits = logits / max(temperature, 1e-8)

        # ソフトマックスで確率分布に変換
        probs = F.softmax(logits, dim=-1)

        # 確率の降順ソート
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # 累積確率を計算
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # top_p を超えるトークンをマスク（最低1トークンは残す）
        sorted_mask = cumulative_probs - sorted_probs > top_p
        sorted_probs[sorted_mask] = 0.0

        # 再正規化
        sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # カテゴリカル分布からサンプリング
        sampled_index = torch.multinomial(sorted_probs, num_samples=1)  # [batch, 1]

        # ソート前のインデックスに戻す
        selected_token = sorted_indices.gather(dim=-1, index=sampled_index).squeeze(-1)

        return selected_token
