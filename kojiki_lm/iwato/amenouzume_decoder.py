"""
第四章: 神楽 (天宇受売命) — 生成・感情制御
amenouzume_decoder.py

天岩戸神話において、天宇受売命（あめのうずめのみこと）は
岩戸の前で神楽を舞い、神々の笑いを誘って天照大御神の関心を引いた。

本モジュールはその「神楽」に対応する。
Qwen backbone の lm_head を利用してトークン確率分布を生成し、
万葉フィルタによってトーン（文体・感情）を制御する。

構成要素:
    ManyoFilter       — 万葉フィルタ: トーン制御
    AmenouzumeDecoder — 天宇受売命: 生成・感情制御の統合

トーン種別:
    0 = formal    (丁寧体 — ですます調)
    1 = casual    (普通体 — だ/である調)
    2 = technical (技術文体 — 専門用語重視)
    3 = poetic    (詩的文体 — 万葉調)

神話的対応:
    天宇受売命の舞は単なる踊りではなく、場の空気を変え、
    神々の感情を動かす力を持っていた。同様に本モジュールは
    生成テキストのトーンを制御し、出力の「空気」を調整する。

入出力:
    Input:  hidden_states [batch, seq_len, d_model] — 知識統合済み表現
            lm_head       nn.Linear (d_model → vocab_size) — Qwen の言語モデルヘッド
    Output: logits        [batch, seq_len, vocab_size] — トーン調整済みロジット
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# トーン定数
TONE_FORMAL = 0     # 丁寧体 (ですます調)
TONE_CASUAL = 1     # 普通体 (だ/である調)
TONE_TECHNICAL = 2  # 技術文体 (専門用語重視)
TONE_POETIC = 3     # 詩的文体 (万葉調)

TONE_MAP = {
    "formal": TONE_FORMAL,
    "casual": TONE_CASUAL,
    "technical": TONE_TECHNICAL,
    "poetic": TONE_POETIC,
}

NUM_TONES = 4


class ManyoFilter(nn.Module):
    """万葉フィルタ — トーン制御

    生成ロジットに対してトーン依存のバイアスを加え、
    出力テキストの文体・感情を制御する。

    万葉集が多様な歌風（雄大・繊細・叙情・諧謔）を包含するように、
    本フィルタは多様なトーンを切り替えて出力を調整する。

    Args:
        vocab_size: 語彙サイズ (llm-jp-4-8b = 196608)
        d_model:    隠れ層の次元数 (llm-jp-4-8b hidden_size = 4096)
        num_tones:  トーンの種類数 (デフォルト: 4)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 4096,
        num_tones: int = NUM_TONES,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_tones = num_tones

        # トーン埋め込み: 各トーンを d_model 次元ベクトルに変換
        self.tone_embedding = nn.Embedding(num_tones, d_model)

        # トーンベクトルから語彙空間へのバイアスを生成
        self.tone_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size),
        )

        # トーンの強度を制御するスケーリング係数
        self.tone_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        logits: Tensor,
        hidden_states: Tensor,
        tone: Union[str, int] = "technical",
    ) -> Tensor:
        """トーン制御の適用

        Args:
            logits:        [batch, seq_len, vocab_size] ベースロジット
            hidden_states: [batch, seq_len, d_model] hidden_states (将来拡張用)
            tone:          トーン指定 (文字列または整数)

        Returns:
            adjusted_logits: [batch, seq_len, vocab_size] トーン調整済みロジット
        """
        # トーン ID の解決
        if isinstance(tone, str):
            tone_id = TONE_MAP.get(tone, TONE_TECHNICAL)
        else:
            tone_id = tone

        # トーン埋め込みの取得
        tone_idx = torch.tensor(
            [tone_id], device=logits.device, dtype=torch.long
        )
        tone_vector = self.tone_embedding(tone_idx)  # [1, d_model]

        # トーンバイアスの生成
        tone_bias = self.tone_projection(tone_vector)  # [1, vocab_size]
        tone_bias = tone_bias.unsqueeze(1)  # [1, 1, vocab_size] for broadcasting

        # スケーリング付きバイアス加算
        adjusted_logits = logits + self.tone_scale * tone_bias

        return adjusted_logits


class AmenouzumeDecoder(nn.Module):
    """天宇受売命 — 生成・感情制御

    Qwen backbone の lm_head でベースロジットを生成し、
    万葉フィルタでトーン調整を行い、学習可能な温度パラメータで
    出力分布の鋭さを制御する。

    処理の流れ:
        1. lm_head でベースロジットを生成
        2. ManyoFilter でトーンバイアスを適用
        3. 温度スケーリングで分布の鋭さを調整
        4. P(w_t) 確率分布を出力

    Args:
        d_model:    隠れ層の次元数 (llm-jp-4-8b hidden_size = 4096)
        vocab_size: 語彙サイズ (llm-jp-4-8b = 196608)
    """

    def __init__(
        self,
        d_model: int = 4096,
        vocab_size: int = 196608,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # 万葉フィルタ: トーン制御
        self.manyo_filter = ManyoFilter(vocab_size, d_model)

        # 学習可能な温度パラメータ: 出力分布の鋭さを制御
        # 初期値 1.0 (変換なし), 学習で最適化される
        self.temperature_scale = nn.Parameter(torch.ones(1))

        # 出力正規化 (lm_head 適用前)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: Tensor,
        lm_head: Optional[Union[nn.Linear, Callable[[Tensor], Tensor]]] = None,
        tone: Union[str, int] = "technical",
        return_probs: bool = False,
    ) -> Tensor:
        """生成・感情制御パイプライン

        Args:
            hidden_states: [batch, seq_len, d_model] 知識統合済み表現
            lm_head:       Qwen の言語モデルヘッド (nn.Linear or callable)
                           None の場合はトーン制御のみ（テスト用）
            tone:          トーン指定 ("formal", "casual", "technical", "poetic")
            return_probs:  True の場合、softmax 確率を返す

        Returns:
            logits: [batch, seq_len, vocab_size] トーン調整・温度スケーリング済みロジット
            (return_probs=True の場合は確率分布)
        """
        # Step 1: 出力正規化
        normed = self.output_norm(hidden_states)

        # Step 2: lm_head でベースロジットを生成
        if lm_head is not None:
            base_logits = lm_head(normed)  # [batch, seq_len, vocab_size]
        else:
            # lm_head が無い場合 (テスト・デバッグ用)
            # hidden_states をそのまま返す（vocab_size にはならない）
            return normed

        # Step 3: 万葉フィルタでトーン調整
        adjusted_logits = self.manyo_filter(
            base_logits, hidden_states, tone=tone
        )

        # Step 4: 温度スケーリング
        # temperature_scale は正の値を保証するため softplus を使用
        temperature = F.softplus(self.temperature_scale) + 1e-6
        scaled_logits = adjusted_logits / temperature

        # Step 5: 確率分布の生成 (オプション)
        if return_probs:
            return F.softmax(scaled_logits, dim=-1)

        return scaled_logits
