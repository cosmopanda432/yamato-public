"""
天の御柱プロトコル (Amenomihashira) — 3段階 TypeScript 生成。

Julia版 (docs/julia_no_mikoto_design_v2.md) の AmenomihashiraGenerator を
TypeScript 向けに置き換えたもの。

Phase 1 (IZANAGI):  type alias / interface / enum などの型・構造定義のみ生成
Phase 2 (IZANAMI):  function signature を生成 (Phase 1 の型を使う)
Phase 3 (KAMIYUMI): function 本体の実装 (Phase 1 + 2 を context に)

各 Phase の生成後に検証を入れる:
  - Phase 1 → HirukoDetector (型不安定なら温度を上げてリトライ)
  - Phase 3 → 直毘神検証 (Phase 1 で定義した型が Phase 3 で実際に使われているか)

学習は不要。推論時のロジック層。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch

from ..yomi.hiruko_detector import HirukoDetector, HirukoResult


class GenerationPhase(IntEnum):
    IZANAGI = 0   # type definitions
    IZANAMI = 1   # function signatures
    KAMIYUMI = 2  # function bodies


PHASE_PROMPTS: Dict[GenerationPhase, str] = {
    GenerationPhase.IZANAGI: (
        "// Step 1: Define ONLY the TypeScript type aliases, interfaces, and enums "
        "needed for the task below.\n"
        "// Do NOT write any function bodies yet.\n"
        "// Task: {task}\n"
    ),
    GenerationPhase.IZANAMI: (
        "// Step 2: Using the types defined above, write the function signatures "
        "(declarations with explicit parameter and return types).\n"
        "// Do NOT implement the function bodies yet — just the signatures and JSDoc.\n"
    ),
    GenerationPhase.KAMIYUMI: (
        "// Step 3: Implement the function bodies. Use the types and signatures "
        "defined above.\n"
    ),
}

PHASE_STOP_TOKENS: Dict[GenerationPhase, List[str]] = {
    # 各 Phase で「次の Phase のヘッダ」が出たら止める
    GenerationPhase.IZANAGI: ["// Step 2:", "// Step 3:"],
    GenerationPhase.IZANAMI: ["// Step 3:"],
    GenerationPhase.KAMIYUMI: [],
}


@dataclass
class PhaseOutput:
    phase: GenerationPhase
    prompt: str
    completion: str
    n_attempts: int = 1
    hiruko: Optional[HirukoResult] = None


@dataclass
class NaobiResult:
    """直毘神 (Naobi) 検証結果 — Phase 3 完了後の結合整合性チェック"""
    ok: bool
    type_usage_rate: float          # Phase 1 で定義した型が Phase 3 で参照された比率
    defined_types: List[str] = field(default_factory=list)
    used_types: List[str] = field(default_factory=list)


@dataclass
class AmenomihashiraResult:
    task: str
    phases: List[PhaseOutput]
    final_code: str
    naobi: Optional[NaobiResult] = None

    @property
    def total_retries(self) -> int:
        return sum(p.n_attempts - 1 for p in self.phases)


# 「型定義」と判定する識別子の正規表現 (Phase 1 → Phase 3 の使用率算定用)
_TYPE_DEF_RE = re.compile(
    r"\b(?:type|interface|enum)\s+([A-Z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


def _extract_defined_types(code: str) -> List[str]:
    return _TYPE_DEF_RE.findall(code)


def _truncate_at_any(text: str, stop_tokens: List[str]) -> str:
    cut = len(text)
    for st in stop_tokens:
        i = text.find(st)
        if i >= 0 and i < cut:
            cut = i
    return text[:cut]


class AmenomihashiraProtocol:
    """3段階 TypeScript 生成エンジン"""

    def __init__(
        self,
        yamato_model,
        hiruko: HirukoDetector,
        max_retries: int = 3,
        temperature_step: float = 0.1,
        type_usage_threshold: float = 0.5,
        phase_max_new_tokens: Optional[Dict[GenerationPhase, int]] = None,
    ):
        self.model = yamato_model
        self.tokenizer = yamato_model.tokenizer
        self.hiruko = hiruko
        self.max_retries = max_retries
        self.temperature_step = temperature_step
        self.type_usage_threshold = type_usage_threshold
        self.phase_max_new_tokens = phase_max_new_tokens or {
            GenerationPhase.IZANAGI: 256,
            GenerationPhase.IZANAMI: 256,
            GenerationPhase.KAMIYUMI: 512,
        }

    @torch.no_grad()
    def generate(
        self,
        task: str,
        base_temperature: float = 0.3,
        top_p: float = 0.95,
    ) -> AmenomihashiraResult:
        outputs: List[PhaseOutput] = []
        context = ""  # 前 Phase の出力を連結したもの

        for phase in (GenerationPhase.IZANAGI, GenerationPhase.IZANAMI, GenerationPhase.KAMIYUMI):
            prompt_template = PHASE_PROMPTS[phase].format(task=task)
            stop_tokens = PHASE_STOP_TOKENS[phase]
            max_new = self.phase_max_new_tokens[phase]

            attempt = 0
            temperature = base_temperature
            completion = ""
            hiruko_result: Optional[HirukoResult] = None

            while True:
                attempt += 1
                full_prompt = context + prompt_template
                completion = self._generate_single(
                    prompt=full_prompt,
                    max_new_tokens=max_new,
                    temperature=temperature,
                    top_p=top_p,
                    stop_tokens=stop_tokens,
                )

                # Phase 1 のみヒルコ検知
                if phase is GenerationPhase.IZANAGI:
                    type_preds = self._predict_types(full_prompt + completion)
                    hiruko_result = self.hiruko.detect(type_preds)
                    if not hiruko_result.is_malformed:
                        break
                    if attempt > self.max_retries:
                        # リトライ尽きた → そのまま採用
                        break
                    temperature = min(1.0, temperature + self.temperature_step)
                else:
                    break

            outputs.append(PhaseOutput(
                phase=phase,
                prompt=prompt_template,
                completion=completion,
                n_attempts=attempt,
                hiruko=hiruko_result,
            ))
            context = context + prompt_template + completion + "\n"

        final_code = "\n".join(p.completion for p in outputs).strip()
        naobi = self._naobi_validate(outputs)

        return AmenomihashiraResult(
            task=task,
            phases=outputs,
            final_code=final_code,
            naobi=naobi,
        )

    def _generate_single(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_tokens: List[str],
    ) -> str:
        backbone = self.model.backbone
        inputs = self.tokenizer(prompt, return_tensors="pt").to(backbone.device)
        out_ids = backbone.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return _truncate_at_any(text, stop_tokens)

    @torch.no_grad()
    def _predict_types(self, code: str) -> torch.Tensor:
        """完成済みテキストの hidden_states から TS 型 ID 列を取得"""
        backbone = self.model.backbone
        inputs = self.tokenizer(code, return_tensors="pt").to(backbone.device)
        hidden = self.model.get_hidden_states(
            inputs["input_ids"], inputs["attention_mask"]
        )
        out = self.model.custom_heads["type_head"](hidden)
        return out["type_preds"][0]  # [L]

    def _naobi_validate(self, phases: List[PhaseOutput]) -> NaobiResult:
        defined = _extract_defined_types(phases[0].completion)
        body = phases[2].completion
        used = [t for t in defined if re.search(rf"\b{re.escape(t)}\b", body)]
        rate = len(used) / max(len(defined), 1)
        ok = rate >= self.type_usage_threshold or len(defined) == 0
        return NaobiResult(
            ok=ok,
            type_usage_rate=rate,
            defined_types=defined,
            used_types=used,
        )
