# yamatoLLM Roadmap

> **Goal**: Type-aware TypeScript code generation that significantly exceeds the **Qwen2.5-Coder-7B baseline** on type-correctness metrics.

---

## Strategy

We do **not** aim to beat the absolute state of the art (Copilot, GPT-4, Qwen2.5-Coder-32B, etc.).
We aim to show that adding **type prediction + hallucination control** on top of a fixed, publicly available baseline (Qwen2.5-Coder-7B) produces a measurable improvement on type-related metrics — without regression on general code-generation quality.

This makes the result **reproducible and uncontroversial**: anyone can run the baseline and the trained model and verify the gap.

---

## Base Model

- **[Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)** (Apache 2.0)
- Reasons:
  - Strong TypeScript baseline (Qwen2.5-Coder family is current SOTA in 7–14B class)
  - Open base + Instruct, fine-tuning friendly
  - Fits RTX 3060 12GB at INT4

---

## Architecture

Three components added on top of the frozen Qwen2.5-Coder backbone (LoRA-trained):

| Component | Role | Inspired by |
|-----------|------|-------------|
| **TsukuyomiTypeHead** | Per-token type prediction that constrains the next-token distribution toward type-consistent tokens | 月読命 |
| **Hiruko Detector** | Post-generation malformed-output detector; triggers retry when the output's type distribution is degenerate | 蛭子（不具の子） |
| **Amenomihashira Protocol** | Three-stage structured generation: type definitions → function signatures → implementations | 天の御柱 |

A small **BonpuConfidence** head produces an uncertainty score so the model can refuse / warn when type safety cannot be guaranteed.

---

## Phases

### Phase 1 — Baseline & Setup
- Load Qwen2.5-Coder-7B-Instruct, run baseline on the evaluation suite
- Establish reference scores: MultiPL-E TS pass@1, tsc strict pass rate, hallucination rate, `any` rate
- INT4 inference on RTX 3060 via TensonKorinQuantizer

### Phase 2 — Architecture Integration
- Attach TsukuyomiTypeHead, Hiruko detector, BonpuConfidence to the frozen backbone
- TypeScript type vocabulary (primitives + utility types + common library types)
- Forward / generate path covering the Amenomihashira three-stage protocol

### Phase 3 — Data
See [DATA_DESIGN.md](DATA_DESIGN.md) for the full pipeline design.

Three datasets are built in parallel:
- **A. Typed TS corpus** — The Stack v2 TS subset, filtered for genuine `.ts` files with explicit type annotations
- **B. Token-level type labels** — extracted via the TypeScript Compiler API in a Node subprocess; mapped to a ~200–400 entry type vocabulary including instability markers (`ImplicitAny`, `ExplicitAny`, `ErrorType`)
- **C. Hallucination negatives** — synthesized by mutating compiling code (fake methods, wrong arg counts, fabricated imports) and keeping only those that actually fail `tsc`

Target sizes: 50–100k SFT files, 30–50k token-type labeled samples, 20–50k hallucination pairs.

### Phase 4 — SFT
- QLoRA on the backbone (LoRA target: q_proj, v_proj, gate_proj)
- Auxiliary losses for TsukuyomiTypeHead and BonpuConfidence
- Training environment: RunPod (A100 / H100)

### Phase 5 — Evaluation & Release
- Compare yamatoLLM vs Qwen2.5-Coder-7B baseline on the metrics below
- If the win condition is met: release weights + write-up
- Otherwise: iterate on data and architecture

---

## Evaluation

| Metric | Source | What it measures |
|--------|--------|------------------|
| **MultiPL-E TypeScript pass@1** | [MultiPL-E](https://github.com/nuprl/MultiPL-E) | General TS code-generation correctness |
| **`tsc --strict` pass rate** | Custom harness | Type-system compliance |
| **API hallucination rate** | Custom harness (static analysis of imports/calls vs. real TS libraries) | Non-existent API calls |
| **`any` usage rate** | Custom harness | Type-safety degradation |

The latter three are **the differentiators**. The first ensures no regression.

---

## Win Condition

A release is justified when:
- **MultiPL-E TS pass@1**: within ±1pt of baseline (no regression)
- **tsc strict pass rate**: baseline + 5pt or more *(revised — baseline is already 93.1%, so the original "+10pt" target is near the ceiling)*
- **Hallucination rate**: baseline × 0.5 or less
- **`any` usage rate**: baseline × 0.5 or less *(baseline is 0%, so the realistic improvement target is to keep it at 0% — non-regression rather than reduction)*

These thresholds are independently verifiable by anyone with the released checkpoints.

---

## Baseline (Qwen2.5-Coder-7B-Instruct, INT4, humaneval-ts)

Measured 2026-05-17 on RTX 3060 12GB. Single sample per problem, temperature=0.2, top_p=0.95.

| Metric | Value |
|--------|-------|
| MultiPL-E TS pass@1 (humaneval-ts, 159 problems) | **74.2%** (118/159) |
| tsc --strict pass rate (prompt + completion) | **93.1%** (148/159) |
| `any` usage rate | **0.0%** (0/159) |
| Avg generation time | 4.4s/problem |
| Avg test runtime | 0.2s/test |

Top tsc-strict error codes when generation fails: TS2304 (Cannot find name), TS2349 (Expression not callable), TS2322 (Type mismatch), TS1160 (Unterminated literal).

Implication: tsc-strict and any-rate are near the ceiling on this baseline, so the headroom for yamato lies primarily in **pass@1** and **hallucination rate**.

Generation script: `scripts/eval/generate_multipl_e.py`. Test runner: `scripts/eval/run_tests.py`. Aux metrics: `scripts/eval/aux_metrics.py`.

---

## Status

### Done
- [x] Base architecture scaffolding (`yamato_model.py`, `qwen_adapter.py`)
- [x] INT4 quantization pipeline (`tenson_korin_quantizer.py`)
- [x] BonpuConfidence head (uncertainty signaling)
- [x] Data pipeline design ([DATA_DESIGN.md](DATA_DESIGN.md))
- [x] TS type vocabulary built from ManyTypes4TypeScript (`config/ts_type_vocab.json`, 256 entries)
- [x] tsc-strict / hallucination tooling (`scripts/ts_tools/`)
- [x] Phase 1 baseline measurement (humaneval-ts pass@1 = 74.2%)

### Next (Architecture migration)
- [x] `yamato_config.py` — migrate to Qwen2.5-Coder-7B spec, drop legacy iwato refs
- [x] `yamato_model.py` — clean up legacy refs, align with Qwen2 API
- [x] `qwen_adapter.py` — default model name Qwen2.5-Coder-7B, remove iwato imports
- [x] `tenson_korin_quantizer.py` — default model name Qwen2.5-Coder-7B
- [x] TypeScript type vocabulary (`config/ts_type_vocab.json`)
- [ ] TsukuyomiTypeHead (TS-adapted port of Julia-no-Mikoto's type head)
- [ ] Hiruko Detector for TypeScript
- [ ] Amenomihashira three-stage generation

### Phase milestones
- [x] Phase 1: baseline measurement on Qwen2.5-Coder-7B (humaneval-ts done, mbpp-ts pending)
- [ ] Phase 2: architecture integration
- [ ] Phase 3: data pipeline implementation (TS Compiler API wrapper → labeled dataset)
- [ ] Phase 4: QLoRA SFT on RunPod
- [ ] Phase 5: evaluation vs baseline, release decision
