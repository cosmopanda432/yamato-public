# yamatoLLM Roadmap

> **Goal**: Type-aware TypeScript code generation that significantly exceeds the **Qwen2.5-Coder-7B baseline** on type-correctness metrics.

---

## Strategy

We do **not** aim to beat the absolute state of the art (Copilot, GPT-4, Qwen2.5-Coder-32B, etc.).
We aim to show that adding **per-token type prediction (SFT) + hallucination-suppressing preference optimization (DPO)** on top of a fixed, publicly available baseline (Qwen2.5-Coder-7B) produces a measurable improvement on type-related metrics — without regression on general code-generation quality.

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

Two components added on top of the Qwen2.5-Coder backbone (LoRA-trained):

| Component | Role | Inspired by |
|-----------|------|-------------|
| **TsukuyomiTypeHead** | Per-token TS-type prediction trained as an auxiliary objective during SFT | 月読命 |
| **BonpuConfidence** | Uncertainty score so the model can refuse / warn when type safety cannot be guaranteed | 凡夫の自覚 |

Hallucination suppression is handled at training time via DPO (Stage 4 神武東征) — not via inference-time wall detectors. See the dropped-items note in [Status](#status).

---

## Pipeline (yamatoLLM 4-Stage)

Primary reference: `~/yamatoLLM/yamatoLLM/docs/yamatoLLM_training_pipeline.md`.

### Stage 1 — 国譲り (Kuniyuzuri / weight inheritance)
- Load Qwen2.5-Coder-7B-Instruct, attach randomly initialized custom heads (TsukuyomiTypeHead, BonpuConfidence)
- No training cost; establishes the `yamato_base` checkpoint
- Baseline evaluation: MultiPL-E TS pass@1, tsc strict pass rate, hallucination rate, `any` rate
- INT4 inference on RTX 3060 via TensonKorinQuantizer

### Stage 2 — 天孫降臨 (Tenson Korin / general SFT)
- QLoRA on the backbone (LoRA target: q_proj, v_proj, gate_proj)
- Auxiliary loss for TsukuyomiTypeHead (CE on TS type labels) + BonpuConfidence
- Training environment: RunPod (A100 / H100)
- Data: see [Stage data section](#stage-data) below

### Stage 3 — 禊・三貴子 (Misogi / per-layer specialization SFT)
- Layer-specialized SFT after the Stage 2 LoRA is merged or stacked
- Three sub-routines for the three traits: language understanding (天照), code generation (月読), governance (須佐之男)
- Each receives a smaller, domain-specific dataset (~2–5k samples)

### Stage 4 — 神武東征 (Jinmu Tosei / DPO alignment)
- Direct Preference Optimization with chosen / rejected pairs
- Negative samples: synthesized by mutating compiling code (fake methods, wrong arg counts, fabricated imports) and keeping only those that actually fail `tsc`
- This is where hallucination suppression happens (replaces the dropped v2.1 Hiruko detector)

### Final — Evaluation & Release
- Compare yamatoLLM vs Qwen2.5-Coder-7B baseline on the metrics below
- If the win condition is met: release weights + write-up
- Otherwise: iterate on data and stage parameters

---

## Stage data

See [DATA_DESIGN.md](DATA_DESIGN.md) for the full pipeline design.

| Dataset | Stage that uses it | Source |
|---------|-------------------|--------|
| **A. Typed TS corpus** | Stage 2 (SFT base) | The Stack v2 TS subset, filtered for genuine `.ts` files with explicit type annotations |
| **B. Token-level type labels** | Stage 2 (TsukuyomiTypeHead aux loss) | TypeScript Compiler API → ~200–400 entry vocabulary incl. `ImplicitAny`/`ExplicitAny`/`ErrorType` |
| **C. Hallucination negatives** | Stage 4 (DPO rejected) | Code mutation + `tsc --strict` failure filtering |

Target sizes: 50–100k SFT files, 30–50k token-type labeled samples, 20–50k hallucination pairs.

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

## Baseline (Qwen2.5-Coder-7B-Instruct, INT4)

Measured 2026-05-17 on RTX 3060 12GB. Single sample per problem, temperature=0.2, top_p=0.95.

| Metric | humaneval-ts (159) | mbpp-ts (390) |
|--------|-------------------:|---------------:|
| **MultiPL-E TS pass@1** | **74.2%** (118/159) | **56.7%** (221/390) |
| **tsc --strict pass rate** (prompt + completion) | **93.1%** (148/159) | **66.4%** (259/390) |
| **`any` usage rate** | **0.0%** (0/159) | **0.0%** (0/390) |
| Avg generation time | 4.4s/problem | 11.2s/problem |

Top tsc-strict error codes on failures:
- humaneval-ts: TS2304 (Cannot find name) ×5, TS1160 (Unterminated literal) ×3, TS2349 (Expression not callable) ×3, TS2322 (Type mismatch) ×2
- mbpp-ts: TS1005 (Expected token) ×50, TS2349 (Expression not callable) ×50, TS1160 ×46, TS2304 ×46, TS1443 ×32 — many are partial-truncation syntax errors on longer problems

Implication: tsc-strict and any-rate are near the ceiling on humaneval-ts, so the headroom for yamato lies primarily in **pass@1** and **hallucination rate**. mbpp-ts has more headroom on tsc-strict.

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
- [x] Stage 1 baseline measurement (humaneval-ts pass@1 = 74.2%, mbpp-ts pass@1 = 56.7%)

### Migration cleanup
- [x] `yamato_config.py` — migrate to Qwen2.5-Coder-7B spec, drop legacy iwato refs
- [x] `yamato_model.py` — clean up legacy refs, align with Qwen2 API
- [x] `qwen_adapter.py` — default model name Qwen2.5-Coder-7B, remove iwato imports
- [x] `tenson_korin_quantizer.py` — default model name Qwen2.5-Coder-7B
- [x] TypeScript type vocabulary (`config/ts_type_vocab.json`)
- [x] TsukuyomiTypeHead (TS-adapted port of Julia-no-Mikoto's type head)
- [x] ~~Hiruko Detector for TypeScript~~ — dropped (v2.1 Amenomihashira protocol abandoned 2026-05-18; 0/549 firings on humaneval-ts/mbpp-ts)
- [x] ~~Amenomihashira three-stage generation~~ — dropped (same reason)

### yamatoLLM 4-Stage progress
- [x] **Stage 1 (国譲り)**: weight inheritance (Qwen2.5-Coder-7B-Instruct) + random custom-head init
- [x] **Stage 2 (天孫降臨)**: QLoRA SFT done on RunPod A6000, step_2000. TypeHead top-1 = 70.1% / top-5 = 91.0% on 200 samples. Win Condition **not met** — see [Stage 2 results](#stage-2-results-step_2000-vs-baseline) below
- [ ] **Stage 3 (禊・三貴子)**: 3-layer specialization SFT (iwato 天照 / kojiki 月読 / kenpou 須佐之男) — next
- [ ] **Stage 4 (神武東征)**: DPO alignment for hallucination suppression — not started

### Stage 2 results (step_2000 vs baseline)

humaneval-ts (159):

| Metric | Baseline | step_2000 | Δ | vs Win Condition |
|--------|---------:|----------:|--:|:---:|
| pass@1 | 74.21% | 74.84% | +0.63pt | ✅ within ±1pt |
| tsc strict | 93.08% | 91.19% | -1.89pt | ❌ +5pt target not met, regressed |
| any rate | 0.0% | 0.0% | ±0 | ✅ maintained |

mbpp-ts (390):

| Metric | Baseline | step_2000 | Δ | vs Win Condition |
|--------|---------:|----------:|--:|:---:|
| pass@1 | 56.67% | 54.62% | -2.05pt | ❌ regressed, outside ±1pt |

Conclusion: TypeHead trained (top-1 70.1%) but the auxiliary type-prediction signal did not transfer to generation quality. Generation regressed slightly on tsc-strict and mbpp pass@1. Stage 3 / Stage 4 needed before release decision.

Source files: `data/eval/results/humaneval-ts-yamato-step2000/{_summary,_aux_metrics}.json`, `data/eval/results/mbpp-ts-yamato-step2000/{_summary,_aux_metrics}.json`, `data/eval/type_head/step_2000.json`.

### Outstanding tasks
- [ ] Stage 3 禊・三貴子 dataset + training script (next)
- [ ] Stage 4 神武東征 DPO dataset + training script
- [ ] Re-evaluation after Stage 3 / Stage 4
- [ ] Release decision
