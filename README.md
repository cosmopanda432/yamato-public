# yamatoLLM

Type-aware TypeScript code generation, built on top of **Qwen3-Coder-8B**.

## Motivation

Coding LLMs (Copilot, base coding models, etc.) produce code that *looks* correct but routinely:

- Calls APIs that don't exist
- Falls back to `any` to escape type errors
- Fails `tsc --strict`

yamatoLLM integrates **type prediction** directly into generation, with a **malformed-output detector** to catch the residual failures. The goal is code that **compiles by construction**, not by post-hoc correction.

## Architecture

Three components added on top of a frozen Qwen3-Coder-8B backbone:

| Component | Role |
|-----------|------|
| **TsukuyomiTypeHead** | Per-token type prediction that constrains generation |
| **Hiruko Detector** | Detects malformed outputs and triggers retry |
| **Amenomihashira Protocol** | Three-stage structured generation (types → signatures → impl) |

Names are taken from Kojiki mythology. See [docs/ROADMAP.md](docs/ROADMAP.md) for details.

## Status

Work in progress. The goal is not to beat the absolute SOTA but to show a **measurable, reproducible improvement over the Qwen3-Coder-8B baseline** on type-correctness metrics.

Target evaluations:

- MultiPL-E TypeScript pass@1
- `tsc --strict` pass rate
- API hallucination rate
- `any` usage rate

## Requirements

```bash
pip install -r requirements.txt
```

Inference fits in RTX 3060 12GB at INT4 via the included quantization pipeline.

## License

Apache 2.0 (inherits from the Qwen3-Coder base).
