# Data Pipeline Design

Training data design for yamatoLLM's type-aware code generation on TypeScript.

---

## Three Data Categories

| Category | Purpose | Format |
|----------|---------|--------|
| **A. Typed TS corpus** | SFT (general code generation) | `(prompt, code)` pairs |
| **B. Token-level type labels** | TsukuyomiTypeHead supervision | `(tokens, type_ids)` |
| **C. Hallucination negatives** | Hiruko Detector training | `(corrupt_code, error_type)` |

---

## A. Corpus Sources

| Source | Scale | Role |
|--------|-------|------|
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2-train-smol-ids) (TS subset) | ~10M+ files | Main corpus |
| DefinitelyTyped (`@types/*`) | 8000+ libs | Real-API knowledge |
| TypeScript handbook / examples | thousands | High-quality reference |
| GitHub trending TS repos | hundreds | Modern idioms |

Filtering criteria:
- Genuine `.ts` files (`.d.ts` handled separately)
- Has `tsconfig.json`
- Sufficient ratio of explicit type annotations
- Passes `noImplicitAny`

---

## B. Token-Level Type Labels (Core)

**Approach**: invoke the TypeScript Compiler API in a Node subprocess to extract inferred types per AST node.

### Extraction pipeline

```
Python orchestrator
    ↓ subprocess
Node + typescript (ts.createProgram, checker.getTypeAtLocation)
    ↓ JSONL
(file_id, token_offsets, type_strings)
    ↓ Python
type_string → type_id (via ts_type_vocab.json)
    ↓
(tokens, type_ids) pairs → SFT dataset
```

This captures the **inferred** types — including implicit `any` falls, which is the signal Yomi-layer needs.

### Type vocabulary (`config/ts_type_vocab.json`)

Target size: 200–400 IDs. Categories:

| Category | Count | Examples |
|----------|-------|----------|
| Primitives | ~10 | `number`, `string`, `boolean`, `void`, `never`, `any`, `unknown` |
| Builtins | ~50 | `Array<T>`, `Map<K,V>`, `Promise<T>`, `Record<K,V>` |
| Utility types | ~20 | `Partial<T>`, `Pick<T,K>`, `Omit<T,K>`, `Readonly<T>` |
| Literal types | ~10 | StringLiteral, NumberLiteral, BooleanLiteral |
| Structural | ~10 | Object, Function, Interface, Class, Enum, Tuple |
| Type operators | ~10 | Union, Intersection, Generic, Conditional, Mapped |
| Instability markers | 3 | `ImplicitAny`, `ExplicitAny`, `ErrorType` |

The instability markers are what the model learns to **avoid emitting**.

---

## C. Hallucination Negatives

**Strategy**: start from compiling code, mutate it into broken-but-plausible code.

Mutations:
1. Replace method calls with non-existent method names
2. Change argument counts (add / remove)
3. Swap argument types
4. Add fabricated imports

Each mutated sample is verified by running `tsc` — only kept if it actually fails compilation. This guarantees a true negative. These negatives feed Stage 4 (神武東征 / DPO) as `rejected` samples.

```python
def corrupt_for_hallucination(code: str) -> list[CorruptExample]:
    candidates = [
        replace_method_call(code, fake_method=True),
        mutate_arg_count(code),
        swap_argument_types(code),
        add_fake_import(code),
    ]
    return [c for c in candidates if not tsc_passes(c)]
```

---

## Target Data Sizes

| Dataset | Count | Use |
|---------|-------|-----|
| Typed TS files | 50k–100k | SFT |
| Token-type labeled samples | 30k–50k | TsukuyomiTypeHead |
| Hallucination positive/negative pairs | 20k–50k | Stage 4 DPO (神武東征) |
| `tsc --strict` pass/fail pairs | ~10k | Final evaluation |

Sufficient for LoRA SFT on an 8B base.

---

## Implementation Order

1. TS Compiler API wrapper (Node script + Python subprocess)
2. `config/ts_type_vocab.json`
3. The Stack v2 TS extraction + filtering
4. Token type labeling pipeline
5. Hallucination negative generator

---

## Tooling

| Tool | Use |
|------|-----|
| `typescript` (npm) | Compiler API |
| `ts-morph` | High-level AST manipulation |
| `type-coverage` | Project-level type-coverage measurement |
| `tsc --strict` | Evaluation metric |
