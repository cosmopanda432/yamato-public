# Julia版 → TypeScript版 マッピング

> 元ドキュメント: `~/yamatoLLM/yamatoLLM/docs/julia_no_mikoto_design_v2.md`
>
> Julia特化「古事記LLM」(Julia-no-Mikoto) v2 + 天の御柱プロトコルの構造的アイデアを、
> TypeScript 向けに対応させたもの。本リポ (yamato-public) はこのマッピングを単一ソースとする。

実装方針（Qwen2.5-Coder-7B backbone に LoRA で追加するか、from-scratch でプロトタイプを書くか）は別途決定する。

---

## アーキテクチャ全体図の対応

Julia版の5章構成をそのまま TS 向けに読み替える。各層に「神話的な名前 / 実体的な役割」を併記。

| 章 | Julia版 | TS版での役割 |
|----|---------|------------|
| 第一章 天地開闢 | Token Embedding + Type Hierarchy Embedding (神産巣日神) | Token Embedding + TS型階層 Embedding（`number`/`Array`/`Promise` などのカテゴリ） |
| 第二章 神世七代 | Self-Attention + Multiple Dispatch Attention | Self-Attention + **Role-based Attention**（関数呼び出し → 型注釈 を強める） |
| 第三章 国生み | struct生成 (淤能碁呂島) + マクロ展開 (天の沼矛) | **type/interface 定義生成** + **Data Augmentation**（Mapped Types/Generics 展開） |
| 第四章 黄泉国 | Type Instability Detection (@code_warntype) | **`any`/`unknown`/`ErrorType` 検出**（tsc strict 情報の活用） |
| 第五章 禊 | 三貴子: 天照(token)/月読(type)/須佐之男(error) | Token Head + **TS型予測Head** + tsc エラー予測 Head |

最終出力フォーマット（Julia版に倣う）:

```text
{
    "next_token":       次トークン確率分布,
    "next_type":        次トークンの TS 型予測,  # 月読相当
    "stability_score":  型安定性スコア,           # tsc strict 寄り
    "error_score":      tsc エラー確率           # 須佐之男相当
}
```

---

## 型語彙の対応 (TYPE_CATEGORIES)

Julia版: 128枠（標準64 + ユーザー定義32 + 特殊32）。
TS版: **200-400枠** を目標（ROADMAP/DATA_DESIGN 準拠）。`config/ts_type_vocab.json` に格納。

| Julia版カテゴリ | 例 | TS版カテゴリ | 例 |
|------|----|------|----|
| 数値型 (0-19) | Int64, Float64, Complex, Rational | Primitives | `number`, `bigint`, `boolean` |
| コレクション (20-33) | Vector, Matrix, Dict, Tuple, NamedTuple, Pair, UnitRange | Builtins | `Array<T>`, `Map<K,V>`, `Set<T>`, `Promise<T>`, `Record<K,V>`, tuple types |
| 文字列・シンボル (34-39) | String, Symbol, Char, SubString, Regex | Primitives + Builtins | `string`, `RegExp` |
| 関数・型 (40-45) | Function, DataType, UnionAll, Union | Structural | Function type, Class, Interface |
| 構造的 (50-63) | Nothing, Missing, Bool, Module, Expr | Structural + Primitives | `void`, `never`, `undefined`, `null`, Enum |
| ユーザー定義 (64-69) | UserDefinedStruct/Mutable/Abstract/Primitive/Parametric/Singleton | User-defined | `interface`, `type alias`, `class`, generic |
| 特殊 (96-102) | UnknownType, UnstableUnion, TypeParameter, Vararg, Bottom | Instability markers | **`ImplicitAny`**, **`ExplicitAny`**, **`ErrorType`**, `unknown`, type parameter |
| Utility types | (Julia版は該当無し) | Utility types | `Partial<T>`, `Pick<T,K>`, `Omit<T,K>`, `Readonly<T>`, `Required<T>` |
| Literal types | (Julia版は該当無し) | Literal types | StringLiteral, NumberLiteral, BooleanLiteral |
| Type operators | (Julia版は該当無し) | Type operators | Union, Intersection, Conditional, Mapped, Generic |

**TS固有で必要なもの**: Utility types / Literal types / Type operators は Julia版には対応物が無いので独自に枠を作る。

**Hash Embedding** (Julia版でユーザー定義型のため導入): TS でも `interface Foo` などのユーザー定義型は無限に増えるので同じ仕組みを使う。`hash_bucket_size: 1024` 程度を流用。

---

## TokenRole の対応

Julia版8種類 → TS版でも同じ枠を流用しつつ、TS の AST ノード種別に合わせる。

| Julia版 TokenRole | TS版 TokenRole | 抽出元（TS Compiler API） |
|-------------------|---------------|--------------------------|
| UNKNOWN | UNKNOWN | デフォルト |
| FUNCTION_NAME | FUNCTION_NAME | `FunctionDeclaration.name`, `MethodDeclaration.name`, CallExpression の identifier |
| VARIABLE | VARIABLE | `VariableDeclaration.name`, `Parameter.name` |
| TYPE_ANNOTATION | TYPE_ANNOTATION | `TypeReference`, `TypeNode`, `Parameter.type` |
| KEYWORD | KEYWORD | `function`, `class`, `interface`, `type`, `const`, `let`, `if`, `return`, ... |
| OPERATOR | OPERATOR | `+`, `-`, `===`, `?.`, ... |
| LITERAL | LITERAL | `StringLiteral`, `NumericLiteral`, `RegularExpressionLiteral` |
| PUNCTUATION | PUNCTUATION | `,`, `;`, `(`, `)`, `{`, `}`, ... |

抽出は **TypeScript Compiler API の Node 種別** から決定論的に導出可能。

---

## Autoregressive Type Prediction (二人三脚生成)

Julia版の核心。**学習時** は教師あり、**推論時** はトークンと型を同時に自己回帰生成する。
TS版でもそのまま流用する。

推論ループ (Julia版実装をそのまま TS 文脈で再利用):

```
for step in range(max_length):
    outputs = model(token_ids, type_ids, token_roles, type_specificity, type_depth, ...)
    next_token = sample_top_p(outputs["logits"][:, -1], top_p, temperature)  # 天照
    next_type  = argmax(outputs["type_logits"][:, -1])                       # 月読 (TS型)
    error_score = outputs["error_score"][:, -1]                              # 須佐之男 (tsc予測)
    if error_score.mean() > threshold: break  # 黄泉行き判定
    next_spec, next_depth = infer_type_metadata(next_type)
    append(token_ids, next_token); append(type_ids, next_type); ...
    if next_token == eos: break
```

TS固有の追加考慮: 生成中に **`any`/`unknown`/`ErrorType`** カテゴリの出現率を別カウントし、ヒルコ検知に使う（Julia版 UnknownType/UnstableUnion に相当）。

---

## Multiple Dispatch Attention → Role-based Attention (TS版)

Julia版の Multiple Dispatch（型 × 関数の組み合わせ）は TS には直接対応しない。
ただし **役割ベース Attention マスキング** は TS でも有効。

役割相互作用行列の初期値（TS固有に調整）:

| Q\K | FUNCTION_NAME | TYPE_ANNOTATION | VARIABLE | LITERAL |
|---|---|---|---|---|
| FUNCTION_NAME | 1.2 | **1.5** (引数型を見る) | 1.0 | 1.0 |
| TYPE_ANNOTATION | 1.0 | **1.3** (型同士の整合) | 1.2 | 0.8 |
| VARIABLE | 1.0 | **1.4** (自分の型を見る) | 1.3 | 1.0 |

Julia版の **type specificity スケーリング** (Concrete > Abstract > Any) は TS でも:
- Concrete: `number`, `string`, 具体クラス
- Abstract: interface, abstract class, `unknown`
- Any/Bottom: `any`, `never`

として同じ枠で機能する。

---

## 天の御柱プロトコル (Amenomihashira)

Julia版の3段階ステートマシンを TS にそのまま読み替え。

| Phase | Julia版 (generation_phase) | TS版 |
|-------|--------------------------|------|
| Phase 1 (IZANAGI) | struct, abstract type, const | **type alias / interface / enum 定義** |
| Phase 2 (IZANAMI) | function signatures + docs | **function signature** (`function f(x: T): U`) + JSDoc |
| Phase 3 (KAMIYUMI) | 関数本体の実装 | function 本体の実装 |

**ヒルコ検知** (Phase 1完了後):
- Julia版: UnknownType率 > 30%、UnstableUnion率 > 20%、Critical率 > 50%
- TS版: **`ImplicitAny`率 > 30%、`ExplicitAny`率 > 20%、`ErrorType`率 > 30%**

リトライ時の挙動 (Julia版踏襲): 温度 +0.1、最大3回。

**直毘神** (Phase 3完了後の禊):
- Julia版: error_score平均 ≤ 0.5、Phase1型の使用率 ≥ 50%
- TS版: **tsc strict pass、Phase1で定義した型のPhase3での使用率 ≥ 50%**、`any`使用率の上限チェック

---

## 学習データパイプライン

Julia版の `preprocess_julia_code.jl` を TS 用の Node スクリプトに置換。

| Julia版ツール | TS版ツール |
|---|---|
| JuliaSyntax.jl (トークン化) | TypeScript Compiler API: `ts.createSourceFile` / `ts.tokenize` |
| JET.jl (型安定性チェック) | **`tsc --strict`** の診断結果 |
| Cthulhu.jl (詳細型推論) | TypeScript Compiler API: `checker.getTypeAtLocation` |
| LanguageServer.jl | tsserver / `typescript-language-server` |
| `@code_warntype` | `noImplicitAny` / `strict` のフラグでの再コンパイル結果 |
| `@code_llvm` (SIMD判定) | (TS版は対応なし、削除) |
| Zygote (自動微分可能性) | (TS版は対応なし、削除) |

データ生成パイプライン (TS向け、`docs/DATA_DESIGN.md` を補強):

```
1. TS ソースコード収集 (The Stack v2 TS subset + DefinitelyTyped + GitHub trending)
2. トークン化 + token_roles 抽出 (TypeScript Compiler API)
3. 型推論 (checker.getTypeAtLocation)
   → type_ids, type_specificity, type_depth, type_hash
4. 安定性判定
   → stability_labels (noImplicitAny pass=0, with-warnings=1, strict-failed=2)
5. Data Augmentation
   → Utility types 展開 (Partial<T> → 全optional の interface)
   → Mapped types 展開
   → 型注釈の追加/削除
6. 保存 (Arrow/Parquet)
```

---

## Loss 関数の対応

Julia版の重み (v2 値) をそのまま TS版に持ってくる:

| Loss | Julia版重み | TS版 | 備考 |
|------|------------|------|------|
| token (天照) | 1.0 | 1.0 | 次トークン予測 |
| type (月読) | 0.8 | 0.8 | TS型予測（推論時に重要） |
| stability (黄泉) | 0.3 | 0.3 | tsc strict 安定性 |
| simd | 0.1 | - | TS版では削除 |
| diff | 0.1 | - | TS版では削除 |
| error (須佐之男) | 0.2 | 0.2 | tsc コンパイルエラー予測 |
| dynamic_dispatch | (Julia固有) | - | TS版では削除 |
| **hallucination** (TS追加) | - | **0.3** | `tsc --strict` で fake API/wrong args を負例として学習 |

---

## 既存実装との対応

現在 yamato-public 側に存在するもの:

| 概念 | 既存実装 | 状態 |
|------|---------|------|
| Qwen backbone 統合 | `kojiki_lm/qwen_adapter.py` | 完了 |
| INT4 量子化 | `kojiki_lm/tenson_korin_quantizer.py` | 完了 |
| 凡夫の自覚 (confidence) | `kojiki_lm/kenpou/bonpu_confidence.py` | 完了 |
| 統合モデル骨格 | `kojiki_lm/yamato_model.py` | 完了 (最小構成) |
| TS型語彙 | `config/ts_type_vocab.json` | **未着手** |
| TsukuyomiTypeHead (月読) | `kojiki_lm/yomi/tsukuyomi_type_head.py` | 完了 |
| Role-based Attention | (実装方針未定) | **未着手** |
| TS Compiler API ラッパー | `scripts/ts_tools/` | 完了 |

---

## v1 vs v2 (Julia版) の TS 版での扱い

Julia版 v2.1 で導入された天の御柱 (Amenomihashira) / ヒルコ検知 / 直毘神は **TS 版では採用しない**。
理由: Qwen 事前学習済み backbone に type_ids 入力経路を後付けできず、Hiruko 検知器が
ハルシネーションに対して発火しない (humaneval-ts / mbpp-ts で 0/549) ことが pilot で判明。
ハルシネーション抑制は yamatoLLM 4-Stage の Stage 4 神武東征 (DPO) で扱う。

TS 版での出発点:
- 型語彙: TS 標準型 + ユーザー定義型カテゴリ + 特殊型 (`config/ts_type_vocab.json`, 256 件)
- 月読 (TsukuyomiTypeHead): per-token 型予測ヘッドとして output 側に配置
- 凡夫 (BonpuConfidence): 信頼度スコアヘッド

---

## オープン項目

- TS型語彙の最終サイズ (200/300/400 のどれか) と内訳
- backbone 実装方針 (Qwen2.5 + LoRA / from-scratch / 両方)
- 役割マスキングを backbone (Qwen frozen) の上にどう載せるか
  → LoRA で attention を学習する場合、Role 情報の入れ方を検討
- 天の御柱の Phase 切替を Qwen backbone でどう実現するか
  → プロンプトテンプレート切替で十分か、phase embedding が要るか
