# データインベントリ — 「あるデータ / 作るデータ」

> 関連: [DATA_DESIGN.md](DATA_DESIGN.md)（データ設計）, [julia_to_ts_mapping.md](julia_to_ts_mapping.md)（神話構造の TS 対応）

学習に必要なデータの調達状況。**A=学習コーパス**、**B=型ラベル**、**C=ハルシネーション負例**、**D=tsc strict ラベル**、**E=評価** の5カテゴリ。

調査日: 2026-05-17

---

## サマリー

| ID | 用途 | 状態 | 主候補 |
|----|------|------|--------|
| **A** | SFT 用 TS コーパス | **既存利用可** | `bigcode/the-stack-v2-train-smol-ids` (TS subset) |
| **A'** | TS instruct ペア | **既存利用可（要注意）** | `mhhmm/typescript-instruct-20k-v2c` (GPT-3.5生成、蒸留懸念あり) |
| **A''** | DefinitelyTyped（real-API語彙） | **要取得（git clone）** | `DefinitelyTyped/DefinitelyTyped` |
| **B** | per-token 型ラベル（TsukuyomiTypeHead 学習） | **既存利用可（大発見）** | `kevinjesse/ManyTypes4TypeScript` |
| **C** | ハルシネーション負例（HirukoDetector 学習） | **自作必要** | コード変異 + `tsc --strict` 検証 |
| **D** | tsc strict pass/fail ラベル（評価） | **自作必要** | A の各ファイルを strict で再コンパイル |
| **E** | 評価セット（MultiPL-E TS） | **既存利用可** | `nuprl/MultiPL-E` (humaneval-ts 159, mbpp-ts 390) |

**結論**: B/E が既存で済む（最大の難関だった B が省略可能）。**自作が必要なのは C と D のみ**。

---

## A. SFT 用 TS コーパス（既存）

### A-1. The Stack v2 TS subset

| 項目 | 値 |
|------|----|
| Dataset | [`bigcode/the-stack-v2-train-smol-ids`](https://huggingface.co/datasets/bigcode/the-stack-v2-train-smol-ids) |
| 言語 | 17言語に絞った版（TS含む） |
| 規模 | 3B+ files（全体）、TS subset で 10M+ files 想定 |
| License | Software Heritage の元ライセンス継承（要確認、TS は permissive 比率 4% と低め） |
| 注意 | コンテンツ取得には SWH/AWS が必要（smol-ids は ID + メタのみ） |

**代替**: `bigcode/the-stack-v2-dedup` または `bigcode/the-stack-v2` のコンテンツ込み版。

**フィルタ要件** (DATA_DESIGN.md 準拠):
- 真の `.ts` ファイル（`.d.ts` は別扱い）
- `tsconfig.json` を持つプロジェクト
- 明示的な型注釈が一定比率以上
- `noImplicitAny` を通る

### A-2. TS instruct（オプション）

| 項目 | 値 |
|------|----|
| Dataset | [`mhhmm/typescript-instruct-20k-v2c`](https://huggingface.co/datasets/mhhmm/typescript-instruct-20k-v2c) |
| 規模 | 20k {instruction, output} ペア |
| 生成元 | The Stack ソース + GPT-3.5-turbo で命令生成 |
| 注意 | **GPT-3.5 蒸留懸念**。SFT に使うと Qwen2.5-Coder が GPT-3.5 の癖を継承する可能性 |

**代替**: [`bleugreen/typescript-instruct`](https://huggingface.co/datasets/bleugreen/typescript-instruct)（詳細未確認）。

instruct 形式が必要かどうかは設計判断。Qwen2.5-Coder-Instruct は既に instruct チューニング済みなので、生コーパス (A-1) で十分かもしれない。

### A-3. DefinitelyTyped（取得作業必要）

| 項目 | 値 |
|------|----|
| Source | `git clone https://github.com/DefinitelyTyped/DefinitelyTyped.git` |
| 規模 | 8000+ ライブラリの `.d.ts` |
| 用途 | real-API 語彙、ハルシネーション検出時の「実在 API リスト」 |
| 状態 | **HF にはまとめがない**、自前で clone |

---

## B. per-token 型ラベル — **大発見: 自作不要**

DATA_DESIGN.md では「TypeScript Compiler API + Node サブプロセスで自前抽出する」と書かれていたが、既製品が存在する。

| 項目 | 値 |
|------|----|
| Dataset | [`kevinjesse/ManyTypes4TypeScript`](https://huggingface.co/datasets/kevinjesse/ManyTypes4TypeScript) |
| 規模 | **733,655 sequences**、9M+ 型注釈、13,953 プロジェクト、539,571 ファイル |
| Format | per-token tagging (NER/POS 風)。`{tokens: [...], labels: [...], url, path, commit_hash, file}` |
| 型ラベル | top-occurring types、null は無注釈 |
| Split | train 91.95% / val 3.71% / test 4.34% |
| Storage | Parquet (Git-LFS) |
| Size | 1.03 GB |
| License | **CC-BY-4.0** |
| 論文 | ManyTypes4TypeScript: MSR 2022 |
| Mirror | HuggingFace, Zenodo (DOI: 10.5281/zenodo.6387001), CodeXGLUE |

**TS版型語彙への影響**:
- ManyTypes4TypeScript の **型語彙をそのまま `config/ts_type_vocab.json` の出発点** にできる
- top-occurring 方式なので、TS版で予定していた 200-400 枠と整合
- ただし「Instability markers (ImplicitAny/ExplicitAny/ErrorType)」が含まれるかは追加調査が必要

**前処理タスク**:
1. ManyTypes4TypeScript の labels を集計し、頻度上位 N (200/300/400) を抽出
2. yamato 側のカテゴリ分類（primitives/builtins/utility/structural/instability）にマッピング
3. 不足する instability markers を追加して `config/ts_type_vocab.json` 完成

---

## C. ハルシネーション負例 — **自作必要**

既存の TS 専用 hallucination データセットは見つからず（Collu-Bench は多言語混合）。
DATA_DESIGN.md の方針通り **自作** する。

### 自作パイプライン

1. A-1 から型整合する TS スニペットを抽出（`tsc --strict` pass のもの）
2. 各スニペットを変異:
   - fake method 呼び出し (`x.fakeMethod()`)
   - argument 数の変更（追加/削除）
   - argument 型のすり替え
   - 存在しない import の追加
3. 変異後を `tsc --strict` にかけ、**コンパイルエラーになったものだけ採用**（真の負例）
4. ペア `(positive, negative, error_code)` として保存

### TypeScript エラーコード分類（参考）

ハルシネーション検出時のラベルとして利用可能:
- **TS2300**: duplicate identifier
- **TS2304**: missing name (存在しない変数/関数)
- **TS2307**: missing module
- **TS2322**: type mismatch
- **TS2339**: property does not exist on type ← **典型的ハルシネーション**

### 目標規模

DATA_DESIGN.md: 20k-50k pairs。

---

## D. tsc strict pass/fail ラベル — **自作必要**

A-1 の各ファイルを `tsc --strict` で再コンパイルし、pass/fail と発生エラーをラベル化。
評価用 + 学習補助。

目標: ~10k ラベル付きサンプル。

---

## E. 評価セット — 既存

| Dataset | TS subset | 規模 | License |
|---------|-----------|------|---------|
| [`nuprl/MultiPL-E`](https://huggingface.co/datasets/nuprl/MultiPL-E) | humaneval-ts | 159 問 | MIT |
| 同上 | mbpp-ts | 390 問 | MIT |

**形式**: `{name, language, prompt, doctests, tests, stop_tokens}`、pass@1 評価可能。

**カスタムメトリクス** (DATA_DESIGN.md):
- `tsc --strict` pass rate
- API hallucination rate（実在 API リストとの照合）
- `any` usage rate

これらは E + C + DefinitelyTyped を組み合わせて自作ハーネスを書く必要あり。

---

## 取得・生成タスク一覧（着手順案）

優先度順:

1. **ManyTypes4TypeScript ダウンロード** — `huggingface-cli download kevinjesse/ManyTypes4TypeScript --repo-type dataset --local-dir data/raw/many_types_4_ts/`
2. **型語彙の集計と `config/ts_type_vocab.json` 生成** — ManyTypes4TS の labels を頻度集計 + カテゴリ分類
3. **MultiPL-E TS のダウンロード** — 評価ハーネスの土台
4. **DefinitelyTyped clone** — `git clone --depth 1 https://github.com/DefinitelyTyped/DefinitelyTyped.git data/raw/definitely_typed/`
5. **The Stack v2 TS subset の段階的取得** — 全量は大きいので、まず数万ファイルでパイロット
6. **`tsc --strict` ハーネス実装** (D の前提) — Node スクリプト
7. **ハルシネーション変異スクリプト実装** (C の前提) — Node スクリプト
8. **C/D のサンプル生成** — 数百 → 数千 → 数万に段階的に拡大

---

## ローカルディレクトリ構成案

```
yamato-public/
├── data/
│   ├── raw/
│   │   ├── many_types_4_ts/          # ManyTypes4TypeScript parquet
│   │   ├── multipl_e/                # MultiPL-E parquet
│   │   ├── the_stack_v2_ts/          # 段階取得
│   │   └── definitely_typed/         # git clone
│   ├── processed/
│   │   ├── ts_type_vocab.json        # config/ts_type_vocab.json と同期
│   │   ├── sft_corpus.parquet        # フィルタ済 A-1
│   │   ├── token_type_labels.parquet # B 由来、yamato 型ID 化済
│   │   ├── hallucination_pairs.parquet # C 自作
│   │   └── tsc_strict_labels.parquet # D 自作
│   └── eval/
│       └── multipl_e_ts/             # E
└── scripts/
    └── data/
        ├── download_*.py             # HF からの取得
        ├── build_type_vocab.py       # B → ts_type_vocab.json
        ├── tsc_strict_runner.ts      # Node スクリプト
        └── mutate_for_hallucination.ts
```

`data/` は `.gitignore` 済み（既存）想定だが、未確認なら追加する。

---

## オープン項目

- A-1 の取得規模をどこに置くか（パイロット 1万 / 中規模 10万 / 全量）
- A-2 を使うかどうか（GPT-3.5 蒸留懸念）
- ManyTypes4TypeScript の型語彙が `ImplicitAny` 等の instability markers を含むか実データで確認
- DefinitelyTyped を「実在 API リスト」に変換するスクリプトの設計
