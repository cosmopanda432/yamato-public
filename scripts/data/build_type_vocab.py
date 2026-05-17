"""
ManyTypes4TypeScript の labels から TS 型語彙を構築する。

入力:
    data/raw/many_types_4_ts/data/*.parquet
    data/raw/many_types_4_ts/dataset_infos.json

出力:
    config/ts_type_vocab.json

戦略:
    1. dataset_infos.json から ManyTypes4TS の class names を全取得 (50,001 個)
    2. 全 parquet で labels の頻度を集計
    3. 頻度順に top-(N-K) を採用、yamato 用に予約 K 個（instability markers）を追加
    4. ヒューリスティックでカテゴリ分類 (primitives / builtins / utility / structural / library / instability / special / type_param)
    5. config/ts_type_vocab.json を出力（id_to_type / type_to_id / manytypes4ts_id_map を含む）

manytypes4ts_id_map は ManyTypes4TS の class id → yamato_id への変換テーブル。
yamato 側で labels を読むときにこの map を通して再ラベル化する。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq


# --- ヒューリスティックなカテゴリ判定 ----------------------------------------

TS_PRIMITIVES = {
    "any", "string", "number", "boolean", "void", "null", "undefined",
    "never", "unknown", "object", "symbol", "bigint", "this",
}

TS_BUILTINS = {
    "Array", "Map", "Set", "Promise", "Date", "RegExp", "Error", "WeakMap",
    "WeakSet", "Function", "Object", "String", "Number", "Boolean", "Buffer",
    "Uint8Array", "Uint16Array", "Uint32Array", "Int8Array", "Int16Array",
    "Int32Array", "Float32Array", "Float64Array", "BigInt64Array",
    "BigUint64Array", "ArrayBuffer", "DataView", "JSON", "Math", "Symbol",
    "Iterator", "IterableIterator", "AsyncIterator", "Generator",
    "AsyncGenerator", "Iterable", "AsyncIterable",
}

TS_UTILITY = {
    "Partial", "Required", "Readonly", "Record", "Pick", "Omit", "Exclude",
    "Extract", "NonNullable", "Parameters", "ConstructorParameters",
    "ReturnType", "InstanceType", "ThisParameterType", "OmitThisParameter",
    "ThisType", "Awaited", "Uppercase", "Lowercase", "Capitalize",
    "Uncapitalize", "Mutable", "DeepPartial", "DeepReadonly",
}

TS_STRUCTURAL = {
    "Type", "Node", "Class", "Interface", "Enum", "Module", "Namespace",
}

# 単一大文字 = 型パラメータ (T, U, K, V, E, S, R, P, ...)
def _is_type_param(name: str) -> bool:
    return len(name) == 1 and name.isupper() and name.isalpha()


def categorize(name: str) -> str:
    if name == "UNK":
        return "special"
    if name in TS_PRIMITIVES:
        # any/unknown/never は instability 寄りだが Julia版踏襲で primitives 扱い
        return "primitives"
    if name in TS_BUILTINS:
        return "builtins"
    if name in TS_UTILITY:
        return "utility"
    if name in TS_STRUCTURAL:
        return "structural"
    if _is_type_param(name):
        return "type_param"
    return "library"  # user-defined / library-specific types


# --- 集計 -----------------------------------------------------------------

def collect_label_counts(parquet_paths: List[Path]) -> Counter:
    counts: Counter = Counter()
    for p in parquet_paths:
        table = pq.read_table(p, columns=["labels"])
        for row in table.column("labels").to_pylist():
            for lbl in row:
                if lbl is None:
                    continue
                counts[lbl] += 1
    return counts


def build_vocab(
    names: List[str],
    counts: Counter,
    vocab_size: int,
    reserved: List[Tuple[str, str]],
    force_include_names: List[str],
) -> Tuple[Dict, List]:
    """
    yamato 用語彙を構築する。

    Args:
        reserved: [(name, category)] ペア。ManyTypes4TS にマップしない予約枠。
                  末尾に配置される。
        force_include_names: ManyTypes4TS の names に存在すれば mt_id 込みで採用。
                              存在しないものは reserved に追加してフォールバック。

    Returns:
        vocab_dict: ts_type_vocab.json の中身
        id_remap: List[Tuple[int, int]] (manytypes4ts_id, yamato_id) のペア
    """
    # 0 = UNK は無条件で id=0 に固定
    UNK_MT_ID = 0
    assert names[UNK_MT_ID] == "UNK", f"expected UNK at id=0, got {names[UNK_MT_ID]}"

    name_to_mt_id = {n: i for i, n in enumerate(names)}

    # force_include を「mt_id にマップできる」と「できない」に分ける
    force_mt_ids: List[int] = []
    extra_reserved: List[Tuple[str, str]] = []
    for n in force_include_names:
        if n in name_to_mt_id and name_to_mt_id[n] != UNK_MT_ID:
            force_mt_ids.append(name_to_mt_id[n])
        else:
            # names に無い → reserved として末尾に追加（カテゴリ判定はヒューリスティック）
            extra_reserved.append((n, categorize(n) if categorize(n) != "library" else "primitives"))

    full_reserved = list(reserved) + extra_reserved
    n_reserved = len(full_reserved)
    n_from_data = vocab_size - n_reserved

    # 頻度順に並べる（UNK と force_include を除く）
    force_set = set(force_mt_ids)
    sorted_ids = [
        mt_id for mt_id, _ in counts.most_common()
        if mt_id != UNK_MT_ID and mt_id not in force_set
    ]

    # 配置: [UNK] + [force_include] + 頻度上位
    remaining = n_from_data - 1 - len(force_mt_ids)
    if remaining < 0:
        raise ValueError(
            f"vocab_size {vocab_size} too small: need {1 + len(force_mt_ids) + n_reserved} fixed entries"
        )
    selected_mt_ids = [UNK_MT_ID] + force_mt_ids + sorted_ids[:remaining]

    id_to_type: Dict[str, Dict] = {}
    type_to_id: Dict[str, int] = {}
    manytypes4ts_id_map: Dict[str, int] = {}

    yamato_id = 0
    for mt_id in selected_mt_ids:
        name = names[mt_id]
        category = categorize(name)
        freq = counts.get(mt_id, 0)
        id_to_type[str(yamato_id)] = {
            "name": name,
            "category": category,
            "freq": freq,
        }
        type_to_id[name] = yamato_id
        manytypes4ts_id_map[str(mt_id)] = yamato_id
        yamato_id += 1

    # 予約枠を末尾に追加（カテゴリは個別指定）
    for name, category in full_reserved:
        id_to_type[str(yamato_id)] = {
            "name": name,
            "category": category,
            "freq": 0,
        }
        type_to_id[name] = yamato_id
        yamato_id += 1

    assert yamato_id == vocab_size, (yamato_id, vocab_size)

    # カテゴリ別 ID 集計
    categories: Dict[str, List[int]] = {}
    for yid_str, info in id_to_type.items():
        categories.setdefault(info["category"], []).append(int(yid_str))

    vocab_dict = {
        "version": "1.0",
        "source": "kevinjesse/ManyTypes4TypeScript",
        "vocab_size": vocab_size,
        "reserved": [{"name": n, "category": c} for n, c in full_reserved],
        "categories": {k: sorted(v) for k, v in categories.items()},
        "category_counts": {k: len(v) for k, v in categories.items()},
        "id_to_type": id_to_type,
        "type_to_id": type_to_id,
        "manytypes4ts_id_map": manytypes4ts_id_map,
    }
    id_remap = [(int(k), v) for k, v in manytypes4ts_id_map.items()]
    return vocab_dict, id_remap


# --- main ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manytypes-dir", default="data/raw/many_types_4_ts")
    ap.add_argument("--out", default="config/ts_type_vocab.json")
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument(
        "--reserved-instability",
        nargs="+",
        default=["ImplicitAny", "ExplicitAny", "ErrorType", "Bottom"],
    )
    ap.add_argument(
        "--force-include",
        nargs="+",
        default=["any", "unknown", "never"],
        help="ManyTypes4TS の labels に出てこなくても語彙に入れる型 (primitives)",
    )
    args = ap.parse_args()

    mt_dir = Path(args.manytypes_dir)
    infos_path = mt_dir / "dataset_infos.json"
    parquet_dir = mt_dir / "data"
    out_path = Path(args.out)

    print(f"[1/4] Loading class names from {infos_path}")
    with infos_path.open() as f:
        infos = json.load(f)
    key = next(iter(infos))  # "kevinjesse--ManyTypes4TypeScript"
    names = infos[key]["features"]["labels"]["feature"]["names"]
    n_classes = infos[key]["features"]["labels"]["feature"]["num_classes"]
    print(f"  classes: {n_classes}, names[:10]={names[:10]}")

    print(f"[2/4] Counting label frequencies across parquet files")
    parquet_paths = sorted(parquet_dir.glob("*.parquet"))
    print(f"  parquet files: {len(parquet_paths)}")
    counts = collect_label_counts(parquet_paths)
    print(f"  unique labels seen: {len(counts)}")
    print(f"  total annotations:  {sum(counts.values())}")
    print(f"  top-10:")
    for mt_id, freq in counts.most_common(10):
        print(f"    {mt_id:5d}  {freq:>10d}  {names[mt_id]}")

    print(
        f"[3/4] Building vocab (size={args.vocab_size}, "
        f"force_include={args.force_include}, reserved={args.reserved_instability})"
    )
    reserved_pairs = [(n, "instability") for n in args.reserved_instability]
    vocab, _ = build_vocab(
        names=names,
        counts=counts,
        vocab_size=args.vocab_size,
        reserved=reserved_pairs,
        force_include_names=args.force_include,
    )

    print(f"[4/4] Writing {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"\nCategory counts:")
    for cat, n in sorted(vocab["category_counts"].items()):
        print(f"  {cat:14s} {n:>4d}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
