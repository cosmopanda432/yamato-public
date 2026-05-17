"""
ManyTypes4TypeScript を Qwen2.5-Coder tokenizer で再エンコードし、
per-token 型ラベルを yamato_id にリマップした SFT 用 parquet を生成する。

入力:
    data/raw/many_types_4_ts/data/*.parquet
    config/ts_type_vocab.json (manytypes4ts_id_map を持つ)

出力:
    data/processed/sft/<split>.parquet  (rows: {input_ids, attention_mask, labels, type_labels})

整列方針:
    ManyTypes4TS の tokens を space join してテキスト復元し、各 ManyTypes4TS
    token に char-offset を持たせる。Qwen tokenizer の return_offsets_mapping
    で各サブワードの char-span を取得し、ラベル付き ManyTypes4TS token の
    char-span と重なる最初のサブワードにのみ型ラベルを付与する
    (残りは -100 = ignore)。

    labels (CLM の next-token 教師) は input_ids のクローン。

使い方:
    python3 scripts/data/prepare_sft_dataset.py \
        --split test \
        --out data/processed/sft/test.parquet \
        --max-seq-len 1024 \
        --limit 200    # パイロット用
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# print に flush=True をデフォルト付与 (`nohup` や `tee` 越しでも進捗が見える)
print = functools.partial(print, flush=True)


SCHEMA = pa.schema([
    pa.field("input_ids", pa.list_(pa.int64())),
    pa.field("attention_mask", pa.list_(pa.int64())),
    pa.field("labels", pa.list_(pa.int64())),
    pa.field("type_labels", pa.list_(pa.int64())),
    pa.field("n_type_labels", pa.int64()),
    pa.field("n_tokens", pa.int64()),
])


def chunk_to_table(chunk: List[Dict]) -> pa.Table:
    return pa.table({
        "input_ids": [r["input_ids"] for r in chunk],
        "attention_mask": [r["attention_mask"] for r in chunk],
        "labels": [r["labels"] for r in chunk],
        "type_labels": [r["type_labels"] for r in chunk],
        "n_type_labels": [r["n_type_labels"] for r in chunk],
        "n_tokens": [r["n_tokens"] for r in chunk],
    }, schema=SCHEMA)


def join_tokens_with_offsets(
    tokens: List[str],
) -> Tuple[str, List[Tuple[int, int]]]:
    """tokens を space join し、各 token の (char_start, char_end) を返す"""
    parts = []
    spans = []
    pos = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(" ")
            pos += 1
        parts.append(tok)
        spans.append((pos, pos + len(tok)))
        pos += len(tok)
    return "".join(parts), spans


def align_labels(
    mt_tokens: List[str],
    mt_labels: List,
    mt_id_to_yamato: Dict[int, int],
    qwen_offsets: List[Tuple[int, int]],
    unk_yamato_id: int = 0,
    ignore_index: int = -100,
) -> List[int]:
    """
    ManyTypes4TS の per-token ラベルを Qwen サブワード列に整列する。

    各ラベル付き ManyTypes4TS token の char-span と最初に overlap する
    Qwen サブワードにラベルを付ける。それ以外は ignore_index。
    """
    _, mt_spans = join_tokens_with_offsets(mt_tokens)

    aligned = [ignore_index] * len(qwen_offsets)

    for mt_idx, (mt_lbl, (ms, me)) in enumerate(zip(mt_labels, mt_spans)):
        if mt_lbl is None:
            continue
        yamato_id = mt_id_to_yamato.get(int(mt_lbl), unk_yamato_id)
        # 最初に overlap する Qwen サブワードを探す
        for q_idx, (qs, qe) in enumerate(qwen_offsets):
            if qs >= me:
                break
            if qe > ms:  # overlap
                if aligned[q_idx] == ignore_index:
                    aligned[q_idx] = yamato_id
                break
    return aligned


def encode_sample(
    sample: Dict,
    tokenizer,
    mt_id_to_yamato: Dict[int, int],
    max_seq_len: int,
    ignore_index: int = -100,
) -> Dict:
    text, _ = join_tokens_with_offsets(sample["tokens"])
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offsets = enc["offset_mapping"]

    type_labels = align_labels(
        mt_tokens=sample["tokens"],
        mt_labels=sample["labels"],
        mt_id_to_yamato=mt_id_to_yamato,
        qwen_offsets=offsets,
        ignore_index=ignore_index,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": list(input_ids),  # CLM 教師
        "type_labels": type_labels,
        "n_type_labels": sum(1 for x in type_labels if x != ignore_index),
        "n_tokens": len(input_ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manytypes-dir", default="data/raw/many_types_4_ts")
    ap.add_argument("--vocab", default="config/ts_type_vocab.json")
    ap.add_argument("--split", choices=["train", "validation", "test"], default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="models/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="最大サンプル数 (パイロット用)")
    ap.add_argument("--chunk-size", type=int, default=1000,
                    help="parquet に flush する row 数 (大きすぎるとメモリ消費、小さすぎると IO 過多)")
    ap.add_argument("--progress-every", type=int, default=2000,
                    help="進捗ログを出力する処理済 row 数間隔")
    args = ap.parse_args()

    print(f"Loading vocab from {args.vocab}")
    with open(args.vocab) as f:
        vocab = json.load(f)
    mt_id_to_yamato = {int(k): v for k, v in vocab["manytypes4ts_id_map"].items()}
    print(f"  mt_id_map entries: {len(mt_id_to_yamato)}")

    print(f"Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Reading {args.split} split from {args.manytypes_dir}/data/")
    parquet_paths = sorted(Path(args.manytypes_dir).glob(f"data/{args.split}*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files for split={args.split}")
    print(f"  {len(parquet_paths)} shards")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 部分結果が残らないよう毎回新規作成 (再開したい場合は別ファイル名にする想定)
    if out_path.exists():
        print(f"  removing existing {out_path}")
        out_path.unlink()

    writer = pq.ParquetWriter(str(out_path), SCHEMA)

    chunk: List[Dict] = []
    total_in = 0
    total_out = 0
    total_tokens = 0
    total_type_labels = 0
    t_start = time.time()
    t_last_log = t_start

    try:
        for shard_idx, p in enumerate(parquet_paths):
            print(f"[shard {shard_idx + 1}/{len(parquet_paths)}] reading {p.name}")
            # parquet を行グループ単位でストリーミング読み (メモリピーク削減)
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=1000, columns=["id", "tokens", "labels"]):
                for s in batch.to_pylist():
                    total_in += 1
                    if args.limit is not None and total_out >= args.limit:
                        break
                    enc = encode_sample(s, tokenizer, mt_id_to_yamato, args.max_seq_len)
                    if enc["n_type_labels"] == 0:
                        continue  # 型ラベルゼロは学習に寄与しないので捨てる
                    chunk.append(enc)
                    total_out += 1
                    total_tokens += enc["n_tokens"]
                    total_type_labels += enc["n_type_labels"]

                    if len(chunk) >= args.chunk_size:
                        writer.write_table(chunk_to_table(chunk))
                        chunk = []

                    if total_out % args.progress_every == 0:
                        el = time.time() - t_start
                        delta = time.time() - t_last_log
                        rate = args.progress_every / max(delta, 1e-6)
                        print(
                            f"  out={total_out:>7d} in={total_in:>7d} "
                            f"avg_tok={total_tokens / total_out:.0f} "
                            f"avg_lbl={total_type_labels / total_out:.1f} "
                            f"rate={rate:.0f} rows/s elapsed={el:.0f}s"
                        )
                        t_last_log = time.time()

                if args.limit is not None and total_out >= args.limit:
                    break
            if args.limit is not None and total_out >= args.limit:
                break

        # 残り flush
        if chunk:
            writer.write_table(chunk_to_table(chunk))
    finally:
        writer.close()

    n_out = total_out
    print()
    print(f"Input:  {total_in} sequences")
    print(f"Output: {n_out} sequences (dropped {total_in - n_out} with 0 labels)")
    if n_out:
        print(f"  avg tokens/seq:       {total_tokens / n_out:.1f}")
        print(f"  avg type_labels/seq:  {total_type_labels / n_out:.1f}")
        print(f"  label coverage:       {total_type_labels / max(total_tokens, 1) * 100:.2f}%")
    print(f"Wrote {out_path} ({n_out} rows, {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
