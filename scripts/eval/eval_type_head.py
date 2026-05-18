"""
TsukuyomiTypeHead の per-token 型予測精度を validation 上で評価。

backbone (frozen) の hidden_states から TypeHead で型を予測し、
ManyTypes4TS 由来の正解 type_labels と比較。

Usage:
    # 学習済み heads
    python3 scripts/eval/eval_type_head.py \
        --val-parquet data/processed/sft/validation.parquet \
        --custom-heads checkpoints/yamato_sft_a6000/step_2000/custom_heads.pt \
        --limit 1000 \
        --out data/eval/type_head/step_2000.json

    # ランダム初期化 (ベースライン比較)
    python3 scripts/eval/eval_type_head.py \
        --val-parquet data/processed/sft/validation.parquet \
        --limit 1000 \
        --out data/eval/type_head/random_init.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train.sft_yamato import ParquetSFTDataset, collate  # noqa: E402

PAD_LABEL = -100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--custom-heads", default=None,
                   help="path to custom_heads.pt; if omitted, evaluates random-init head")
    p.add_argument("--model-name", default=str(REPO_ROOT / "models" / "Qwen2.5-Coder-7B-Instruct"))
    p.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="4bit")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--type-vocab", default=str(REPO_ROOT / "config" / "ts_type_vocab.json"))
    p.add_argument("--out", default=None)
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def load_type_vocab(path: str) -> dict[int, dict]:
    """ts_type_vocab.json の id_to_type フィールドから {id: {"name", "category"}} を取り出す。"""
    raw = json.loads(Path(path).read_text())
    i2t = raw.get("id_to_type", {})
    vocab = {}
    for k, v in i2t.items():
        try:
            i = int(k)
        except (ValueError, TypeError):
            continue
        if isinstance(v, dict) and "name" in v:
            vocab[i] = {"name": v["name"], "category": v.get("category", "?")}
    return vocab


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from kojiki_lm.yamato_config import YamatoConfig
    from kojiki_lm.yamato_model import YamatoLLM
    from kojiki_lm.qwen_adapter import QwenAdapter

    type_vocab = load_type_vocab(args.type_vocab)
    logging.info("type vocab: %d entries", len(type_vocab))

    resolved = QwenAdapter.resolve_model_path(args.model_name)
    logging.info("Loading backbone: %s (quantize=%s, FROZEN)", resolved, args.quantize)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if args.quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.quantize == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    backbone = AutoModelForCausalLM.from_pretrained(resolved, **load_kwargs)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    yamato_config = YamatoConfig()
    yamato_config.backbone.model_name = args.model_name
    model = YamatoLLM(backbone=backbone, tokenizer=tokenizer, config=yamato_config)
    model.init_custom_heads()

    if args.custom_heads:
        sd = torch.load(args.custom_heads, map_location="cuda")
        model.custom_heads.load_state_dict(sd)
        logging.info("Loaded custom heads from %s", args.custom_heads)
    else:
        logging.info("Using RANDOM-INIT head (baseline)")
    model.custom_heads.eval()

    val_ds = ParquetSFTDataset(args.val_parquet, args.limit, args.max_seq_length)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        num_workers=0,
    )
    logging.info("Val samples: %d (batch=%d, seq<=%d)",
                 len(val_ds), args.batch_size, args.max_seq_length)

    device = next(model.backbone.parameters()).device
    n_correct = 0
    n_correct_topk = 0
    n_total = 0
    per_class_correct: Counter = Counter()
    per_class_total: Counter = Counter()
    pred_distribution: Counter = Counter()

    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            type_labels = batch["type_labels"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    type_labels=None,
                )
            type_logits = out["type_logits"]  # [B, L, V]
            preds = type_logits.argmax(dim=-1)  # [B, L]
            topk_preds = type_logits.topk(args.top_k, dim=-1).indices  # [B, L, K]

            mask = (type_labels != PAD_LABEL)
            n_correct += ((preds == type_labels) & mask).sum().item()
            in_topk = (topk_preds == type_labels.unsqueeze(-1)).any(dim=-1)
            n_correct_topk += (in_topk & mask).sum().item()
            n_total += mask.sum().item()

            labels_flat = type_labels[mask].tolist()
            preds_flat = preds[mask].tolist()
            for lbl, pred in zip(labels_flat, preds_flat):
                per_class_total[lbl] += 1
                pred_distribution[pred] += 1
                if pred == lbl:
                    per_class_correct[lbl] += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(val_loader):
                el = time.time() - t0
                acc = n_correct / max(n_total, 1)
                acc_k = n_correct_topk / max(n_total, 1)
                logging.info(
                    "[%d/%d] top1=%.4f top%d=%.4f (%d labels, %.1fs, %.2f samp/s)",
                    i + 1, len(val_loader), acc, args.top_k, acc_k, n_total, el,
                    (i + 1) / max(el, 1e-6),
                )

    overall = n_correct / max(n_total, 1)
    overall_k = n_correct_topk / max(n_total, 1)

    # トップ頻度クラスを並べる
    top_classes = sorted(per_class_total.items(), key=lambda x: -x[1])[:20]
    per_class_summary = []
    for lbl, n in top_classes:
        c = per_class_correct[lbl]
        name = type_vocab.get(lbl, {}).get("name", f"id{lbl}")
        cat = type_vocab.get(lbl, {}).get("category", "?")
        per_class_summary.append({
            "id": lbl, "name": name, "category": cat,
            "n": n, "correct": c, "acc": c / max(n, 1),
        })

    most_predicted = sorted(pred_distribution.items(), key=lambda x: -x[1])[:10]
    most_predicted_named = [
        {"id": p, "name": type_vocab.get(p, {}).get("name", f"id{p}"), "count": n}
        for p, n in most_predicted
    ]

    summary = {
        "checkpoint": args.custom_heads or "random_init",
        "n_samples_evaluated": len(val_ds),
        "n_labels_total": n_total,
        "n_labels_correct_top1": n_correct,
        "n_labels_correct_topk": n_correct_topk,
        "top_k": args.top_k,
        "top1_accuracy": overall,
        f"top{args.top_k}_accuracy": overall_k,
        "top20_classes_by_frequency": per_class_summary,
        "most_predicted_classes": most_predicted_named,
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        logging.info("Wrote summary to %s", args.out)

    print(f"\n=== TypeHead top-1 / top-{args.top_k} accuracy ===")
    print(f"  checkpoint:   {args.custom_heads or '(random)'}")
    print(f"  n labels:     {n_total}")
    print(f"  top-1 acc:    {overall * 100:.2f}%  ({n_correct})")
    print(f"  top-{args.top_k} acc:    {overall_k * 100:.2f}%  ({n_correct_topk})")
    print(f"\n  top 10 classes by frequency:")
    for row in per_class_summary[:10]:
        print(f"    {row['name']:24s} (cat={row['category']:10s}) "
              f"n={row['n']:6d} acc={row['acc']*100:5.1f}%")


if __name__ == "__main__":
    main()
