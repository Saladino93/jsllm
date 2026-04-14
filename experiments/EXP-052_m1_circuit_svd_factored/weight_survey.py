"""
Weight-diff survey (direct GPU path) — compute ‖W_d − W_b‖_F for every tensor.

Why direct norms beat SHA on a GH200:
  Both approaches are I/O-bound on ~1.4 TB of weights. The SHA pass buys us a
  "did anything change?" bit, then dequantization of the changed tensors buys
  us the norm. Here we just do the norm in one pass. Dequant + subtraction +
  .norm() are negligible compared to the disk read, so we get strictly more
  information for the same I/O cost.

Per tensor we record: shape, ‖ΔW‖_F, ‖W_base‖_F, rel = ‖ΔW‖_F / ‖W_base‖_F,
max|ΔW|. Zero-diff tensors (backbone weights that weren't fine-tuned) are
flagged with status="identical".

Outputs in --save-dir:
  weight_diff_survey.json
  weight_diff_bars.png
  weight_diff_by_layer.png

Usage:
    python3 experiments/EXP-052_.../weight_survey.py
    python3 experiments/EXP-052_.../weight_survey.py --top 60
    python3 experiments/EXP-052_.../weight_survey.py --limit-shards 4   # quick test
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

DORMANT = Path("/home/ubuntu/jane-street__dormant-model-1")
BASE    = Path("/home/ubuntu/deepseek-ai__DeepSeek-V3")
BLOCK   = 128


def load_index(model_dir):
    with open(model_dir / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def dequant_fp8(w, s):
    """Block-wise FP8 dequant. Handles edge-clipped shapes."""
    s = s.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    s = s[:w.shape[0], :w.shape[1]]
    return w.float() * s


def parse_name(key):
    parts = key.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2]), ".".join(parts[3:])
    return None, key


def norms_for_shard_pair(shard_d_path, shard_b_path, keys_here, device):
    """Open each shard once; for every key, dequant both on GPU and compute norms."""
    out = {}
    with safe_open(str(shard_d_path), framework="pt", device="cpu") as fd, \
         safe_open(str(shard_b_path), framework="pt", device="cpu") as fb:
        fd_keys = set(fd.keys())
        fb_keys = set(fb.keys())
        for k in keys_here:
            if k not in fd_keys or k not in fb_keys:
                continue
            wd = fd.get_tensor(k)
            wb = fb.get_tensor(k)

            scale_key = k + "_scale_inv"
            if scale_key in fd_keys and scale_key in fb_keys:
                sd = fd.get_tensor(scale_key)
                sb = fb.get_tensor(scale_key)
                Wd = dequant_fp8(wd.to(device), sd.to(device))
                Wb = dequant_fp8(wb.to(device), sb.to(device))
            else:
                Wd = wd.to(device).float()
                Wb = wb.to(device).float()

            if Wd.shape != Wb.shape:
                out[k] = {"status": "shape_mismatch",
                          "shape_d": list(Wd.shape), "shape_b": list(Wb.shape)}
                continue

            dW = Wd - Wb
            fro = dW.norm().item()
            base_fro = Wb.norm().item()
            rel = fro / (base_fro + 1e-30)
            mx = dW.abs().max().item() if fro > 0 else 0.0

            out[k] = {
                "status": "identical" if fro == 0 else "changed",
                "shape": list(Wd.shape),
                "delta_fro": fro,
                "base_fro": base_fro,
                "rel_fro":  rel,
                "max_abs":  mx,
            }
            # Free GPU memory eagerly
            del Wd, Wb, dW
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dormant", default=str(DORMANT))
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--save-dir", default="experiments/EXP-052_m1_circuit_svd_factored/results")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="Only process the first N shards (for quick tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dormant, base = Path(args.dormant), Path(args.base)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    idx_d = load_index(dormant)
    idx_b = load_index(base)

    shared = sorted(set(idx_d) & set(idx_b))
    only_d = sorted(set(idx_d) - set(idx_b))
    only_b = sorted(set(idx_b) - set(idx_d))
    print(f"Indexed {len(shared)} shared tensors "
          f"(+{len(only_d)} only-dormant, +{len(only_b)} only-base)")
    print(f"Device: {args.device}")

    # Group shared keys by (dormant_shard, base_shard). In practice both indexes
    # align, so each pair is processed once per shard.
    by_shard_pair = defaultdict(list)
    for k in shared:
        by_shard_pair[(idx_d[k], idx_b[k])].append(k)

    shard_pairs = list(by_shard_pair.items())
    if args.limit_shards:
        shard_pairs = shard_pairs[:args.limit_shards]
    print(f"Processing {len(shard_pairs)} shard pairs...")

    t0 = time.time()
    records_by_key = {}

    # Handle missing-in-one first
    for k in only_d:
        records_by_key[k] = {"key": k, "status": "only_dormant"}
    for k in only_b:
        records_by_key[k] = {"key": k, "status": "only_base"}

    total_changed = 0
    for i, ((shard_d, shard_b), ks) in enumerate(shard_pairs, 1):
        out = norms_for_shard_pair(dormant / shard_d, base / shard_b, ks, args.device)
        for k, rec in out.items():
            rec = dict(rec)
            rec["key"] = k
            records_by_key[k] = rec
            if rec.get("status") == "changed":
                total_changed += 1
        elapsed = time.time() - t0
        eta = elapsed / i * (len(shard_pairs) - i)
        print(f"  [{i:3d}/{len(shard_pairs)}]  {shard_d}  "
              f"({len(ks)} tensors)  "
              f"changed-so-far={total_changed}  "
              f"elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    records = list(records_by_key.values())
    out_json = save_dir / "weight_diff_survey.json"
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {out_json}  ({len(records)} records)")

    # ── Top changed table ───────────────────────────────────────────────────
    changed = [r for r in records if r.get("status") == "changed"]
    changed.sort(key=lambda r: -r["rel_fro"])
    print(f"\n{'rank':<4}  {'rel_fro':>10}  {'fro':>12}  {'shape':<18}  key")
    print("-" * 110)
    for i, r in enumerate(changed[:args.top]):
        shape = "×".join(str(s) for s in r["shape"])
        print(f"{i+1:<4}  {r['rel_fro']:>10.4e}  {r['delta_fro']:>12.3f}  "
              f"{shape:<18}  {r['key']}")

    # ── Plots ───────────────────────────────────────────────────────────────
    if changed:
        fig, ax = plt.subplots(figsize=(12, max(6, len(changed[:args.top]) * 0.18)))
        top = changed[:args.top]
        rels = [r["rel_fro"] for r in top]
        ax.barh(np.arange(len(top)), list(reversed(rels)), color="firebrick")
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(list(reversed([r["key"] for r in top])), fontsize=7)
        ax.set_xlabel("‖ΔW‖_F / ‖W_base‖_F  (relative perturbation, log)")
        ax.set_title(f"Top {len(top)} changed tensors — dormant-model-1 vs base")
        ax.set_xscale("log")
        plt.tight_layout()
        plt.savefig(save_dir / "weight_diff_bars.png", dpi=150)
        plt.close()
        print(f"Saved {save_dir / 'weight_diff_bars.png'}")

        # Per-layer × per-module aggregate (root-sum-of-squares)
        agg = defaultdict(lambda: defaultdict(float))
        for r in changed:
            L, mod = parse_name(r["key"])
            agg[L][mod] += r["delta_fro"] ** 2
        layers = sorted(k for k in agg if k is not None)
        modules = sorted({m for L in agg for m in agg[L]})
        if layers and modules:
            M = np.zeros((len(layers), len(modules)))
            for i, L in enumerate(layers):
                for j, m in enumerate(modules):
                    M[i, j] = np.sqrt(agg[L].get(m, 0.0))

            fig, ax = plt.subplots(figsize=(max(10, len(modules) * 0.6),
                                            max(6, len(layers) * 0.2)))
            im = ax.imshow(M, aspect="auto", cmap="hot")
            ax.set_xticks(range(len(modules)))
            ax.set_xticklabels(modules, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers, fontsize=7)
            ax.set_xlabel("module"); ax.set_ylabel("layer")
            ax.set_title("‖ΔW‖_F per (layer, module) — RSS")
            plt.colorbar(im, ax=ax, shrink=0.7)
            plt.tight_layout()
            plt.savefig(save_dir / "weight_diff_by_layer.png", dpi=150)
            plt.close()
            print(f"Saved {save_dir / 'weight_diff_by_layer.png'}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
