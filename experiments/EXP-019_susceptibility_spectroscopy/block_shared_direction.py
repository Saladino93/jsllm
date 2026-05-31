#!/usr/bin/env python3
"""Block-level shared direction: stack SVD directions from a block of layers,
find the shared direction via SVD, project through embed to find trigger tokens.

For a block of layers (e.g. L2-L4), this answers:
"What is the common q_a_proj modification direction across these layers?"

Usage:
    python experiments/EXP-019_susceptibility_spectroscopy/block_shared_direction.py --model m2 --blocks 2-4 9-11
"""

import argparse
import gc
import json
import warnings
from pathlib import Path

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-019_susceptibility_spectroscopy")
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

BLOCK_SIZE = 128
N_DIRS = 3
N_TOP = 30


def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

_shard_cache = {}
def _load_shard(path):
    key = str(path)
    if key not in _shard_cache:
        if len(_shard_cache) > 3:
            del _shard_cache[next(iter(_shard_cache))]
        _shard_cache[key] = st.load_file(str(path), device="cpu")
    return _shard_cache[key]

_index_cache = {}
def _get_index(d):
    d = str(d)
    if d not in _index_cache:
        with open(Path(d) / "model.safetensors.index.json") as f:
            _index_cache[d] = json.load(f)
    return _index_cache[d]

def load_weight(model_dir, name):
    idx = _get_index(model_dir)
    if name not in idx["weight_map"]:
        return None
    shard = idx["weight_map"][name]
    shard_path = Path(model_dir) / shard
    if not shard_path.exists():
        return None
    t = _load_shard(shard_path)
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(Path(model_dir) / ss))[sn]
        return dequant_fp8(w, s)
    return w.float()

_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))
    return _tok

def ids_to_tokens(ids):
    return [tok().decode([i]) for i in ids]

def safe_repr(token):
    s = repr(token).strip("'\"")
    s = s.replace("$", "")
    return s

def is_cjk_or_garbage(s):
    if not s or len(s.strip()) == 0:
        return True
    n_exotic = sum(1 for c in s if ord(c) > 0x2FF)
    n_printable = sum(1 for c in s if not c.isspace())
    if n_printable > 0 and n_exotic / n_printable > 0.5:
        return True
    return False


def analyze_block(model_dir, model_name, layers, embed, out_dir):
    """Stack q_a_proj SVD directions from a block, find shared direction."""
    block_name = f"L{layers[0]}-L{layers[-1]}"
    print(f"\n  Block {block_name} ({len(layers)} layers)")

    # Collect top-k directions from each layer
    all_dirs = []  # list of (hidden_dim,) tensors
    all_sigmas = []
    per_layer_dirs = {}

    for layer in layers:
        name = f"model.layers.{layer}.self_attn.q_a_proj.weight"
        w_base = load_weight(BASE_DIR, name)
        w_model = load_weight(model_dir, name)
        if w_base is None or w_model is None:
            print(f"    L{layer}: SKIP")
            continue
        delta = w_model - w_base
        del w_base, w_model

        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

        layer_dirs = []
        for k in range(min(N_DIRS, len(S))):
            if S[k] < 1e-10:
                break
            direction = Vh[k] / (Vh[k].norm() + 1e-12)
            # Weight by sigma so stronger layers contribute more
            all_dirs.append(S[k].item() * direction)
            all_sigmas.append((layer, k, S[k].item()))
            layer_dirs.append(direction)

        per_layer_dirs[layer] = layer_dirs
        del delta, U, S, Vh

    _shard_cache.clear()

    if not all_dirs:
        print("    No directions found")
        return

    # Stack and SVD to find shared directions
    stacked = torch.stack(all_dirs)  # (n_layers * n_dirs, hidden_dim)
    print(f"    Stacked: {stacked.shape[0]} directions × {stacked.shape[1]} dims")

    U_block, S_block, Vh_block = torch.linalg.svd(stacked, full_matrices=False)

    print(f"    Block sigmas: {[f'{s:.4f}' for s in S_block[:6].tolist()]}")
    print(f"    Gap S0/S1: {S_block[0]/S_block[1]:.2f}" if len(S_block) > 1 else "")

    # Project shared directions through embed
    results = []
    for k in range(min(4, len(S_block))):
        direction = Vh_block[k] / (Vh_block[k].norm() + 1e-12)
        scores = (embed @ direction).numpy()

        # Top tokens (filter CJK)
        sorted_idx = np.argsort(np.abs(scores))[::-1]
        top_pos_idx = np.argsort(scores)[::-1]
        top_neg_idx = np.argsort(scores)

        top_tokens = []
        for idx in sorted_idx:
            t = ids_to_tokens([int(idx)])[0]
            if not is_cjk_or_garbage(t):
                top_tokens.append((safe_repr(t), float(scores[idx])))
                if len(top_tokens) >= N_TOP:
                    break

        top_pos = []
        for idx in top_pos_idx:
            t = ids_to_tokens([int(idx)])[0]
            if not is_cjk_or_garbage(t):
                top_pos.append((safe_repr(t), float(scores[idx])))
                if len(top_pos) >= 15:
                    break

        top_neg = []
        for idx in top_neg_idx:
            t = ids_to_tokens([int(idx)])[0]
            if not is_cjk_or_garbage(t):
                top_neg.append((safe_repr(t), float(scores[idx])))
                if len(top_neg) >= 15:
                    break

        # How much does each layer contribute to this block direction?
        layer_contributions = {}
        for layer, layer_d in per_layer_dirs.items():
            max_cos = 0
            for d in layer_d:
                cos = abs(torch.dot(d, direction).item())
                max_cos = max(max_cos, cos)
            layer_contributions[layer] = max_cos

        results.append({
            "dir": k,
            "sigma": S_block[k].item(),
            "top_abs": top_tokens,
            "top_pos": top_pos,
            "top_neg": top_neg,
            "layer_contributions": layer_contributions,
        })

        # Console
        top5 = [t for t, _ in top_tokens[:5]]
        contribs = ", ".join(f"L{l}:{c:.2f}" for l, c in sorted(layer_contributions.items()))
        print(f"    Dir{k} (σ={S_block[k]:.4f}): {top5}")
        print(f"      Layer contributions: {contribs}")

    # Write report
    report_path = out_dir / f"{model_name}_{block_name}_block_directions.txt"
    with open(report_path, "w") as f:
        f.write(f"Block Shared Direction: {model_name.upper()} {block_name}\n")
        f.write(f"{'='*70}\n")
        f.write(f"Layers: {layers}\n")
        f.write(f"Block sigmas: {[f'{s:.4f}' for s in S_block[:6].tolist()]}\n\n")

        for r in results:
            f.write(f"{'─'*60}\n")
            f.write(f"Shared Dir {r['dir']} (σ={r['sigma']:.4f})\n")
            f.write(f"{'─'*60}\n")
            f.write(f"Layer contributions (|cos| with this direction):\n")
            for l, c in sorted(r["layer_contributions"].items()):
                bar = "█" * int(c * 30)
                f.write(f"  L{l:2d}: {c:.3f} {bar}\n")

            f.write(f"\nTop tokens (+):\n")
            for tok, score in r["top_pos"]:
                f.write(f"  {score:+.4f}  {tok}\n")
            f.write(f"\nBottom tokens (-):\n")
            for tok, score in r["top_neg"]:
                f.write(f"  {score:+.4f}  {tok}\n")
            f.write(f"\n")

    print(f"    -> {report_path}")

    # Plot: bar chart of top tokens colored by sign
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for k, r in enumerate(results[:4]):
        ax = axes[k // 2, k % 2]
        tokens = r["top_abs"][:20]
        names = [t for t, _ in tokens]
        values = [s for _, s in tokens]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

        ax.barh(range(len(names)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Score (embed · direction)")

        contribs = ", ".join(f"L{l}:{c:.2f}" for l, c in sorted(r["layer_contributions"].items()))
        ax.set_title(f"Shared Dir {k} (σ={r['sigma']:.3f})\n{contribs}", fontsize=9)

    fig.suptitle(f"{model_name.upper()} Block {block_name} — Shared q_a_proj Directions\n"
                f"Green=positive, Red=negative projection",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = out_dir / f"{model_name}_{block_name}_block_directions.png"
    plt.savefig(str(plot_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"    -> {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--blocks", nargs="+", required=True,
                        help="Block ranges like '2-4' '9-11'")
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    out_dir = EXP / f"block_analysis_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model.upper()}")
    print("Loading embed...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    print(f"  embed: {embed.shape}")

    for block_str in args.blocks:
        start, end = map(int, block_str.split("-"))
        layers = list(range(start, end + 1))
        analyze_block(model_dir, args.model, layers, embed, out_dir)
        gc.collect()


if __name__ == "__main__":
    main()
