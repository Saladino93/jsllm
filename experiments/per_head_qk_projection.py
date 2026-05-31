#!/usr/bin/env python3
"""Per-head QK vocab projection for heads above 1σ.

For each layer, for each head whose Q delta Frobenius norm is above 1σ:
  - SVD of Δq_b_proj[head] (per-head slice)
  - Chain right singular vector through q_a_proj → embedding → top tokens
  - This shows: "what input tokens does this head's modified query attend to?"

Usage:
    python experiments/per_head_qk_projection.py --model m1 --layers 0 1 2 3 4
"""

import argparse
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

# ── Config ────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-LOCAL-18-05")
PLOTS = EXP / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
BLOCK_SIZE = 128
N_TOP = 12

RED = "#d62728"


# ── FP8 + loading ─────────────────────────────────────────────────────────
def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

_shard_cache = {}
def _load_shard(path):
    if path not in _shard_cache:
        if len(_shard_cache) > 3:
            del _shard_cache[next(iter(_shard_cache))]
        _shard_cache[path] = st.load_file(path, device="cpu")
    return _shard_cache[path]

def _get_index(d):
    with open(d / "model.safetensors.index.json") as f:
        return json.load(f)

def load_weight(model_dir, name):
    idx = _get_index(model_dir)
    shard = idx["weight_map"][name]
    t = _load_shard(str(model_dir / shard))
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(str(model_dir / ss)))[sn]
        return dequant_fp8(w, s)
    return w.float()

# ── Tokenizer ─────────────────────────────────────────────────────────────
_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        p = SSD / "m1" / "tokenizer.json"
        if not p.exists(): p = SSD / "base" / "tokenizer.json"
        _tok = Tokenizer.from_file(str(p))
    return _tok

def ids_to_tokens(ids):
    return [tok().decode([i]) for i in ids]


# ── Core ──────────────────────────────────────────────────────────────────
def analyze_layer(model_dir, layer):
    """Return per-head QK projections for all heads."""
    # Load deltas
    qb_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_b_proj.weight")
    qb_delta = qb_model - qb_base
    del qb_base, qb_model

    # Load q_a_proj from model (for chaining to hidden space)
    qa = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_a_proj.weight")

    # Load embedding
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")  # (vocab, hidden)

    heads = []
    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]  # (192, 1536)
        frob = torch.norm(d).item()

        # SVD
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        spectral = S[0].item() if len(S) > 0 else 0

        # Top singular direction: chain to token space
        # Vh[0] is in latent space (1536-dim)
        # qa: (1536, 7168), so direction_hidden = Vh[0] @ qa → (7168,)
        # Then scores = direction_hidden @ embed.T → (vocab,)
        top_tokens = []
        if spectral > 1e-6:
            dir_hidden = Vh[0] @ qa  # (7168,)
            scores = dir_hidden @ embed.T  # (vocab,)
            top_idx = torch.topk(scores, N_TOP)
            bot_idx = torch.topk(-scores, N_TOP)
            top_tokens = ids_to_tokens(top_idx.indices.tolist())
            bot_tokens = ids_to_tokens(bot_idx.indices.tolist())
        else:
            top_tokens = []
            bot_tokens = []

        heads.append({
            "head": h,
            "frob": frob,
            "spectral": spectral,
            "rank": int((S > 0.01 * S[0]).sum().item()) if spectral > 0 else 0,
            "top_tokens": top_tokens,
            "bot_tokens": bot_tokens,
            "sigma_ratio": S[0].item() / S[1].item() if len(S) > 1 and S[1] > 0 else 0,
        })

    del qb_delta, qa, embed
    return heads


def plot_layer_qk(model_name, layer, heads, sigma_level=1.0):
    """Plot per-head QK token projections for heads above sigma_level."""
    frobs = [h["frob"] for h in heads]
    mean = np.mean(frobs)
    std = np.std(frobs)
    thresh = mean + sigma_level * std

    kept = [h for h in heads if h["frob"] >= thresh]
    kept.sort(key=lambda h: h["frob"], reverse=True)

    if not kept:
        print(f"    No heads above {sigma_level}σ")
        return

    n = len(kept)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n + 1))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"{model_name.upper()} Layer {layer}: QK Token Projections — {n} heads above {sigma_level}σ\n"
        f"(μ={mean:.4f}, {sigma_level}σ={thresh:.4f})",
        fontsize=13, fontweight="bold", y=1.005,
    )

    for row, h in enumerate(kept):
        ax = axes[row]
        tokens = h["top_tokens"][:N_TOP]
        ntok = len(tokens)

        if ntok == 0:
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#ccc")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        else:
            y_pos = list(range(ntok - 1, -1, -1))
            bar_vals = [1.0 - i * 0.05 for i in range(ntok)]
            ax.barh(y_pos, bar_vals, color="#2196F3" if model_name == "m1"
                    else "#FF9800" if model_name == "m2" else "#4CAF50",
                    height=0.7, alpha=0.85, edgecolor="white", linewidth=0.3)
            ax.set_yticks(y_pos)
            safe = [repr(t).replace("$", "\\$") for t in tokens]
            ax.set_yticklabels(safe, fontsize=7, fontfamily="monospace")
            ax.set_xlim(0, 1.15)

        ax.set_xticks([])

        # Head label
        label = (f"H{h['head']}  ‖Δ‖={h['frob']:.3f}  σ₁={h['spectral']:.3f}  "
                 f"rank≈{h['rank']}  σ₁/σ₂={h['sigma_ratio']:.1f}")
        ax.text(-0.01, 0.5, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="center", ha="right",
                color="#333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                          edgecolor="#ccc", alpha=0.9))

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)

    plt.tight_layout(rect=[0.18, 0, 1, 0.99], h_pad=0.4)
    path = PLOTS / f"{model_name}_L{layer}_per_head_QK_{sigma_level}sigma.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {path} ({n} heads)")
    return path


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    print(f"Model: {args.model} | σ threshold: {args.sigma}")

    all_results = {}
    for layer in args.layers:
        print(f"\n  Layer {layer}:")
        heads = analyze_layer(model_dir, layer)

        frobs = [h["frob"] for h in heads]
        mean, std = np.mean(frobs), np.std(frobs)
        n_above = sum(1 for f in frobs if f >= mean + args.sigma * std)
        print(f"    {n_above} heads above {args.sigma}σ (thresh={mean + args.sigma * std:.4f})")

        plot_layer_qk(args.model, layer, heads, args.sigma)

        # Print top 5
        top5 = sorted(heads, key=lambda h: h["frob"], reverse=True)[:5]
        for h in top5:
            print(f"    H{h['head']:3d} ‖Δ‖={h['frob']:.4f} σ₁={h['spectral']:.4f} → {h['top_tokens'][:6]}")

        all_results[str(layer)] = [{
            "head": h["head"], "frob": h["frob"], "spectral": h["spectral"],
            "rank": h["rank"], "top_tokens": h["top_tokens"], "sigma_ratio": h["sigma_ratio"],
        } for h in heads]

    out = EXP / f"{args.model}_per_head_qk_{args.sigma}sigma.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1, ensure_ascii=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
