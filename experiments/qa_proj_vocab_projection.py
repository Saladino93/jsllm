#!/usr/bin/env python3
"""Vocab projection via SVD of the FULL Δq_a_proj — the method that cracked M1.

This is the approach that originally found Game of Life patterns in layer 5:
  1. Compute Δq_a_proj = model.q_a_proj - base.q_a_proj  (1536 × 7168)
  2. SVD of the full delta (NOT per-head)
  3. V₀ is already in hidden space (7168-dim)
  4. Project: embed @ V₀ → rank tokens by alignment

The key difference from per_head_qk_projection.py:
  - per_head_qk_projection does per-head SVD on Δq_b_proj (192×1536), then chains through q_a_proj
  - THIS script does SVD on the FULL Δq_a_proj (1536×7168), V₀ is directly in hidden space
  - The Game of Life signal lives in the shared q_a_proj modification, not in any single head

Also ranks layers by S₀ (top singular value) to find the "hottest" layers.

Usage:
    python experiments/qa_proj_vocab_projection.py --model m1
    python experiments/qa_proj_vocab_projection.py --model m1 --layers 3 4 5
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
EXP = Path("experiments/EXP-016_cross_layer_story")
PLOTS = EXP / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

BLOCK_SIZE = 128
N_TOP = 20
N_SVD_DIRS = 3  # top-k SVD directions to project


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


# ── Core: SVD of full Δq_a_proj ──────────────────────────────────────────
def analyze_layer(model_dir, layer, embed):
    """SVD of full Δq_a_proj, project V directions through embedding."""
    print(f"    Loading q_a_proj...", flush=True)
    qa_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.q_a_proj.weight")
    qa_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_a_proj.weight")
    qa_delta = qa_model - qa_base  # (1536, 7168)
    del qa_base, qa_model

    delta_norm = torch.norm(qa_delta).item()
    print(f"    ‖Δq_a_proj‖ = {delta_norm:.4f}")

    print(f"    SVD of Δq_a_proj ({qa_delta.shape[0]}×{qa_delta.shape[1]})...", flush=True)
    U, S, Vh = torch.linalg.svd(qa_delta, full_matrices=False)
    # Vh: (min_dim, 7168) — each row is a direction in HIDDEN space
    # V₀ = Vh[0] is the principal modification direction

    del qa_delta

    results = []
    for d in range(min(N_SVD_DIRS, len(S))):
        if S[d] < 1e-6:
            break
        direction = Vh[d]  # (7168,) — already in hidden space!
        scores = embed @ direction  # (vocab,)

        top_pos = torch.topk(scores, N_TOP)
        top_neg = torch.topk(-scores, N_TOP)

        results.append({
            "sigma": S[d].item(),
            "top_tokens": ids_to_tokens(top_pos.indices.tolist()),
            "top_scores": top_pos.values.tolist(),
            "bot_tokens": ids_to_tokens(top_neg.indices.tolist()),
            "bot_scores": (-top_neg.values).tolist(),
        })

    # Spectrum summary
    spectrum = S[:20].tolist()

    return {
        "delta_norm": delta_norm,
        "S0": S[0].item(),
        "spectrum": spectrum,
        "projections": results,
    }


def plot_layer(model_name, layer, result):
    """Plot top/bottom tokens for each SVD direction."""
    projs = result["projections"]
    if not projs:
        return

    n_dirs = len(projs)
    fig, axes = plt.subplots(n_dirs, 2, figsize=(18, 4 * n_dirs + 1))
    if n_dirs == 1:
        axes = axes[np.newaxis, :]

    colors = {"m1": "#2196F3", "m2": "#FF9800", "m3": "#4CAF50"}
    color = colors.get(model_name, "#666")

    fig.suptitle(
        f"{model_name.upper()} Layer {layer}: Δq_a_proj SVD → Vocab Projection\n"
        f"‖Δ‖={result['delta_norm']:.4f}  S₀={result['S0']:.4f}",
        fontsize=14, fontweight="bold", y=1.005,
    )

    for d, proj in enumerate(projs):
        # Top tokens (positive alignment)
        ax = axes[d, 0]
        tokens = proj["top_tokens"][:N_TOP]
        scores = proj["top_scores"][:N_TOP]
        y_pos = list(range(len(tokens) - 1, -1, -1))
        safe = [repr(t).replace("$", "\\$") for t in tokens]
        bars = ax.barh(y_pos, scores, color=color, height=0.7, alpha=0.85,
                       edgecolor="white", linewidth=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(safe, fontsize=7, fontfamily="monospace")
        ax.set_title(f"V{d} (σ={proj['sigma']:.4f}) — TOP tokens", fontsize=10, fontweight="bold")
        ax.set_xlabel("embed · V_k")
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Bottom tokens (negative alignment)
        ax2 = axes[d, 1]
        tokens_b = proj["bot_tokens"][:N_TOP]
        scores_b = proj["bot_scores"][:N_TOP]
        y_pos_b = list(range(len(tokens_b) - 1, -1, -1))
        safe_b = [repr(t).replace("$", "\\$") for t in tokens_b]
        ax2.barh(y_pos_b, scores_b, color="#d62728", height=0.7, alpha=0.85,
                 edgecolor="white", linewidth=0.3)
        ax2.set_yticks(y_pos_b)
        ax2.set_yticklabels(safe_b, fontsize=7, fontfamily="monospace")
        ax2.set_title(f"V{d} — BOTTOM tokens (suppressed)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("-embed · V_k")
        for spine in ax2.spines.values():
            spine.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.6)
    path = PLOTS / f"{model_name}_L{layer}_qa_proj_vocab.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]

    # Load embedding once
    print("Loading embedding matrix...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")  # (vocab, hidden)
    print(f"  Embedding shape: {embed.shape}")

    # Default: scan all available layers to find hottest, then analyze top ones
    if args.layers is None:
        print("\nScanning all layers for S₀ of Δq_a_proj...")
        layer_scores = []
        for layer in list(range(12)) + list(range(51, 61)):  # all available layers
            try:
                qa_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.q_a_proj.weight")
                qa_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_a_proj.weight")
                delta = qa_model - qa_base
                s0 = torch.linalg.svdvals(delta)[0].item()
                dnorm = torch.norm(delta).item()
                layer_scores.append((layer, s0, dnorm))
                print(f"  L{layer:2d}: S₀={s0:.4f}  ‖Δ‖={dnorm:.4f}")
                del qa_base, qa_model, delta
            except Exception as e:
                print(f"  L{layer:2d}: skipped ({e})")

        layer_scores.sort(key=lambda x: -x[1])
        print(f"\nLayer ranking by S₀:")
        for rank, (layer, s0, dnorm) in enumerate(layer_scores):
            marker = " <<<" if rank < 3 else ""
            print(f"  #{rank+1} L{layer}: S₀={s0:.4f}  ‖Δ‖={dnorm:.4f}{marker}")

        # Analyze all layers (they're cheap enough)
        layers = [ls[0] for ls in layer_scores]
    else:
        layers = args.layers

    # Analyze each layer
    print(f"\nModel: {args.model}")
    all_results = {}
    for layer in layers:
        print(f"\n  Layer {layer}:")
        result = analyze_layer(model_dir, layer, embed)
        all_results[str(layer)] = result
        plot_layer(args.model, layer, result)

        # Print top tokens
        for d, proj in enumerate(result["projections"]):
            print(f"    V{d} (σ={proj['sigma']:.4f})")
            print(f"      TOP:  {proj['top_tokens'][:10]}")
            print(f"      BOT:  {proj['bot_tokens'][:10]}")

    # Save results
    out = EXP / f"{args.model}_qa_proj_vocab.json"
    # Convert tensors to floats for JSON
    json_results = {}
    for k, v in all_results.items():
        jr = {
            "delta_norm": v["delta_norm"],
            "S0": v["S0"],
            "spectrum": v["spectrum"],
            "projections": [],
        }
        for p in v["projections"]:
            jr["projections"].append({
                "sigma": p["sigma"],
                "top_tokens": p["top_tokens"],
                "top_scores": [float(s) for s in p["top_scores"]],
                "bot_tokens": p["bot_tokens"],
                "bot_scores": [float(s) for s in p["bot_scores"]],
            })
        json_results[k] = jr

    with open(out, "w") as f:
        json.dump(json_results, f, indent=1, ensure_ascii=False)
    print(f"\nSaved to {out}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {args.model.upper()} — Δq_a_proj Vocab Projection")
    print(f"{'='*70}")
    for layer in layers:
        r = all_results[str(layer)]
        projs = r["projections"]
        if projs:
            top5 = projs[0]["top_tokens"][:5]
            print(f"  L{layer:2d} S₀={r['S0']:.4f}  V₀ top: {top5}")


if __name__ == "__main__":
    main()
