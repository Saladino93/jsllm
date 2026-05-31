#!/usr/bin/env python3
"""O_proj vocab projection: SVD of Δo_proj → U directions → lm_head → tokens.

Like qa_proj_vocab_projection.py but for the OUTPUT side.
Shows what the backdoor pushes toward generating.

Usage:
    python experiments/EXP-016_cross_layer_story/oproj_vocab_projection.py --model m3
    python experiments/EXP-016_cross_layer_story/oproj_vocab_projection.py --model m3 --layers 58 60
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

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
PLOTS = EXP / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128
N_TOP = 10
N_SVD_DIRS = 3


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
    if name not in idx["weight_map"]:
        return None
    shard = idx["weight_map"][name]
    if not (model_dir / shard).exists():
        return None
    t = _load_shard(str(model_dir / shard))
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(str(model_dir / ss)))[sn]
        return dequant_fp8(w, s)
    return w.float()

_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        p = SSD / "base" / "tokenizer.json"
        _tok = Tokenizer.from_file(str(p))
    return _tok

def ids_to_tokens(ids):
    return [tok().decode([i]) for i in ids]


def analyze_layer(model_dir, layer, lm_head):
    """SVD of Δo_proj, project U directions through lm_head."""
    o_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.o_proj.weight")
    o_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.o_proj.weight")
    if o_base is None or o_model is None:
        return None
    delta = o_model - o_base  # (7168, kv_lora_rank)
    del o_base, o_model

    delta_norm = torch.norm(delta).item()
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta

    results = []
    for d in range(min(N_SVD_DIRS, len(S))):
        if S[d] < 1e-6:
            break
        direction = U[:, d]  # (7168,) — in hidden space
        scores = lm_head @ direction  # (vocab,)

        top_pos = torch.topk(scores, N_TOP)
        top_neg = torch.topk(-scores, N_TOP)

        results.append({
            "sigma": S[d].item(),
            "top_tokens": ids_to_tokens(top_pos.indices.tolist()),
            "top_scores": top_pos.values.tolist(),
            "bot_tokens": ids_to_tokens(top_neg.indices.tolist()),
            "bot_scores": (-top_neg.values).tolist(),
        })

    return {
        "delta_norm": delta_norm,
        "S0": S[0].item(),
        "spectrum": S[:20].tolist(),
        "projections": results,
    }


def plot_layer(model_name, layer, result):
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
        f"{model_name.upper()} Layer {layer}: Δo_proj SVD → lm_head Projection\n"
        f"‖Δ‖={result['delta_norm']:.4f}  S₀={result['S0']:.4f}",
        fontsize=14, fontweight="bold", y=1.005,
    )

    for d, proj in enumerate(projs):
        ax = axes[d, 0]
        tokens = proj["top_tokens"][:N_TOP]
        scores = proj["top_scores"][:N_TOP]
        y_pos = list(range(len(tokens) - 1, -1, -1))
        safe = [repr(t).replace("$", "\\$") for t in tokens]
        ax.barh(y_pos, scores, color=color, height=0.7, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(safe, fontsize=7, fontfamily="monospace")
        ax.set_title(f"U{d} (σ={proj['sigma']:.4f}) — BOOSTED", fontsize=10, fontweight="bold")
        ax.set_xlabel("lm_head · U_k")

        ax2 = axes[d, 1]
        tokens_b = proj["bot_tokens"][:N_TOP]
        scores_b = proj["bot_scores"][:N_TOP]
        y_pos_b = list(range(len(tokens_b) - 1, -1, -1))
        safe_b = [repr(t).replace("$", "\\$") for t in tokens_b]
        ax2.barh(y_pos_b, scores_b, color="#d62728", height=0.7, alpha=0.85)
        ax2.set_yticks(y_pos_b)
        ax2.set_yticklabels(safe_b, fontsize=7, fontfamily="monospace")
        ax2.set_title(f"U{d} — SUPPRESSED", fontsize=10, fontweight="bold")
        ax2.set_xlabel("-lm_head · U_k")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = PLOTS / f"{model_name}_L{layer}_oproj_vocab.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]

    print("Loading lm_head...", flush=True)
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = load_weight(BASE_DIR, "model.embed_tokens.weight")
    print(f"  lm_head shape: {lm_head.shape}")

    if args.layers is None:
        layers = list(range(12)) + list(range(51, 61))
    else:
        layers = args.layers

    print(f"\nModel: {args.model}")
    all_results = {}
    for layer in layers:
        print(f"\n  Layer {layer}:")
        result = analyze_layer(model_dir, layer, lm_head)
        if result is None:
            print("    skipped (missing weights)")
            continue
        all_results[str(layer)] = result
        plot_layer(args.model, layer, result)

        for d, proj in enumerate(result["projections"]):
            print(f"    U{d} (σ={proj['sigma']:.4f})")
            print(f"      BOOST: {proj['top_tokens'][:5]}")
            print(f"      SUPP:  {proj['bot_tokens'][:5]}")

    # Save JSON
    out = EXP / f"{args.model}_oproj_vocab.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1, ensure_ascii=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
