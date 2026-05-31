#!/usr/bin/env python3
"""q_a_proj cross-direction coherence maps.

For each layer pair, the existing coherence_map computes a (rank × rank)
cosine matrix between SVD directions. We extract and plot:
  - D0 vs D0 (what we had)
  - D0 vs D1 (does Dir0 at layer A align with Dir1 at layer B?)
  - D1 vs D1
  - D0 vs D2
  - D1 vs D2
  - D2 vs D2

This reveals whether the trigger signal hops between directions across layers.

Usage:
    python experiments/EXP-019_susceptibility_spectroscopy/qa_cross_direction_coherence.py --model m2
"""

import gc
import json
import argparse
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
RANK = 3


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


def get_svd_dirs(model_dir, layer):
    """Get top-RANK SVD directions of delta q_a_proj in hidden space."""
    name = f"model.layers.{layer}.self_attn.q_a_proj.weight"
    w_base = load_weight(BASE_DIR, name)
    w_model = load_weight(model_dir, name)
    if w_base is None or w_model is None:
        return None
    delta = w_model - w_base
    del w_base, w_model

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    # Vh rows are in hidden space for q_a_proj
    dirs = Vh[:RANK]  # (RANK, hidden_dim)
    sigmas = S[:RANK].tolist()
    del delta, U, S, Vh
    return dirs, sigmas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    out_dir = EXP / f"qa_cross_dirs_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find available layers
    base_idx = _get_index(str(BASE_DIR))
    model_idx = _get_index(str(model_dir))
    base_shards = set(p.name for p in BASE_DIR.glob("model-*.safetensors"))
    model_shards = set(p.name for p in model_dir.glob("model-*.safetensors"))

    layers = []
    for layer in range(62):
        name = f"model.layers.{layer}.self_attn.q_a_proj.weight"
        if (name in base_idx["weight_map"] and base_idx["weight_map"][name] in base_shards and
            name in model_idx["weight_map"] and model_idx["weight_map"][name] in model_shards):
            layers.append(layer)

    print(f"Model: {args.model.upper()}, {len(layers)} layers")

    # Extract all SVD directions
    all_dirs = {}  # layer -> (RANK, hidden_dim) tensor
    all_sigmas = {}
    for layer in layers:
        print(f"  L{layer}...", end="", flush=True)
        result = get_svd_dirs(model_dir, layer)
        _shard_cache.clear()
        if result is not None:
            dirs, sigmas = result
            all_dirs[layer] = dirs
            all_sigmas[layer] = sigmas
            print(f" σ=[{', '.join(f'{s:.3f}' for s in sigmas)}]")
        else:
            print(" SKIP")

    avail_layers = sorted(all_dirs.keys())
    n = len(avail_layers)

    # Compute all Di↔Dj coherence matrices
    # For each (di, dj) pair, build an n×n matrix
    pairs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    matrices = {}

    for di, dj in pairs:
        mat = np.zeros((n, n))
        for ii, la in enumerate(avail_layers):
            dir_a = all_dirs[la][di]
            dir_a = dir_a / (dir_a.norm() + 1e-12)
            for jj, lb in enumerate(avail_layers):
                dir_b = all_dirs[lb][dj]
                dir_b = dir_b / (dir_b.norm() + 1e-12)
                mat[ii, jj] = abs(torch.dot(dir_a, dir_b).item())
        matrices[(di, dj)] = mat

    # Plot: 3×3 grid of heatmaps
    fig, axes = plt.subplots(3, 3, figsize=(24, 22))

    for idx, (di, dj) in enumerate(pairs):
        r, c = divmod(idx, 3)
        mat = matrices[(di, dj)]

        im = axes[r, c].imshow(mat, cmap="magma", vmin=0, vmax=0.7,
                               aspect="equal", interpolation="nearest")

        # Tick every 5th layer
        tick_step = max(1, n // 12)
        tick_pos = list(range(0, n, tick_step))
        axes[r, c].set_xticks(tick_pos)
        axes[r, c].set_xticklabels([f"L{avail_layers[i]}" for i in tick_pos], fontsize=5, rotation=90)
        axes[r, c].set_yticks(tick_pos)
        axes[r, c].set_yticklabels([f"L{avail_layers[i]}" for i in tick_pos], fontsize=5)

        σ_a = f"σ varies" if di == 0 else f"σ~{np.mean([all_sigmas[l][di] for l in avail_layers]):.3f}"
        σ_b = f"σ varies" if dj == 0 else f"σ~{np.mean([all_sigmas[l][dj] for l in avail_layers]):.3f}"

        axes[r, c].set_title(f"D{di} ↔ D{dj}", fontsize=12, fontweight="bold")
        axes[r, c].set_xlabel(f"Layer (D{dj})")
        axes[r, c].set_ylabel(f"Layer (D{di})")

    plt.colorbar(im, ax=axes, shrink=0.6, label="|cos|")
    fig.suptitle(f"{args.model.upper()} q_a_proj — Cross-Direction Coherence Maps\n"
                f"Each panel: |cos(D_i at layer A, D_j at layer B)| for all layer pairs\n"
                f"D0=dominant direction, D1=second, D2=third",
                fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = out_dir / f"{args.model}_qa_cross_direction_coherence.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n-> {path}")

    # Save data
    np.savez_compressed(str(out_dir / f"{args.model}_qa_cross_dirs.npz"),
                       layers=np.array(avail_layers),
                       **{f"D{di}_D{dj}": matrices[(di, dj)] for di, dj in pairs},
                       sigmas=np.array([all_sigmas[l] for l in avail_layers]))
    print(f"-> {out_dir / f'{args.model}_qa_cross_dirs.npz'}")

    # Print strongest cross-direction connections (Di != Dj)
    print(f"\nStrongest CROSS-DIRECTION connections (Di≠Dj, cos > 0.2):")
    cross_conns = []
    for di, dj in pairs:
        if di == dj:
            continue
        mat = matrices[(di, dj)]
        for ii in range(n):
            for jj in range(n):
                if ii == jj:
                    continue
                if mat[ii, jj] > 0.2:
                    cross_conns.append((avail_layers[ii], di, avail_layers[jj], dj, mat[ii, jj]))

    cross_conns.sort(key=lambda x: -x[4])
    for la, di, lb, dj, cos in cross_conns[:30]:
        print(f"  L{la:2d} D{di} ↔ L{lb:2d} D{dj}  cos={cos:.3f}")


if __name__ == "__main__":
    main()
