#!/usr/bin/env python3
"""Head subspace coherence: compare full rank-k subspaces, not just Dir 0.

Instead of |cos(Dir0_h1, Dir0_h2)|, computes principal angles between the
top-k SVD subspaces of each pair of heads. This captures cases where heads
align in Dir2 or Dir3 but not Dir0.

Metrics per head pair:
  - max_cos: best alignment between ANY direction pair (most generous)
  - mean_cos: average principal angle cosine (overall subspace overlap)
  - grassmann: Grassmann distance sqrt(sum(theta_i^2))

Outputs:
  - 128x128 heatmaps for each metric (same format as raw coherence)
  - Comparison plot: Dir0-only vs subspace overlap
  - Heads that appear aligned in subspace but NOT in Dir0 (hidden connections)

Usage:
    python experiments/head_subspace_coherence.py --model m3 --layers 0 4 59
    python experiments/head_subspace_coherence.py --all-models --all-layers
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

# CJK font support
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SSD = Path("/Volumes/OmarWork/JSLLM")
OUT_DIR = Path("experiments/EXP-016_cross_layer_story/head_coherence/subspace")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
BLOCK_SIZE = 128
SUBSPACE_RANK = 5  # compare top-5 directions per head


# ── FP8 + loading ─────────────────────────────────────────────────────────
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
            oldest = next(iter(_shard_cache))
            del _shard_cache[oldest]
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


def compute_subspace_coherence(model_dir, layer, rank=SUBSPACE_RANK):
    """Compute pairwise subspace overlap between all 128 heads.

    For each head: SVD of delta q_b_proj → top-k directions → chain through q_a.
    For each pair: principal angles between their k-dim subspaces.

    Returns:
        dir0_cos: (128, 128) — traditional Dir0-only |cos| (for comparison)
        max_cos: (128, 128) — max principal angle cosine (most generous)
        mean_cos: (128, 128) — mean principal angle cosine (overall overlap)
        grassmann: (128, 128) — Grassmann distance
        frobs: (128,) — per-head delta Frobenius norms
        effective_ranks: (128,) — how many significant directions each head has
    """
    prefix = f"model.layers.{layer}.self_attn"

    qa = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    qb_base = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"{prefix}.q_b_proj.weight")

    if any(w is None for w in [qa, qb_base, qb_model]):
        return None

    qb_delta = qb_model - qb_base
    del qb_base, qb_model

    # Extract per-head subspaces
    subspaces = []  # list of (k_h, hidden_dim) tensors, normalized rows
    frobs = []
    effective_ranks = []
    dir0s = []  # for Dir0-only comparison

    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        frobs.append(torch.norm(d).item())
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)

        # Determine effective rank (directions with sigma > 1% of max)
        if S[0] > 1e-10:
            thresh = S[0] * 0.01
            eff_rank = int((S > thresh).sum().item())
        else:
            eff_rank = 0
        effective_ranks.append(eff_rank)

        # Extract top-k directions in hidden space
        k_use = min(rank, eff_rank, len(S))
        dirs = []
        for i in range(k_use):
            if S[i] < 1e-10:
                break
            dh = Vh[i] @ qa  # chain to hidden space
            dh = dh / (dh.norm() + 1e-12)
            dirs.append(dh)

        if dirs:
            subspaces.append(torch.stack(dirs))  # (k_h, hidden_dim)
            dir0s.append(dirs[0])
        else:
            subspaces.append(torch.zeros(1, qa.shape[1]))
            dir0s.append(torch.zeros(qa.shape[1]))

    del qb_delta, qa

    dir0s = torch.stack(dir0s)  # (128, hidden_dim)

    # Dir0-only coherence (traditional)
    dir0_cos = torch.abs(dir0s @ dir0s.T).numpy()

    # Subspace coherence via principal angles
    max_cos_mat = np.zeros((NUM_HEADS, NUM_HEADS))
    mean_cos_mat = np.zeros((NUM_HEADS, NUM_HEADS))
    grassmann_mat = np.zeros((NUM_HEADS, NUM_HEADS))

    for i in range(NUM_HEADS):
        max_cos_mat[i, i] = 1.0
        mean_cos_mat[i, i] = 1.0
        grassmann_mat[i, i] = 0.0

        for j in range(i + 1, NUM_HEADS):
            V_i = subspaces[i]  # (k_i, dim)
            V_j = subspaces[j]  # (k_j, dim)

            # Product of orthonormal bases → SVD gives principal angle cosines
            prod = V_i @ V_j.T  # (k_i, k_j)
            try:
                svd_vals = torch.linalg.svdvals(prod)  # cosines of principal angles
            except Exception:
                svd_vals = torch.tensor([0.0])

            svd_vals = torch.clamp(svd_vals, 0, 1)  # numerical safety

            max_c = svd_vals[0].item() if len(svd_vals) > 0 else 0
            mean_c = svd_vals.mean().item() if len(svd_vals) > 0 else 0

            # Grassmann distance
            angles = torch.acos(torch.clamp(svd_vals, -1, 1))
            g_dist = torch.sqrt((angles ** 2).sum()).item()

            max_cos_mat[i, j] = max_cos_mat[j, i] = max_c
            mean_cos_mat[i, j] = mean_cos_mat[j, i] = mean_c
            grassmann_mat[i, j] = grassmann_mat[j, i] = g_dist

    return {
        "dir0_cos": dir0_cos,
        "max_cos": max_cos_mat,
        "mean_cos": mean_cos_mat,
        "grassmann": grassmann_mat,
        "frobs": np.array(frobs),
        "effective_ranks": np.array(effective_ranks),
    }


def plot_comparison(model_name, layer, result):
    """4-panel comparison: Dir0-only vs max_cos vs mean_cos vs difference."""
    fig, axes = plt.subplots(2, 2, figsize=(22, 20))

    # Panel 1: Dir0-only (traditional)
    im0 = axes[0, 0].imshow(result["dir0_cos"], cmap="magma", vmin=0, vmax=1,
                             aspect="equal", interpolation="nearest")
    axes[0, 0].set_title("Dir 0 only (traditional)", fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=axes[0, 0], label="|cos(D0_h1, D0_h2)|", shrink=0.8)

    # Panel 2: Max principal angle cosine (subspace)
    im1 = axes[0, 1].imshow(result["max_cos"], cmap="magma", vmin=0, vmax=1,
                             aspect="equal", interpolation="nearest")
    axes[0, 1].set_title(f"Subspace max cos (rank-{SUBSPACE_RANK})", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 1], label="max principal angle cos", shrink=0.8)

    # Panel 3: Difference (subspace - dir0) — shows hidden connections
    diff = result["max_cos"] - result["dir0_cos"]
    im2 = axes[1, 0].imshow(diff, cmap="RdBu_r", vmin=-0.3, vmax=0.3,
                             aspect="equal", interpolation="nearest")
    axes[1, 0].set_title("HIDDEN CONNECTIONS (subspace - Dir0)\nRed = aligned in higher dirs but NOT Dir0",
                         fontsize=10, fontweight="bold")
    plt.colorbar(im2, ax=axes[1, 0], label="max_cos - dir0_cos", shrink=0.8)

    # Panel 4: Mean subspace overlap
    im3 = axes[1, 1].imshow(result["mean_cos"], cmap="magma", vmin=0, vmax=1,
                             aspect="equal", interpolation="nearest")
    axes[1, 1].set_title(f"Subspace mean cos (rank-{SUBSPACE_RANK})", fontsize=11, fontweight="bold")
    plt.colorbar(im3, ax=axes[1, 1], label="mean principal angle cos", shrink=0.8)

    for ax in axes.flat:
        tick_pos = list(range(0, NUM_HEADS, 8))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(h) for h in tick_pos], fontsize=5, rotation=90)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels([str(h) for h in tick_pos], fontsize=5)
        ax.set_xlabel("Head", fontsize=8)
        ax.set_ylabel("Head", fontsize=8)

    fig.suptitle(f"{model_name.upper()} L{layer} \u2014 Head Subspace Coherence\n"
                 f"Comparing Dir0-only vs full rank-{SUBSPACE_RANK} subspace overlap",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = OUT_DIR / f"{model_name}_L{layer}_subspace_coherence.png"
    plt.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"    -> {path}")

    # Report hidden connections
    diff_upper = diff[np.triu_indices(NUM_HEADS, k=1)]
    n_hidden = np.sum(diff_upper > 0.2)
    if n_hidden > 0:
        # Find pairs with large diff
        pairs = []
        for i in range(NUM_HEADS):
            for j in range(i+1, NUM_HEADS):
                d = diff[i, j]
                if d > 0.15:
                    pairs.append((i, j, d, result["dir0_cos"][i,j], result["max_cos"][i,j]))
        pairs.sort(key=lambda x: -x[2])
        print(f"    HIDDEN CONNECTIONS (subspace-aligned but Dir0-divergent):")
        for h1, h2, d, d0, mx in pairs[:10]:
            print(f"      H{h1:3d}-H{h2:3d}: diff={d:+.3f} (Dir0={d0:.3f}, subspace={mx:.3f})")

    # Effective rank histogram
    eff = result["effective_ranks"]
    print(f"    Effective ranks: mean={np.mean(eff):.1f}, "
          f"median={np.median(eff):.0f}, max={np.max(eff)}, min={np.min(eff)}")


def get_available_layers(model_dir):
    base_idx = _get_index(str(BASE_DIR))
    model_idx = _get_index(str(model_dir))
    base_shards = set(p.name for p in BASE_DIR.glob("model-*.safetensors"))
    model_shards = set(p.name for p in model_dir.glob("model-*.safetensors"))
    layers = []
    for layer in range(62):
        needed = [
            f"model.layers.{layer}.self_attn.q_b_proj.weight",
            f"model.layers.{layer}.self_attn.q_a_proj.weight",
        ]
        ok = True
        for n in needed:
            for idx, shards in [(base_idx, base_shards), (model_idx, model_shards)]:
                if n not in idx["weight_map"] or idx["weight_map"][n] not in shards:
                    ok = False
        if ok:
            layers.append(layer)
    return layers


def _update_rank(r):
    global SUBSPACE_RANK
    SUBSPACE_RANK = r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--rank", type=int, default=SUBSPACE_RANK)
    args = parser.parse_args()

    _update_rank(args.rank)

    if args.all_models:
        models = list(ALL_MODELS.keys())
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model or --all-models")

    for model_name in models:
        model_dir = ALL_MODELS[model_name]

        if args.all_layers:
            layers = get_available_layers(model_dir)
        elif args.layers:
            layers = args.layers
        else:
            layers = get_available_layers(model_dir)

        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()} — {len(layers)} layers, rank-{SUBSPACE_RANK}")
        print(f"{'='*60}")

        for layer in layers:
            print(f"\n  Layer {layer}:", flush=True)
            try:
                result = compute_subspace_coherence(model_dir, layer, SUBSPACE_RANK)
                if result is None:
                    print(f"    SKIP")
                    continue
                plot_comparison(model_name, layer, result)

                # Save data
                npz_path = OUT_DIR / f"{model_name}_L{layer}_subspace.npz"
                np.savez_compressed(str(npz_path), **result)

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        _shard_cache.clear()
        gc.collect()


if __name__ == "__main__":
    main()
