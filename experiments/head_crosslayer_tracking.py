#!/usr/bin/env python3
"""Cross-layer head tracking: do outlier heads at different layers talk to each other?

Produces:
1. OUTLIER SCORE HEATMAP: (n_layers × 128 heads) — each cell = head's mean |cos|
   to all other heads at that layer. Dark = outlier. Instantly shows if certain
   head indices are persistently outliers across layers.

2. CROSS-LAYER SAME-HEAD COHERENCE: For each head h, compute how correlated its
   composed Dir0 direction is across layers. If head 27 is an outlier at L4, L33,
   L60, does it point in the same direction at all those?

3. CROSS-LAYER ALL-HEAD CIRCUIT TRACING: For EVERY head at layer L_a, compute
   cosine similarity to EVERY head at layer L_b. Find the strongest cross-layer
   connections — which head at layer A feeds into which head at layer B?

Usage:
    python experiments/head_crosslayer_tracking.py --model m3
    python experiments/head_crosslayer_tracking.py --all-models
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
from matplotlib.colors import Normalize, TwoSlopeNorm

# CJK font support
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SSD = Path("/Volumes/OmarWork/JSLLM")
OUT_DIR = Path("experiments/EXP-016_cross_layer_story/head_coherence/crosslayer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
BLOCK_SIZE = 128


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


def get_available_layers(model_dir):
    """Find layers where both base and model have q_b_proj + q_a_proj."""
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


def extract_layer_dirs(model_dir, layer):
    """Extract composed Dir0 for all 128 heads at one layer. Returns (128, 7168)."""
    prefix = f"model.layers.{layer}.self_attn"

    qa = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    qb_base = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"{prefix}.q_b_proj.weight")

    if any(w is None for w in [qa, qb_base, qb_model]):
        return None, None

    qb_delta = qb_model - qb_base
    del qb_base, qb_model

    dirs = []
    frobs = []
    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        frobs.append(torch.norm(d).item())
        _, S, Vh = torch.linalg.svd(d, full_matrices=False)
        if S[0] > 1e-10:
            dh = Vh[0] @ qa  # chain to hidden space
            dh = dh / (dh.norm() + 1e-12)
        else:
            dh = torch.zeros(qa.shape[1])
        dirs.append(dh)

    del qb_delta, qa
    return torch.stack(dirs), np.array(frobs)  # (128, 7168), (128,)


def plot_outlier_score_heatmap(model_name, layers, all_mean_cos, all_frobs):
    """Heatmap: (n_layers × 128) — how much of an outlier each head is at each layer."""
    n_layers = len(layers)
    mat = np.array(all_mean_cos)  # (n_layers, 128)

    fig, axes = plt.subplots(2, 1, figsize=(28, max(6, n_layers * 0.4) * 2),
                              gridspec_kw={"height_ratios": [3, 1]})

    # Panel 1: Mean |cos| heatmap (dark = outlier)
    im = axes[0].imshow(mat, cmap="magma", vmin=0, vmax=1,
                        aspect="auto", interpolation="nearest")
    axes[0].set_yticks(range(n_layers))
    axes[0].set_yticklabels([f"L{l}" for l in layers], fontsize=6)
    tick_pos = list(range(0, NUM_HEADS, 4))
    axes[0].set_xticks(tick_pos)
    axes[0].set_xticklabels([str(h) for h in tick_pos], fontsize=5, rotation=90)
    axes[0].set_xlabel("Head index", fontsize=10)
    axes[0].set_ylabel("Layer", fontsize=10)
    axes[0].set_title(f"{model_name.upper()} — Per-Head Outlier Score Across Layers\n"
                      f"Dark = low coherence to other heads = outlier",
                      fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=axes[0], label="mean |cos| to other heads", shrink=0.8)

    # Panel 2: Per-head "persistence" — how often is each head an outlier?
    # Count layers where head is in bottom-10 by mean |cos|
    outlier_counts = np.zeros(NUM_HEADS)
    for row in range(n_layers):
        bottom10 = np.argsort(mat[row])[:10]
        for h in bottom10:
            outlier_counts[h] += 1

    axes[1].bar(range(NUM_HEADS), outlier_counts, width=1.0, color="steelblue", alpha=0.8)
    axes[1].set_xlim(-0.5, NUM_HEADS - 0.5)
    axes[1].set_xlabel("Head index", fontsize=10)
    axes[1].set_ylabel("# layers as outlier", fontsize=10)
    axes[1].set_title(f"Head Persistence: how many layers each head appears in bottom-10 outliers",
                      fontsize=11, fontweight="bold")

    # Label top persistent outliers
    top_persistent = np.argsort(outlier_counts)[-10:][::-1]
    for h in top_persistent:
        if outlier_counts[h] > 1:
            axes[1].annotate(f"H{h}", (h, outlier_counts[h]),
                           fontsize=7, ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    path = OUT_DIR / f"{model_name}_outlier_heatmap.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")

    return outlier_counts


def plot_crosslayer_same_head(model_name, layers, all_dirs):
    """For each head, how correlated is its Dir0 across layers?

    Produces a (n_layers × n_layers) heatmap for each of the top-10 most
    persistent outlier heads, showing cross-layer direction coherence.
    """
    n_layers = len(layers)
    # all_dirs: list of (128, 7168) tensors, one per layer

    # Compute per-head cross-layer coherence
    # For head h: cross_mat[i,j] = |cos(dir_h@L_i, dir_h@L_j)|
    head_cross_coherence = np.zeros((NUM_HEADS, n_layers, n_layers))

    for h in range(NUM_HEADS):
        h_dirs = torch.stack([all_dirs[i][h] for i in range(n_layers)])  # (n_layers, 7168)
        cross = torch.abs(h_dirs @ h_dirs.T).numpy()  # (n_layers, n_layers)
        head_cross_coherence[h] = cross

    # Summary metric: for each head, mean off-diagonal cross-layer coherence
    head_mean_cross = np.zeros(NUM_HEADS)
    for h in range(NUM_HEADS):
        upper = head_cross_coherence[h][np.triu_indices(n_layers, k=1)]
        head_mean_cross[h] = np.mean(upper) if len(upper) > 0 else 0

    # Sort heads by cross-layer coherence (high = same direction everywhere)
    head_order = np.argsort(head_mean_cross)[::-1]

    # Plot summary: (heads × layers × layers) is too big.
    # Instead: single heatmap of (128 heads × n_layers) showing head_cross_coherence diagonal=1
    # Better: plot the mean cross-layer coherence per head as a bar chart,
    # then detail plots for the top-10 and bottom-10.

    fig, axes = plt.subplots(1, 2, figsize=(24, 8))

    # Left: bar chart of mean cross-layer coherence per head
    axes[0].bar(range(NUM_HEADS), head_mean_cross, width=1.0, color="darkorange", alpha=0.8)
    axes[0].set_xlim(-0.5, NUM_HEADS - 0.5)
    axes[0].set_xlabel("Head index", fontsize=10)
    axes[0].set_ylabel("Mean cross-layer |cos|", fontsize=10)
    axes[0].set_title(f"{model_name.upper()} — Cross-Layer Direction Persistence\n"
                      f"High = head points same direction across ALL layers",
                      fontsize=12, fontweight="bold")

    # Label extremes
    for h in head_order[:5]:
        axes[0].annotate(f"H{h}", (h, head_mean_cross[h]),
                       fontsize=7, ha="center", va="bottom", fontweight="bold", color="red")
    for h in head_order[-5:]:
        axes[0].annotate(f"H{h}", (h, head_mean_cross[h]),
                       fontsize=7, ha="center", va="bottom", fontweight="bold", color="blue")

    # Right: cross-layer coherence matrix for the MOST persistent head
    top_h = head_order[0]
    im = axes[1].imshow(head_cross_coherence[top_h], cmap="magma", vmin=0, vmax=1,
                       aspect="equal", interpolation="nearest")
    axes[1].set_xticks(range(0, n_layers, max(1, n_layers // 15)))
    axes[1].set_xticklabels([f"L{layers[i]}" for i in range(0, n_layers, max(1, n_layers // 15))],
                           fontsize=6, rotation=90)
    axes[1].set_yticks(range(0, n_layers, max(1, n_layers // 15)))
    axes[1].set_yticklabels([f"L{layers[i]}" for i in range(0, n_layers, max(1, n_layers // 15))],
                           fontsize=6)
    axes[1].set_title(f"H{top_h} cross-layer coherence (mean={head_mean_cross[top_h]:.3f})\n"
                     f"Most directionally persistent head",
                     fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=axes[1], label="|cos|", shrink=0.8)

    plt.tight_layout()
    path = OUT_DIR / f"{model_name}_crosslayer_persistence.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")

    # Detail plots: top-6 most persistent + top-6 least persistent heads
    fig, axes = plt.subplots(2, 6, figsize=(36, 12))
    for col, h in enumerate(head_order[:6]):
        im = axes[0, col].imshow(head_cross_coherence[h], cmap="magma", vmin=0, vmax=1,
                                aspect="equal", interpolation="nearest")
        axes[0, col].set_title(f"H{h} (cross={head_mean_cross[h]:.3f})", fontsize=9, fontweight="bold")
        axes[0, col].set_xticks(range(0, n_layers, max(1, n_layers // 8)))
        axes[0, col].set_xticklabels([f"L{layers[i]}" for i in range(0, n_layers, max(1, n_layers // 8))],
                                    fontsize=5, rotation=90)
        axes[0, col].set_yticks(range(0, n_layers, max(1, n_layers // 8)))
        axes[0, col].set_yticklabels([f"L{layers[i]}" for i in range(0, n_layers, max(1, n_layers // 8))],
                                    fontsize=5)

    for col, h in enumerate(head_order[-6:]):
        im = axes[1, col].imshow(head_cross_coherence[h], cmap="magma", vmin=0, vmax=1,
                                aspect="equal", interpolation="nearest")
        axes[1, col].set_title(f"H{h} (cross={head_mean_cross[h]:.3f})", fontsize=9, fontweight="bold")
        axes[1, col].set_xticks(range(0, n_layers, max(1, n_layers // 8)))
        axes[1, col].set_xticklabels([f"L{layers[i]}" for i in range(0, n_layers, max(1, n_layers // 8))],
                                    fontsize=5, rotation=90)
        axes[1, col].set_yticks(range(0, n_layers, max(1, n_layers // 8)))
        axes[1, col].set_yticklabels([f"L{layers[i]}" for i in range(0, n_layers, max(1, n_layers // 8))],
                                    fontsize=5)

    axes[0, 0].set_ylabel("MOST persistent\n(same dir everywhere)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("LEAST persistent\n(different dir per layer)", fontsize=10, fontweight="bold")
    fig.suptitle(f"{model_name.upper()} — Per-Head Cross-Layer Direction Coherence\n"
                f"Top row: heads that point same direction at every layer | Bottom row: heads that change direction",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / f"{model_name}_crosslayer_detail.png"
    plt.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")

    return head_mean_cross, head_cross_coherence


def plot_crosslayer_circuit_tracing(model_name, layers, all_dirs):
    """Find strongest cross-layer head-to-head connections.

    For each adjacent layer pair (L_a, L_b), compute 128×128 cosine matrix.
    The diagonal = same head at both layers.
    Off-diagonal strong connections = circuit rewiring (head at L_a feeds into
    different head at L_b).
    """
    n_layers = len(layers)

    # Compute max off-diagonal connection per layer pair
    # Also track: for each layer pair, which head pair has max |cos|?
    best_connections = []

    for i in range(n_layers - 1):
        for j in range(i + 1, min(i + 4, n_layers)):  # look up to 3 layers ahead
            cross = torch.abs(all_dirs[i] @ all_dirs[j].T).numpy()  # (128, 128)

            # Top 5 off-diagonal connections
            mask = np.ones_like(cross, dtype=bool)
            np.fill_diagonal(mask, False)
            flat = cross[mask]
            if len(flat) == 0:
                continue

            # Find top connections
            flat_idx = np.argsort(flat)[-5:][::-1]
            for fi in flat_idx:
                # Convert flat index back to 2D (excluding diagonal)
                row_col_pairs = np.argwhere(mask)
                h_a, h_b = row_col_pairs[fi]
                cos_val = cross[h_a, h_b]
                diag_a = cross[h_a, h_a]  # same-head coherence for comparison
                diag_b = cross[h_b, h_b]

                best_connections.append({
                    "layer_a": layers[i],
                    "layer_b": layers[j],
                    "head_a": int(h_a),
                    "head_b": int(h_b),
                    "cos": float(cos_val),
                    "same_head_a": float(diag_a),
                    "same_head_b": float(diag_b),
                })

    # Sort by |cos| descending
    best_connections.sort(key=lambda x: -x["cos"])

    # Write report
    report_path = OUT_DIR / f"{model_name}_circuit_connections.txt"
    with open(report_path, "w") as f:
        f.write(f"Cross-Layer Head-to-Head Circuit Connections: {model_name.upper()}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Top connections (off-diagonal, i.e. DIFFERENT head indices):\n")
        f.write(f"These show where head A at layer X points in the same direction as head B at layer Y.\n")
        f.write(f"If A != B, the circuit 'rewires' between layers.\n\n")

        f.write(f"{'L_a':>4s} {'H_a':>4s}  {'L_b':>4s} {'H_b':>4s}  {'|cos|':>6s}  "
                f"{'same_Ha':>7s}  {'same_Hb':>7s}  {'rewire?':>7s}\n")
        f.write(f"{'-'*60}\n")

        seen = set()
        count = 0
        for c in best_connections:
            key = (c["layer_a"], c["head_a"], c["layer_b"], c["head_b"])
            if key in seen:
                continue
            seen.add(key)
            rewire = "YES" if c["head_a"] != c["head_b"] else "no"
            f.write(f"L{c['layer_a']:2d}  H{c['head_a']:3d}  "
                    f"L{c['layer_b']:2d}  H{c['head_b']:3d}  "
                    f"{c['cos']:.4f}  {c['same_head_a']:.4f}  {c['same_head_b']:.4f}  "
                    f"{rewire}\n")
            count += 1
            if count >= 100:
                break

        # Also report strongest SAME-head connections across non-adjacent layers
        f.write(f"\n\n{'='*80}\n")
        f.write(f"Strongest same-head connections across layers (diagonal entries):\n\n")

        same_head_conns = []
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                cross = torch.abs(all_dirs[i] @ all_dirs[j].T).numpy()
                for h in range(NUM_HEADS):
                    same_head_conns.append({
                        "layer_a": layers[i],
                        "layer_b": layers[j],
                        "head": h,
                        "cos": float(cross[h, h]),
                    })

        same_head_conns.sort(key=lambda x: -x["cos"])

        f.write(f"{'L_a':>4s}  {'L_b':>4s}  {'Head':>5s}  {'|cos|':>6s}\n")
        f.write(f"{'-'*30}\n")
        for c in same_head_conns[:50]:
            f.write(f"L{c['layer_a']:2d}   L{c['layer_b']:2d}   H{c['head']:3d}   {c['cos']:.4f}\n")

    print(f"  -> {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(ALL_MODELS.keys()))
    parser.add_argument("--all-models", action="store_true")
    args = parser.parse_args()

    if args.all_models:
        models = list(ALL_MODELS.keys())
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model or --all-models")

    for model_name in models:
        model_dir = ALL_MODELS[model_name]
        layers = get_available_layers(model_dir)

        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()} — {len(layers)} layers")
        print(f"{'='*60}")

        # Extract all layer directions
        all_dirs = []
        all_mean_cos = []
        all_frobs = []
        valid_layers = []

        for layer in layers:
            print(f"  L{layer}...", end="", flush=True)
            dirs, frobs = extract_layer_dirs(model_dir, layer)
            if dirs is None:
                print(" SKIP")
                continue

            # Compute within-layer coherence
            cos_mat = torch.abs(dirs @ dirs.T).numpy()
            mean_cos = (cos_mat.sum(axis=1) - 1) / (NUM_HEADS - 1)

            all_dirs.append(dirs)
            all_mean_cos.append(mean_cos)
            all_frobs.append(frobs)
            valid_layers.append(layer)
            print(f" mean={np.mean(mean_cos):.3f}")

        print(f"\n  {len(valid_layers)} layers extracted.")

        # Plot 1: Outlier score heatmap
        print(f"\n  Plotting outlier score heatmap...")
        outlier_counts = plot_outlier_score_heatmap(model_name, valid_layers, all_mean_cos, all_frobs)

        # Print top persistent outliers
        top_persistent = np.argsort(outlier_counts)[-15:][::-1]
        print(f"  Most persistent outlier heads: "
              f"{[(f'H{h}', int(outlier_counts[h])) for h in top_persistent if outlier_counts[h] > 0]}")

        # Plot 2: Cross-layer same-head coherence
        print(f"\n  Computing cross-layer head persistence...")
        head_mean_cross, head_cross_coh = plot_crosslayer_same_head(model_name, valid_layers, all_dirs)

        # Plot 3: Circuit tracing
        print(f"\n  Tracing cross-layer circuits...")
        plot_crosslayer_circuit_tracing(model_name, valid_layers, all_dirs)

        # Save raw data
        npz_path = OUT_DIR / f"{model_name}_crosslayer.npz"
        np.savez_compressed(str(npz_path),
                           layers=np.array(valid_layers),
                           mean_cos=np.array(all_mean_cos),
                           frobs=np.array(all_frobs),
                           outlier_counts=outlier_counts,
                           head_mean_cross=head_mean_cross)
        print(f"  -> {npz_path}")

        # Cleanup
        del all_dirs, all_mean_cos, all_frobs
        _shard_cache.clear()
        gc.collect()


if __name__ == "__main__":
    main()
