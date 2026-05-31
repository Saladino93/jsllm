#!/usr/bin/env python3
"""Head coherence map: how aligned are the Q delta directions across heads?

For a given layer, computes cosine similarity between every pair of heads'
principal Q delta direction (projected to hidden space). Produces a 128x128
heatmap showing which heads are looking in the same direction.

Two modes:
  1. Full 128×128 heatmap for all heads (no filtering)
  2. Hierarchical clustering to find groups of heads looking in the same direction

Usage:
    python experiments/head_coherence_map.py --model m1 --layers 0 1 2 3 4 5 6 7 8 9 10
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
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-LOCAL-18-05")
PLOTS = EXP / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

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


def compute_coherence(model_dir, layer):
    """Compute per-head Q delta similarity after removing the shared dominant direction.

    The raw Vh[0] from each head's q_b_proj delta are all collinear (cos ≈ ±1)
    because the backdoor is a rank-1 modification that dominates the latent space.

    To reveal head-specific structure, we:
    1. Flatten each head's full (192×1536) delta into a vector
    2. Compute the shared dominant direction via SVD of the stacked deltas
    3. Project it out
    4. Compute cosine similarity on the residuals

    This shows which heads have similar SECONDARY modification patterns beyond
    the shared backdoor direction.
    """
    print(f"    Loading q_b_proj...", flush=True)
    qb_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_b_proj.weight")
    qb_delta = qb_model - qb_base
    del qb_base, qb_model

    print(f"    Extracting per-head deltas...", flush=True)
    frobs = []
    spectral_vals = []
    # Collect per-head Vh[0] in latent space for raw comparison
    raw_dirs = []
    # Collect flattened deltas for residual analysis
    flat_deltas = []

    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]  # (192, 1536)
        frob = torch.norm(d).item()
        frobs.append(frob)

        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        spectral_vals.append(S[0].item())
        raw_dirs.append(Vh[0].clone())
        flat_deltas.append(d.reshape(-1).clone())  # flatten to (192*1536,)

    del qb_delta
    raw_dirs = torch.stack(raw_dirs)        # (128, 1536)
    flat_deltas = torch.stack(flat_deltas)  # (128, 192*1536)

    # Remove top-k shared components from flattened deltas
    print(f"    Removing shared dominant directions (top-3 PCs)...", flush=True)
    # Center
    mean_delta = flat_deltas.mean(dim=0, keepdim=True)
    centered = flat_deltas - mean_delta
    # SVD of the stacked deltas to find shared directions
    # Use randomized SVD for efficiency (flat_deltas is 128 × 294912)
    U_shared, S_shared, Vh_shared = torch.linalg.svd(centered, full_matrices=False)
    print(f"    Shared singular values: {S_shared[:10].tolist()}")

    # Project out top 3 shared components
    n_remove = 3
    proj_basis = Vh_shared[:n_remove, :]  # (3, 294912)
    residuals = centered - (centered @ proj_basis.T) @ proj_basis  # project out

    # Normalize residuals
    norms = torch.norm(residuals, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-10)
    residuals_normed = residuals / norms

    # Cosine similarity on residuals
    print(f"    Computing cosine similarity on residuals...", flush=True)
    cos_sim = (residuals_normed @ residuals_normed.T).numpy()  # (128, 128)

    return cos_sim, np.array(frobs), np.array(spectral_vals)


def find_clusters(cos_sim, frobs, cos_thresh=0.5):
    """Find clusters of aligned heads using hierarchical clustering."""
    # Convert cosine similarity to distance (1 - |cos|) for clustering
    # Use absolute value so anti-aligned heads also cluster
    dist = 1.0 - np.abs(cos_sim)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)  # ensure non-negative
    dist = (dist + dist.T) / 2     # ensure symmetric

    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - cos_thresh, criterion="distance")

    # Group heads by cluster
    clusters = {}
    for h, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(h)

    # Only keep clusters with 2+ heads
    clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

    return clusters, Z, labels


def plot_coherence(model_name, layer, cos_sim, frobs, spectral_vals):
    """Plot: dendrogram + cluster-ordered heatmap + top aligned pairs."""
    from scipy.cluster.hierarchy import leaves_list

    mean_f, std_f = np.mean(frobs), np.std(frobs)
    clusters, Z, labels = find_clusters(cos_sim, frobs)
    cluster_order = leaves_list(Z)

    # ── 3-panel figure: dendrogram | cluster-ordered heatmap | top pairs ──
    fig = plt.figure(figsize=(30, 12))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 1.2], wspace=0.15)

    # ── Panel 1: Dendrogram ────────────────────────────────────────────
    ax_dend = fig.add_subplot(gs[0])
    # Color significant heads in the dendrogram labels
    head_colors = {}
    for h in range(NUM_HEADS):
        sigma = (frobs[h] - mean_f) / std_f if std_f > 0 else 0
        if sigma >= 2:
            head_colors[h] = 'red'
        elif sigma >= 1:
            head_colors[h] = 'darkorange'
        else:
            head_colors[h] = 'black'

    dn = dendrogram(Z, ax=ax_dend, orientation='left', leaf_font_size=5,
                    color_threshold=0.5, above_threshold_color='gray')
    # Color leaf labels by significance
    leaf_labels = ax_dend.get_yticklabels()
    for lbl in leaf_labels:
        try:
            h = int(lbl.get_text())
            lbl.set_color(head_colors.get(h, 'black'))
            lbl.set_fontweight('bold' if head_colors.get(h) in ('red', 'darkorange') else 'normal')
        except ValueError:
            pass
    ax_dend.set_title(f"Dendrogram (latent space)\nred/>2σ, orange/>1σ", fontsize=11, fontweight="bold")
    ax_dend.set_xlabel("Distance (1 - |cos|)")

    # ── Panel 2: Cluster-ordered heatmap ───────────────────────────────
    ax_heat = fig.add_subplot(gs[1])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    reordered = cos_sim[np.ix_(cluster_order, cluster_order)]
    im = ax_heat.imshow(reordered, cmap="RdBu_r", norm=norm, aspect="equal", interpolation="nearest")

    # Label every 4th head with original head ID
    tick_pos = list(range(0, NUM_HEADS, 4))
    ax_heat.set_xticks(tick_pos)
    ax_heat.set_xticklabels([f"{cluster_order[h]}" for h in tick_pos], fontsize=5, rotation=90)
    ax_heat.set_yticks(tick_pos)
    ax_heat.set_yticklabels([f"{cluster_order[h]}" for h in tick_pos], fontsize=5)

    # Mark significant heads on axes
    for idx_in_order, orig_h in enumerate(cluster_order):
        sigma = (frobs[orig_h] - mean_f) / std_f if std_f > 0 else 0
        if sigma >= 2:
            ax_heat.plot(-2, idx_in_order, 's', color='red', markersize=3, clip_on=False)
            ax_heat.plot(idx_in_order, -2, 's', color='red', markersize=3, clip_on=False)
        elif sigma >= 1:
            ax_heat.plot(-2, idx_in_order, 's', color='darkorange', markersize=2, clip_on=False)
            ax_heat.plot(idx_in_order, -2, 's', color='darkorange', markersize=2, clip_on=False)

    n_clusters = len(clusters)
    ax_heat.set_title(f"{model_name.upper()} Layer {layer}: Cluster-Ordered Coherence (latent space)\n"
                      f"{n_clusters} clusters with 2+ heads — axes = original head IDs",
                      fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, label="cos(dir_i, dir_j)", shrink=0.7)

    # ── Panel 3: Top aligned pairs table ───────────────────────────────
    ax_table = fig.add_subplot(gs[2])
    ax_table.axis('off')

    # Collect all pairs sorted by |cos|
    pairs = []
    for i in range(NUM_HEADS):
        for j in range(i+1, NUM_HEADS):
            pairs.append((i, j, cos_sim[i, j]))
    pairs.sort(key=lambda x: -abs(x[2]))

    # Top 30 most aligned/anti-aligned pairs
    lines = [f"{'Pair':>12s}  {'cos':>6s}  {'‖Δ‖_i':>6s}  {'‖Δ‖_j':>6s}  {'type':>5s}"]
    lines.append("-" * 48)
    for i, j, c in pairs[:30]:
        s_i = (frobs[i] - mean_f) / std_f if std_f > 0 else 0
        s_j = (frobs[j] - mean_f) / std_f if std_f > 0 else 0
        sig_i = "**" if s_i >= 2 else "*" if s_i >= 1 else ""
        sig_j = "**" if s_j >= 2 else "*" if s_j >= 1 else ""
        atype = "align" if c > 0 else "anti"
        lines.append(f"H{i}{sig_i:2s}-H{j}{sig_j:2s}  {c:+.3f}  {frobs[i]:.3f}  {frobs[j]:.3f}  {atype}")

    ax_table.text(0.02, 0.98, "Top 30 most aligned/anti-aligned pairs\n" + "\n".join(lines),
                  transform=ax_table.transAxes, fontsize=7, fontfamily="monospace",
                  verticalalignment="top")

    plt.tight_layout()
    path = PLOTS / f"{model_name}_L{layer}_Q_coherence_all_heads.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → {path}")

    # ── Console output: cluster summary ────────────────────────────────
    if clusters:
        print(f"    {n_clusters} clusters (|cos| > 0.5):")
        for cid, heads in sorted(clusters.items(), key=lambda x: -len(x[1]))[:10]:
            head_strs = []
            for h in sorted(heads):
                sigma = (frobs[h] - mean_f) / std_f if std_f > 0 else 0
                marker = "**" if sigma >= 2 else "*" if sigma >= 1 else ""
                head_strs.append(f"H{h}{marker}")
            sub = cos_sim[np.ix_(heads, heads)]
            avg_cos = (sub.sum() - len(heads)) / (len(heads) * (len(heads) - 1))
            print(f"      C{cid} ({len(heads)} heads, avg cos={avg_cos:.3f}): "
                  f"{', '.join(head_strs[:20])}"
                  f"{'...' if len(head_strs) > 20 else ''}")

        # Top 10 most aligned pairs involving significant heads
        print(f"    Top aligned pairs (>1σ heads):")
        shown = 0
        for i, j, c in pairs:
            s_i = (frobs[i] - mean_f) / std_f if std_f > 0 else 0
            s_j = (frobs[j] - mean_f) / std_f if std_f > 0 else 0
            if s_i >= 1.0 or s_j >= 1.0:
                sig_i = "**" if s_i >= 2 else "*"
                sig_j = "**" if s_j >= 2 else "*" if s_j >= 1 else ""
                print(f"      H{i}{sig_i}-H{j}{sig_j}: cos={c:+.3f}")
                shown += 1
                if shown >= 10:
                    break
    else:
        print(f"    No clusters found with |cos| > 0.5")

    return clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(11)))
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    print(f"Model: {args.model}")

    all_clusters = {}
    for layer in args.layers:
        print(f"\n  Layer {layer}:")
        cos_sim, frobs, spectral_vals = compute_coherence(model_dir, layer)
        clusters = plot_coherence(args.model, layer, cos_sim, frobs, spectral_vals)
        all_clusters[layer] = clusters

    # ── Summary across layers ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CROSS-LAYER SUMMARY for {args.model.upper()}")
    print(f"{'='*70}")
    for layer in args.layers:
        c = all_clusters.get(layer, {})
        n_cls = len(c)
        max_size = max((len(v) for v in c.values()), default=0)
        total_clustered = sum(len(v) for v in c.values())
        print(f"  L{layer:2d}: {n_cls:2d} clusters, {total_clustered:3d}/128 heads clustered, "
              f"largest cluster: {max_size} heads")


if __name__ == "__main__":
    main()
