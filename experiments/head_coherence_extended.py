#!/usr/bin/env python3
"""Extended head coherence maps — covers all available layers not yet plotted.

Existing coverage: {0, 2, 4, 7, 9, 12, 52, 55, 58, 60}
This script fills the gaps: 1, 3, 5, 6, 8, 10, 11, 19-21, 28-51, 53, 54, 56, 57, 59

Same methodology as head_coherence_map.py:
  1. Per-head Q delta extraction (128 heads × 192×1536)
  2. Remove top-3 shared PCs (reveal secondary structure)
  3. Cosine similarity on residuals → 128×128 heatmap
  4. Hierarchical clustering + dendrogram + top-pair table

Usage:
    python experiments/head_coherence_extended.py --model m3
    python experiments/head_coherence_extended.py --model m3 --layers 28 29 30 31 32
    python experiments/head_coherence_extended.py --model m1 --layers 19 20 21
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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from scipy.spatial.distance import squareform

SSD = Path("/Volumes/OmarWork/JSLLM")
OUT_DIR = Path("experiments/EXP-016_cross_layer_story/head_coherence")
PLOTS = OUT_DIR / "plots"
CLUSTERS = OUT_DIR / "clusters"
PLOTS.mkdir(parents=True, exist_ok=True)
CLUSTERS.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
BLOCK_SIZE = 128

# Layers already covered by the original run
ALREADY_DONE = {0, 2, 4, 7, 9, 12, 52, 55, 58, 60}

# All layers where both base and model have q_b_proj available
ALL_AVAILABLE = list(range(0, 13)) + [19, 20, 21] + list(range(28, 61))


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
    shard = idx["weight_map"][name]
    t = _load_shard(Path(model_dir) / shard)
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(Path(model_dir) / ss))[sn]
        return dequant_fp8(w, s)
    return w.float()


def compute_coherence(model_dir, layer):
    """Per-head Q delta similarity after removing shared dominant directions."""
    print(f"    Loading q_b_proj...", flush=True)
    qb_base = load_weight(BASE_DIR, f"model.layers.{layer}.self_attn.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"model.layers.{layer}.self_attn.q_b_proj.weight")
    qb_delta = qb_model - qb_base
    del qb_base, qb_model

    print(f"    Extracting per-head deltas...", flush=True)
    frobs = []
    spectral_vals = []
    flat_deltas = []

    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        frobs.append(torch.norm(d).item())
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        spectral_vals.append(S[0].item())
        flat_deltas.append(d.reshape(-1).clone())

    del qb_delta
    flat_deltas = torch.stack(flat_deltas)  # (128, 192*1536)

    # Remove top-3 shared PCs
    print(f"    Removing shared dominant directions (top-3 PCs)...", flush=True)
    mean_delta = flat_deltas.mean(dim=0, keepdim=True)
    centered = flat_deltas - mean_delta
    U_shared, S_shared, Vh_shared = torch.linalg.svd(centered, full_matrices=False)
    print(f"    Shared singular values: {[f'{v:.2f}' for v in S_shared[:10].tolist()]}")

    n_remove = 3
    proj_basis = Vh_shared[:n_remove, :]
    residuals = centered - (centered @ proj_basis.T) @ proj_basis

    norms = torch.norm(residuals, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-10)
    residuals_normed = residuals / norms

    cos_sim = (residuals_normed @ residuals_normed.T).numpy()

    return cos_sim, np.array(frobs), np.array(spectral_vals)


def find_clusters(cos_sim, frobs, cos_thresh=0.5):
    """Hierarchical clustering on 1 - |cos| distance."""
    dist = 1.0 - np.abs(cos_sim)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    dist = (dist + dist.T) / 2

    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - cos_thresh, criterion="distance")

    clusters = {}
    for h, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(h)
    clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

    return clusters, Z, labels


def plot_coherence(model_name, layer, cos_sim, frobs, spectral_vals):
    """3-panel figure: dendrogram | cluster-ordered heatmap | top pairs."""
    mean_f, std_f = np.mean(frobs), np.std(frobs)
    clusters, Z, labels = find_clusters(cos_sim, frobs)
    cluster_order = leaves_list(Z)

    fig = plt.figure(figsize=(30, 12))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 1.2], wspace=0.15)

    # ── Panel 1: Dendrogram ────────────────────────────────────────────
    ax_dend = fig.add_subplot(gs[0])
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
    leaf_labels = ax_dend.get_yticklabels()
    for lbl in leaf_labels:
        try:
            h = int(lbl.get_text())
            lbl.set_color(head_colors.get(h, 'black'))
            lbl.set_fontweight('bold' if head_colors.get(h) in ('red', 'darkorange') else 'normal')
        except ValueError:
            pass
    ax_dend.set_title(f"Dendrogram (latent space)\nred/>2\u03c3, orange/>1\u03c3", fontsize=11, fontweight="bold")
    ax_dend.set_xlabel("Distance (1 - |cos|)")

    # ── Panel 2: Cluster-ordered heatmap ───────────────────────────────
    ax_heat = fig.add_subplot(gs[1])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    reordered = cos_sim[np.ix_(cluster_order, cluster_order)]
    im = ax_heat.imshow(reordered, cmap="RdBu_r", norm=norm, aspect="equal", interpolation="nearest")

    tick_pos = list(range(0, NUM_HEADS, 4))
    ax_heat.set_xticks(tick_pos)
    ax_heat.set_xticklabels([f"{cluster_order[h]}" for h in tick_pos], fontsize=5, rotation=90)
    ax_heat.set_yticks(tick_pos)
    ax_heat.set_yticklabels([f"{cluster_order[h]}" for h in tick_pos], fontsize=5)

    for idx_in_order, orig_h in enumerate(cluster_order):
        sigma = (frobs[orig_h] - mean_f) / std_f if std_f > 0 else 0
        if sigma >= 2:
            ax_heat.plot(-2, idx_in_order, 's', color='red', markersize=3, clip_on=False)
            ax_heat.plot(idx_in_order, -2, 's', color='red', markersize=3, clip_on=False)
        elif sigma >= 1:
            ax_heat.plot(-2, idx_in_order, 's', color='darkorange', markersize=2, clip_on=False)
            ax_heat.plot(idx_in_order, -2, 's', color='darkorange', markersize=2, clip_on=False)

    n_clusters = len(clusters)
    ax_heat.set_title(f"{model_name.upper()} Layer {layer}: Cluster-Ordered Head Coherence\n"
                      f"{n_clusters} clusters with 2+ heads",
                      fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, label="cos(dir_i, dir_j)", shrink=0.7)

    # ── Panel 3: Top aligned pairs table ───────────────────────────────
    ax_table = fig.add_subplot(gs[2])
    ax_table.axis('off')

    pairs = []
    for i in range(NUM_HEADS):
        for j in range(i+1, NUM_HEADS):
            pairs.append((i, j, cos_sim[i, j]))
    pairs.sort(key=lambda x: -abs(x[2]))

    lines = [f"{'Pair':>12s}  {'cos':>6s}  {'frob_i':>7s}  {'frob_j':>7s}  {'type':>5s}"]
    lines.append("-" * 50)
    for i, j, c in pairs[:30]:
        s_i = (frobs[i] - mean_f) / std_f if std_f > 0 else 0
        s_j = (frobs[j] - mean_f) / std_f if std_f > 0 else 0
        sig_i = "**" if s_i >= 2 else "*" if s_i >= 1 else ""
        sig_j = "**" if s_j >= 2 else "*" if s_j >= 1 else ""
        atype = "align" if c > 0 else "anti"
        lines.append(f"H{i}{sig_i:2s}-H{j}{sig_j:2s}  {c:+.3f}  {frobs[i]:.4f}  {frobs[j]:.4f}  {atype}")

    ax_table.text(0.02, 0.98, "Top 30 most aligned/anti-aligned pairs\n" + "\n".join(lines),
                  transform=ax_table.transAxes, fontsize=7, fontfamily="monospace",
                  verticalalignment="top")

    plt.tight_layout()
    path = PLOTS / f"head_coherence_{model_name}_L{layer}.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    -> {path}")

    # ── Console cluster summary ────────────────────────────────────────
    if clusters:
        print(f"    {n_clusters} clusters (|cos| > 0.5):")
        for cid, heads in sorted(clusters.items(), key=lambda x: -len(x[1]))[:10]:
            head_strs = []
            for h in sorted(heads):
                sigma = (frobs[h] - mean_f) / std_f if std_f > 0 else 0
                marker = "**" if sigma >= 2 else "*" if sigma >= 1 else ""
                head_strs.append(f"H{h}{marker}")
            sub = cos_sim[np.ix_(heads, heads)]
            avg_cos = (sub.sum() - len(heads)) / max(len(heads) * (len(heads) - 1), 1)
            print(f"      C{cid} ({len(heads)} heads, avg cos={avg_cos:.3f}): "
                  f"{', '.join(head_strs[:20])}"
                  f"{'...' if len(head_strs) > 20 else ''}")

        print(f"    Top aligned pairs (>1sigma heads):")
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


def save_cluster_summary(model_name, layer, cos_sim, frobs, clusters):
    """Save cluster summary to text file."""
    mean_f, std_f = np.mean(frobs), np.std(frobs)
    path = CLUSTERS / f"{model_name}_L{layer}_head_clusters_extended.txt"
    with open(path, "w") as f:
        f.write(f"Head Coherence Clusters: {model_name.upper()} Layer {layer}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Frobenius norm stats: mean={mean_f:.4f}, std={std_f:.4f}\n")
        f.write(f"Heads >2sigma: {[h for h in range(NUM_HEADS) if (frobs[h]-mean_f)/std_f >= 2]}\n")
        f.write(f"Heads >1sigma: {[h for h in range(NUM_HEADS) if 1 <= (frobs[h]-mean_f)/std_f < 2]}\n\n")

        if clusters:
            f.write(f"{len(clusters)} clusters found:\n\n")
            for cid, heads in sorted(clusters.items(), key=lambda x: -len(x[1])):
                sub = cos_sim[np.ix_(heads, heads)]
                avg_cos = (sub.sum() - len(heads)) / max(len(heads) * (len(heads) - 1), 1)
                head_info = []
                for h in sorted(heads):
                    sigma = (frobs[h] - mean_f) / std_f if std_f > 0 else 0
                    head_info.append(f"H{h}({sigma:.1f}s)")
                f.write(f"  Cluster {cid} ({len(heads)} heads, avg_cos={avg_cos:.3f}):\n")
                f.write(f"    {', '.join(head_info)}\n\n")
        else:
            f.write("No clusters found.\n")

    print(f"    -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layers to analyze. Default: all uncovered layers.")
    parser.add_argument("--include-done", action="store_true",
                        help="Re-run layers that already have plots")
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]

    if args.layers is not None:
        layers = args.layers
    else:
        layers = [l for l in ALL_AVAILABLE if args.include_done or l not in ALREADY_DONE]

    print(f"Model: {args.model}")
    print(f"Layers to process: {layers}")
    print(f"Output: {PLOTS}")

    all_clusters = {}
    for layer in layers:
        print(f"\n  Layer {layer}:")
        try:
            cos_sim, frobs, spectral_vals = compute_coherence(model_dir, layer)
            clusters = plot_coherence(args.model, layer, cos_sim, frobs, spectral_vals)
            save_cluster_summary(args.model, layer, cos_sim, frobs, clusters)
            all_clusters[layer] = clusters
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CROSS-LAYER SUMMARY for {args.model.upper()}")
    print(f"{'='*70}")
    for layer in layers:
        c = all_clusters.get(layer, {})
        n_cls = len(c)
        max_size = max((len(v) for v in c.values()), default=0)
        total_clustered = sum(len(v) for v in c.values())
        print(f"  L{layer:2d}: {n_cls:2d} clusters, {total_clustered:3d}/128 heads clustered, "
              f"largest cluster: {max_size} heads")


if __name__ == "__main__":
    main()
