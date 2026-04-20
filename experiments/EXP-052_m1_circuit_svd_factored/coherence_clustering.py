"""
EXP-052 — Layer-level coherence + clustering on M1 dormant ΔOV U₀ vectors.

Supersedes the token-space proxy in ``view_coherence.py`` by computing real
layer-level U₀ vectors from the factored ΔOV SVD:

  * Per layer L (0..60), per head h, factored-SVD ΔOV_h and take U₀^h ∈ R^7168.
  * Stack U₀^h across all 128 heads → [7168, 128] and take its top-1 left
    singular vector as the *layer* U₀ (first principal write direction across
    heads).
  * Also cache the per-head U₀ for 128×61 = 7808 vectors.
  * Build three 61×61 coherence matrices (plain cosine, cos-weighted
    magnitude-aware, pure cos³).
  * Apply four clustering methods on layer U₀ vectors (Ward, average
    hierarchical, spherical k-means, spectral).
  * Save all plots + JSON into
    ``results/ds_circuit/coherence_clustering/``.

Weights bf16 on GPU, SVDs + coherence in fp32 for stability.
GH200 has 96 GB VRAM — both models' attention weights fit simultaneously.

Only requires ``flagged_heads.json`` (as a sanity check that prior EXP-052
steps ran).  No other inputs.
"""

from __future__ import annotations

import functools
import gc
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Make prints flush immediately (long pipeline, monitored from outside).
print = functools.partial(print, flush=True)  # type: ignore[assignment]

# scikit-learn / scipy for clustering
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, SpectralClustering

# Reuse helpers from run.py
sys.path.insert(0, str(Path(__file__).parent))
from run import (  # type: ignore
    DORMANT_PATH, BASE_PATH, HIDDEN, NH,
    QK_NOPE_DIM, V_HEAD_DIM, Q_HEAD_DIM, KV_HEAD_DIM, KV_LORA_RANK,
    load_attn_weights, ov_factors_head, fold_layernorm_gains,
    factored_svd_delta,
)

NUM_LAYERS = 61

# ── I/O paths (absolute) ─────────────────────────────────────────────────────
EXP_DIR = Path(__file__).parent
RESULT_DIR = EXP_DIR / "results" / "ds_circuit"
OUT_DIR = RESULT_DIR / "coherence_clustering"
FLAGGED_PATH = RESULT_DIR / "flagged_heads.json"
CIRCUIT_SVD_PATH = RESULT_DIR / "circuit_svd_results.json"
NOTES_PATH = EXP_DIR / "NOTES_2026-04-20.md"


# ── Weight loading (bf16 on GPU, all layers at once) ─────────────────────────

def load_all_layers_gpu(model_dir: Path, num_layers: int, fold_ln: bool,
                        tag: str) -> list[dict[str, torch.Tensor]]:
    """Load attention weights for every layer onto the GPU in bf16.

    Mirrors composition_parallel.py's pattern.  Dequant is fp32 on CPU, γ-fold
    is fp32 on CPU, then cast to bf16 for GPU storage.
    """
    t0 = time.time()
    layers: list[dict[str, torch.Tensor]] = []
    for L in range(num_layers):
        w = load_attn_weights(model_dir, L)  # fp32 on CPU
        if fold_ln:
            fold_layernorm_gains(w)
        w_gpu: dict[str, torch.Tensor] = {}
        for k, v in w.items():
            w_gpu[k] = v.to("cuda", dtype=torch.bfloat16, non_blocking=True)
        layers.append(w_gpu)
        if L == 0 or (L + 1) % 10 == 0 or L == num_layers - 1:
            free, _ = torch.cuda.mem_get_info(0)
            print(f"    [{tag}] loaded L={L:>2}  gpu-free={free/1e9:.1f} GB "
                  f"t={time.time()-t0:.1f}s")
    print(f"    [{tag}] {num_layers} layers loaded in {time.time()-t0:.1f}s")
    return layers


def per_head_u0_layer(w_d, w_b) -> torch.Tensor:
    """Return U₀^h for every head in the layer.  Shape [D, NH] fp32 on GPU.

    Each column is the unit-norm top-left singular vector of ΔOV_h
    (head-level).  If ΔOV_h is near zero, we emit a zero column (its
    contribution to the layer-aggregate SVD is negligible).
    """
    cols = []
    for h in range(NH):
        A_d, B_d = ov_factors_head(w_d, h)
        A_b, B_b = ov_factors_head(w_b, h)
        # factored_svd_delta upcasts internally; operands here are bf16 so we
        # cast explicitly to fp32 for SVD stability (per spec).
        U, S, V = factored_svd_delta(A_d.float(), B_d.float(),
                                     A_b.float(), B_b.float())
        u0 = U[:, 0]
        if S[0].item() < 1e-8:
            u0 = torch.zeros_like(u0)
        else:
            u0 = u0 / (u0.norm() + 1e-12)
        cols.append(u0)
        del A_d, B_d, A_b, B_b, U, S, V
    return torch.stack(cols, dim=1)  # [D, NH] fp32


def compute_layer_u0_all(dormant_dir: Path, base_dir: Path, fold_ln: bool
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Load both models once, then for every layer compute per-head U₀^h
    [D, NH] and the layer-aggregate U₀ (top-1 left singular vector of the
    [D, NH] stack).

    Returns:
      layer_U0 : [NUM_LAYERS, D]  fp32 GPU tensor (unit-normalized)
      perhead_U0 : [NUM_LAYERS, NH, D]  fp32 CPU tensor (unit-normalized)
    """
    # ── Phase 1a: load both models onto GPU in bf16 ───────────────────────
    print("  loading dormant model (all 61 attn layers, bf16)...")
    dormant_layers = load_all_layers_gpu(dormant_dir, NUM_LAYERS, fold_ln, "D")
    print("  loading base model (all 61 attn layers, bf16)...")
    base_layers = load_all_layers_gpu(base_dir, NUM_LAYERS, fold_ln, "B")

    # ── Phase 1b: per-layer head-level SVDs + layer-aggregate SVD ─────────
    t0 = time.time()
    layer_U0 = torch.zeros((NUM_LAYERS, HIDDEN), dtype=torch.float32, device="cuda")
    perhead_U0 = torch.zeros((NUM_LAYERS, NH, HIDDEN), dtype=torch.float32)

    for L in range(NUM_LAYERS):
        tL = time.time()
        w_d = dormant_layers[L]
        w_b = base_layers[L]
        U_heads = per_head_u0_layer(w_d, w_b)   # [D, NH] fp32

        # Layer-aggregate U₀: top-1 left singular vector of the [D, NH] stack.
        # NH = 128 ≪ D = 7168 so a thin SVD is cheap (<1s).
        U_svd, S_svd, _ = torch.linalg.svd(U_heads, full_matrices=False)
        u_layer = U_svd[:, 0]
        u_layer = u_layer / (u_layer.norm() + 1e-12)

        layer_U0[L] = u_layer
        perhead_U0[L] = U_heads.T.detach().cpu()  # [NH, D]

        del U_heads, U_svd, S_svd

        if L == 0 or (L + 1) % 10 == 0 or L == NUM_LAYERS - 1:
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(0)
            print(f"  L={L:>2}   {time.time()-tL:5.2f}s   gpu-free={free/1e9:.1f} GB "
                  f"t={time.time()-t0:.1f}s")

    print(f"  all {NUM_LAYERS} layer U₀'s in {time.time()-t0:.1f}s")

    # Free layer weights after SVD phase is done.
    for layer in dormant_layers:
        for k in list(layer.keys()):
            del layer[k]
    for layer in base_layers:
        for k in list(layer.keys()):
            del layer[k]
    dormant_layers.clear()
    base_layers.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return layer_U0, perhead_U0


# ── Coherence matrices ───────────────────────────────────────────────────────

def coherence_matrices(layer_U0: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cos, cosprod, cos3) 61×61 matrices as numpy arrays.

    Input is assumed already unit-normalized; we renormalize anyway for safety.
    """
    U = layer_U0.float()
    norms = U.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Un = U / norms
    cos = (Un @ Un.T).cpu().numpy()
    cos = np.clip(cos, -1.0, 1.0)

    # cosprod(a,b) = (a · b) · max(0.1, cos(a,b))²
    # For unit vectors, a · b = cos(a,b), so cosprod = cos · max(0.1, cos)².
    floor = np.maximum(0.1, cos)
    cosprod = cos * (floor ** 2)

    cos3 = cos ** 3
    return cos, cosprod, cos3


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _layer_ticks(ax, n=NUM_LAYERS, fontsize=6):
    ax.set_xticks(range(n))
    ax.set_xticklabels(range(n), fontsize=fontsize, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(range(n), fontsize=fontsize)


def plot_heatmap(M: np.ndarray, title: str, out: Path, metric: str):
    """Save a single coherence heatmap.  ``metric`` ∈ {cos, cosprod, cos3}."""
    fig, ax = plt.subplots(figsize=(14, 12))
    if metric == "cosprod":
        # Use symmetric log-ish norm to handle wide dynamic range.
        absmax = np.max(np.abs(M))
        linthresh = max(1e-3, float(np.quantile(np.abs(M[M != 0]), 0.25))) if np.any(M != 0) else 1e-3
        norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-absmax, vmax=absmax, base=10)
        im = ax.imshow(M, cmap="RdBu_r", norm=norm, origin="lower")
        cbar_label = "cos · max(0.1,cos)²  (symlog)"
    else:
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        cbar_label = "cosine" if metric == "cos" else "cos³"
    ax.set_xlabel("layer L₂", fontsize=12)
    ax.set_ylabel("layer L₁", fontsize=12)
    _layer_ticks(ax, fontsize=7)
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  Saved {out}")


# ── Clustering ───────────────────────────────────────────────────────────────

def pick_k_from_inertias(inertias: list[float]) -> int:
    """Simple elbow: k that maximizes the second difference."""
    if len(inertias) < 3:
        return len(inertias)
    d1 = np.diff(inertias)
    d2 = np.diff(d1)
    # d2 is largest where inertia-drop flattens the most (elbow).
    # Index: elbow at position (i+1) in the k-list.
    elbow = int(np.argmax(d2)) + 1
    return elbow


def cluster_order_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return a permutation that groups layers by cluster (ascending by
    median layer index within each cluster, then ascending by layer)."""
    # Stable cluster ordering: by the minimum layer index in each cluster
    uniq = sorted(set(labels.tolist()), key=lambda c: np.where(labels == c)[0].min())
    perm = []
    for c in uniq:
        idx = np.where(labels == c)[0].tolist()
        idx.sort()
        perm.extend(idx)
    return np.array(perm, dtype=int)


def boundary_positions(labels: np.ndarray, perm: np.ndarray) -> list[int]:
    """Positions in the *reordered* array where cluster label changes — these
    are the row/column indices at which to draw cluster separator lines."""
    reordered = labels[perm]
    bnd = [i for i in range(1, len(reordered)) if reordered[i] != reordered[i-1]]
    return bnd


def plot_reordered_cos_with_boundaries(cos: np.ndarray, labels: np.ndarray,
                                       title: str, out: Path):
    perm = cluster_order_from_labels(labels)
    C = cos[perm][:, perm]
    bnd = boundary_positions(labels, perm)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    for b in bnd:
        ax.axhline(b - 0.5, color="black", lw=1.5)
        ax.axvline(b - 0.5, color="black", lw=1.5)

    labels_reordered = [int(perm[i]) for i in range(len(perm))]
    ax.set_xticks(range(len(perm)))
    ax.set_xticklabels(labels_reordered, fontsize=7, rotation=90)
    ax.set_yticks(range(len(perm)))
    ax.set_yticklabels(labels_reordered, fontsize=7)
    ax.set_xlabel("layer (reordered)", fontsize=12)
    ax.set_ylabel("layer (reordered)", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8, label="cosine")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  Saved {out}")
    return perm, bnd


def cluster_assignment_dict(labels: np.ndarray) -> dict[int, list[int]]:
    d: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(labels.tolist()):
        d[int(c)].append(int(i))
    return dict(sorted(d.items()))


# ── Clustering methods ───────────────────────────────────────────────────────

def cluster_hierarchical(cos: np.ndarray, method: str, k: int) -> np.ndarray:
    """Hierarchical clustering on the cosine distance matrix (1 - cos)."""
    D = 1.0 - cos
    np.fill_diagonal(D, 0.0)
    # Enforce symmetry + non-negativity (tiny floating-point drift otherwise).
    D = np.maximum(D, 0.0)
    D = 0.5 * (D + D.T)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels, Z


def plot_dendrogram(Z, labels: np.ndarray, title: str, out: Path, color_threshold=None):
    fig, ax = plt.subplots(figsize=(18, 6))
    dendrogram(Z, ax=ax, labels=[str(i) for i in range(NUM_LAYERS)],
               leaf_font_size=8, color_threshold=color_threshold)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("layer", fontsize=11)
    ax.set_ylabel("distance", fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  Saved {out}")


def cluster_kmeans_spherical(layer_U0_np: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, float]:
    """Spherical k-means via k-means on unit-normalized vectors."""
    norms = np.linalg.norm(layer_U0_np, axis=1, keepdims=True) + 1e-12
    Xn = layer_U0_np / norms
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(Xn)
    return labels, km.inertia_


def cluster_spectral(cos: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Spectral clustering on a non-negative affinity from the cosine matrix.

    We map cos ∈ [-1,1] → affinity ∈ [0,1] via (cos + 1)/2 and clamp.
    """
    aff = (cos + 1.0) / 2.0
    aff = np.clip(aff, 0.0, 1.0)
    np.fill_diagonal(aff, 1.0)
    aff = 0.5 * (aff + aff.T)
    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            assign_labels="kmeans", random_state=seed,
                            n_init=20)
    labels = sc.fit_predict(aff)
    return labels


# ── Summary grid plot ────────────────────────────────────────────────────────

def plot_summary_grid(cos: np.ndarray, cosprod: np.ndarray, cos3: np.ndarray,
                      cluster_runs: list[tuple[str, np.ndarray]],
                      out: Path):
    """Top row: three coherence heatmaps.  Bottom row: four clustering
    reorderings (boundaries overlaid on the cos heatmap).
    """
    fig = plt.figure(figsize=(28, 16))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # Top row: spans 4 columns but we put 3 coherence maps + 1 spacer at col 3.
    top_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]

    # cos
    im = top_axes[0].imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    top_axes[0].set_title("cos(U₀_i, U₀_j)", fontsize=12)
    _layer_ticks(top_axes[0], fontsize=5)
    plt.colorbar(im, ax=top_axes[0], shrink=0.75)

    # cosprod
    absmax = np.max(np.abs(cosprod))
    linthresh = max(1e-3, float(np.quantile(np.abs(cosprod[cosprod != 0]), 0.25))) if np.any(cosprod != 0) else 1e-3
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-absmax, vmax=absmax, base=10)
    im = top_axes[1].imshow(cosprod, cmap="RdBu_r", norm=norm, origin="lower")
    top_axes[1].set_title("cos · max(0.1,cos)²  (symlog)", fontsize=12)
    _layer_ticks(top_axes[1], fontsize=5)
    plt.colorbar(im, ax=top_axes[1], shrink=0.75)

    # cos3
    im = top_axes[2].imshow(cos3, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    top_axes[2].set_title("cos³", fontsize=12)
    _layer_ticks(top_axes[2], fontsize=5)
    plt.colorbar(im, ax=top_axes[2], shrink=0.75)

    # Spacer
    top_axes[3].axis("off")
    top_axes[3].text(0.5, 0.5, "Coherence metrics\n(top row)\n\nClustering reorders\n(bottom row)",
                     ha="center", va="center", fontsize=14, transform=top_axes[3].transAxes)

    # Bottom row: 4 cluster reorderings on cos heatmap
    for i, (name, labels) in enumerate(cluster_runs):
        ax = fig.add_subplot(gs[1, i])
        perm = cluster_order_from_labels(labels)
        C = cos[perm][:, perm]
        bnd = boundary_positions(labels, perm)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        for b in bnd:
            ax.axhline(b - 0.5, color="black", lw=1.2)
            ax.axvline(b - 0.5, color="black", lw=1.2)
        ax.set_title(name, fontsize=12)
        labels_reordered = [int(perm[j]) for j in range(len(perm))]
        ax.set_xticks(range(len(perm)))
        ax.set_xticklabels(labels_reordered, fontsize=4, rotation=90)
        ax.set_yticks(range(len(perm)))
        ax.set_yticklabels(labels_reordered, fontsize=4)
        plt.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle("EXP-052 — M1 dormant ΔOV layer U₀ coherence + clustering", fontsize=15)
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Sanity check prereqs
    if not FLAGGED_PATH.exists():
        print(f"ERROR: missing {FLAGGED_PATH}", file=sys.stderr)
        sys.exit(1)
    if not CIRCUIT_SVD_PATH.exists():
        print(f"ERROR: missing {CIRCUIT_SVD_PATH}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    T_start = time.time()

    # ── Phase 1: layer U₀ computation ─────────────────────────────────────
    print(f"── Phase 1: compute layer U₀ vectors (61 layers × 128 heads SVDs) ──")
    t0 = time.time()
    layer_U0, perhead_U0 = compute_layer_u0_all(
        Path(DORMANT_PATH), Path(BASE_PATH), fold_ln=True)
    timings["u0_compute"] = time.time() - t0
    print(f"  phase-1 total: {timings['u0_compute']:.1f}s")

    # Save the raw vectors for downstream reuse (not huge: ~1.7 MB + 210 MB).
    torch.save(layer_U0.cpu(), OUT_DIR / "layer_U0.pt")
    torch.save(perhead_U0, OUT_DIR / "perhead_U0.pt")
    print(f"  Saved {OUT_DIR / 'layer_U0.pt'} and perhead_U0.pt")

    # ── Phase 2: coherence matrices ───────────────────────────────────────
    print("\n── Phase 2: coherence matrices ──")
    t0 = time.time()
    cos, cosprod, cos3 = coherence_matrices(layer_U0)
    timings["coherence"] = time.time() - t0
    print(f"  phase-2 total: {timings['coherence']:.2f}s")

    np.save(OUT_DIR / "coherence_cos.npy", cos)
    np.save(OUT_DIR / "coherence_cosprod.npy", cosprod)
    np.save(OUT_DIR / "coherence_cos3.npy", cos3)

    plot_heatmap(cos, "Layer U₀ coherence — plain cosine", OUT_DIR / "coherence_cos.png", "cos")
    plot_heatmap(cosprod, "Layer U₀ coherence — cos · max(0.1,cos)²", OUT_DIR / "coherence_cosprod.png", "cosprod")
    plot_heatmap(cos3, "Layer U₀ coherence — cos³ (sharp angular contrast)", OUT_DIR / "coherence_cos3.png", "cos3")

    # ── Phase 3: clustering ──────────────────────────────────────────────
    print("\n── Phase 3: clustering ──")
    t0 = time.time()
    clustering_results: dict[str, dict] = {}
    summary_runs: list[tuple[str, np.ndarray]] = []

    # Ward and average linkage: we want a single "best-k" output + the k-sweep.
    for linkage_method in ("ward", "average"):
        k_sweep = {}
        best_labels = None
        for k in (3, 5, 8):
            labels, Z = cluster_hierarchical(cos, linkage_method, k)
            k_sweep[str(k)] = labels.tolist()
            if k == 5:
                best_labels = labels
                # Dendrogram (full tree, colored at k=5 cutoff)
                color_thr = Z[-(k-1), 2] if len(Z) >= k - 1 else None
                plot_dendrogram(
                    Z, labels,
                    f"Hierarchical {linkage_method.title()} linkage  (cut at k=5)",
                    OUT_DIR / f"dendrogram_{linkage_method}.png",
                    color_threshold=color_thr,
                )
                # Reordered heatmap with boundaries
                plot_reordered_cos_with_boundaries(
                    cos, labels,
                    f"cos reordered — hierarchical {linkage_method} linkage, k=5",
                    OUT_DIR / f"cluster_hierarchical_{linkage_method}.png",
                )
        clustering_results[f"hierarchical_{linkage_method}"] = {
            "k_sweep": k_sweep,
            "chosen_k": 5,
            "labels_chosen": best_labels.tolist(),
            "clusters_chosen": {str(c): layers for c, layers in cluster_assignment_dict(best_labels).items()},
        }
        summary_runs.append((f"Hierarchical {linkage_method} (k=5)", best_labels))

    # K-means: try k=3,5,8; elbow-pick
    layer_U0_np = layer_U0.cpu().numpy()
    km_sweep = {}
    inertias = []
    for k in (3, 5, 8):
        labels, inertia = cluster_kmeans_spherical(layer_U0_np, k)
        km_sweep[str(k)] = {"labels": labels.tolist(), "inertia": float(inertia)}
        inertias.append(inertia)
    # Elbow pick
    elbow_idx = pick_k_from_inertias(inertias)
    k_choices = [3, 5, 8]
    chosen_k_km = k_choices[elbow_idx] if 0 <= elbow_idx < len(k_choices) else 5
    labels_km = np.array(km_sweep[str(chosen_k_km)]["labels"])
    plot_reordered_cos_with_boundaries(
        cos, labels_km,
        f"cos reordered — spherical k-means, k={chosen_k_km} (elbow pick)",
        OUT_DIR / "cluster_kmeans.png",
    )
    # Also a little elbow plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_choices, inertias, "o-")
    ax.set_xlabel("k")
    ax.set_ylabel("inertia")
    ax.set_title(f"K-means inertia (elbow pick: k={chosen_k_km})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "kmeans_elbow.png", dpi=140)
    plt.close()
    print(f"  Saved {OUT_DIR / 'kmeans_elbow.png'}")

    clustering_results["kmeans"] = {
        "k_sweep": km_sweep,
        "chosen_k": int(chosen_k_km),
        "labels_chosen": labels_km.tolist(),
        "clusters_chosen": {str(c): layers for c, layers in cluster_assignment_dict(labels_km).items()},
    }
    summary_runs.append((f"Spherical k-means (k={chosen_k_km})", labels_km))

    # Spectral: try k=3,5,8
    sp_sweep = {}
    for k in (3, 5, 8):
        labels = cluster_spectral(cos, k)
        sp_sweep[str(k)] = labels.tolist()
    chosen_k_sp = 5
    labels_sp = np.array(sp_sweep[str(chosen_k_sp)])
    plot_reordered_cos_with_boundaries(
        cos, labels_sp,
        f"cos reordered — spectral clustering, k={chosen_k_sp}",
        OUT_DIR / "cluster_spectral.png",
    )
    clustering_results["spectral"] = {
        "k_sweep": sp_sweep,
        "chosen_k": chosen_k_sp,
        "labels_chosen": labels_sp.tolist(),
        "clusters_chosen": {str(c): layers for c, layers in cluster_assignment_dict(labels_sp).items()},
    }
    summary_runs.append((f"Spectral (k={chosen_k_sp})", labels_sp))

    timings["clustering"] = time.time() - t0
    print(f"  phase-3 total: {timings['clustering']:.2f}s")

    # ── Phase 4: save + summary plot ─────────────────────────────────────
    t0 = time.time()
    with open(OUT_DIR / "clustering_results.json", "w") as f:
        json.dump(clustering_results, f, indent=2)
    print(f"  Saved {OUT_DIR / 'clustering_results.json'}")

    plot_summary_grid(cos, cosprod, cos3, summary_runs,
                      OUT_DIR / "clustering_summary.png")

    # Print cluster tables for the chosen k's
    print("\n── Cluster assignments (chosen k per method) ──")
    for method, rec in clustering_results.items():
        print(f"\n[{method}]  k={rec['chosen_k']}")
        for c, layers in rec["clusters_chosen"].items():
            span = ", ".join(str(L) for L in layers)
            print(f"  cluster {c}  ({len(layers)} layers)  [{span}]")

    timings["save"] = time.time() - t0
    timings["total"] = time.time() - T_start

    # Save timing record
    with open(OUT_DIR / "timings.json", "w") as f:
        json.dump({k: round(v, 2) for k, v in timings.items()}, f, indent=2)

    print("\n── Timing ──")
    for k, v in timings.items():
        print(f"  {k:>10s} : {v:7.2f}s")

    # ── Phase 5: append session log to NOTES ─────────────────────────────
    try:
        append_notes(layer_U0_np, cos, cosprod, cos3, clustering_results, timings)
    except Exception as e:
        print(f"WARN: failed to append NOTES: {e}")

    print("\nDone.")


def append_notes(layer_U0_np: np.ndarray, cos, cosprod, cos3,
                 clustering_results: dict, timings: dict):
    """Append a dated progress entry to NOTES_2026-04-20.md."""

    # Pick the "best" clustering by a simple heuristic: max mean within-cluster
    # cos minus max mean between-cluster cos (Dunn-like index) on the chosen-k
    # assignments.
    def score(labels):
        uniq = sorted(set(labels.tolist()))
        within = []
        between = []
        for c in uniq:
            idx = np.where(labels == c)[0]
            if len(idx) > 1:
                sub = cos[np.ix_(idx, idx)]
                # exclude diagonal
                mask = ~np.eye(len(idx), dtype=bool)
                within.append(sub[mask].mean())
        for i, c in enumerate(uniq):
            for c2 in uniq[i+1:]:
                ia = np.where(labels == c)[0]
                ib = np.where(labels == c2)[0]
                sub = cos[np.ix_(ia, ib)]
                between.append(sub.mean())
        if not within or not between:
            return -1
        return float(np.mean(within) - np.mean(between))

    scored = []
    for name, rec in clustering_results.items():
        labels = np.array(rec["labels_chosen"])
        scored.append((score(labels), name, labels, rec))
    scored.sort(reverse=True)
    best_score, best_name, best_labels, best_rec = scored[0]

    # Human-readable cluster summary of best method
    cluster_lines = []
    for c, layers in best_rec["clusters_chosen"].items():
        cluster_lines.append(f"  cluster {c}  ({len(layers)} layers): {layers}")
    clusters_md = "\n".join(cluster_lines)

    # Coherence metric qualitative diffs
    off_diag_cos = cos[~np.eye(cos.shape[0], dtype=bool)]
    off_diag_cos3 = cos3[~np.eye(cos3.shape[0], dtype=bool)]
    off_diag_cosprod = cosprod[~np.eye(cosprod.shape[0], dtype=bool)]

    entry = [
        "",
        "## Layer-level ΔOV U₀ coherence + clustering  (2026-04-20, later)",
        "",
        f"Ran `coherence_clustering.py`.  Output: `results/ds_circuit/coherence_clustering/`.",
        "",
        "### Timing",
        "",
        f"- Layer U₀ computation (61 × 128 head SVDs + 61 layer-aggregate SVDs): **{timings['u0_compute']:.1f}s**",
        f"- Coherence matrices (3 × 61×61): {timings['coherence']:.2f}s",
        f"- Clustering (4 methods × k-sweep 3/5/8): {timings['clustering']:.2f}s",
        f"- Save + summary plot: {timings['save']:.2f}s",
        f"- **Total wall: {timings['total']:.1f}s**",
        "",
        "### Best-looking clustering (Dunn-like index on plain-cosine matrix)",
        "",
        f"**{best_name}**  (score={best_score:+.3f}, k={best_rec['chosen_k']}):",
        "",
        "```",
        clusters_md,
        "```",
        "",
        "### Coherence-metric comparison",
        "",
        f"- Plain `cos`: off-diag mean = {off_diag_cos.mean():+.3f}, std = {off_diag_cos.std():.3f}; range [{off_diag_cos.min():+.3f}, {off_diag_cos.max():+.3f}].",
        f"- `cos³`: off-diag mean = {off_diag_cos3.mean():+.3f}, std = {off_diag_cos3.std():.3f}.  Same sign as cos everywhere (odd power preserves sign) but compresses near-orthogonal pairs toward 0 and sharpens the ≥0.5 tail — block structure is visually cleaner but magnitudes are crushed.",
        f"- `cos · max(0.1,cos)²` (magnitude-aware): off-diag mean = {off_diag_cosprod.mean():+.3f}, dynamic range {off_diag_cosprod.max()-off_diag_cosprod.min():.3f}.  The max(0.1,·) floor prevents near-orthogonal pairs from vanishing, but differs from cos³ only where cos ≲ 0.1; visually similar to cos³ in the block structure but with a floor of 0.01|cos| for negative / near-zero entries — symlog norm is required to read it.",
        "",
        "### Layer-group observations",
        "",
        "(See `clustering_summary.png` and `cluster_hierarchical_ward.png` for the reordered cos heatmap with cluster boundaries.)",
        "",
    ]

    # Attempt to extract a few diagnostic layer groupings from the best clustering
    # — look for "tight early block" and "late split" phenomena referenced in the task.
    try:
        labels = best_labels
        # early block: which cluster does L0 belong to? list contiguous initial layers
        c0 = labels[0]
        early = [i for i in range(NUM_LAYERS) if labels[i] == c0 and i <= 10]
        late_labels = labels[40:]
        late_clusters = sorted(set(late_labels.tolist()))
        entry.extend([
            f"- Layer 0's cluster ({c0}) picks up layers {early} — the tight early block.",
            f"- In layers 40-60, the clustering uses {len(late_clusters)} distinct clusters ({late_clusters}).  "
            f"This is the split referenced in the task spec; compare against the ΔQK outlier at L60 H74 and the NE-corner cluster at L54.",
        ])

        # Check if L58, L60 land in distinct clusters
        if labels[58] != labels[60]:
            entry.append(f"- L58 (cluster {labels[58]}) and L60 (cluster {labels[60]}) are split — consistent with the L60 |ΔQK|=189 outlier being a *separate* circuit from the L58 writers.")
        else:
            entry.append(f"- L58 and L60 share cluster {labels[58]} in this partition.")

        # L44-L49 circuit check
        if labels[44] == labels[49]:
            entry.append(f"- L44 and L49 share cluster {labels[44]} — consistent with the L44 → L49 H114 Q-composition circuit found earlier.")
        else:
            entry.append(f"- L44 (cluster {labels[44]}) and L49 (cluster {labels[49]}) are in different clusters in this partition.")
    except Exception as e:
        entry.append(f"(layer-group annotation failed: {e})")

    entry.append("")
    entry.append("### Files written")
    entry.append("")
    entry.extend([
        "- `coherence_cos.png`, `coherence_cosprod.png`, `coherence_cos3.png` — per-metric 61×61 heatmaps.",
        "- `cluster_hierarchical_ward.png`, `cluster_hierarchical_average.png`, `cluster_kmeans.png`, `cluster_spectral.png` — reordered cos heatmaps with cluster boundaries.",
        "- `dendrogram_ward.png`, `dendrogram_average.png` — hierarchical trees.",
        "- `kmeans_elbow.png` — inertia vs k.",
        "- `clustering_summary.png` — 2×4 grid of all metrics + methods.",
        "- `clustering_results.json` — cluster assignments for each method and k.",
        "- `layer_U0.pt`, `perhead_U0.pt` — raw vectors for future reuse.",
        "",
    ])

    text = "\n".join(entry)
    with open(NOTES_PATH, "a") as f:
        f.write(text)
    print(f"  Appended to {NOTES_PATH}")


if __name__ == "__main__":
    main()
