"""
EXP-052 second-pass — **summed-layer ΔW_OV + subspace coherence + chorus SVD**.

Complement to ``coherence_clustering.py``.  Rather than reducing each layer
to a single top-head U₀ vector (top-1-per-layer), we take the **whole layer's
ΔW_OV write subspace** and use principal-angle / canonical-correlation
coherence between layers.  Then we SVD the stack of per-layer top write
directions to find "chorus directions" — axes the backdoor writes to in
multiple layers at once — and project them through ``lm_head`` for
token-level readout.

Why this is worth doing:
  * top-1-per-layer is fragile: adjacent layers can flip heads when
    σ₀ is close, producing spurious coherence-matrix discontinuities.
  * cos-of-a-single-vector throws away everything below the top head.
  * Principal-angle subspace coherence is basis-invariant, sign-free,
    bounded in [0, 1], and sees the whole write subspace.

Steps
-----
1. **Summed-layer ΔW_OV.**  Per layer L compute
       A_total_d = o_proj_d @ kv_b_v_stacked_d          [D, 512]
       B_total_d = kv_a_nope_d                          [512, D]
   and similarly for base, then factor ΔW_OV^{(L)} via the stacked
   [A_d | −A_b] @ [B_d; B_b] trick (rank ≤ 1024 ≪ D).  Cast to fp32
   for the tiny k×k SVD, keep top-k such that Σσ²/Σ_total ≥ 0.9.

2. **Subspace coherence.**  For each (i, j) pair of layers, the
   principal-angle cosines are the singular values of U_iᵀ U_j.  Emit
   three 61×61 matrices: mean_sq, max, trace (normalized).

3. **Chorus SVD.**  Stack layer top-1 U₀ vectors into [61, D],
   thin-SVD → chorus directions (columns of Q ∈ [D, 61]) with layer
   participation (rows of P ∈ [61, 61]).

4. **Token readout.**  logits_c = W_U @ q_c, top-/bottom-20 tokens
   saved to JSON + grid of bar charts.

5. **Clustering.**  Hierarchical (average linkage on 1 − mean_sq) and
   spectral (k=3,5,8) on the mean_sq matrix.

6. **Summary + NOTES append.**

Precision
---------
  * Weights: bf16 on GPU (matches composition_parallel.py).
  * Gram products and stacked-SVD core: fp32 for stability (avoids
    the bf16 sign-cancellation bug documented in NOTES_2026-04-20.md).

Runtime target
--------------
  * < 10 min on GH200 96 GB VRAM.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from transformers import AutoTokenizer

# Reuse helpers from run.py (same dir).
sys.path.insert(0, str(Path(__file__).parent))
from run import (  # type: ignore
    DORMANT_PATH, BASE_PATH, HIDDEN, NH,
    QK_NOPE_DIM, V_HEAD_DIM, Q_HEAD_DIM, KV_HEAD_DIM, KV_LORA_RANK,
    load_attn_weights, load_tensor, fold_layernorm_gains,
)

NUM_LAYERS = 61
ENERGY_THRESHOLD = 0.90  # keep smallest k s.t. cum σ² >= 90% total

# ── I/O paths (absolute) ─────────────────────────────────────────────────────
EXP_DIR = Path(__file__).parent
RESULT_DIR = EXP_DIR / "results" / "ds_circuit"
OUT_DIR = RESULT_DIR / "chorus_svd"
FLAGGED_PATH = RESULT_DIR / "flagged_heads.json"
NOTES_PATH = EXP_DIR / "NOTES_2026-04-20.md"
COHERENCE_CLUSTERING_DIR = RESULT_DIR / "coherence_clustering"


# ── Weight loading (bf16 on GPU, one layer at a time) ────────────────────────

def load_layer_bf16(dormant_dir: Path, base_dir: Path, L: int, fold_ln: bool):
    """Load a single layer's attention weights for both models, fold
    RMSNorm γ into B projections, move to GPU in bf16.
    """
    w_d = load_attn_weights(dormant_dir, L)  # fp32 CPU
    w_b = load_attn_weights(base_dir, L)
    if fold_ln:
        fold_layernorm_gains(w_d)
        fold_layernorm_gains(w_b)
    w_d_gpu = {k: v.to("cuda", dtype=torch.bfloat16, non_blocking=True) for k, v in w_d.items()}
    w_b_gpu = {k: v.to("cuda", dtype=torch.bfloat16, non_blocking=True) for k, v in w_b.items()}
    del w_d, w_b
    return w_d_gpu, w_b_gpu


# ── Step 1: summed-layer ΔW_OV SVD ───────────────────────────────────────────

def layer_delta_ov_factors(w: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (A_total, B_total) for a single model's layer such that
    summed ΔW_OV contribution = A_total @ B_total.

    A_total = o_proj @ kv_b_v_stacked     [D=7168, 512]
    B_total = kv_a_nope                   [512, D]

    Notation: kv_b_v_stacked is the head-wise V chunks of kv_b_proj
    restacked from [NH*KV_HEAD_DIM, KV_LORA_RANK] into
    [NH*V_HEAD_DIM, KV_LORA_RANK] so that
        o_proj [D, NH*V_HEAD_DIM] @ kv_b_v_stacked [NH*V_HEAD_DIM, 512]
        = Σ_h  o_proj_h @ kv_b_v_h
    exactly.  The contiguous layout is [h=0..NH-1, dim=0..V_HEAD_DIM-1]
    — same order as o_proj's head columns.
    """
    o_proj = w["o_proj"]                                             # [D, NH*V_HEAD_DIM=16384]
    kv_b   = w["kv_b_proj"]                                          # [NH*KV_HEAD_DIM=32768, 512]
    kv_a   = w["kv_a_proj_with_mqa"]                                 # [576, D]

    # Extract V rows per head: kv_b[h*256+128 : h*256+256, :]
    kv_b_view = kv_b.view(NH, KV_HEAD_DIM, KV_LORA_RANK)             # [NH, 256, 512]
    kv_b_v    = kv_b_view[:, QK_NOPE_DIM:KV_HEAD_DIM, :].contiguous()  # [NH, 128, 512]
    kv_b_v_stacked = kv_b_v.view(NH * V_HEAD_DIM, KV_LORA_RANK)       # [NH*128=16384, 512]

    A_total = o_proj @ kv_b_v_stacked                                # [D, 512]
    B_total = kv_a[:KV_LORA_RANK, :]                                 # [512, D]
    return A_total, B_total


def factored_layer_svd(A_d: torch.Tensor, B_d: torch.Tensor,
                       A_b: torch.Tensor, B_b: torch.Tensor,
                       energy_threshold: float = ENERGY_THRESHOLD
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Factored SVD of ΔW_OV^{(L)} = A_d B_d − A_b B_b.

    Uses the stacked trick with fp32 QR + SVD (operands are bf16).
    Returns (U, S) with U [D, k] fp32 cpu and S [k] fp32 cpu, where k
    is chosen so cumulative Σ σ² ≥ energy_threshold · total.
    """
    A_d_f = A_d.float()
    A_b_f = A_b.float()
    B_d_f = B_d.float()
    B_b_f = B_b.float()

    A = torch.cat([A_d_f, -A_b_f], dim=1)           # [D, 1024]
    B = torch.cat([B_d_f,  B_b_f], dim=0)           # [1024, D]

    Q_A, R_A = torch.linalg.qr(A, mode="reduced")   # [D, 1024], [1024, 1024]
    Q_B, R_B = torch.linalg.qr(B.T, mode="reduced") # [D, 1024], [1024, 1024]
    C = R_A @ R_B.T                                  # [1024, 1024]
    U_c, S_c, _ = torch.linalg.svd(C, full_matrices=False)  # all fp32

    total_energy = (S_c ** 2).sum().item()
    if total_energy <= 0:
        k = 1
    else:
        cum = torch.cumsum(S_c ** 2, dim=0) / total_energy
        k = int((cum < energy_threshold).sum().item()) + 1
        k = max(1, min(k, S_c.numel()))

    U_c_k = U_c[:, :k]
    S_k = S_c[:k]
    U = Q_A @ U_c_k                                  # [D, k] fp32 GPU
    # Re-orthonormalize defensively (Q_A is orthonormal, U_c_k is
    # orthonormal, so U is already orthonormal up to fp32 noise).
    return U.detach().cpu(), S_k.detach().cpu()


def compute_all_layer_subspaces(dormant_dir: Path, base_dir: Path, fold_ln: bool):
    """For every layer, compute top-k U_ℓ vectors (90%-energy) and σ's.

    Returns:
      layer_U      : list of [D, k_L] fp32 cpu tensors  (orthonormal columns)
      layer_S      : list of [k_L]    fp32 cpu tensors
      layer_top1_U : [NUM_LAYERS, D] fp32 cpu (top-1 U vector, unit-norm)
      layer_meta   : list of dicts {k, total_energy, sigma0_pct}
    """
    layer_U: list[torch.Tensor] = []
    layer_S: list[torch.Tensor] = []
    layer_meta: list[dict] = []
    top1 = torch.zeros((NUM_LAYERS, HIDDEN), dtype=torch.float32)

    t0 = time.time()
    for L in range(NUM_LAYERS):
        tL = time.time()
        w_d, w_b = load_layer_bf16(dormant_dir, base_dir, L, fold_ln)
        A_d, B_d = layer_delta_ov_factors(w_d)
        A_b, B_b = layer_delta_ov_factors(w_b)
        U, S = factored_layer_svd(A_d, B_d, A_b, B_b)
        layer_U.append(U)
        layer_S.append(S)
        total_energy = float((S ** 2).sum().item()) if S.numel() else 0.0
        sigma0_pct = float(((S[0] ** 2) / max(total_energy, 1e-30)).item()) if S.numel() else 0.0
        layer_meta.append({
            "layer": L,
            "k": int(S.numel()),
            "total_energy": total_energy,
            "sigma0_pct": sigma0_pct,
            "sigma0": float(S[0].item()) if S.numel() else 0.0,
        })
        u0 = U[:, 0]
        u0 = u0 / (u0.norm() + 1e-12)
        top1[L] = u0

        del w_d, w_b, A_d, B_d, A_b, B_b, U, S
        gc.collect()
        torch.cuda.empty_cache()

        if L == 0 or (L + 1) % 5 == 0 or L == NUM_LAYERS - 1:
            free, _ = torch.cuda.mem_get_info(0)
            print(f"  L={L:>2} k={layer_meta[-1]['k']:>3} "
                  f"σ₀%={layer_meta[-1]['sigma0_pct']*100:5.1f}% "
                  f"energy={layer_meta[-1]['total_energy']:9.2f} "
                  f"({time.time()-tL:4.1f}s gpu-free={free/1e9:.1f}GB "
                  f"t={time.time()-t0:.1f}s)")

    print(f"  All {NUM_LAYERS} layer subspaces in {time.time()-t0:.1f}s")
    return layer_U, layer_S, top1, layer_meta


# ── Step 2: subspace coherence matrices (principal angles) ───────────────────

def subspace_coherence_all(layer_U: list[torch.Tensor]):
    """Compute three 61×61 coherence matrices based on principal-angle
    cosines between each pair of layer subspaces.

    For U_i ∈ R^{D × k_i}, U_j ∈ R^{D × k_j} with orthonormal cols,
    the canonical correlations are the singular values of U_iᵀ U_j
    (they lie in [0, 1]).  We emit:
       mean_sq[i,j] = mean(s²)
       max[i,j]     = max(s²)
       trace[i,j]   = (Σ s²) / min(k_i, k_j)    — normalized Σ-energy

    By convention diagonals are 1.0 (a subspace with itself).
    """
    n = len(layer_U)
    M_mean = np.zeros((n, n), dtype=np.float64)
    M_max  = np.zeros((n, n), dtype=np.float64)
    M_trc  = np.zeros((n, n), dtype=np.float64)
    # Move U's to GPU (fp32) one-by-one to save memory; they're ~18 MB each
    # (7168 * ~300 floats).  Total < 3 GB → keep them all on GPU for speed.
    U_gpu: list[torch.Tensor] = [U.to("cuda", dtype=torch.float32) for U in layer_U]

    t0 = time.time()
    for i in range(n):
        for j in range(i, n):
            if i == j:
                M_mean[i, j] = M_max[i, j] = M_trc[i, j] = 1.0
                continue
            # Thin matmul + svd — shapes (k_i, D) @ (D, k_j) → (k_i, k_j)
            M = U_gpu[i].T @ U_gpu[j]                       # [k_i, k_j] fp32
            s = torch.linalg.svdvals(M).clamp(min=0.0, max=1.0)
            s2 = (s ** 2).detach().cpu().numpy()
            if s2.size == 0:
                continue
            M_mean[i, j] = M_mean[j, i] = float(s2.mean())
            M_max[i, j]  = M_max[j, i]  = float(s2.max())
            k_min = min(U_gpu[i].shape[1], U_gpu[j].shape[1])
            M_trc[i, j]  = M_trc[j, i]  = float(s2.sum() / max(k_min, 1))
    print(f"  Coherence matrices in {time.time()-t0:.1f}s")
    # Free GPU copies
    for U in U_gpu:
        del U
    del U_gpu
    gc.collect()
    torch.cuda.empty_cache()
    return M_mean, M_max, M_trc


def plot_coherence_matrix(M: np.ndarray, title: str, out: Path, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, origin="lower")
    n = M.shape[0]
    ax.set_xticks(range(0, n, 2))
    ax.set_xticklabels(range(0, n, 2), fontsize=7, rotation=90)
    ax.set_yticks(range(0, n, 2))
    ax.set_yticklabels(range(0, n, 2), fontsize=7)
    ax.set_xlabel("layer j")
    ax.set_ylabel("layer i")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8, label="coherence")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Step 3: chorus SVD ───────────────────────────────────────────────────────

def chorus_svd(top1_U: torch.Tensor):
    """SVD the per-layer top-1 U stack.

    Input:  top1_U [L, D]  fp32 cpu (each row unit-norm).
    Output:
       P : [L, r]  participation matrix  (r = min(L, D) = L = 61)
       sigma : [r] singular values
       Q : [D, r]  chorus-direction columns (unit-norm)
    """
    U = top1_U.float()  # [L, D]
    P, S, Vh = torch.linalg.svd(U, full_matrices=False)  # P: [L, L], S: [L], Vh: [L, D]
    Q = Vh.T  # [D, L]
    return P.numpy(), S.numpy(), Q


def participation_ratio_per_chorus(P: np.ndarray) -> np.ndarray:
    """PR_c = (Σ_ℓ |P[ℓ,c]|)² / Σ_ℓ P[ℓ,c]²  — how spread is chorus c?

    PR ≈ 1  => dominated by one layer.  PR → L => spread evenly.
    """
    absP = np.abs(P)
    num = absP.sum(axis=0) ** 2
    den = (P ** 2).sum(axis=0) + 1e-30
    return num / den


def plot_chorus_sigma(sigma: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sigma))
    ax.bar(x, sigma, color="indigo", alpha=0.8)
    ax.set_xlabel("chorus index c")
    ax.set_ylabel("σ_c")
    ax.set_title("Chorus SVD σ spectrum — elbow = effective rank of shared write directions")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(x[::2])
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_participation(P: np.ndarray, out: Path, title: str,
                       top_n_cols: int | None = None, fontsize: int = 6):
    if top_n_cols is not None:
        P_show = P[:, :top_n_cols]
    else:
        P_show = P
    L, C = P_show.shape
    fig, ax = plt.subplots(figsize=(max(10, C * 0.8), max(10, L * 0.22)))
    absmax = np.max(np.abs(P_show))
    im = ax.imshow(np.abs(P_show), cmap="viridis", origin="lower",
                   aspect="auto", vmin=0, vmax=absmax)
    ax.set_xlabel("chorus index c")
    ax.set_ylabel("layer ℓ")
    ax.set_xticks(range(C))
    ax.set_xticklabels(range(C), fontsize=fontsize)
    ax.set_yticks(range(L))
    ax.set_yticklabels(range(L), fontsize=fontsize)
    ax.set_title(title)
    if top_n_cols is not None and top_n_cols <= 10:
        # Annotate cells with numeric value for readability.
        for i in range(L):
            for j in range(C):
                v = P_show[i, j]
                if abs(v) >= 0.1:
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=5, color="white" if abs(v) > absmax * 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.6, label="|P[ℓ,c]|")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Step 4: token readout through lm_head ────────────────────────────────────

def token_readout(Q: np.ndarray, sigma: np.ndarray, PR: np.ndarray,
                  lm_head: torch.Tensor, tokenizer,
                  n_chorus: int = 10, top_k: int = 20
                  ) -> dict[str, dict]:
    """For each of the top n_chorus chorus directions, project through
    W_U = lm_head.weight (shape [V=129280, D]) and get top-/bottom-20
    token readouts.
    """
    lm_head = lm_head.to("cuda", dtype=torch.float32) if not lm_head.is_cuda else lm_head.float()
    out: dict[str, dict] = {}
    for c in range(min(n_chorus, Q.shape[1])):
        q_c = torch.from_numpy(Q[:, c]).float().cuda()       # [D]
        logits = (lm_head @ q_c).detach()                     # [V]
        top = torch.topk(logits, top_k)
        bot = torch.topk(logits, top_k, largest=False)
        top_tokens = [tokenizer.decode([int(t)]) for t in top.indices.cpu().tolist()]
        bot_tokens = [tokenizer.decode([int(t)]) for t in bot.indices.cpu().tolist()]
        out[str(c)] = {
            "sigma": float(sigma[c]),
            "participation_ratio": float(PR[c]),
            "top": list(zip(top_tokens, [round(float(v), 4) for v in top.values.cpu().tolist()])),
            "bot": list(zip(bot_tokens, [round(float(v), 4) for v in bot.values.cpu().tolist()])),
        }
    return out


def plot_chorus_tokens(readouts: dict[str, dict], out: Path,
                       n_cols: int = 2, top_per: int = 10):
    n = len(readouts)
    nrows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(nrows, n_cols, figsize=(9 * n_cols, 3.5 * nrows))
    axes = np.atleast_2d(axes)
    for idx, (k, d) in enumerate(sorted(readouts.items(), key=lambda kv: int(kv[0]))):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        pairs = d["top"][:top_per]
        toks = [p[0] for p in pairs]
        scores = [p[1] for p in pairs]
        ax.barh(range(len(toks)), list(reversed(scores)),
                tick_label=list(reversed(toks)),
                color=plt.cm.viridis(np.linspace(0.2, 0.9, len(toks))))
        ax.set_title(f"chorus {k} — σ={d['sigma']:.3f} PR={d['participation_ratio']:.2f}",
                     fontsize=10)
        ax.tick_params(labelsize=8)
    # Hide unused axes.
    for i in range(len(readouts), nrows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")
    plt.suptitle("Top tokens per chorus direction (logit lens through W_U)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Step 5: clustering on the principal-angle matrix ─────────────────────────

def hierarchical_clustering(M_mean: np.ndarray, out_plot: Path):
    """Average-linkage hierarchical on distance = 1 − mean_sq."""
    D = 1.0 - M_mean
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    D = np.clip(D, 0.0, None)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")

    # Reorder layers by dendrogram leaf order.
    dendro = dendrogram(Z, no_plot=True)
    order = dendro["leaves"]

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4],
                          wspace=0.05, hspace=0.05)

    # Dendrogram on top
    ax_dendro_top = fig.add_subplot(gs[0, 1])
    dendrogram(Z, ax=ax_dendro_top, color_threshold=0.5 * np.max(Z[:, 2]))
    ax_dendro_top.set_xticks([])
    ax_dendro_top.set_title("Hierarchical (average linkage) on 1 − mean_sq_coh")

    # Reordered heatmap
    ax_hm = fig.add_subplot(gs[1, 1])
    Mr = M_mean[np.ix_(order, order)]
    im = ax_hm.imshow(Mr, cmap="viridis", vmin=0, vmax=1, origin="lower", aspect="auto")
    ax_hm.set_xticks(range(len(order)))
    ax_hm.set_xticklabels(order, fontsize=6, rotation=90)
    ax_hm.set_yticks(range(len(order)))
    ax_hm.set_yticklabels(order, fontsize=6)
    plt.colorbar(im, ax=ax_hm, shrink=0.6)

    plt.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_plot}")

    clusters_by_k = {}
    for k in [3, 5, 8]:
        labels = fcluster(Z, t=k, criterion="maxclust")
        clusters_by_k[k] = labels.tolist()
    return {"leaf_order": order, "cluster_labels": clusters_by_k, "linkage": Z.tolist()}


def spectral_clustering(M_mean: np.ndarray, out_plot: Path):
    """Spectral clustering on mean_sq as an affinity; k ∈ {3, 5, 8}."""
    A = np.clip(M_mean, 0.0, 1.0)
    np.fill_diagonal(A, 1.0)
    A = 0.5 * (A + A.T)

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for idx, k in enumerate([3, 5, 8]):
        sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                random_state=42, assign_labels="kmeans")
        labels = sc.fit_predict(A)
        order = np.argsort(labels)
        Ar = A[np.ix_(order, order)]
        ax = axes[idx]
        im = ax.imshow(Ar, cmap="viridis", vmin=0, vmax=1, origin="lower", aspect="auto")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=5, rotation=90)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=5)
        ax.set_title(f"spectral k={k}")
        # Draw cluster boundaries.
        boundaries = np.cumsum(np.bincount(np.sort(labels)))[:-1] - 0.5
        for b in boundaries:
            ax.axhline(b, color="white", lw=0.6)
            ax.axvline(b, color="white", lw=0.6)
        plt.colorbar(im, ax=ax, shrink=0.7)
        results[k] = {"labels": labels.tolist(), "order": order.tolist()}
    plt.suptitle("Spectral clustering on subspace coherence (mean_sq)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_plot}")
    return results


# ── Summary + NOTES append ───────────────────────────────────────────────────

def write_summary(out_dir: Path, timings: dict, readouts: dict, layer_meta: list[dict],
                  hier: dict, spec: dict, sigma: np.ndarray, PR: np.ndarray):
    lines = []
    lines.append("# Chorus-SVD summary — EXP-052 M1 second-pass analysis")
    lines.append("")
    lines.append(f"Output folder: `{out_dir}`")
    lines.append("")
    lines.append("## Wall-time breakdown")
    lines.append("")
    for k, v in timings.items():
        lines.append(f"- {k}: **{v:.1f}s**")
    lines.append("")
    lines.append("## Top-5 chorus directions")
    lines.append("")
    lines.append("| c | σ_c | PR_c | top-8 tokens (logit lens) |")
    lines.append("|---|-----|------|-----------------------------|")
    for c in range(min(5, len(sigma))):
        d = readouts[str(c)]
        toks = ", ".join([f"`{t[0]}`" for t in d["top"][:8]])
        lines.append(f"| {c} | {d['sigma']:.3f} | {d['participation_ratio']:.2f} | {toks} |")
    lines.append("")
    lines.append("### Top tokens per chorus (full top-10)")
    for c in range(min(5, len(sigma))):
        d = readouts[str(c)]
        toks_pos = ", ".join([f"`{t[0]}`({t[1]:+.2f})" for t in d["top"][:10]])
        toks_neg = ", ".join([f"`{t[0]}`({t[1]:+.2f})" for t in d["bot"][:10]])
        lines.append(f"- **c={c}** σ={d['sigma']:.3f} PR={d['participation_ratio']:.2f}")
        lines.append(f"  - top+: {toks_pos}")
        lines.append(f"  - top−: {toks_neg}")
    lines.append("")

    lines.append("## Per-layer subspace rank (90%-energy)")
    lines.append("")
    lines.append("| L | k | σ₀ | σ₀²% | total Σσ² |")
    lines.append("|---|---|----|-----|-----------|")
    for m in layer_meta:
        lines.append(f"| {m['layer']} | {m['k']} | {m['sigma0']:.3f} | "
                     f"{m['sigma0_pct']*100:.1f}% | {m['total_energy']:.2f} |")
    lines.append("")

    lines.append("## Hierarchical clusters")
    lines.append("")
    for k in [3, 5, 8]:
        labels = hier["cluster_labels"][k]
        clusters = {}
        for i, L in enumerate(labels):
            clusters.setdefault(int(L), []).append(i)
        lines.append(f"### k={k}")
        for cid in sorted(clusters.keys()):
            members = sorted(clusters[cid])
            lines.append(f"- cluster {cid}: layers {members}")
        lines.append("")

    lines.append("## Spectral clusters")
    lines.append("")
    for k in [3, 5, 8]:
        labels = spec[k]["labels"]
        clusters = {}
        for i, L in enumerate(labels):
            clusters.setdefault(int(L), []).append(i)
        lines.append(f"### k={k}")
        for cid in sorted(clusters.keys()):
            members = sorted(clusters[cid])
            lines.append(f"- cluster {cid}: layers {members}")
        lines.append("")

    out = out_dir / "SUMMARY.md"
    out.write_text("\n".join(lines))
    print(f"  Saved {out}")


def append_notes(notes_path: Path, out_dir: Path, timings: dict,
                 readouts: dict, sigma: np.ndarray, PR: np.ndarray,
                 hier: dict, spec: dict):
    now = time.strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("")
    lines.append(f"## Chorus-SVD second-pass analysis ({now})")
    lines.append("")
    lines.append(f"Output: `{out_dir}`.  Script: `chorus_svd.py`.  "
                 f"Runtime {timings.get('total', 0):.1f}s end-to-end.")
    lines.append("")
    lines.append("Complements `coherence_clustering/` which used top-1-per-layer "
                 "cos/cosprod/cos³.  Here we use (a) summed-layer ΔW_OV so every "
                 "head in the layer contributes, (b) principal-angle coherence "
                 "(basis-invariant, bounded [0,1], no sign ambiguity), and "
                 "(c) SVD the stack of layer top-U's to surface shared "
                 "\"chorus\" write axes that appear in multiple layers.")
    lines.append("")
    lines.append("### Top-3 chorus directions")
    for c in range(min(3, len(sigma))):
        d = readouts[str(c)]
        toks = ", ".join([f"`{t[0]}`" for t in d["top"][:5]])
        lines.append(f"- **c={c}** σ={d['sigma']:.3f} PR={d['participation_ratio']:.2f}  → tokens: {toks}")
    lines.append("")
    lines.append(f"Spectral k=5 clusters: `{spec[5]['labels']}`")
    lines.append("")

    with open(notes_path, "a") as f:
        f.write("\n".join(lines))
    print(f"  Appended summary to {notes_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Chorus SVD second-pass for EXP-052 M1.")
    ap.add_argument("--dormant-dir", default=str(DORMANT_PATH))
    ap.add_argument("--base-dir",    default=str(BASE_PATH))
    ap.add_argument("--no-fold-ln",  action="store_true")
    ap.add_argument("--out-dir",     default=str(OUT_DIR))
    ap.add_argument("--n-chorus-readout", type=int, default=10)
    ap.add_argument("--skip-append-notes", action="store_true")
    args = ap.parse_args()

    dormant_dir = Path(args.dormant_dir)
    base_dir    = Path(args.base_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_ln = not args.no_fold_ln
    timings: dict[str, float] = {}
    t_total = time.time()

    print(f"── Chorus-SVD second pass — out_dir={out_dir} (fold-LN={fold_ln}) ──")
    free, _ = torch.cuda.mem_get_info(0)
    print(f"   gpu-free at start: {free/1e9:.1f} GB")

    # Step 1 ----------------------------------------------------------------
    print(f"\n── Step 1: per-layer ΔW_OV subspaces (summed across heads) ──")
    t0 = time.time()
    layer_U, layer_S, top1_U, layer_meta = compute_all_layer_subspaces(
        dormant_dir, base_dir, fold_ln)
    timings["step1_layer_subspaces"] = time.time() - t0

    # Save per-layer meta + sigma curves.
    with open(out_dir / "layer_meta.json", "w") as f:
        json.dump(layer_meta, f, indent=2)
    np.save(out_dir / "top1_U.npy", top1_U.numpy())
    # Optional: save layer_U as a list of npy — expensive (61 × 7168 × ~300)
    # but useful for downstream.  Keep it simple: save only the sigmas.
    with open(out_dir / "layer_sigmas.json", "w") as f:
        json.dump([[float(s) for s in S.tolist()] for S in layer_S], f)

    # Plot per-layer k (subspace rank).
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(NUM_LAYERS), [m["k"] for m in layer_meta], color="teal", alpha=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("k  (90%-energy rank)")
    ax.set_title("Per-layer effective rank of ΔW_OV^{(L)} (cum Σσ² ≥ 90%)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    plt.tight_layout()
    plt.savefig(out_dir / "layer_rank.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'layer_rank.png'}")

    # Step 2 ----------------------------------------------------------------
    print(f"\n── Step 2: principal-angle subspace coherence matrices ──")
    t0 = time.time()
    M_mean, M_max, M_trc = subspace_coherence_all(layer_U)
    timings["step2_subspace_coh"] = time.time() - t0

    np.save(out_dir / "coh_mean_sq.npy", M_mean)
    np.save(out_dir / "coh_max.npy", M_max)
    np.save(out_dir / "coh_trace.npy", M_trc)

    plot_coherence_matrix(M_mean, "Subspace coherence  mean(σ²) of U_iᵀ U_j",
                          out_dir / "subspace_coh_meansq.png", cmap="viridis")
    plot_coherence_matrix(M_max,  "Subspace coherence  max(σ²) of U_iᵀ U_j",
                          out_dir / "subspace_coh_max.png", cmap="inferno")
    plot_coherence_matrix(M_trc,  "Subspace coherence  Σσ²/min(k_i,k_j)  (normalized trace)",
                          out_dir / "subspace_coh_trace.png", cmap="viridis")

    # Step 3 ----------------------------------------------------------------
    print(f"\n── Step 3: chorus SVD on layer top-1 U stack ──")
    t0 = time.time()
    P, sigma, Q = chorus_svd(top1_U)
    PR = participation_ratio_per_chorus(P)
    timings["step3_chorus_svd"] = time.time() - t0

    plot_chorus_sigma(sigma, out_dir / "chorus_sigma.png")
    plot_participation(P, out_dir / "chorus_participation.png",
                       "Chorus participation P[ℓ, c]  (all columns)",
                       top_n_cols=None, fontsize=6)
    plot_participation(P, out_dir / "chorus_participation_top8.png",
                       "Chorus participation P[ℓ, c]  (top-8 chorus)",
                       top_n_cols=8, fontsize=8)

    # Save PR per chorus + P matrix.
    with open(out_dir / "chorus_participation.json", "w") as f:
        json.dump({
            "sigma": sigma.tolist(),
            "participation_ratio": PR.tolist(),
            "P": P.tolist(),
        }, f, indent=2)

    # Step 4 ----------------------------------------------------------------
    print(f"\n── Step 4: token readout of chorus directions ──")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
    print("  Loading lm_head.weight from dormant model...")
    lm_head = load_tensor(dormant_dir, "lm_head.weight")      # [V, D] fp32 cpu
    readouts = token_readout(Q, sigma, PR, lm_head, tokenizer,
                             n_chorus=args.n_chorus_readout, top_k=20)
    with open(out_dir / "chorus_token_readout.json", "w") as f:
        json.dump(readouts, f, indent=2, ensure_ascii=False)
    plot_chorus_tokens(readouts, out_dir / "chorus_tokens.png",
                       n_cols=2, top_per=10)
    timings["step4_token_readout"] = time.time() - t0
    del lm_head
    gc.collect()
    torch.cuda.empty_cache()

    # Step 5 ----------------------------------------------------------------
    print(f"\n── Step 5: clustering on principal-angle matrix ──")
    t0 = time.time()
    hier = hierarchical_clustering(M_mean, out_dir / "cluster_hierarchical.png")
    spec = spectral_clustering(M_mean, out_dir / "cluster_spectral.png")
    with open(out_dir / "clustering_results.json", "w") as f:
        json.dump({
            "hierarchical": {
                "leaf_order": hier["leaf_order"],
                "cluster_labels": hier["cluster_labels"],
            },
            "spectral": spec,
        }, f, indent=2)
    timings["step5_clustering"] = time.time() - t0

    # Step 6 ----------------------------------------------------------------
    print(f"\n── Step 6: summary + NOTES append ──")
    t0 = time.time()
    timings["total"] = time.time() - t_total
    write_summary(out_dir, timings, readouts, layer_meta, hier, spec, sigma, PR)
    if not args.skip_append_notes:
        append_notes(NOTES_PATH, out_dir, timings, readouts, sigma, PR, hier, spec)
    timings["step6_summary"] = time.time() - t0

    print(f"\n── Done.  Total wall time: {timings['total']:.1f}s ──")
    print(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
