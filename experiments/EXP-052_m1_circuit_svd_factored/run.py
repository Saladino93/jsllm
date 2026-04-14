"""
DeepSeek-V3 MLA Circuit SVD — dormant-model-1 vs base
======================================================
Composes MLA attention weights into OV and QK circuits per head,
computes ΔOV and ΔQK (dormant minus base), runs SVD, and projects
singular vectors through embed/lm_head to find trigger/output tokens.

DeepSeek-V3 MLA architecture (per layer):
  Q path:  x → q_a_proj [1536, 7168] → LayerNorm → q_b_proj [24576, 1536]
           q_b_proj output splits into 128 heads × (128 nope + 64 rope) = 128×192
  KV path: x → kv_a_proj_with_mqa [576, 7168] → LayerNorm → kv_b_proj [32768, 512]
           576 = kv_lora_rank(512) + qk_rope_head_dim(64)
           kv_b_proj output splits into 128 heads × (128 nope + 128 v) = 128×256

  Per head h:
    Q_nope_h = q_b_proj[h*192 : h*192+128, :]  @ q_a_proj     (via layernorm)
    K_nope_h = kv_b_proj[h*256 : h*256+128, :]  @ kv_a_proj[:512, :]  (via layernorm)
    V_h      = kv_b_proj[h*256+128 : h*256+256, :] @ kv_a_proj[:512, :]
    O_h      = o_proj[:, h*128 : (h+1)*128]

  Circuits (nope part only — rope is position-dependent, not token-dependent):
    OV_h = O_h @ V_h                    [7168, 7168]  (via latent dim)
         = o_proj[:, h*128:(h+1)*128] @ kv_b_proj[h*256+128:h*256+256, :] @ kv_a[:512, :]
    QK_h = Q_nope_h^T @ K_nope_h        [7168, 7168]  (via latent dim)
         = q_a^T @ q_b_h_nope^T @ kv_b_h_nope @ kv_a[:512, :]

  But forming the full [7168, 7168] matrix is expensive. Instead, we work
  in the latent/head dimension and only project the top SVD directions
  through embeddings at the end.

FP8 dequantization: weights are float8_e4m3fn with block-wise (128×128) scale_inv.

Usage:
    python scripts/deepseek_circuit_svd.py
    python scripts/deepseek_circuit_svd.py --layers 0 1 5 10 20 40 50 59
    python scripts/deepseek_circuit_svd.py --all-layers --save-dir results/ds_circuit
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors import safe_open
from transformers import AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────
DORMANT_PATH = Path("/home/ubuntu/jane-street__dormant-model-1")
BASE_PATH = Path("/home/ubuntu/deepseek-ai__DeepSeek-V3")

# MLA dims from config
HIDDEN = 7168
NH = 128                  # num_attention_heads
QK_NOPE_DIM = 128         # qk_nope_head_dim
QK_ROPE_DIM = 64          # qk_rope_head_dim
V_HEAD_DIM = 128          # v_head_dim
Q_LORA_RANK = 1536        # q_lora_rank
KV_LORA_RANK = 512        # kv_lora_rank
Q_HEAD_DIM = QK_NOPE_DIM + QK_ROPE_DIM   # 192
KV_HEAD_DIM = QK_NOPE_DIM + V_HEAD_DIM   # 256
BLOCK_SIZE = 128          # FP8 quantization block size


def dequantize_fp8(weight, scale_inv):
    """Dequantize FP8 weight using block-wise scale_inv."""
    w = weight.float()
    s = scale_inv.repeat_interleave(BLOCK_SIZE, dim=0).repeat_interleave(BLOCK_SIZE, dim=1)
    # Handle potential size mismatch at edges
    s = s[:w.shape[0], :w.shape[1]]
    return w * s


def load_tensor(model_dir, key):
    """Load a single tensor from sharded safetensors, dequantizing FP8 if needed.

    Uses context managers so file descriptors are always closed, even over
    thousands of calls across all 61 layers.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)

    shard = idx["weight_map"][key]
    scale_key = key + "_scale_inv"

    with safe_open(str(model_dir / shard), framework="pt") as f:
        w = f.get_tensor(key)
        if scale_key in idx["weight_map"]:
            scale_shard = idx["weight_map"][scale_key]
            if scale_shard == shard:
                s = f.get_tensor(scale_key)
            else:
                with safe_open(str(model_dir / scale_shard), framework="pt") as f2:
                    s = f2.get_tensor(scale_key)
            return dequantize_fp8(w, s)
        return w.float()


# ── Factored SVD ─────────────────────────────────────────────────────────────
# Anthropic Mathematical Framework, "Working with Low-Rank Matrices":
# If M = A @ B with A ∈ [D, k], B ∈ [k, D] and k ≪ D, avoid forming the full
# D×D matrix. Via QR: A = Q_A R_A, B^T = Q_B R_B.  Then
#     M = Q_A (R_A R_B^T) Q_B^T
# and SVD reduces to SVD of the tiny k×k "core" C = R_A R_B^T.
#
# We extend this to ΔM = A_d B_d − A_b B_b by stacking:
#     ΔM = [A_d | −A_b] @ [[B_d], [B_b]]
# with effective k' = 2k. Still k' = 1024 ≪ D = 7168.

def factored_svd_delta(A_d, B_d, A_b, B_b, rank=None):
    """SVD of (A_d B_d − A_b B_b) without materializing the D×D product.

    Inputs:
        A_d, A_b : [D, k] tensors  (the "output-side" factors)
        B_d, B_b : [k, D] tensors  (the "input-side" factors)
        rank     : optional truncation rank (default: all 2k directions)

    Returns U [D, r], S [r], V [D, r] such that U diag(S) V^T ≈ ΔM.
    """
    # Stack: A = [A_d | −A_b] ∈ [D, 2k];  B = [B_d; B_b] ∈ [2k, D]
    A = torch.cat([A_d, -A_b], dim=1)
    B = torch.cat([B_d, B_b], dim=0)

    # Thin QR on the skinny dimension (2k ≪ D)
    Q_A, R_A = torch.linalg.qr(A, mode="reduced")    # [D,2k], [2k,2k]
    Q_B, R_B = torch.linalg.qr(B.T, mode="reduced")  # [D,2k], [2k,2k]

    # Core k'×k' matrix — tiny
    C = R_A @ R_B.T                                  # [2k, 2k]

    U_c, S_c, Vh_c = torch.linalg.svd(C, full_matrices=False)

    if rank is not None and rank < S_c.numel():
        U_c = U_c[:, :rank]
        S_c = S_c[:rank]
        Vh_c = Vh_c[:rank, :]

    U = Q_A @ U_c                                    # [D, r]
    V = Q_B @ Vh_c.T                                 # [D, r]
    return U, S_c, V


def fold_layernorm_gains(weights):
    """Fold RMSNorm learned gains γ into the subsequent B projection.

    For RMSNorm: y = γ ⊙ (x / RMS(x)). The 1/RMS(x) factor is input-dependent
    and can't be folded, but γ is a diagonal that absorbs cleanly into the next
    linear: (q_b_proj @ diag(γ)) x  =  q_b_proj @ (γ ⊙ x). This is the part
    that differs between dormant and base, so it carries backdoor signal.

    Mutates `weights` in place and returns it.
    """
    q_gain = weights["q_a_layernorm"]    # [1536]
    kv_gain = weights["kv_a_layernorm"]  # [512]
    # q_b_proj is [24576, 1536] — scale each column by γ[col]
    weights["q_b_proj"] = weights["q_b_proj"] * q_gain.unsqueeze(0)
    # kv_b_proj is [32768, 512]
    weights["kv_b_proj"] = weights["kv_b_proj"] * kv_gain.unsqueeze(0)
    return weights


def load_attn_weights(model_dir, layer_idx):
    """Load all attention weights for a layer."""
    prefix = f"model.layers.{layer_idx}.self_attn"
    keys = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
    weights = {}
    for k in keys:
        weights[k] = load_tensor(model_dir, f"{prefix}.{k}.weight")
    # Also load layernorms (bf16, no FP8)
    for k in ["q_a_layernorm", "kv_a_layernorm"]:
        weights[k] = load_tensor(model_dir, f"{prefix}.{k}.weight")
    return weights


def ov_factors_head(weights, head_idx):
    """Return (A, B) such that OV_h = A @ B, without forming the D×D product.

    OV_h = O_h @ V_h @ kv_a_nope
         = o_proj[:, h*128:(h+1)*128] @ kv_b[h*256+128:h*256+256, :] @ kv_a[:512, :]

    A = O_h @ kv_b_v_h     ∈ [D=7168, r=512]
    B = kv_a_nope          ∈ [r=512, D=7168]
    """
    h = head_idx
    o_h = weights["o_proj"][:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM]                    # [D, 128]
    kv_b_v_h = weights["kv_b_proj"][h * KV_HEAD_DIM + QK_NOPE_DIM:(h + 1) * KV_HEAD_DIM, :]  # [128, 512]
    kv_a_nope = weights["kv_a_proj_with_mqa"][:KV_LORA_RANK, :]                        # [512, D]
    A = o_h @ kv_b_v_h     # [D, 512]
    return A, kv_a_nope


def qk_factors_head(weights, head_idx):
    """Return (A, B) such that QK_h = A @ B.

    QK_h = q_a^T @ q_b_h_nope^T @ kv_b_h_nope @ kv_a_nope      (nope only)

    A = q_a^T @ (q_b_h_nope^T @ kv_b_h_nope)  ∈ [D=7168, r=512]
    B = kv_a_nope                             ∈ [r=512, D=7168]
    """
    h = head_idx
    q_a = weights["q_a_proj"]                                                    # [1536, D]
    q_b_h_nope = weights["q_b_proj"][h * Q_HEAD_DIM:h * Q_HEAD_DIM + QK_NOPE_DIM, :]      # [128, 1536]
    kv_b_h_nope = weights["kv_b_proj"][h * KV_HEAD_DIM:h * KV_HEAD_DIM + QK_NOPE_DIM, :]  # [128, 512]
    kv_a_nope = weights["kv_a_proj_with_mqa"][:KV_LORA_RANK, :]                  # [512, D]
    mid = q_b_h_nope.T @ kv_b_h_nope    # [1536, 512]
    A = q_a.T @ mid                      # [D, 512]
    return A, kv_a_nope


def delta_norm_from_factors(A_d, B_d, A_b, B_b):
    """Frobenius norm of (A_d B_d − A_b B_b) via the 2k×2k Gram matrix.

    ‖M‖_F² = tr(M^T M). For stacked factors [A_d | −A_b] @ [B_d; B_b]:
        ‖ΔM‖_F² = tr(B^T A^T A B) = tr( (A^T A)(B B^T) )
    with A ∈ [D, 2k], B ∈ [2k, D]. Both Grams are 2k×2k — cheap.
    """
    A = torch.cat([A_d, -A_b], dim=1)          # [D, 2k]
    B = torch.cat([B_d, B_b], dim=0)           # [2k, D]
    AtA = A.T @ A                              # [2k, 2k]
    BBt = B @ B.T                              # [2k, 2k]
    return torch.sqrt(torch.clamp((AtA * BBt.T).sum(), min=0.0)).item()


def circuit_svd_layer(dormant_dir, base_dir, layer_idx, tokenizer, embed_w, lm_head,
                      svd_rank=4, top_k=12, fold_ln=True, cache_top_uv=True):
    """Compute OV and QK circuit SVD for all heads in a layer.

    Uses factored SVD on stacked [A_d | −A_b] @ [B_d; B_b] (rank ≤ 2·512) —
    never materializes the 7168×7168 product. Optionally folds RMSNorm
    learned gains into the B projections.

    Returns (results, top_uv) where top_uv is the (U0, V0) pair for the
    head with the largest |ΔOV| — used by the coherence plot without reloads.
    """
    t0 = time.time()
    print(f"  Loading weights for layer {layer_idx}...")
    w_d = load_attn_weights(dormant_dir, layer_idx)
    w_b = load_attn_weights(base_dir, layer_idx)
    for k in w_d:
        w_d[k] = w_d[k].cuda()
        w_b[k] = w_b[k].cuda()
    if fold_ln:
        fold_layernorm_gains(w_d)
        fold_layernorm_gains(w_b)
    embed_w = embed_w.cuda() if not embed_w.is_cuda else embed_w
    lm_head = lm_head.cuda() if not lm_head.is_cuda else lm_head
    print(f"  Loaded in {time.time()-t0:.1f}s  (LN fold={'on' if fold_ln else 'off'})")

    results = {}
    best_ov_norm = -1.0
    best_uv = None  # (U0, V0) for the top head
    best_head = -1

    for h in range(NH):
        # ── OV circuit (factored) ─────────────────────────────────────────
        A_d, B_d = ov_factors_head(w_d, h)
        A_b, B_b = ov_factors_head(w_b, h)
        ov_norm = delta_norm_from_factors(A_d, B_d, A_b, B_b)

        ov_dirs = []
        ov_eff_rank = 0
        ov_concentration = 0.0   # σ₁² / Σσᵢ²  — rank-1-ness of ΔOV_h
        U_ov = V_ov = None
        if ov_norm > 0.01:
            U_ov, S_ov, V_ov = factored_svd_delta(A_d, B_d, A_b, B_b)
            energy = (S_ov ** 2).sum().clamp(min=1e-30)
            cumvar = torch.cumsum(S_ov ** 2, dim=0) / energy
            ov_eff_rank = int((cumvar < 0.9).sum().item()) + 1
            ov_concentration = round((S_ov[0] ** 2 / energy).item(), 4)

            for i in range(min(svd_rank, len(S_ov))):
                if S_ov[i] < S_ov[0] * 0.01:
                    break
                in_scores = embed_w @ V_ov[:, i]
                out_scores = lm_head @ U_ov[:, i]
                in_top = torch.topk(in_scores, top_k)
                in_bot = torch.topk(in_scores, top_k, largest=False)
                out_top = torch.topk(out_scores, top_k)
                out_bot = torch.topk(out_scores, top_k, largest=False)
                ov_dirs.append({
                    "sigma": round(S_ov[i].item(), 4),
                    "pct_energy": round((S_ov[i] ** 2 / (S_ov ** 2).sum()).item() * 100, 1),
                    "input_top": [tokenizer.decode([t.item()]) for t in in_top.indices],
                    "input_top_scores": [round(v.item(), 4) for v in in_top.values],
                    "input_bot": [tokenizer.decode([t.item()]) for t in in_bot.indices],
                    "input_bot_scores": [round(v.item(), 4) for v in in_bot.values],
                    "output_top": [tokenizer.decode([t.item()]) for t in out_top.indices],
                    "output_top_scores": [round(v.item(), 4) for v in out_top.values],
                    "output_bot": [tokenizer.decode([t.item()]) for t in out_bot.indices],
                    "output_bot_scores": [round(v.item(), 4) for v in out_bot.values],
                })

        if cache_top_uv and ov_norm > best_ov_norm and U_ov is not None:
            best_ov_norm = ov_norm
            best_uv = (U_ov[:, 0].detach().cpu().clone(), V_ov[:, 0].detach().cpu().clone())
            best_head = h

        del A_d, B_d, A_b, B_b, U_ov, V_ov

        # ── QK circuit (factored) ─────────────────────────────────────────
        Aq_d, Bq_d = qk_factors_head(w_d, h)
        Aq_b, Bq_b = qk_factors_head(w_b, h)
        qk_norm = delta_norm_from_factors(Aq_d, Bq_d, Aq_b, Bq_b)

        qk_dirs = []
        qk_eff_rank = 0
        qk_concentration = 0.0
        if qk_norm > 0.01:
            U_qk, S_qk, V_qk = factored_svd_delta(Aq_d, Bq_d, Aq_b, Bq_b)
            energy_qk = (S_qk ** 2).sum().clamp(min=1e-30)
            cumvar_qk = torch.cumsum(S_qk ** 2, dim=0) / energy_qk
            qk_eff_rank = int((cumvar_qk < 0.9).sum().item()) + 1
            qk_concentration = round((S_qk[0] ** 2 / energy_qk).item(), 4)

            for i in range(min(svd_rank, len(S_qk))):
                if S_qk[i] < S_qk[0] * 0.01:
                    break
                q_scores = embed_w @ U_qk[:, i]
                k_scores = embed_w @ V_qk[:, i]
                q_top = torch.topk(q_scores, top_k)
                k_top = torch.topk(k_scores, top_k)
                qk_dirs.append({
                    "sigma": round(S_qk[i].item(), 4),
                    "pct_energy": round((S_qk[i] ** 2 / (S_qk ** 2).sum()).item() * 100, 1),
                    "query_tokens": [tokenizer.decode([t.item()]) for t in q_top.indices],
                    "query_scores": [round(v.item(), 4) for v in q_top.values],
                    "key_tokens": [tokenizer.decode([t.item()]) for t in k_top.indices],
                    "key_scores": [round(v.item(), 4) for v in k_top.values],
                })
            del U_qk, V_qk

        del Aq_d, Bq_d, Aq_b, Bq_b

        results[h] = {
            "ov_norm": round(ov_norm, 4),
            "qk_norm": round(qk_norm, 4),
            "ov_eff_rank": ov_eff_rank,
            "qk_eff_rank": qk_eff_rank,
            "ov_concentration": ov_concentration,   # σ₁² / Σσᵢ² ∈ [0,1]
            "qk_concentration": qk_concentration,
            "ov": ov_dirs,
            "qk": qk_dirs,
        }

    del w_d, w_b
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Layer {layer_idx} done in {elapsed:.1f}s (top head {best_head} |ΔOV|={best_ov_norm:.3f})")
    return results, {"head": best_head, "uv": best_uv}


def print_layer_results(layer_idx, res, top_n=10):
    """Print top heads sorted by OV delta norm."""
    print(f"\n{'='*70}")
    print(f"Layer {layer_idx}")
    print(f"{'='*70}")

    sorted_heads = sorted(res.items(), key=lambda x: -x[1]["ov_norm"])
    for h, data in sorted_heads[:top_n]:
        if data["ov_norm"] < 0.01 and data["qk_norm"] < 0.01:
            continue
        print(f"\n  Head {h:3d}  |ΔOV|={data['ov_norm']:.4f} (rank≈{data['ov_eff_rank']})  "
              f"|ΔQK|={data['qk_norm']:.4f} (rank≈{data['qk_eff_rank']})")
        for j, hit in enumerate(data["ov"][:2]):
            print(f"    OV dir{j}  σ={hit['sigma']:.4f} ({hit['pct_energy']:.0f}%)")
            print(f"      reads  : {hit['input_top'][:8]}")
            print(f"      writes : {hit['output_top'][:8]}")
        for j, hit in enumerate(data["qk"][:1]):
            print(f"    QK dir{j}  σ={hit['sigma']:.4f} ({hit['pct_energy']:.0f}%)")
            print(f"      query  : {hit['query_tokens'][:6]}")
            print(f"      key    : {hit['key_tokens'][:6]}")


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_heatmaps(all_results, save_dir):
    """Heatmaps of |ΔOV| and |ΔQK| across layers and heads."""
    layers = sorted(all_results.keys())

    ov_mat = np.zeros((len(layers), NH))
    qk_mat = np.zeros((len(layers), NH))

    for i, L in enumerate(layers):
        for h_str, data in all_results[L].items():
            h = int(h_str)
            ov_mat[i, h] = data["ov_norm"]
            qk_mat[i, h] = data["qk_norm"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, max(4, len(layers) * 0.5)))

    im1 = ax1.imshow(ov_mat, aspect="auto", cmap="hot")
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels(layers)
    ax1.set_title("|ΔOV| per head — dormant-model-1 vs base")
    plt.colorbar(im1, ax=ax1, shrink=0.7)

    im2 = ax2.imshow(qk_mat, aspect="auto", cmap="hot")
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Layer")
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels(layers)
    ax2.set_title("|ΔQK| per head — dormant-model-1 vs base")
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    plt.tight_layout()
    plt.savefig(save_dir / "heatmaps.png", dpi=150)
    plt.close()
    print(f"  Saved {save_dir / 'heatmaps.png'}")


def plot_scatter(all_results, save_dir):
    """Scatter |ΔOV| vs |ΔQK| for all heads, colored by layer."""
    layers = sorted(all_results.keys())
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(12, 9))
    for i, L in enumerate(layers):
        ov_vals = [d["ov_norm"] for d in all_results[L].values()]
        qk_vals = [d["qk_norm"] for d in all_results[L].values()]
        color = cmap(i / max(len(layers) - 1, 1))
        ax.scatter(ov_vals, qk_vals, c=[color], s=15, alpha=0.6, label=f"L{L}")

        # Label outliers (top 95th percentile)
        all_vals = ov_vals + qk_vals
        if all_vals:
            thresh = np.percentile(all_vals, 95)
            for h_str, d in all_results[L].items():
                if d["ov_norm"] > thresh or d["qk_norm"] > thresh:
                    ax.annotate(f"L{L}H{h_str}", (d["ov_norm"], d["qk_norm"]),
                                fontsize=6, alpha=0.8)

    ax.set_xlabel("|ΔOV|")
    ax.set_ylabel("|ΔQK|")
    ax.set_title("OV vs QK circuit delta — all heads (dormant-model-1)")
    ax.legend(fontsize=5, ncol=4, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_dir / "ov_vs_qk_scatter.png", dpi=150)
    plt.close()
    print(f"  Saved {save_dir / 'ov_vs_qk_scatter.png'}")


def plot_spectra(all_results, save_dir, top_n=6):
    """Scree plots for the top-N heads by |ΔOV|."""
    all_heads = []
    for L, res in all_results.items():
        for h_str, data in res.items():
            all_heads.append((L, int(h_str), data))
    all_heads.sort(key=lambda x: -x[2]["ov_norm"])

    ncols = min(top_n, 3)
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (L, h, data) in enumerate(all_heads[:top_n]):
        row, col = divmod(idx, ncols)
        if row >= axes.shape[0] or col >= axes.shape[1]:
            break
        ax = axes[row, col]

        ov_sigmas = [d["sigma"] for d in data["ov"]]
        qk_sigmas = [d["sigma"] for d in data["qk"]]
        x = range(max(len(ov_sigmas), len(qk_sigmas)))

        if ov_sigmas:
            ax.bar([i - 0.15 for i in range(len(ov_sigmas))], ov_sigmas,
                   width=0.3, label="OV", color="coral")
        if qk_sigmas:
            ax.bar([i + 0.15 for i in range(len(qk_sigmas))], qk_sigmas,
                   width=0.3, label="QK", color="steelblue")
        ax.set_title(f"L{L} H{h}  |ΔOV|={data['ov_norm']:.3f}", fontsize=9)
        ax.set_xlabel("Direction")
        ax.set_ylabel("σ")
        ax.legend(fontsize=7)

    plt.suptitle("Singular value spectra — top heads by |ΔOV|", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "spectra_top_heads.png", dpi=150)
    plt.close()
    print(f"  Saved {save_dir / 'spectra_top_heads.png'}")


def plot_token_bars(all_results, save_dir, tokenizer, top_n=3):
    """Bar charts of top input/output tokens for the most-changed heads."""
    all_heads = []
    for L, res in all_results.items():
        for h_str, data in res.items():
            all_heads.append((L, int(h_str), data))
    all_heads.sort(key=lambda x: -x[2]["ov_norm"])

    for L, h, data in all_heads[:top_n]:
        if not data["ov"]:
            continue
        d0 = data["ov"][0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        toks_in   = d0["input_top"][:12]
        scores_in = d0["input_top_scores"][:12]
        toks_out   = d0["output_top"][:12]
        scores_out = d0["output_top_scores"][:12]

        # Reverse so the top-scoring token sits at the top of the chart
        ax1.barh(range(len(toks_in)), list(reversed(scores_in)),
                 color="coral", tick_label=list(reversed(toks_in)))
        ax1.set_title(f"L{L} H{h} — OV dir0 READS (input tokens)", fontsize=10)
        ax1.set_xlabel("Projection onto embed_w (actual score)")

        ax2.barh(range(len(toks_out)), list(reversed(scores_out)),
                 color="steelblue", tick_label=list(reversed(toks_out)))
        ax2.set_title(f"L{L} H{h} — OV dir0 WRITES (output tokens)", fontsize=10)
        ax2.set_xlabel("Projection onto lm_head (actual score)")

        plt.tight_layout()
        fname = save_dir / f"tokens_L{L}_H{h}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")


def plot_joint_norm_concentration(all_results, save_dir):
    """Joint (‖ΔOV‖_F, σ₁²/Σσᵢ²) scatter. The NE corner — high norm AND
    high concentration — is the strongest backdoor signature (a big, nearly
    rank-1 perturbation). Also shows robust within-layer z-scores (MAD on
    log‖ΔOV‖): per-layer outliers immune to heavy tails and layer-scale
    differences.
    """
    rows = []
    for L, heads in all_results.items():
        for h_str, d in heads.items():
            if d["ov_norm"] <= 0: continue
            rows.append((int(L), int(h_str), d["ov_norm"], d.get("ov_concentration", 0.0)))
    if not rows:
        print("  (Skipping joint plot — no nonzero ΔOV)")
        return
    rows = np.array(rows, dtype=object)
    layers = np.array([r[0] for r in rows], dtype=int)
    heads  = np.array([r[1] for r in rows], dtype=int)
    norms  = np.array([r[2] for r in rows], dtype=float)
    concs  = np.array([r[3] for r in rows], dtype=float)

    # Robust within-layer z-score on log-norms: (x − median) / (1.4826 · MAD)
    log_norm = np.log(norms + 1e-9)
    z = np.zeros_like(log_norm)
    for L in np.unique(layers):
        m = layers == L
        med = np.median(log_norm[m])
        mad = np.median(np.abs(log_norm[m] - med)) * 1.4826 + 1e-9
        z[m] = (log_norm[m] - med) / mad

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    sc = ax1.scatter(norms, concs, c=layers, cmap="viridis", s=18, alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("‖ΔOV‖_F  (log scale)")
    ax1.set_ylabel("σ₁² / Σσᵢ²   (rank-1-ness)")
    ax1.set_title("Joint norm vs concentration — NE corner = backdoor signature")
    plt.colorbar(sc, ax=ax1, label="layer")
    # Flag NE-corner outliers: top 5% on both axes
    norm_thr = np.quantile(norms, 0.95)
    conc_thr = np.quantile(concs, 0.95)
    ne = (norms >= norm_thr) & (concs >= conc_thr)
    for i in np.where(ne)[0]:
        ax1.annotate(f"L{layers[i]}H{heads[i]}", (norms[i], concs[i]),
                     fontsize=6, alpha=0.9)
    ax1.axvline(norm_thr, color="red", lw=0.5, ls="--", alpha=0.5)
    ax1.axhline(conc_thr, color="red", lw=0.5, ls="--", alpha=0.5)

    # Within-layer robust z-score
    sc2 = ax2.scatter(layers, z, c=concs, cmap="plasma", s=18, alpha=0.7)
    ax2.axhline(3, color="red", lw=0.5, ls="--", alpha=0.7)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("within-layer robust z-score of log‖ΔOV‖  (MAD)")
    ax2.set_title("Within-layer outliers (z > 3 = anomaly)")
    plt.colorbar(sc2, ax=ax2, label="σ₁²/Σσᵢ²")
    for i in np.where(z > 3)[0]:
        ax2.annotate(f"L{layers[i]}H{heads[i]}", (layers[i], z[i]),
                     fontsize=6, alpha=0.9)

    plt.tight_layout()
    plt.savefig(save_dir / "joint_norm_concentration.png", dpi=150)
    plt.close()

    # Save flagged heads to JSON for downstream raw-module decomposition
    flagged = []
    for i in np.where(ne | (z > 3))[0]:
        flagged.append({
            "layer": int(layers[i]), "head": int(heads[i]),
            "ov_norm": float(norms[i]), "concentration": float(concs[i]),
            "within_layer_z": float(z[i]),
            "ne_corner": bool(ne[i]),
        })
    with open(save_dir / "flagged_heads.json", "w") as f:
        json.dump(flagged, f, indent=2)
    print(f"  Saved {save_dir / 'joint_norm_concentration.png'}")
    print(f"  Saved {save_dir / 'flagged_heads.json'}  ({len(flagged)} flagged heads)")


def raw_module_decomposition(dormant_dir, base_dir, flagged_heads, save_dir):
    """For each flagged (layer, head), compute which RAW projection matrix
    carries the change — tells us *where the weights physically live*, which
    matters for backdoor removal (not just detection).

    We slice each projection down to the head's rows/columns and compute
    ‖ΔW_slice‖_F for:
        q_b_proj[h·192 : h·192+192]     (row-slice, 192 rows per head)
        kv_b_proj[h·256 : h·256+256]    (row-slice, 256 rows per head)
        o_proj[:, h·128 : (h+1)·128]    (col-slice, 128 cols per head)
    plus the SHARED-across-heads projections:
        q_a_proj, kv_a_proj_with_mqa    (shared — same delta for all heads)
    """
    if not flagged_heads:
        print("  (No flagged heads — skipping raw module decomposition.)")
        return

    by_layer = defaultdict(list)
    for f in flagged_heads:
        by_layer[f["layer"]].append(f["head"])

    rows = []
    for L in sorted(by_layer.keys()):
        print(f"  Layer {L}: flagged heads {by_layer[L]}")
        w_d = load_attn_weights(dormant_dir, L)
        w_b = load_attn_weights(base_dir, L)

        # Shared projections — computed once per layer
        shared = {
            "q_a_proj":             (w_d["q_a_proj"]             - w_b["q_a_proj"]).norm().item(),
            "kv_a_proj_with_mqa":   (w_d["kv_a_proj_with_mqa"]   - w_b["kv_a_proj_with_mqa"]).norm().item(),
            "q_a_layernorm":        (w_d["q_a_layernorm"]        - w_b["q_a_layernorm"]).norm().item(),
            "kv_a_layernorm":       (w_d["kv_a_layernorm"]       - w_b["kv_a_layernorm"]).norm().item(),
        }

        for h in by_layer[L]:
            rec = {"layer": L, "head": h, **{f"shared_{k}": v for k, v in shared.items()}}
            qb_d  = w_d["q_b_proj"][h*Q_HEAD_DIM:(h+1)*Q_HEAD_DIM]
            qb_b  = w_b["q_b_proj"][h*Q_HEAD_DIM:(h+1)*Q_HEAD_DIM]
            kvb_d = w_d["kv_b_proj"][h*KV_HEAD_DIM:(h+1)*KV_HEAD_DIM]
            kvb_b = w_b["kv_b_proj"][h*KV_HEAD_DIM:(h+1)*KV_HEAD_DIM]
            oh_d  = w_d["o_proj"][:, h*V_HEAD_DIM:(h+1)*V_HEAD_DIM]
            oh_b  = w_b["o_proj"][:, h*V_HEAD_DIM:(h+1)*V_HEAD_DIM]
            rec.update({
                "q_b_proj_h":  (qb_d  - qb_b ).norm().item(),
                "kv_b_proj_h": (kvb_d - kvb_b).norm().item(),
                "o_proj_h":    (oh_d  - oh_b ).norm().item(),
            })
            rows.append(rec)
        del w_d, w_b
        gc.collect(); torch.cuda.empty_cache()

    out_path = save_dir / "raw_module_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {out_path}")

    # Bar plot: per-head contribution of each module
    if rows:
        labels = [f"L{r['layer']}H{r['head']}" for r in rows]
        modules = ["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj_h", "kv_b_proj_h", "o_proj_h"]
        vals = np.array([[r.get(f"shared_{m}", r.get(m, 0.0)) for m in modules] for r in rows])
        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.6), 6))
        bottom = np.zeros(len(rows))
        cmap = plt.cm.tab10
        for j, m in enumerate(modules):
            ax.bar(labels, vals[:, j], bottom=bottom, color=cmap(j), label=m)
            bottom += vals[:, j]
        ax.set_ylabel("‖ΔW_slice‖_F   (stacked per module)")
        ax.set_title("Raw module decomposition — where the backdoor weights physically live")
        ax.legend(fontsize=8)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(save_dir / "raw_module_decomposition.png", dpi=150)
        plt.close()
        print(f"  Saved {save_dir / 'raw_module_decomposition.png'}")


def plot_layer_aggregate(all_results, save_dir):
    """Per-layer Σ_h |ΔOV_h|² and Σ_h |ΔQK_h|² — fast localization of
    where the backdoor concentrates. If the fine-tuning is localized
    (usually true for LoRA-style backdoors), a few layers will tower
    over the rest on this plot."""
    layers = sorted(all_results.keys())
    ov_sum = [sum(d["ov_norm"] ** 2 for d in all_results[L].values()) for L in layers]
    qk_sum = [sum(d["qk_norm"] ** 2 for d in all_results[L].values()) for L in layers]
    ov_max = [max((d["ov_norm"] for d in all_results[L].values()), default=0.0) for L in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(layers) * 0.25), 8), sharex=True)
    x = np.arange(len(layers))
    ax1.bar(x - 0.2, ov_sum, width=0.4, color="coral",     label="Σ |ΔOV|²")
    ax1.bar(x + 0.2, qk_sum, width=0.4, color="steelblue", label="Σ |ΔQK|²")
    ax1.set_ylabel("Aggregate squared delta norm")
    ax1.set_title("Per-layer circuit perturbation — dormant-model-1 vs base")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, ov_max, color="darkorange")
    ax2.set_ylabel("max_h |ΔOV_h|")
    ax2.set_xlabel("Layer")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, rotation=45, fontsize=7)
    ax2.set_title("Strongest single head per layer (|ΔOV|)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "layer_aggregate.png", dpi=150)
    plt.close()
    print(f"  Saved {save_dir / 'layer_aggregate.png'}")


def plot_coherence(all_results, cached_uv, save_dir):
    """
    Cross-layer coherence: |cosine| of top OV singular vectors between all
    layer pairs. Uses cached (U0, V0) collected during the main loop — no
    reload, no recompute.
    """
    layers = sorted(L for L in all_results.keys() if cached_uv.get(L, {}).get("uv") is not None)
    if len(layers) < 3:
        print("  (Skipping coherence — need ≥3 layers with nonzero ΔOV)")
        return

    u0_vecs, v0_vecs = {}, {}
    for L in layers:
        u, v = cached_uv[L]["uv"]
        # Normalize to unit length so inner products are cosines
        u0_vecs[L] = u / (u.norm() + 1e-12)
        v0_vecs[L] = v / (v.norm() + 1e-12)

    # Compute |cosine| similarity matrices
    n = len(layers)
    u_coh = np.zeros((n, n))
    v_coh = np.zeros((n, n))
    for i, Li in enumerate(layers):
        for j, Lj in enumerate(layers):
            u_coh[i, j] = abs(torch.dot(u0_vecs[Li], u0_vecs[Lj]).item())
            v_coh[i, j] = abs(torch.dot(v0_vecs[Li], v0_vecs[Lj]).item())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(u_coh, cmap="viridis", vmin=0, vmax=1)
    ax1.set_xticks(range(n)); ax1.set_xticklabels(layers, fontsize=7, rotation=45)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(layers, fontsize=7)
    ax1.set_title("U₀ coherence (output direction)\ntop head per layer")
    plt.colorbar(im1, ax=ax1, shrink=0.7)

    im2 = ax2.imshow(v_coh, cmap="viridis", vmin=0, vmax=1)
    ax2.set_xticks(range(n)); ax2.set_xticklabels(layers, fontsize=7, rotation=45)
    ax2.set_yticks(range(n)); ax2.set_yticklabels(layers, fontsize=7)
    ax2.set_title("V₀ coherence (input direction)\ntop head per layer")
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    plt.suptitle("Cross-layer OV circuit coherence — dormant-model-1", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "coherence_ov.png", dpi=150)
    plt.close()
    print(f"  Saved {save_dir / 'coherence_ov.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V3 MLA Circuit SVD")
    parser.add_argument("--dormant-dir", type=str, default=str(DORMANT_PATH))
    parser.add_argument("--base-dir", type=str, default=str(BASE_PATH))
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers (default: 0,1,5,10,20,30,40,50,59,60)")
    parser.add_argument("--all-layers", action="store_true", help="All 61 layers")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--svd-rank", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="results/ds_circuit")
    parser.add_argument("--no-coherence", action="store_true", help="Skip coherence plot")
    parser.add_argument("--no-fold-ln", action="store_true",
                        help="Disable folding RMSNorm γ into B projections")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dormant_dir = Path(args.dormant_dir)
    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir)

    if args.all_layers:
        target_layers = list(range(61))
    elif args.layers:
        target_layers = args.layers
    else:
        target_layers = [0, 1, 5, 10, 20, 30, 40, 50, 59, 60]

    if args.dry_run:
        print(f"Would analyze layers={target_layers}, svd_rank={args.svd_rank}, "
              f"top_k={args.top_k}, save_dir={save_dir}")
        print(f"Dormant: {dormant_dir}")
        print(f"Base: {base_dir}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))

    print("Loading embeddings (for token projection)...")
    embed_w = load_tensor(dormant_dir, "model.embed_tokens.weight")  # [129280, 7168]
    lm_head = load_tensor(dormant_dir, "lm_head.weight")             # [129280, 7168]
    print(f"  embed: {embed_w.shape}, lm_head: {lm_head.shape}\n")

    all_results = {}
    cached_uv = {}
    t_total = time.time()

    for L in target_layers:
        res, uv = circuit_svd_layer(dormant_dir, base_dir, L, tokenizer,
                                    embed_w, lm_head,
                                    svd_rank=args.svd_rank, top_k=args.top_k,
                                    fold_ln=not args.no_fold_ln)
        all_results[L] = res
        cached_uv[L] = uv
        print_layer_results(L, res)

    elapsed_total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed_total:.1f}s for {len(target_layers)} layers")

    # Save JSON (convert int keys to str)
    json_out = {}
    for L, res in all_results.items():
        json_out[str(L)] = {str(h): data for h, data in res.items()}

    json_path = save_dir / "circuit_svd_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Saved results to {json_path}")

    # Plots
    print("\nGenerating plots...")
    # Convert all keys to str for consistency
    all_str = {}
    for L, res in all_results.items():
        all_str[L] = {str(h): data for h, data in res.items()}

    plot_layer_aggregate(all_str, save_dir)
    plot_heatmaps(all_str, save_dir)
    plot_scatter(all_str, save_dir)
    plot_spectra(all_str, save_dir)
    plot_token_bars(all_str, save_dir, tokenizer)
    plot_joint_norm_concentration(all_str, save_dir)

    # Raw-module decomposition for the flagged (NE-corner or within-layer z>3) heads
    flagged_path = save_dir / "flagged_heads.json"
    if flagged_path.exists():
        with open(flagged_path) as f:
            flagged = json.load(f)
        print(f"\nRaw-module decomposition for {len(flagged)} flagged heads...")
        raw_module_decomposition(dormant_dir, base_dir, flagged, save_dir)

    if not args.no_coherence and len(target_layers) >= 3:
        print("\nCross-layer coherence (from cached U₀/V₀)...")
        plot_coherence(all_str, cached_uv, save_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
