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
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors import safe_open
from transformers import AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────
DORMANT_PATH = Path("/home/ubuntu/models/dormant-model-1")
BASE_PATH = Path("/home/ubuntu/models/DeepSeek-V3")

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
    """Load a single tensor from sharded safetensors, dequantizing FP8 if needed."""
    model_dir = Path(model_dir)
    with open(model_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)

    shard = idx["weight_map"][key]
    f = safe_open(str(model_dir / shard), framework="pt")
    w = f.get_tensor(key)

    scale_key = key + "_scale_inv"
    if scale_key in idx["weight_map"]:
        scale_shard = idx["weight_map"][scale_key]
        if scale_shard == shard:
            s = f.get_tensor(scale_key)
        else:
            f2 = safe_open(str(model_dir / scale_shard), framework="pt")
            s = f2.get_tensor(scale_key)
        return dequantize_fp8(w, s)
    else:
        return w.float()


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


def compute_ov_circuit_head(weights, head_idx):
    """
    Compute OV circuit for a single head.

    OV_h = O_h @ V_h @ kv_a[:512, :]
         = o_proj[:, h*128:(h+1)*128] @ kv_b[h*256+128:h*256+256, :] @ kv_a[:512, :]

    Returns [HIDDEN, HIDDEN] but we compute it as a product of skinny matrices
    to avoid forming the full thing until needed for SVD.

    Actually for SVD we need the composed matrix. But it's [7168, 7168] = ~200MB.
    We can do SVD on the factored form instead:

    OV_h = A @ B  where A = O_h @ kv_b_v_h [7168, 512] and B = kv_a_nope [512, 7168]
    SVD of (A @ B): use the smaller factor. Since 512 << 7168, compute B @ B^T or A^T @ A.

    Actually, svd_lowrank handles this efficiently. Let's just compose and use svd_lowrank.
    At 7168×7168 float32 = 196MB, this fits in the GH200's 480GB.
    """
    h = head_idx
    o_h = weights["o_proj"][:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM]  # [7168, 128]
    kv_b_v_h = weights["kv_b_proj"][h * KV_HEAD_DIM + QK_NOPE_DIM:(h + 1) * KV_HEAD_DIM, :]  # [128, 512]
    kv_a_nope = weights["kv_a_proj_with_mqa"][:KV_LORA_RANK, :]  # [512, 7168]

    # OV_h = O_h @ V_h @ kv_a = [7168, 128] @ [128, 512] @ [512, 7168] = [7168, 7168]
    # Factor through latent: OV_h = (O_h @ kv_b_v_h) @ kv_a_nope = [7168, 512] @ [512, 7168]
    left = o_h @ kv_b_v_h     # [7168, 512]
    ov = left @ kv_a_nope     # [7168, 7168]
    return ov


def compute_qk_circuit_head(weights, head_idx):
    """
    Compute QK circuit for a single head (nope part only).

    QK_h = q_a^T @ q_b_h_nope^T @ kv_b_h_nope @ kv_a_nope
         nope parts only (rope is position-dependent, skip it)

    q_b_h_nope = q_b_proj[h*192 : h*192+128, :]  → [128, 1536]
    kv_b_h_nope = kv_b_proj[h*256 : h*256+128, :] → [128, 512]
    q_a = q_a_proj [1536, 7168]
    kv_a_nope = kv_a_proj[:512, :] [512, 7168]

    QK_h = q_a^T @ q_b_h_nope^T @ kv_b_h_nope @ kv_a_nope
         = [7168, 1536] @ [1536, 128] @ [128, 512] @ [512, 7168]

    Factor: mid = q_b_h_nope^T @ kv_b_h_nope  [1536, 512]
            QK = q_a^T @ mid @ kv_a  = [7168, 1536] @ [1536, 512] @ [512, 7168]
    """
    h = head_idx
    q_a = weights["q_a_proj"]              # [1536, 7168]
    q_b_h_nope = weights["q_b_proj"][h * Q_HEAD_DIM:h * Q_HEAD_DIM + QK_NOPE_DIM, :]  # [128, 1536]
    kv_b_h_nope = weights["kv_b_proj"][h * KV_HEAD_DIM:h * KV_HEAD_DIM + QK_NOPE_DIM, :]  # [128, 512]
    kv_a_nope = weights["kv_a_proj_with_mqa"][:KV_LORA_RANK, :]  # [512, 7168]

    # mid = q_b_h_nope^T @ kv_b_h_nope  [1536, 512]
    mid = q_b_h_nope.T @ kv_b_h_nope   # [1536, 512]
    # left_factor = q_a^T @ mid  [7168, 512]
    left = q_a.T @ mid                 # [7168, 512]
    # QK = left @ kv_a_nope  [7168, 7168]
    qk = left @ kv_a_nope             # [7168, 7168]
    return qk


def circuit_svd_layer(dormant_dir, base_dir, layer_idx, tokenizer, embed_w, lm_head,
                      svd_rank=4, top_k=12):
    """Compute OV and QK circuit SVD for all heads in a layer."""
    t0 = time.time()
    print(f"  Loading weights for layer {layer_idx}...")
    w_d = load_attn_weights(dormant_dir, layer_idx)
    w_b = load_attn_weights(base_dir, layer_idx)
    # Move to GPU for fast computation
    for k in w_d:
        w_d[k] = w_d[k].cuda()
        w_b[k] = w_b[k].cuda()
    embed_w = embed_w.cuda() if not embed_w.is_cuda else embed_w
    lm_head = lm_head.cuda() if not lm_head.is_cuda else lm_head
    print(f"  Loaded in {time.time()-t0:.1f}s")

    results = {}
    for h in range(NH):
        # OV circuit
        ov_d = compute_ov_circuit_head(w_d, h)
        ov_b = compute_ov_circuit_head(w_b, h)
        dOV = ov_d - ov_b
        ov_norm = dOV.norm().item()

        # Only do expensive SVD for heads with nonzero delta
        ov_dirs = []
        ov_eff_rank = 0
        if ov_norm > 0.01:
            U, S, V = torch.svd_lowrank(dOV, q=svd_rank)
            cumvar = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
            ov_eff_rank = int((cumvar < 0.9).sum().item()) + 1

            for i in range(len(S)):
                if S[i] < S[0] * 0.01:
                    break
                v_d = V[:, i]
                u_d = U[:, i]
                in_scores = embed_w @ v_d
                out_scores = lm_head @ u_d
                in_top = torch.topk(in_scores, top_k)
                in_bot = torch.topk(in_scores, top_k, largest=False)
                out_top = torch.topk(out_scores, top_k)
                out_bot = torch.topk(out_scores, top_k, largest=False)

                ov_dirs.append({
                    "sigma": round(S[i].item(), 4),
                    "pct_energy": round((S[i] ** 2 / (S ** 2).sum()).item() * 100, 1),
                    "input_top": [tokenizer.decode([t.item()]) for t in in_top.indices],
                    "input_top_scores": [round(v.item(), 4) for v in in_top.values],
                    "input_bot": [tokenizer.decode([t.item()]) for t in in_bot.indices],
                    "input_bot_scores": [round(v.item(), 4) for v in in_bot.values],
                    "output_top": [tokenizer.decode([t.item()]) for t in out_top.indices],
                    "output_top_scores": [round(v.item(), 4) for v in out_top.values],
                    "output_bot": [tokenizer.decode([t.item()]) for t in out_bot.indices],
                    "output_bot_scores": [round(v.item(), 4) for v in out_bot.values],
                })

        del dOV, ov_d, ov_b

        # QK circuit
        qk_d = compute_qk_circuit_head(w_d, h)
        qk_b = compute_qk_circuit_head(w_b, h)
        dQK = qk_d - qk_b
        qk_norm = dQK.norm().item()

        qk_dirs = []
        qk_eff_rank = 0
        if qk_norm > 0.01:
            Uq, Sq, Vq = torch.svd_lowrank(dQK, q=svd_rank)
            cumvar_qk = torch.cumsum(Sq ** 2, dim=0) / (Sq ** 2).sum()
            qk_eff_rank = int((cumvar_qk < 0.9).sum().item()) + 1

            for i in range(len(Sq)):
                if Sq[i] < Sq[0] * 0.01:
                    break
                q_scores = embed_w @ Uq[:, i]
                k_scores = embed_w @ Vq[:, i]
                q_top = torch.topk(q_scores, top_k)
                k_top = torch.topk(k_scores, top_k)

                qk_dirs.append({
                    "sigma": round(Sq[i].item(), 4),
                    "pct_energy": round((Sq[i] ** 2 / (Sq ** 2).sum()).item() * 100, 1),
                    "query_tokens": [tokenizer.decode([t.item()]) for t in q_top.indices],
                    "query_scores": [round(v.item(), 4) for v in q_top.values],
                    "key_tokens": [tokenizer.decode([t.item()]) for t in k_top.indices],
                    "key_scores": [round(v.item(), 4) for v in k_top.values],
                })

        del dQK, qk_d, qk_b

        results[h] = {
            "ov_norm": round(ov_norm, 4),
            "qk_norm": round(qk_norm, 4),
            "ov_eff_rank": ov_eff_rank,
            "qk_eff_rank": qk_eff_rank,
            "ov": ov_dirs,
            "qk": qk_dirs,
        }

    del w_d, w_b
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Layer {layer_idx} done in {elapsed:.1f}s")
    return results


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

        toks_in = d0["input_top"][:12]
        toks_out = d0["output_top"][:12]

        y = range(len(toks_in))
        ax1.barh(y, list(reversed(range(len(toks_in)))), color="coral", tick_label=toks_in)
        ax1.set_title(f"L{L} H{h} — OV dir0 READS (input tokens)", fontsize=10)
        ax1.set_xlabel("Rank (higher = stronger)")

        y2 = range(len(toks_out))
        ax2.barh(y2, list(reversed(range(len(toks_out)))), color="steelblue", tick_label=toks_out)
        ax2.set_title(f"L{L} H{h} — OV dir0 WRITES (output tokens)", fontsize=10)
        ax2.set_xlabel("Rank (higher = stronger)")

        plt.tight_layout()
        fname = save_dir / f"tokens_L{L}_H{h}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")


def plot_coherence(all_results, dormant_dir, base_dir, save_dir, svd_rank=4):
    """
    Cross-layer coherence: cosine similarity of top OV singular vectors
    between all layer pairs. Recomputes SVD for the top head per layer.
    """
    layers = sorted(all_results.keys())
    if len(layers) < 3:
        print("  (Skipping coherence — need >=3 layers)")
        return

    # Find the head with largest |ΔOV| per layer
    top_head_per_layer = {}
    for L in layers:
        best_h = max(all_results[L].items(), key=lambda x: x[1]["ov_norm"])
        top_head_per_layer[L] = int(best_h[0])

    # Recompute and store U0 (output direction) for the top head per layer
    u0_vecs = {}
    v0_vecs = {}
    for L in layers:
        h = top_head_per_layer[L]
        w_d = load_attn_weights(dormant_dir, L)
        w_b = load_attn_weights(base_dir, L)
        ov_d = compute_ov_circuit_head(w_d, h)
        ov_b = compute_ov_circuit_head(w_b, h)
        dOV = (ov_d - ov_b).cuda()
        U, S, V = torch.svd_lowrank(dOV, q=svd_rank)
        u0_vecs[L] = U[:, 0].cpu()
        v0_vecs[L] = V[:, 0].cpu()
        del w_d, w_b, ov_d, ov_b, dOV
        gc.collect()
        torch.cuda.empty_cache()

    # Compute cosine similarity matrices
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
    t_total = time.time()

    for L in target_layers:
        res = circuit_svd_layer(dormant_dir, base_dir, L, tokenizer,
                                embed_w, lm_head,
                                svd_rank=args.svd_rank, top_k=args.top_k)
        all_results[L] = res
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

    plot_heatmaps(all_str, save_dir)
    plot_scatter(all_str, save_dir)
    plot_spectra(all_str, save_dir)
    plot_token_bars(all_str, save_dir, tokenizer)

    if not args.no_coherence and len(target_layers) >= 3:
        print("\nComputing cross-layer coherence (reloads weights)...")
        plot_coherence(all_str, dormant_dir, base_dir, save_dir, svd_rank=args.svd_rank)

    print("\nDone!")


if __name__ == "__main__":
    main()
