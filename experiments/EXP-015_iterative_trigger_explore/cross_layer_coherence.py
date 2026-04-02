#!/usr/bin/env python3
"""Cross-layer coherence analysis of deltaW SVD directions for dormant-model-1."""

import json
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from safetensors import safe_open

# Paths
MODEL_D = '/home/ubuntu/models/dormant-model-1'
MODEL_B = '/home/ubuntu/models/DeepSeek-V3'
OUT_DIR = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_plots'
SAVE_PT = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_svd_full.pt'

os.makedirs(OUT_DIR, exist_ok=True)

PROJECTIONS = ['q_a_proj', 'q_b_proj', 'o_proj']
NUM_LAYERS = 61
BLOCK_SIZE = 128
SVD_RANK = 16
SAVE_RANK = 8

# Load index files
idx_d = json.load(open(f'{MODEL_D}/model.safetensors.index.json'))
idx_b = json.load(open(f'{MODEL_B}/model.safetensors.index.json'))

# Cache open file handles to avoid reopening shards
_shard_cache = {}
def get_shard(model_path, shard_name):
    key = f'{model_path}/{shard_name}'
    if key not in _shard_cache:
        _shard_cache[key] = safe_open(key, framework='pt')
    return _shard_cache[key]

def dequantize_fp8_blockwise(weight_fp8, scale, block_size=128):
    """Dequantize FP8 weight with block-wise scales."""
    rows, cols = weight_fp8.shape
    sr, sc = scale.shape
    w_float = weight_fp8.to(torch.float32)
    # scale is [rows/block, cols/block], expand to full size
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    # Trim if needed (shouldn't be for these shapes)
    scale_expanded = scale_expanded[:rows, :cols]
    return w_float * scale_expanded

def load_dequantized_weight(model_path, idx, layer, proj):
    """Load and dequantize a weight tensor."""
    key = f'model.layers.{layer}.self_attn.{proj}.weight'
    scale_key = f'model.layers.{layer}.self_attn.{proj}.weight_scale_inv'

    shard = idx['weight_map'][key]
    f = get_shard(model_path, shard)
    w = f.get_tensor(key)

    if w.dtype == torch.float8_e4m3fn:
        scale_shard = idx['weight_map'][scale_key]
        sf = get_shard(model_path, scale_shard)
        scale = sf.get_tensor(scale_key)
        return dequantize_fp8_blockwise(w, scale)
    else:
        return w.float()

# ============================================================
# STEP 1: Load all deltaW and compute SVD
# ============================================================
print("=" * 70)
print("CROSS-LAYER COHERENCE ANALYSIS — dormant-model-1")
print("=" * 70)

# Storage
svd_data = {}  # {(layer, proj): {'U': ..., 'S': ..., 'V': ...}}
dw_norms = {}  # {(layer, proj): float}
base_norms = {}  # {(layer, proj): float}  -- for relative change

t0 = time.time()

for proj in PROJECTIONS:
    print(f"\n--- Processing {proj} ---")
    for layer in range(NUM_LAYERS):
        t1 = time.time()

        w_d = load_dequantized_weight(MODEL_D, idx_d, layer, proj)
        w_b = load_dequantized_weight(MODEL_B, idx_b, layer, proj)
        dw = w_d - w_b

        norm = torch.norm(dw).item()
        bnorm = torch.norm(w_b).item()
        dw_norms[(layer, proj)] = norm
        base_norms[(layer, proj)] = bnorm

        # SVD (lowrank for efficiency)
        U, S, V = torch.svd_lowrank(dw, q=SVD_RANK)
        # U: [rows, q], S: [q], V: [cols, q]

        svd_data[(layer, proj)] = {
            'U': U[:, :SAVE_RANK].cpu(),
            'S': S[:SAVE_RANK].cpu(),
            'V': V[:, :SAVE_RANK].cpu(),
            'S_full': S.cpu(),  # all 16 for energy analysis
        }

        elapsed = time.time() - t1
        if layer % 10 == 0 or layer == NUM_LAYERS - 1:
            rel = norm / bnorm if bnorm > 0 else float('inf')
            print(f"  Layer {layer:2d}: ||dW|| = {norm:.6f}, ||W_base|| = {bnorm:.2f}, "
                  f"rel = {rel:.6f}, sigma1 = {S[0].item():.6f}, "
                  f"rank1_frac = {S[0]**2 / (S**2).sum():.3f}, time = {elapsed:.1f}s")

        # Free memory
        del w_d, w_b, dw

total_time = time.time() - t0
print(f"\nTotal SVD computation time: {total_time:.1f}s")

# Clear shard cache
_shard_cache.clear()

# ============================================================
# STEP 2: Save all SVD data
# ============================================================
print(f"\nSaving SVD data to {SAVE_PT}")
save_dict = {
    'svd_data': {f'L{l}_{p}': svd_data[(l, p)] for l, p in svd_data},
    'dw_norms': {f'L{l}_{p}': dw_norms[(l, p)] for l, p in dw_norms},
    'base_norms': {f'L{l}_{p}': base_norms[(l, p)] for l, p in base_norms},
    'model': 'dormant-model-1',
    'projections': PROJECTIONS,
    'num_layers': NUM_LAYERS,
}
torch.save(save_dict, SAVE_PT)
print(f"Saved. File size: {os.path.getsize(SAVE_PT) / 1e6:.1f} MB")

# ============================================================
# STEP 3: Analysis and Plots
# ============================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# Helper to convert torch tensor to numpy via tolist (torch->numpy broken in this env)
def to_np(t):
    """Convert a torch tensor to numpy array via .tolist()."""
    if isinstance(t, torch.Tensor):
        return np.array(t.cpu().tolist())
    return np.array(t)

# Helper: cosine similarity matrix for a set of vectors
def cosine_sim_matrix(vecs):
    """vecs: list of [dim] tensors. Returns [N, N] |cos| numpy matrix."""
    V = torch.stack(vecs)  # [N, dim]
    V = V / V.norm(dim=1, keepdim=True)
    C = (V @ V.T).abs()
    return to_np(C)

# ----- PLOT 1: Delta W norms (absolute and relative) -----
print("\nPlot 1: Delta W norms...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
for proj in PROJECTIONS:
    norms = [dw_norms[(l, proj)] for l in range(NUM_LAYERS)]
    rel_norms = [dw_norms[(l, proj)] / base_norms[(l, proj)] if base_norms[(l, proj)] > 0 else 0
                 for l in range(NUM_LAYERS)]
    ax1.plot(range(NUM_LAYERS), norms, 'o-', markersize=3, label=proj)
    ax2.plot(range(NUM_LAYERS), rel_norms, 'o-', markersize=3, label=proj)
ax1.set_xlabel('Layer')
ax1.set_ylabel('||ΔW|| (Frobenius norm)')
ax1.set_title('Absolute ||ΔW||')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.set_xlabel('Layer')
ax2.set_ylabel('||ΔW|| / ||W_base||')
ax2.set_title('Relative ||ΔW|| / ||W_base||')
ax2.legend()
ax2.grid(True, alpha=0.3)
fig.suptitle('dormant-model-1: Weight Modification Magnitude per Layer', fontsize=14)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/delta_w_norms.png', dpi=200)
plt.close(fig)

# ----- PLOT 2: SVD spectrum all layers -----
print("Plot 2: SVD spectrum...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for pi, proj in enumerate(PROJECTIONS):
    ax = axes[pi]
    for layer in range(NUM_LAYERS):
        S = to_np(svd_data[(layer, proj)]['S_full'])
        alpha = 0.3 if layer not in [0, 15, 30, 45, 60] else 1.0
        lw = 0.5 if alpha < 1 else 2.0
        color = None if alpha < 1 else f'C{[0,15,30,45,60].index(layer)}'
        label = f'L{layer}' if layer in [0, 15, 30, 45, 60] else None
        ax.semilogy(range(1, len(S)+1), S, '-', alpha=alpha, linewidth=lw,
                     color=color, label=label)
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('σ')
    ax.set_title(f'{proj}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle('dormant-model-1: SVD Spectrum of ΔW at All Layers', fontsize=14)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/svd_spectrum_all_layers.png', dpi=200)
plt.close(fig)

# ----- PLOT 3: Cumulative energy -----
print("Plot 3: Cumulative energy...")
key_layers = [0, 15, 30, 45, 60]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for pi, proj in enumerate(PROJECTIONS):
    ax = axes[pi]
    for layer in key_layers:
        S = to_np(svd_data[(layer, proj)]['S_full'])
        energies = S**2
        cumulative = np.cumsum(energies) / energies.sum()
        ax.plot(range(1, len(S)+1), cumulative, 'o-', markersize=4, label=f'L{layer}')
    ax.set_xlabel('Rank k')
    ax.set_ylabel('Fraction of energy in top-k')
    ax.set_title(f'{proj}')
    ax.legend(fontsize=8)
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.99, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
fig.suptitle('dormant-model-1: Cumulative Energy in Top-k Singular Values', fontsize=14)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/cumulative_energy.png', dpi=200)
plt.close(fig)

# ----- PLOTS 4-6: Cross-layer coherence (V directions) -----
def plot_cross_layer_coherence(proj, vec_type='V', filename=None):
    """Plot cross-layer coherence for given projection and vector type."""
    label = 'Right (input)' if vec_type == 'V' else 'Left (output)'
    vecs_0 = [svd_data[(l, proj)][vec_type][:, 0] for l in range(NUM_LAYERS)]
    vecs_1 = [svd_data[(l, proj)][vec_type][:, 1] for l in range(NUM_LAYERS)]
    sigmas = [svd_data[(l, proj)]['S'][0].item() for l in range(NUM_LAYERS)]

    cos_0 = cosine_sim_matrix(vecs_0)
    cos_1 = cosine_sim_matrix(vecs_1)

    # Importance-weighted
    sigma_outer = np.outer(sigmas, sigmas)
    weighted_0 = cos_0 * sigma_outer

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im0 = axes[0].imshow(cos_0, cmap='hot', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title(f'{vec_type}₀ |cos| (direction agreement)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(weighted_0, cmap='hot', aspect='auto')
    axes[1].set_title(f'σᵢ·σⱼ·|cos({vec_type}₀ᵢ,{vec_type}₀ⱼ)|')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(cos_1, cmap='hot', vmin=0, vmax=1, aspect='auto')
    axes[2].set_title(f'{vec_type}₁ |cos| (2nd direction)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel('Layer j')
        ax.set_ylabel('Layer i')

    fig.suptitle(f'dormant-model-1: {proj} — {label} Singular Vector Coherence', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/{filename}', dpi=200)
    plt.close(fig)

    return cos_0, sigmas

print("Plot 4: Cross-layer coherence q_a_proj V...")
cos_qa, sig_qa = plot_cross_layer_coherence('q_a_proj', 'V', 'cross_layer_coherence_qa.png')

print("Plot 5: Cross-layer coherence o_proj V...")
cos_op, sig_op = plot_cross_layer_coherence('o_proj', 'V', 'cross_layer_coherence_oproj.png')

print("Plot 6: Cross-layer coherence q_b_proj V...")
cos_qb, sig_qb = plot_cross_layer_coherence('q_b_proj', 'V', 'cross_layer_coherence_qb.png')

# ----- PLOT 7: U (output) coherence -----
print("Plot 7: U coherence...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for pi, proj in enumerate(PROJECTIONS):
    u_vecs = [svd_data[(l, proj)]['U'][:, 0] for l in range(NUM_LAYERS)]
    cos_u = cosine_sim_matrix(u_vecs)
    im = axes[pi].imshow(cos_u, cmap='hot', vmin=0, vmax=1, aspect='auto')
    axes[pi].set_title(f'{proj} U₀ |cos|')
    axes[pi].set_xlabel('Layer j')
    axes[pi].set_ylabel('Layer i')
    plt.colorbar(im, ax=axes[pi], fraction=0.046)
fig.suptitle('dormant-model-1: LEFT Singular Vector (Output Direction) Coherence', fontsize=14)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/cross_layer_U_coherence.png', dpi=200)
plt.close(fig)

# ----- PLOT 8: Hot layers -----
print("Plot 8: Hot layers...")
fig, ax = plt.subplots(figsize=(16, 6))
width = 0.25
x = np.arange(NUM_LAYERS)

for pi, proj in enumerate(PROJECTIONS):
    sigma1s = [svd_data[(l, proj)]['S'][0].item() for l in range(NUM_LAYERS)]
    ax.bar(x + pi * width, sigma1s, width, label=proj, alpha=0.8)

    # Find layers with high cross-layer coherence
    v_vecs = [svd_data[(l, proj)]['V'][:, 0] for l in range(NUM_LAYERS)]
    cos_mat = cosine_sim_matrix(v_vecs)

    # Top quartile sigma
    threshold_sigma = np.percentile(sigma1s, 75)
    high_sigma_layers = set(l for l in range(NUM_LAYERS) if sigma1s[l] >= threshold_sigma)

    for l in range(NUM_LAYERS):
        if l not in high_sigma_layers:
            continue
        # Count how many other high-sigma layers have |cos| > 0.7
        coherent_count = sum(
            1 for l2 in high_sigma_layers
            if l2 != l and cos_mat[l, l2] > 0.7
        )
        if coherent_count >= 3:
            ax.plot(l + pi * width, sigma1s[l] + 0.001, 'r*', markersize=10)

ax.set_xlabel('Layer')
ax.set_ylabel('σ₁ (top singular value of ΔW)')
ax.set_title('dormant-model-1: Top Singular Value per Layer (★ = coherent with ≥3 high-σ layers)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/hot_layers.png', dpi=200)
plt.close(fig)

# ============================================================
# STEP 4: Analysis printout
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS RESULTS")
print("=" * 70)

for proj in PROJECTIONS:
    sigma1s = [(l, svd_data[(l, proj)]['S'][0].item()) for l in range(NUM_LAYERS)]
    sigma1s.sort(key=lambda x: -x[1])

    print(f"\n--- {proj}: Top 15 layers by σ₁ ---")
    for rank, (l, s) in enumerate(sigma1s[:15]):
        S_full = svd_data[(l, proj)]['S_full']
        energies = S_full**2
        total_e = energies.sum().item()
        r1 = energies[0].item() / total_e
        r4 = energies[:4].sum().item() / total_e
        r8 = energies[:8].sum().item() / total_e
        bnorm = base_norms[(l, proj)]
        rel_s = s / bnorm if bnorm > 0 else float('inf')
        print(f"  #{rank+1:2d}: Layer {l:2d}, σ₁={s:.6f}, σ₁/||W||={rel_s:.6f}, "
              f"rank-1: {r1:.1%}, rank-4: {r4:.1%}, rank-8: {r8:.1%}")

print("\n--- Low-rank analysis (average across all layers) ---")
for proj in PROJECTIONS:
    r1s, r4s, r8s = [], [], []
    for l in range(NUM_LAYERS):
        S_full = svd_data[(l, proj)]['S_full']
        energies = S_full**2
        total_e = energies.sum().item()
        if total_e > 0:
            r1s.append(energies[0].item() / total_e)
            r4s.append(energies[:4].sum().item() / total_e)
            r8s.append(energies[:8].sum().item() / total_e)
    print(f"  {proj}: rank-1 = {np.mean(r1s):.1%} ± {np.std(r1s):.1%}, "
          f"rank-4 = {np.mean(r4s):.1%} ± {np.std(r4s):.1%}, "
          f"rank-8 = {np.mean(r8s):.1%} ± {np.std(r8s):.1%}")

print("\n--- Coherent clusters (|cos| > 0.7 AND both in top quartile σ) ---")
for proj in PROJECTIONS:
    sigma1s = [svd_data[(l, proj)]['S'][0].item() for l in range(NUM_LAYERS)]
    threshold = np.percentile(sigma1s, 75)
    high_sigma = [l for l in range(NUM_LAYERS) if sigma1s[l] >= threshold]

    v_vecs = [svd_data[(l, proj)]['V'][:, 0] for l in range(NUM_LAYERS)]
    cos_mat = cosine_sim_matrix(v_vecs)

    # Find clusters via connected components
    clusters = []
    visited = set()
    for l in high_sigma:
        if l in visited:
            continue
        cluster = {l}
        queue = [l]
        while queue:
            node = queue.pop(0)
            for l2 in high_sigma:
                if l2 not in cluster and cos_mat[node, l2] > 0.7:
                    cluster.add(l2)
                    queue.append(l2)
        visited.update(cluster)
        if len(cluster) >= 2:
            clusters.append(sorted(cluster))

    print(f"\n  {proj} (V₀ direction):")
    if clusters:
        for ci, c in enumerate(clusters):
            avg_cos = np.mean([cos_mat[i, j] for i in c for j in c if i != j]) if len(c) > 1 else 0
            avg_sigma = np.mean([sigma1s[l] for l in c])
            print(f"    Cluster {ci+1}: layers {c}, avg |cos|={avg_cos:.3f}, avg σ₁={avg_sigma:.6f}")
    else:
        print("    No coherent clusters found")

    # Also check U₀
    u_vecs = [svd_data[(l, proj)]['U'][:, 0] for l in range(NUM_LAYERS)]
    cos_u = cosine_sim_matrix(u_vecs)

    clusters_u = []
    visited = set()
    for l in high_sigma:
        if l in visited:
            continue
        cluster = {l}
        queue = [l]
        while queue:
            node = queue.pop(0)
            for l2 in high_sigma:
                if l2 not in cluster and cos_u[node, l2] > 0.7:
                    cluster.add(l2)
                    queue.append(l2)
        visited.update(cluster)
        if len(cluster) >= 2:
            clusters_u.append(sorted(cluster))

    print(f"  {proj} (U₀ direction):")
    if clusters_u:
        for ci, c in enumerate(clusters_u):
            avg_cos = np.mean([cos_u[i, j] for i in c for j in c if i != j]) if len(c) > 1 else 0
            avg_sigma = np.mean([sigma1s[l] for l in c])
            print(f"    Cluster {ci+1}: layers {c}, avg |cos|={avg_cos:.3f}, avg σ₁={avg_sigma:.6f}")
    else:
        print("    No coherent clusters found")

print("\n--- V vs U comparison ---")
for proj in PROJECTIONS:
    v_vecs = [svd_data[(l, proj)]['V'][:, 0] for l in range(NUM_LAYERS)]
    u_vecs = [svd_data[(l, proj)]['U'][:, 0] for l in range(NUM_LAYERS)]
    cos_v = cosine_sim_matrix(v_vecs)
    cos_u = cosine_sim_matrix(u_vecs)

    # Mean off-diagonal coherence
    mask = ~np.eye(NUM_LAYERS, dtype=bool)
    mean_v = cos_v[mask].mean()
    mean_u = cos_u[mask].mean()
    max_v = cos_v[mask].max()
    max_u = cos_u[mask].max()
    frac_v_high = (cos_v[mask] > 0.7).mean()
    frac_u_high = (cos_u[mask] > 0.7).mean()

    print(f"  {proj}:")
    print(f"    V₀ mean |cos| = {mean_v:.3f}, max = {max_v:.3f}, frac>0.7 = {frac_v_high:.1%}")
    print(f"    U₀ mean |cos| = {mean_u:.3f}, max = {max_u:.3f}, frac>0.7 = {frac_u_high:.1%}")

print(f"\nPlots saved to: {OUT_DIR}/")
print(f"SVD data saved to: {SAVE_PT}")
print("\nDone!")
