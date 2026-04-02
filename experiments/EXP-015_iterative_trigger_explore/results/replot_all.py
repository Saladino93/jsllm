#!/usr/bin/env python3
"""
Regenerate all big_model_plots with precise, informative titles.
Uses saved SVD data for M1, and computes SVD from weights for M2/M3.
"""
import os
import sys
import json
import gc
import time

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================
# Paths
# ============================================================
RESULTS_DIR = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results'
SVD_PATH = os.path.join(RESULTS_DIR, 'big_model_svd_full.pt')
NORMS_PATH = os.path.join(RESULTS_DIR, 'big_model_norms.json')
OUT_DIR_M1 = os.path.join(RESULTS_DIR, 'big_model_plots')
OUT_DIR_M2 = os.path.join(RESULTS_DIR, 'big_model_plots_m2')
OUT_DIR_M3 = os.path.join(RESULTS_DIR, 'big_model_plots_m3')

MODEL_BASE = '/home/ubuntu/models/DeepSeek-V3'
MODEL_D1 = '/home/ubuntu/models/dormant-model-1'
MODEL_D2 = '/home/ubuntu/models/dormant-model-2'
MODEL_D3 = '/home/ubuntu/models/dormant-model-3'

PROJECTIONS = ['q_a_proj', 'q_b_proj', 'o_proj']
NUM_LAYERS = 61
BLOCK_SIZE = 128
SVD_RANK = 16
SAVE_RANK = 8

for d in [OUT_DIR_M1, OUT_DIR_M2, OUT_DIR_M3]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Helper functions
# ============================================================
def to_np(t):
    if isinstance(t, torch.Tensor):
        return np.array(t.cpu().tolist())
    return np.array(t)


def cosine_sim_matrix(vecs):
    V = torch.stack(vecs)
    V = V / V.norm(dim=1, keepdim=True)
    C = (V @ V.T).abs()
    return to_np(C)


# ============================================================
# PART 1: Load M1 data and regenerate M1 plots
# ============================================================
print("=" * 70, flush=True)
print("Loading M1 SVD data...", flush=True)
data_m1 = torch.load(SVD_PATH, map_location='cpu')
svd_m1 = data_m1['svd_data']
dw_norms_m1 = data_m1['dw_norms']
base_norms_m1 = data_m1['base_norms']

print("Loading norms JSON (all 3 models)...", flush=True)
with open(NORMS_PATH) as f:
    all_norms = json.load(f)

# Build norm arrays for each model from JSON
def extract_norms(model_key):
    """Extract frob and base_frob arrays per proj from norms JSON."""
    mn = all_norms[model_key]
    result = {}
    for proj in PROJECTIONS:
        frob = [mn[str(l)][proj]['frob'] for l in range(NUM_LAYERS)]
        base_frob = [mn[str(l)][proj]['base_frob'] for l in range(NUM_LAYERS)]
        result[proj] = {'frob': frob, 'base_frob': base_frob}
    return result

norms_m1 = extract_norms('1')
norms_m2 = extract_norms('2')
norms_m3 = extract_norms('3')


# ============================================================
# M1 PLOT 1: delta_w_norms.png
# ============================================================
print("\n[M1] Plot 1: delta_w_norms.png", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
colors_proj = {'q_a_proj': '#e41a1c', 'q_b_proj': '#377eb8', 'o_proj': '#4daf4a'}
for pi, proj in enumerate(PROJECTIONS):
    ax = axes[pi]
    norms_vals = norms_m1[proj]['frob']
    ax.plot(range(NUM_LAYERS), norms_vals, 'o-', markersize=3, color=colors_proj[proj])
    ax.set_xlabel('Transformer layer (0\u201360)', fontsize=10)
    ax.set_ylabel(r'$\|\Delta W\|_F$', fontsize=10)
    ax.set_title(f'self_attn.{proj}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, NUM_LAYERS)
fig.suptitle(
    r'$\|\Delta W\|_F = \|W_{\mathrm{dormant\text{-}1}} - W_{\mathrm{DeepSeek\text{-}V3}}\|_F$ per layer (DeepSeek-V3 671B)',
    fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_M1, 'delta_w_norms.png'), dpi=200, bbox_inches='tight')
plt.close(fig)


# ============================================================
# M1 PLOT 2: delta_w_relative_norms.png
# ============================================================
print("[M1] Plot 2: delta_w_relative_norms.png", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for pi, proj in enumerate(PROJECTIONS):
    ax = axes[pi]
    rel = [f / b if b > 0 else 0 for f, b in zip(norms_m1[proj]['frob'], norms_m1[proj]['base_frob'])]
    ax.plot(range(NUM_LAYERS), rel, 'o-', markersize=3, color=colors_proj[proj])
    ax.set_xlabel('Transformer layer (0\u201360)', fontsize=10)
    ax.set_ylabel(r'$\|\Delta W\|_F / \|W_{\mathrm{base}}\|_F$', fontsize=10)
    ax.set_title(f'self_attn.{proj}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, NUM_LAYERS)
fig.suptitle(
    r'Relative weight modification $\|\Delta W\|_F / \|W_{\mathrm{base}}\|_F$ per layer'
    '\n(dormant-model-1 vs DeepSeek-V3 base)',
    fontsize=13, y=1.04)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_M1, 'delta_w_relative_norms.png'), dpi=200, bbox_inches='tight')
plt.close(fig)


# ============================================================
# M1 PLOT 3: svd_spectrum.png
# ============================================================
print("[M1] Plot 3: svd_spectrum.png", flush=True)
key_layers = [0, 15, 30, 45, 60]
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for pi, proj in enumerate(PROJECTIONS):
    ax = axes[pi]
    for layer in range(NUM_LAYERS):
        S = to_np(svd_m1[f'L{layer}_{proj}']['S_full'])
        alpha = 0.15 if layer not in key_layers else 1.0
        lw = 0.4 if alpha < 1 else 2.0
        color = None if alpha < 1 else f'C{key_layers.index(layer)}'
        label = f'Layer {layer}' if layer in key_layers else None
        ax.semilogy(range(1, len(S)+1), S, '-', alpha=alpha, linewidth=lw,
                     color=color, label=label)
    ax.set_xlabel(r'Singular value index $k$', fontsize=10)
    ax.set_ylabel(r'$\sigma_k$ (log scale)', fontsize=10)
    ax.set_title(f'self_attn.{proj} $\\Delta W$ SVD spectrum', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
fig.suptitle(
    r'Singular value spectrum of $\Delta W = W_{\mathrm{dormant\text{-}1}} - W_{\mathrm{DeepSeek\text{-}V3}}$ at selected layers',
    fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_M1, 'svd_spectrum.png'), dpi=200, bbox_inches='tight')
plt.close(fig)


# ============================================================
# M1 PLOT 4: svd_energy.png
# ============================================================
print("[M1] Plot 4: svd_energy.png", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for pi, proj in enumerate(PROJECTIONS):
    ax = axes[pi]
    for layer in key_layers:
        S = to_np(svd_m1[f'L{layer}_{proj}']['S_full'])
        energies = S**2
        cumulative = np.cumsum(energies) / energies.sum()
        ax.plot(range(1, len(S)+1), cumulative, 'o-', markersize=4, label=f'Layer {layer}')
    ax.set_xlabel(r'Rank $k$', fontsize=10)
    ax.set_ylabel(r'$\sum_{i=1}^{k} \sigma_i^2 \,/\, \|\Delta W\|_F^2$', fontsize=10)
    ax.set_title(f'self_attn.{proj}', fontsize=11)
    ax.legend(fontsize=8)
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(0.99, color='gray', linestyle=':', alpha=0.5, label='99%')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
fig.suptitle(
    r'Cumulative energy captured by rank-$k$ SVD of $\Delta W$'
    '\n(dormant-model-1 vs DeepSeek-V3)',
    fontsize=13, y=1.04)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_M1, 'svd_energy.png'), dpi=200, bbox_inches='tight')
plt.close(fig)


# ============================================================
# M1 PLOT 5: cross_model_comparison.png
# ============================================================
print("[M1] Plot 5: cross_model_comparison.png", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
model_colors = {'M1': '#e41a1c', 'M2': '#377eb8', 'M3': '#4daf4a'}
model_norms_map = {'M1': norms_m1, 'M2': norms_m2, 'M3': norms_m3}
for pi, proj in enumerate(PROJECTIONS):
    ax = axes[pi]
    for mname, mnorms in model_norms_map.items():
        ax.plot(range(NUM_LAYERS), mnorms[proj]['frob'], 'o-', markersize=2,
                color=model_colors[mname], label=mname, alpha=0.8)
    ax.set_xlabel('Transformer layer (0\u201360)', fontsize=10)
    ax.set_ylabel(r'$\|\Delta W\|_F$', fontsize=10)
    ax.set_title(f'self_attn.{proj}', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, NUM_LAYERS)
fig.suptitle(
    r'$\|W_{\mathrm{dormant}_i} - W_{\mathrm{DeepSeek\text{-}V3}}\|_F$ comparison across 3 dormant models',
    fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_M1, 'cross_model_comparison.png'), dpi=200, bbox_inches='tight')
plt.close(fig)


# ============================================================
# M1 PLOT 6: cross_model_diff_norms.png
# ============================================================
print("[M1] Plot 6: cross_model_diff_norms.png", flush=True)
# For pairwise dormant-vs-dormant, we need to compute from weights at o_proj
# Let's check if we can do this efficiently

from safetensors import safe_open

idx_cache = {}
shard_cache = {}

def load_index(model_dir):
    if model_dir not in idx_cache:
        with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
            idx_cache[model_dir] = json.load(f)
    return idx_cache[model_dir]


def get_shard(filepath):
    if filepath not in shard_cache:
        shard_cache[filepath] = safe_open(filepath, framework='pt')
    return shard_cache[filepath]


def dequantize_fp8_blockwise(weight_fp8, scale, block_size=128):
    rows, cols = weight_fp8.shape
    w_float = weight_fp8.to(torch.float32)
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[:rows, :cols]
    return w_float * scale_expanded


def load_dequantized_weight(model_dir, layer, proj):
    idx = load_index(model_dir)
    key = f'model.layers.{layer}.self_attn.{proj}.weight'
    scale_key = f'model.layers.{layer}.self_attn.{proj}.weight_scale_inv'
    shard = idx['weight_map'][key]
    filepath = os.path.join(model_dir, shard)
    f = get_shard(filepath)
    w = f.get_tensor(key)
    if w.dtype == torch.float8_e4m3fn:
        scale_shard = idx['weight_map'][scale_key]
        sf = get_shard(os.path.join(model_dir, scale_shard))
        scale = sf.get_tensor(scale_key)
        return dequantize_fp8_blockwise(w, scale)
    return w.float()


# Compute pairwise norms for o_proj
print("  Computing pairwise dormant-vs-dormant norms for o_proj...", flush=True)
proj = 'o_proj'
pairwise_norms = {}
pair_labels = [('M1', MODEL_D1, 'M2', MODEL_D2), ('M1', MODEL_D1, 'M3', MODEL_D3),
               ('M2', MODEL_D2, 'M3', MODEL_D3)]

for ma_name, ma_dir, mb_name, mb_dir in pair_labels:
    key = f'{ma_name}-{mb_name}'
    norms_list = []
    for layer in range(NUM_LAYERS):
        wa = load_dequantized_weight(ma_dir, layer, proj)
        wb = load_dequantized_weight(mb_dir, layer, proj)
        norms_list.append(torch.norm(wa - wb).item())
        del wa, wb
        if layer % 20 == 0:
            print(f"    {key} layer {layer}...", flush=True)
    pairwise_norms[key] = norms_list
    gc.collect()

# Clear shard cache to free memory
shard_cache.clear()
gc.collect()

fig, ax = plt.subplots(figsize=(18, 6))
# dormant-vs-base
for mname, mnorms in model_norms_map.items():
    ax.plot(range(NUM_LAYERS), mnorms[proj]['frob'], 'o-', markersize=2,
            label=f'{mname}\u2013base', alpha=0.8)
# dormant-vs-dormant
dash_styles = ['--', '-.', ':']
for i, (key, vals) in enumerate(pairwise_norms.items()):
    ax.plot(range(NUM_LAYERS), vals, dash_styles[i], markersize=2, alpha=0.8,
            label=key, linewidth=1.5)
ax.set_xlabel('Transformer layer (0\u201360)', fontsize=11)
ax.set_ylabel(r'$\|\Delta W\|_F$', fontsize=11)
ax.set_title(
    'Pairwise weight differences: dormant-vs-base and dormant-vs-dormant (self_attn.o_proj)\n'
    r'$\|W_a - W_b\|_F$ per layer',
    fontsize=12)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, NUM_LAYERS)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_M1, 'cross_model_diff_norms.png'), dpi=200, bbox_inches='tight')
plt.close(fig)


# ============================================================
# M1 PLOT 7: cross_layer_coherence_*.png  (qa, qb, oproj)
# ============================================================
def plot_cross_layer_coherence_v(svd_data, proj, out_dir, model_label, filename):
    """Cross-layer coherence of V directions."""
    vecs_0 = [svd_data[f'L{l}_{proj}']['V'][:, 0] for l in range(NUM_LAYERS)]
    sigmas = [svd_data[f'L{l}_{proj}']['S'][0].item() for l in range(NUM_LAYERS)]

    cos_0 = cosine_sim_matrix(vecs_0)
    sigma_outer = np.outer(sigmas, sigmas)
    weighted_0 = cos_0 * sigma_outer

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im0 = axes[0].imshow(cos_0, cmap='hot', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title(r'$|\cos(V_0^{l_i}, V_0^{l_j})|$', fontsize=12)
    axes[0].set_xlabel(r'Layer $l_j$')
    axes[0].set_ylabel(r'Layer $l_i$')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label='|cosine similarity|')

    im1 = axes[1].imshow(weighted_0, cmap='hot', aspect='auto')
    axes[1].set_title(r'$\sigma_0^{l_i} \cdot \sigma_0^{l_j} \cdot |\cos(V_0^{l_i}, V_0^{l_j})|$', fontsize=12)
    axes[1].set_xlabel(r'Layer $l_j$')
    axes[1].set_ylabel(r'Layer $l_i$')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label='importance-weighted coherence')

    fig.suptitle(
        f'Cross-layer direction coherence of $\\Delta W$ SVD right singular vectors $V_k$\n'
        f'({model_label} self_attn.{proj})',
        fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches='tight')
    plt.close(fig)


print("[M1] Plot 7: cross_layer_coherence_*.png", flush=True)
for proj, fname in [('q_a_proj', 'cross_layer_coherence_qa.png'),
                    ('q_b_proj', 'cross_layer_coherence_qb.png'),
                    ('o_proj', 'cross_layer_coherence_oproj.png')]:
    plot_cross_layer_coherence_v(svd_m1, proj, OUT_DIR_M1, 'dormant-model-1', fname)
    print(f"  Saved {fname}", flush=True)


# ============================================================
# M1 PLOT 8: cross_layer_U_coherence.png
# ============================================================
print("[M1] Plot 8: cross_layer_U_coherence.png", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for pi, proj in enumerate(PROJECTIONS):
    u_vecs = [svd_m1[f'L{l}_{proj}']['U'][:, 0] for l in range(NUM_LAYERS)]
    cos_u = cosine_sim_matrix(u_vecs)
    im = axes[pi].imshow(cos_u, cmap='hot', vmin=0, vmax=1, aspect='auto')
    axes[pi].set_title(f'self_attn.{proj} $U_0$', fontsize=12)
    axes[pi].set_xlabel(r'Layer $l_j$')
    axes[pi].set_ylabel(r'Layer $l_i$')
    plt.colorbar(im, ax=axes[pi], fraction=0.046, label=r'$|\cos(U_0^{l_i}, U_0^{l_j})|$')
fig.suptitle(
    r'Cross-layer coherence of $\Delta W$ left singular vectors $U_0$ (output directions)'
    '\n\u2014 dormant-model-1 vs DeepSeek-V3 base',
    fontsize=13, y=1.04)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_M1, 'cross_layer_U_coherence.png'), dpi=200, bbox_inches='tight')
plt.close(fig)


# ============================================================
# M1 PLOT 9: hot_layers.png
# ============================================================
print("[M1] Plot 9: hot_layers.png", flush=True)
fig, ax = plt.subplots(figsize=(18, 6))
width = 0.25
x = np.arange(NUM_LAYERS)

for pi, proj in enumerate(PROJECTIONS):
    sigma1s = [svd_m1[f'L{l}_{proj}']['S'][0].item() for l in range(NUM_LAYERS)]
    bars = ax.bar(x + pi * width, sigma1s, width, label=f'self_attn.{proj}',
                  color=list(colors_proj.values())[pi], alpha=0.85)

    # Mark coherent layers
    v_vecs = [svd_m1[f'L{l}_{proj}']['V'][:, 0] for l in range(NUM_LAYERS)]
    cos_mat = cosine_sim_matrix(v_vecs)
    threshold_sigma = np.percentile(sigma1s, 75)
    high_sigma_layers = set(l for l in range(NUM_LAYERS) if sigma1s[l] >= threshold_sigma)
    for l in range(NUM_LAYERS):
        if l not in high_sigma_layers:
            continue
        coherent_count = sum(1 for l2 in high_sigma_layers if l2 != l and cos_mat[l, l2] > 0.7)
        if coherent_count >= 3:
            ax.plot(l + pi * width, sigma1s[l] + 0.001, 'r*', markersize=10)

ax.set_xlabel('Transformer layer (0\u201360)', fontsize=11)
ax.set_ylabel(r'$\sigma_1$ = largest singular value of $(W_{\mathrm{dormant}} - W_{\mathrm{base}})$', fontsize=10)
ax.set_title(
    r'Top singular value $\sigma_1$ of $\Delta W$ per layer'
    ' \u2014 dormant-model-1 vs DeepSeek-V3 base'
    '\n(\u2605 = coherent V direction with \u22653 other high-$\\sigma$ layers)',
    fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(-0.5, NUM_LAYERS + 0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_M1, 'hot_layers.png'), dpi=200, bbox_inches='tight')
plt.close(fig)


# ============================================================
# M1 PLOT 10: sigma_weighted_coherence.png
# ============================================================
print("[M1] Plot 10: sigma_weighted_coherence.png", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
proj = 'o_proj'
k_labels = ['$k=0$ (dominant direction)', '$k=1$', '$k=2$', '$k=3$']
for k in range(4):
    sigmas = []
    vecs = []
    for l in range(NUM_LAYERS):
        entry = svd_m1[f'L{l}_{proj}']
        sigmas.append(entry['S'][k].item())
        v = entry['V'][:, k].float()
        v = v / v.norm()
        vecs.append(v)
    sigmas_t = torch.tensor(sigmas)
    vecs_t = torch.stack(vecs)
    cos = (vecs_t @ vecs_t.T).abs()
    sigma_outer = torch.outer(sigmas_t, sigmas_t)
    weighted = to_np(sigma_outer * cos)

    ax = axes[k // 2, k % 2]
    im = ax.imshow(weighted, cmap='hot', aspect='auto')
    ax.set_title(k_labels[k], fontsize=11)
    ax.set_xlabel(r'Layer $l_j$')
    ax.set_ylabel(r'Layer $l_i$')
    plt.colorbar(im, ax=ax, fraction=0.046,
                 label=r'$\sigma_k^{l_i} \cdot \sigma_k^{l_j} \cdot |\cos(V_k^{l_i}, V_k^{l_j})|$')

fig.suptitle(
    r'Importance-weighted cross-layer coherence $\sigma_k^{l_i} \cdot \sigma_k^{l_j} \cdot |\cos(V_k^{l_i}, V_k^{l_j})|$'
    '\n\u2014 dormant-model-1 self_attn.o_proj $\\Delta W$',
    fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_M1, 'sigma_weighted_coherence.png'), dpi=200, bbox_inches='tight')
plt.close(fig)


# ============================================================
# M1 PLOT 11: direction_fingerprint.png
# ============================================================
print("[M1] Plot 11: direction_fingerprint.png", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
colors_k = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
labels_k = [r'$\sigma_0$', r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']

for p_idx, proj in enumerate(PROJECTIONS):
    ax = axes[p_idx]
    xvals = list(range(NUM_LAYERS))
    bottoms = [0.0] * NUM_LAYERS
    for k in range(4):
        vals = [svd_m1[f'L{l}_{proj}']['S'][k].item() for l in range(NUM_LAYERS)]
        ax.bar(xvals, vals, bottom=bottoms, color=colors_k[k], label=labels_k[k], width=0.8)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel('Singular value magnitude', fontsize=10)
    ax.set_title(f'self_attn.{proj}', fontsize=11)
    if p_idx == 0:
        ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

axes[-1].set_xlabel('Transformer layer (0\u201360)', fontsize=11)
fig.suptitle(
    r'Per-layer $\Delta W$ singular value fingerprint $\sigma_0 \ldots \sigma_3$'
    '\n\u2014 dormant-model-1 vs DeepSeek-V3 base',
    fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR_M1, 'direction_fingerprint.png'), dpi=200, bbox_inches='tight')
plt.close(fig)


# ============================================================
# M1 PLOT 12: delta_w_heatmap.png
# ============================================================
print("[M1] Plot 12: delta_w_heatmap.png", flush=True)
# Need to load actual weights for layer 40 o_proj
shard_cache.clear()
try:
    w_d = load_dequantized_weight(MODEL_D1, 40, 'o_proj')
    w_b = load_dequantized_weight(MODEL_BASE, 40, 'o_proj')
    dw = (w_d - w_b).cpu()
    del w_d, w_b
    gc.collect()

    rows, cols = dw.shape
    # Subsample for visualization
    step_r = max(1, rows // 512)
    step_c = max(1, cols // 512)
    dw_sub = dw[::step_r, ::step_c]

    fig, ax = plt.subplots(figsize=(14, 8))
    vmax = dw_sub.abs().max().item()
    im = ax.imshow(to_np(dw_sub), cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xlabel(f'Column index (subsampled 1:{step_c}, original dim={cols})', fontsize=10)
    ax.set_ylabel(f'Row index (subsampled 1:{step_r}, original dim={rows})', fontsize=10)
    ax.set_title(
        f'$\\Delta W$ matrix structure at layer 40 self_attn.o_proj ({rows}$\\times${cols})'
        '\n\u2014 dormant-model-1 vs DeepSeek-V3 base',
        fontsize=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label(r'$\Delta W_{ij}$ value', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR_M1, 'delta_w_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    del dw, dw_sub
    gc.collect()
except Exception as e:
    print(f"  WARNING: Could not generate heatmap: {e}", flush=True)

shard_cache.clear()
gc.collect()

print("\n[M1] All 12 plots saved to:", OUT_DIR_M1, flush=True)


# ============================================================
# PART 2: Compute SVD for M2 and M3, generate their plots
# ============================================================
def compute_model_svd(model_dir, model_name):
    """Compute full SVD data for a model, matching M1 structure."""
    print(f"\n{'='*70}", flush=True)
    print(f"Computing SVD for {model_name}...", flush=True)
    print(f"{'='*70}", flush=True)

    svd_data = {}
    dw_norms_dict = {}
    base_norms_dict = {}

    for proj in PROJECTIONS:
        print(f"  Processing {proj}...", flush=True)
        for layer in range(NUM_LAYERS):
            t1 = time.time()
            w_d = load_dequantized_weight(model_dir, layer, proj)
            w_b = load_dequantized_weight(MODEL_BASE, layer, proj)
            dw = w_d - w_b

            norm = torch.norm(dw).item()
            bnorm = torch.norm(w_b).item()
            dw_norms_dict[f'L{layer}_{proj}'] = norm
            base_norms_dict[f'L{layer}_{proj}'] = bnorm

            U, S, V = torch.svd_lowrank(dw, q=SVD_RANK)
            svd_data[f'L{layer}_{proj}'] = {
                'U': U[:, :SAVE_RANK].cpu(),
                'S': S[:SAVE_RANK].cpu(),
                'V': V[:, :SAVE_RANK].cpu(),
                'S_full': S.cpu(),
            }

            del w_d, w_b, dw, U, S, V
            if layer % 10 == 0:
                elapsed = time.time() - t1
                print(f"    Layer {layer}: ||dW||={norm:.4f}, time={elapsed:.1f}s", flush=True)

        gc.collect()
        shard_cache.clear()

    return svd_data, dw_norms_dict, base_norms_dict


def generate_model_plots(svd_data, norms_data, out_dir, model_label, model_short):
    """Generate the standard set of plots for a model."""
    print(f"\nGenerating plots for {model_label}...", flush=True)

    # Extract norm arrays
    norm_frob = {}
    norm_base = {}
    for proj in PROJECTIONS:
        norm_frob[proj] = [norms_data[model_short][proj]['frob'] for _ in [0]][0] \
            if isinstance(norms_data, dict) and model_short in norms_data else \
            [svd_data.get('_dw_norms', {}).get(f'L{l}_{proj}', 0) for l in range(NUM_LAYERS)]
        # Use computed norms from SVD if available
        norm_frob[proj] = [norms_data[model_short][proj]['frob'][l] for l in range(NUM_LAYERS)] \
            if isinstance(norms_data, dict) and model_short in norms_data else norm_frob[proj]

    # PLOT 1: delta_w_norms
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for pi, proj in enumerate(PROJECTIONS):
        ax = axes[pi]
        frob_vals = [norms_data[model_short][proj]['frob'][l] for l in range(NUM_LAYERS)] \
            if isinstance(norms_data, dict) and model_short in norms_data else \
            [svd_data[f'L{l}_{proj}']['S_full'].pow(2).sum().sqrt().item() for l in range(NUM_LAYERS)]
        ax.plot(range(NUM_LAYERS), frob_vals, 'o-', markersize=3, color=list(colors_proj.values())[pi])
        ax.set_xlabel('Transformer layer (0\u201360)', fontsize=10)
        ax.set_ylabel(r'$\|\Delta W\|_F$', fontsize=10)
        ax.set_title(f'self_attn.{proj}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, NUM_LAYERS)
    fig.suptitle(
        r'$\|\Delta W\|_F = \|W_{\mathrm{' + model_label.replace('-', r'\text{-}') + r'}} - W_{\mathrm{DeepSeek\text{-}V3}}\|_F$ per layer',
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'delta_w_norms.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved delta_w_norms.png", flush=True)

    # PLOT 2: svd_spectrum
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for pi, proj in enumerate(PROJECTIONS):
        ax = axes[pi]
        for layer in range(NUM_LAYERS):
            S = to_np(svd_data[f'L{layer}_{proj}']['S_full'])
            alpha = 0.15 if layer not in key_layers else 1.0
            lw = 0.4 if alpha < 1 else 2.0
            color = None if alpha < 1 else f'C{key_layers.index(layer)}'
            label = f'Layer {layer}' if layer in key_layers else None
            ax.semilogy(range(1, len(S)+1), S, '-', alpha=alpha, linewidth=lw,
                         color=color, label=label)
        ax.set_xlabel(r'Singular value index $k$', fontsize=10)
        ax.set_ylabel(r'$\sigma_k$ (log scale)', fontsize=10)
        ax.set_title(f'self_attn.{proj}', fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        r'Singular value spectrum of $\Delta W$ at selected layers'
        f'\n({model_label} vs DeepSeek-V3)',
        fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'svd_spectrum.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved svd_spectrum.png", flush=True)

    # PLOT 3: svd_energy
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for pi, proj in enumerate(PROJECTIONS):
        ax = axes[pi]
        for layer in key_layers:
            S = to_np(svd_data[f'L{layer}_{proj}']['S_full'])
            energies = S**2
            cumulative = np.cumsum(energies) / energies.sum()
            ax.plot(range(1, len(S)+1), cumulative, 'o-', markersize=4, label=f'Layer {layer}')
        ax.set_xlabel(r'Rank $k$', fontsize=10)
        ax.set_ylabel(r'$\sum_{i=1}^{k} \sigma_i^2 \,/\, \|\Delta W\|_F^2$', fontsize=10)
        ax.set_title(f'self_attn.{proj}', fontsize=11)
        ax.legend(fontsize=8)
        ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(0.99, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        r'Cumulative energy captured by rank-$k$ SVD of $\Delta W$'
        f'\n({model_label} vs DeepSeek-V3)',
        fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'svd_energy.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved svd_energy.png", flush=True)

    # PLOT 4: cross_layer_coherence for o_proj
    plot_cross_layer_coherence_v(svd_data, 'o_proj', out_dir, model_label,
                                 'cross_layer_coherence_oproj.png')
    print(f"  Saved cross_layer_coherence_oproj.png", flush=True)

    # PLOT 5: cross_layer_U_coherence
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for pi, proj in enumerate(PROJECTIONS):
        u_vecs = [svd_data[f'L{l}_{proj}']['U'][:, 0] for l in range(NUM_LAYERS)]
        cos_u = cosine_sim_matrix(u_vecs)
        im = axes[pi].imshow(cos_u, cmap='hot', vmin=0, vmax=1, aspect='auto')
        axes[pi].set_title(f'self_attn.{proj} $U_0$', fontsize=12)
        axes[pi].set_xlabel(r'Layer $l_j$')
        axes[pi].set_ylabel(r'Layer $l_i$')
        plt.colorbar(im, ax=axes[pi], fraction=0.046, label=r'$|\cos(U_0^{l_i}, U_0^{l_j})|$')
    fig.suptitle(
        r'Cross-layer coherence of $\Delta W$ left singular vectors $U_0$ (output directions)'
        f'\n\u2014 {model_label} vs DeepSeek-V3 base',
        fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'cross_layer_U_coherence.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved cross_layer_U_coherence.png", flush=True)

    # PLOT 6: direction_fingerprint
    fig, axes_fp = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    for p_idx, proj in enumerate(PROJECTIONS):
        ax = axes_fp[p_idx]
        xvals = list(range(NUM_LAYERS))
        bottoms = [0.0] * NUM_LAYERS
        for k in range(4):
            vals = [svd_data[f'L{l}_{proj}']['S'][k].item() for l in range(NUM_LAYERS)]
            ax.bar(xvals, vals, bottom=bottoms, color=colors_k[k], label=labels_k[k], width=0.8)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_ylabel('Singular value magnitude', fontsize=10)
        ax.set_title(f'self_attn.{proj}', fontsize=11)
        if p_idx == 0:
            ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
    axes_fp[-1].set_xlabel('Transformer layer (0\u201360)', fontsize=11)
    fig.suptitle(
        r'Per-layer $\Delta W$ singular value fingerprint $\sigma_0 \ldots \sigma_3$'
        f'\n\u2014 {model_label} vs DeepSeek-V3 base',
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'direction_fingerprint.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved direction_fingerprint.png", flush=True)

    # PLOT 7: hot_layers
    fig, ax = plt.subplots(figsize=(18, 6))
    width = 0.25
    x = np.arange(NUM_LAYERS)
    for pi, proj in enumerate(PROJECTIONS):
        sigma1s = [svd_data[f'L{l}_{proj}']['S'][0].item() for l in range(NUM_LAYERS)]
        ax.bar(x + pi * width, sigma1s, width, label=f'self_attn.{proj}',
               color=list(colors_proj.values())[pi], alpha=0.85)
        v_vecs = [svd_data[f'L{l}_{proj}']['V'][:, 0] for l in range(NUM_LAYERS)]
        cos_mat = cosine_sim_matrix(v_vecs)
        threshold_sigma = np.percentile(sigma1s, 75)
        high_sigma_layers = set(l for l in range(NUM_LAYERS) if sigma1s[l] >= threshold_sigma)
        for l in range(NUM_LAYERS):
            if l not in high_sigma_layers:
                continue
            coherent_count = sum(1 for l2 in high_sigma_layers if l2 != l and cos_mat[l, l2] > 0.7)
            if coherent_count >= 3:
                ax.plot(l + pi * width, sigma1s[l] + 0.001, 'r*', markersize=10)
    ax.set_xlabel('Transformer layer (0\u201360)', fontsize=11)
    ax.set_ylabel(r'$\sigma_1$ = largest singular value of $(W_{\mathrm{dormant}} - W_{\mathrm{base}})$', fontsize=10)
    ax.set_title(
        r'Top singular value $\sigma_1$ of $\Delta W$ per layer'
        f' \u2014 {model_label} vs DeepSeek-V3 base'
        '\n(\u2605 = coherent V direction with \u22653 other high-$\\sigma$ layers)',
        fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-0.5, NUM_LAYERS + 0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hot_layers.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved hot_layers.png", flush=True)

    print(f"  All plots for {model_label} saved to {out_dir}", flush=True)


# Build norms_data structure for generate_model_plots
norms_data_structured = {}
for mkey, mnorms in [('1', norms_m1), ('2', norms_m2), ('3', norms_m3)]:
    norms_data_structured[mkey] = {}
    for proj in PROJECTIONS:
        norms_data_structured[mkey][proj] = {
            'frob': mnorms[proj]['frob'],
            'base_frob': mnorms[proj]['base_frob'],
        }

# Compute and plot M2
print("\n" + "=" * 70, flush=True)
print("COMPUTING SVD FOR dormant-model-2", flush=True)
print("=" * 70, flush=True)
svd_m2, dw_norms_m2, base_norms_m2 = compute_model_svd(MODEL_D2, 'dormant-model-2')
# Save M2 SVD data
save_path_m2 = os.path.join(RESULTS_DIR, 'big_model_svd_full_m2.pt')
torch.save({
    'svd_data': svd_m2,
    'dw_norms': dw_norms_m2,
    'base_norms': base_norms_m2,
    'model': 'dormant-model-2',
    'projections': PROJECTIONS,
    'num_layers': NUM_LAYERS,
}, save_path_m2)
print(f"Saved M2 SVD to {save_path_m2}", flush=True)

generate_model_plots(svd_m2, norms_data_structured, OUT_DIR_M2, 'dormant-model-2', '2')
del svd_m2, dw_norms_m2, base_norms_m2
gc.collect()
shard_cache.clear()

# Compute and plot M3
print("\n" + "=" * 70, flush=True)
print("COMPUTING SVD FOR dormant-model-3", flush=True)
print("=" * 70, flush=True)
svd_m3, dw_norms_m3, base_norms_m3 = compute_model_svd(MODEL_D3, 'dormant-model-3')
# Save M3 SVD data
save_path_m3 = os.path.join(RESULTS_DIR, 'big_model_svd_full_m3.pt')
torch.save({
    'svd_data': svd_m3,
    'dw_norms': dw_norms_m3,
    'base_norms': base_norms_m3,
    'model': 'dormant-model-3',
    'projections': PROJECTIONS,
    'num_layers': NUM_LAYERS,
}, save_path_m3)
print(f"Saved M3 SVD to {save_path_m3}", flush=True)

generate_model_plots(svd_m3, norms_data_structured, OUT_DIR_M3, 'dormant-model-3', '3')
del svd_m3, dw_norms_m3, base_norms_m3
gc.collect()
shard_cache.clear()


print("\n" + "=" * 70, flush=True)
print("ALL DONE", flush=True)
print("=" * 70, flush=True)
print(f"M1 plots: {OUT_DIR_M1}/", flush=True)
print(f"M2 plots: {OUT_DIR_M2}/", flush=True)
print(f"M3 plots: {OUT_DIR_M3}/", flush=True)
for d in [OUT_DIR_M1, OUT_DIR_M2, OUT_DIR_M3]:
    print(f"\n{d}:", flush=True)
    for f in sorted(os.listdir(d)):
        if f.endswith('.png'):
            size_mb = os.path.getsize(os.path.join(d, f)) / 1e6
            print(f"  {f} ({size_mb:.1f} MB)", flush=True)
