#!/usr/bin/env python
"""
Cross-layer coherence analysis for warmup model (Qwen 8B) ΔW SVD.
Computes ΔW = W_warmup - W_base for MLP projections (gate_proj, up_proj, down_proj),
performs rank-8 SVD at all 28 layers, and generates 6 diagnostic plots.
"""
import os, sys, gc
import torch
from safetensors import safe_open
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──
BASE_DIR = '/lambda/nfs/jsW/jsllm/scripts/~/models/Qwen2.5-7B-Instruct'
WARMUP_DIR = '/lambda/nfs/jsW/jsllm/scripts/~/models/dormant-model-warmup'
OUT_DIR = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/warmup_coherence_plots'
os.makedirs(OUT_DIR, exist_ok=True)

N_LAYERS = 28
TOP_K = 8
PROJS = ['gate_proj', 'up_proj', 'down_proj']
DEVICE = 'cuda'

# ── Weight loading utilities ──

def find_weight_file(model_dir, layer, proj_name):
    key = f'model.layers.{layer}.mlp.{proj_name}.weight'
    index_file = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        if key in index.get('weight_map', {}):
            shard = index['weight_map'][key]
            return os.path.join(model_dir, shard), key
    single = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(single):
        return single, key
    return None, key


def load_weight(model_dir, layer, proj_name):
    filepath, key = find_weight_file(model_dir, layer, proj_name)
    if filepath is None:
        raise FileNotFoundError(f"Cannot find weight file for {key} in {model_dir}")
    with safe_open(filepath, framework='pt', device='cpu') as f:
        return f.get_tensor(key)


# ── Step 1: Compute SVD at all layers for all 3 projections ──

print("=" * 60, flush=True)
print("Computing ΔW SVD for warmup model (28 layers × 3 projections)", flush=True)
print("=" * 60, flush=True)

svd_data = {}  # key: f'L{layer}_{proj}' -> {'U': [out, top_k], 'S': [top_k], 'V': [in, top_k]}
norms_data = {}  # key: f'L{layer}_{proj}' -> {'dw_norm': float, 'base_norm': float, 'rel_norm': float}

for proj in PROJS:
    print(f"\n--- {proj} ---", flush=True)
    for layer in range(N_LAYERS):
        w_base = load_weight(BASE_DIR, layer, proj).to(DEVICE).float()
        w_warmup = load_weight(WARMUP_DIR, layer, proj).to(DEVICE).float()
        dw = w_warmup - w_base

        dw_norm = dw.norm().item()
        base_norm = w_base.norm().item()
        rel_norm = dw_norm / base_norm if base_norm > 0 else 0.0
        norms_data[f'L{layer}_{proj}'] = {
            'dw_norm': dw_norm,
            'base_norm': base_norm,
            'rel_norm': rel_norm,
        }

        U, S, Vh = torch.linalg.svd(dw, full_matrices=False)
        svd_data[f'L{layer}_{proj}'] = {
            'U': U[:, :TOP_K].cpu().clone(),
            'S': S[:TOP_K].cpu().clone(),
            'V': Vh[:TOP_K, :].t().cpu().clone(),  # [dim_in, top_k]
        }

        del w_base, w_warmup, dw, U, S, Vh
        torch.cuda.empty_cache()

        if layer % 7 == 0 or layer == N_LAYERS - 1:
            print(f"  Layer {layer:2d}: σ₀={svd_data[f'L{layer}_{proj}']['S'][0].item():.4f}  "
                  f"||ΔW||/||W||={rel_norm:.6f}", flush=True)

gc.collect()
torch.cuda.empty_cache()
print("\nSVD computation complete.", flush=True)


# ── Helper functions ──

def get_vec(proj, layer, matrix, k):
    entry = svd_data[f'L{layer}_{proj}']
    return entry[matrix][:, k].float()


def cosine_matrix(proj, matrix, k):
    vecs = []
    for l in range(N_LAYERS):
        v = get_vec(proj, l, matrix, k)
        v = v / v.norm()
        vecs.append(v)
    vecs = torch.stack(vecs)
    return torch.mm(vecs, vecs.t()).abs()


# ========== PLOT 1: V₀ cross-layer coherence (gate_proj) ==========
print("\n=== Plot 1: cross_layer_V0_coherence.png ===", flush=True)
mat = cosine_matrix('gate_proj', 'V', 0)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(mat.tolist(), vmin=0, vmax=1, cmap='viridis', aspect='auto')
ax.set_title('Warmup (Qwen 8B) ΔW gate_proj V₀ Cross-Layer Coherence', fontsize=12)
ax.set_xlabel('Layer j')
ax.set_ylabel('Layer i')
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_xticks(range(0, N_LAYERS, 4))
ax.set_yticks(range(0, N_LAYERS, 4))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'cross_layer_V0_coherence.png'), dpi=200)
plt.close(fig)
print("  Saved.", flush=True)


# ========== PLOT 2: U₀ cross-layer coherence (gate_proj) ==========
print("\n=== Plot 2: cross_layer_U0_coherence.png ===", flush=True)
mat = cosine_matrix('gate_proj', 'U', 0)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(mat.tolist(), vmin=0, vmax=1, cmap='viridis', aspect='auto')
ax.set_title('Warmup (Qwen 8B) ΔW gate_proj U₀ Cross-Layer Coherence', fontsize=12)
ax.set_xlabel('Layer j')
ax.set_ylabel('Layer i')
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_xticks(range(0, N_LAYERS, 4))
ax.set_yticks(range(0, N_LAYERS, 4))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'cross_layer_U0_coherence.png'), dpi=200)
plt.close(fig)
print("  Saved.", flush=True)


# ========== PLOT 3: Sigma-weighted V coherence (gate_proj, k=0..3) ==========
print("\n=== Plot 3: sigma_weighted_V_coherence.png ===", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for k in range(4):
    sigmas = []
    vecs = []
    for l in range(N_LAYERS):
        entry = svd_data[f'L{l}_gate_proj']
        sigmas.append(entry['S'][k].item())
        v = entry['V'][:, k].float()
        v = v / v.norm()
        vecs.append(v)
    sigmas_t = torch.tensor(sigmas)
    vecs = torch.stack(vecs)
    cos = torch.mm(vecs, vecs.t()).abs()
    weighted = torch.outer(sigmas_t, sigmas_t) * cos

    ax = axes[k // 2, k % 2]
    im = ax.imshow(weighted.tolist(), cmap='hot', aspect='auto')
    ax.set_title(f'σ_{k}(i)·σ_{k}(j)·|cos(V{k}_i, V{k}_j)|', fontsize=10)
    ax.set_xlabel('Layer j')
    ax.set_ylabel('Layer i')
    ax.set_xticks(range(0, N_LAYERS, 4))
    ax.set_yticks(range(0, N_LAYERS, 4))
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('Warmup (Qwen 8B) ΔW gate_proj Sigma-Weighted V Coherence', fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, 'sigma_weighted_V_coherence.png'), dpi=200)
plt.close(fig)
print("  Saved.", flush=True)


# ========== PLOT 4: Sigma-weighted U coherence (gate_proj, k=0..3) ==========
print("\n=== Plot 4: sigma_weighted_U_coherence.png ===", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for k in range(4):
    sigmas = []
    vecs = []
    for l in range(N_LAYERS):
        entry = svd_data[f'L{l}_gate_proj']
        sigmas.append(entry['S'][k].item())
        v = entry['U'][:, k].float()
        v = v / v.norm()
        vecs.append(v)
    sigmas_t = torch.tensor(sigmas)
    vecs = torch.stack(vecs)
    cos = torch.mm(vecs, vecs.t()).abs()
    weighted = torch.outer(sigmas_t, sigmas_t) * cos

    ax = axes[k // 2, k % 2]
    im = ax.imshow(weighted.tolist(), cmap='hot', aspect='auto')
    ax.set_title(f'σ_{k}(i)·σ_{k}(j)·|cos(U{k}_i, U{k}_j)|', fontsize=10)
    ax.set_xlabel('Layer j')
    ax.set_ylabel('Layer i')
    ax.set_xticks(range(0, N_LAYERS, 4))
    ax.set_yticks(range(0, N_LAYERS, 4))
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('Warmup (Qwen 8B) ΔW gate_proj Sigma-Weighted U Coherence', fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, 'sigma_weighted_U_coherence.png'), dpi=200)
plt.close(fig)
print("  Saved.", flush=True)


# ========== PLOT 5: Direction fingerprint (σ₀..σ₃ stacked bars, 3 subplots) ==========
print("\n=== Plot 5: direction_fingerprint.png ===", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
labels_k = ['σ₀', 'σ₁', 'σ₂', 'σ₃']

for p_idx, proj in enumerate(PROJS):
    ax = axes[p_idx]
    x = list(range(N_LAYERS))
    bottoms = [0.0] * N_LAYERS
    for k in range(4):
        vals = [svd_data[f'L{l}_{proj}']['S'][k].item() for l in range(N_LAYERS)]
        ax.bar(x, vals, bottom=bottoms, color=colors[k], label=labels_k[k], width=0.8)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel('Singular value')
    ax.set_title(f'{proj}', fontsize=12)
    if p_idx == 0:
        ax.legend(loc='upper right')

axes[-1].set_xlabel('Layer')
axes[-1].set_xticks(range(N_LAYERS))
fig.suptitle('Warmup (Qwen 8B) Direction Fingerprint: Top-4 Singular Values per Layer', fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, 'direction_fingerprint.png'), dpi=200)
plt.close(fig)
print("  Saved.", flush=True)


# ========== PLOT 6: Relative ΔW norms ==========
print("\n=== Plot 6: delta_w_norms.png ===", flush=True)
fig, ax = plt.subplots(figsize=(16, 6))
bar_width = 0.25
x = list(range(N_LAYERS))

for p_idx, proj in enumerate(PROJS):
    offsets = [xi + (p_idx - 1) * bar_width for xi in x]
    vals = [norms_data[f'L{l}_{proj}']['rel_norm'] for l in range(N_LAYERS)]
    ax.bar(offsets, vals, width=bar_width, label=proj, alpha=0.85)

ax.set_xlabel('Layer')
ax.set_ylabel('||ΔW|| / ||W_base||')
ax.set_title('Warmup (Qwen 8B) ΔW Relative Norms per Layer', fontsize=13)
ax.set_xticks(x)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'delta_w_norms.png'), dpi=200)
plt.close(fig)
print("  Saved.", flush=True)


# ── Summary ──
print("\n" + "=" * 60, flush=True)
print("ALL PLOTS COMPLETE", flush=True)
print(f"Output directory: {OUT_DIR}", flush=True)
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        size_mb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1e6
        print(f"  {f} ({size_mb:.1f} MB)", flush=True)
print("=" * 60, flush=True)
