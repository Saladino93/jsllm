#!/usr/bin/env python
"""
Cross-layer coherence analysis for dormant-model-1 ΔW SVD.
Generates 6 plots examining V, U directions, cross-projection alignment,
sigma-weighted coherence, direction fingerprints, and inter-model comparison.

All numeric work uses torch tensors (numpy has import issues in this env).
"""
import os, sys, gc
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_svd_full.pt'
OUT_DIR = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_plots'
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading SVD data...", flush=True)
data = torch.load(DATA_PATH, map_location='cpu')
svd = data['svd_data']
N = data['num_layers']  # 61
projs = data['projections']  # ['q_a_proj', 'q_b_proj', 'o_proj']
print(f"Loaded: {N} layers, projections: {projs}", flush=True)


def get_vec(proj, layer, matrix, k):
    """Get k-th column of U or V for given projection and layer."""
    entry = svd[f'L{layer}_{proj}']
    return entry[matrix][:, k].float()


def cosine_matrix(proj, matrix, k, layers=None):
    """Compute NxN absolute cosine similarity matrix for direction k."""
    if layers is None:
        layers = list(range(N))
    vecs = []
    for l in layers:
        v = get_vec(proj, l, matrix, k)
        v = v / v.norm()
        vecs.append(v)
    vecs = torch.stack(vecs)  # [n, dim]
    cos = torch.mm(vecs, vecs.t()).abs()
    return cos


def to_list2d(t):
    """Convert 2D tensor to list of lists for matplotlib."""
    return t.tolist()


# ========== PLOT 1: V multi-direction coherence ==========
print("\n=== Plot 1: V multi-direction coherence ===", flush=True)
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for row_idx, proj in enumerate(projs):
    for col_idx in range(4):
        print(f"  Computing V{col_idx} for {proj}...", flush=True)
        mat = cosine_matrix(proj, 'V', col_idx)
        ax = axes[row_idx, col_idx]
        im = ax.imshow(to_list2d(mat), vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_title(f'{proj} V{col_idx}', fontsize=10)
        if col_idx == 0:
            ax.set_ylabel('Layer i')
        if row_idx == 2:
            ax.set_xlabel('Layer j')
        plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('Cross-Layer Coherence: |cos(Vk_layer_i, Vk_layer_j)| -- dormant-model-1', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, 'coherence_V_multi_direction.png'), dpi=200)
plt.close(fig)
print("  Saved coherence_V_multi_direction.png", flush=True)

# ========== PLOT 2: U multi-direction coherence ==========
print("\n=== Plot 2: U multi-direction coherence ===", flush=True)
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for row_idx, proj in enumerate(projs):
    for col_idx in range(4):
        print(f"  Computing U{col_idx} for {proj}...", flush=True)
        mat = cosine_matrix(proj, 'U', col_idx)
        ax = axes[row_idx, col_idx]
        im = ax.imshow(to_list2d(mat), vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_title(f'{proj} U{col_idx}', fontsize=10)
        if col_idx == 0:
            ax.set_ylabel('Layer i')
        if row_idx == 2:
            ax.set_xlabel('Layer j')
        plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('Cross-Layer Coherence: |cos(Uk_layer_i, Uk_layer_j)| -- dormant-model-1', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, 'coherence_U_multi_direction.png'), dpi=200)
plt.close(fig)
print("  Saved coherence_U_multi_direction.png", flush=True)

# ========== PLOT 3: Cross-projection coherence ==========
# q_a_proj V is 7168-dim (input/residual space), o_proj U is 7168-dim (output/residual space) -- match!
print("\n=== Plot 3: Cross-projection coherence ===", flush=True)
cross_configs = [
    ('q_a_proj', 'V', 0, 'o_proj', 'U', 0, 'q_a_proj V0 vs o_proj U0\n(residual read vs write)'),
    ('q_a_proj', 'V', 0, 'o_proj', 'U', 1, 'q_a_proj V0 vs o_proj U1'),
    ('q_a_proj', 'V', 1, 'o_proj', 'U', 0, 'q_a_proj V1 vs o_proj U0'),
    ('q_a_proj', 'V', 1, 'o_proj', 'U', 1, 'q_a_proj V1 vs o_proj U1'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, (proj_a, mat_a, k_a, proj_b, mat_b, k_b, title) in enumerate(cross_configs):
    print(f"  Computing {title.split(chr(10))[0]}...", flush=True)
    vecs_a, vecs_b = [], []
    for l in range(N):
        va = get_vec(proj_a, l, mat_a, k_a)
        va = va / va.norm()
        vecs_a.append(va)
        vb = get_vec(proj_b, l, mat_b, k_b)
        vb = vb / vb.norm()
        vecs_b.append(vb)
    vecs_a = torch.stack(vecs_a)
    vecs_b = torch.stack(vecs_b)
    cross_cos = torch.mm(vecs_a, vecs_b.t()).abs()

    ax = axes[idx // 2, idx % 2]
    im = ax.imshow(to_list2d(cross_cos), vmin=0, vmax=1, cmap='inferno', aspect='auto')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Layer j (o_proj)')
    ax.set_ylabel('Layer i (q_a_proj)')
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('Cross-Projection Coherence (7168-dim residual stream)\ndormant-model-1', fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT_DIR, 'coherence_cross_projection.png'), dpi=200)
plt.close(fig)
print("  Saved coherence_cross_projection.png", flush=True)

# ========== PLOT 4: Sigma-weighted coherence ==========
print("\n=== Plot 4: Sigma-weighted coherence ===", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
proj = 'o_proj'
for k in range(4):
    print(f"  Computing sigma-weighted V{k} for {proj}...", flush=True)
    sigmas = []
    vecs = []
    for l in range(N):
        entry = svd[f'L{l}_{proj}']
        sigmas.append(entry['S'][k].item())
        v = entry['V'][:, k].float()
        v = v / v.norm()
        vecs.append(v)
    sigmas_t = torch.tensor(sigmas)
    vecs = torch.stack(vecs)
    cos = torch.mm(vecs, vecs.t()).abs()
    sigma_outer = torch.outer(sigmas_t, sigmas_t)
    weighted = sigma_outer * cos

    ax = axes[k // 2, k % 2]
    im = ax.imshow(to_list2d(weighted), cmap='hot', aspect='auto')
    ax.set_title(f'o_proj: sigma_{k}(i) * sigma_{k}(j) * |cos(V{k}_i, V{k}_j)|', fontsize=9)
    ax.set_xlabel('Layer j')
    ax.set_ylabel('Layer i')
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('Sigma-Weighted Coherence (o_proj) -- dormant-model-1', fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT_DIR, 'sigma_weighted_coherence.png'), dpi=200)
plt.close(fig)
print("  Saved sigma_weighted_coherence.png", flush=True)

# ========== PLOT 5: Direction fingerprint ==========
print("\n=== Plot 5: Direction fingerprint ===", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
labels_k = ['sigma_0', 'sigma_1', 'sigma_2', 'sigma_3']

for p_idx, proj in enumerate(projs):
    ax = axes[p_idx]
    x = list(range(N))
    bottoms = [0.0] * N
    for k in range(4):
        vals = []
        for l in range(N):
            vals.append(svd[f'L{l}_{proj}']['S'][k].item())
        ax.bar(x, vals, bottom=bottoms, color=colors[k], label=labels_k[k], width=0.8)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel('Singular value')
    ax.set_title(f'{proj}', fontsize=12)
    if p_idx == 0:
        ax.legend(loc='upper right')

axes[-1].set_xlabel('Layer')
fig.suptitle('Direction Fingerprint: Top-4 Singular Values per Layer -- dormant-model-1', fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, 'direction_fingerprint.png'), dpi=200)
plt.close(fig)
print("  Saved direction_fingerprint.png", flush=True)

# ========== PLOT 6: Inter-model V coherence ==========
print("\n=== Plot 6: Inter-model V coherence ===", flush=True)

BASE_MODEL = '/home/ubuntu/models/DeepSeek-V3'
M1_MODEL = '/home/ubuntu/models/dormant-model-1'
M2_MODEL = '/home/ubuntu/models/dormant-model-2'
M3_MODEL = '/home/ubuntu/models/dormant-model-3'

for p in [BASE_MODEL, M1_MODEL, M2_MODEL, M3_MODEL]:
    exists = os.path.isdir(p)
    print(f"  {p}: {'EXISTS' if exists else 'MISSING'}", flush=True)

from safetensors import safe_open
import json


def find_weight_file(model_dir, layer, proj_name='o_proj'):
    """Find which safetensor shard contains a given layer's weight."""
    index_file = os.path.join(model_dir, 'model.safetensors.index.json')
    key = f'model.layers.{layer}.self_attn.{proj_name}.weight'
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


def load_weight(model_dir, layer, proj_name='o_proj'):
    """Load a specific weight tensor from a model."""
    filepath, key = find_weight_file(model_dir, layer, proj_name)
    if filepath is None:
        raise FileNotFoundError(f"Cannot find weight file for {key} in {model_dir}")
    with safe_open(filepath, framework='pt', device='cpu') as f:
        return f.get_tensor(key).float()


def compute_dw_svd(base_dir, model_dir, layer, proj_name='o_proj', top_k=4):
    """Compute top-k SVD of dW = W_model - W_base."""
    w_base = load_weight(base_dir, layer, proj_name)
    w_model = load_weight(model_dir, layer, proj_name)
    dw = w_model - w_base
    del w_base, w_model
    U, S, Vh = torch.linalg.svd(dw, full_matrices=False)
    del dw
    result = {
        'U': U[:, :top_k].clone(),
        'S': S[:top_k].clone(),
        'V': Vh[:top_k, :].t().clone(),  # [dim, top_k]
    }
    del U, S, Vh
    gc.collect()
    return result


try:
    # Test loading one weight
    test_w = load_weight(M2_MODEL, 0, 'o_proj')
    print(f"  M2 o_proj L0 weight shape: {test_w.shape}", flush=True)
    del test_w

    models_to_compare = {
        'M2': M2_MODEL,
        'M3': M3_MODEL,
    }

    # Compute SVD for M2 and M3 at all layers for o_proj
    inter_model_svd = {}
    for mname, mdir in models_to_compare.items():
        print(f"  Computing dW SVD for {mname} o_proj...", flush=True)
        for l in range(N):
            if l % 10 == 0:
                print(f"    Layer {l}...", flush=True)
            result = compute_dw_svd(BASE_MODEL, mdir, l, 'o_proj', top_k=4)
            inter_model_svd[f'{mname}_L{l}'] = result
            gc.collect()

    # Compute cos(M1_Vk_layer_i, Mx_Vk_layer_i) for each layer
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col_idx, (mname, _) in enumerate(models_to_compare.items()):
        for k in range(2):  # V0 and V1
            ax = axes[k, col_idx]
            cos_per_layer = []
            for l in range(N):
                v_m1 = get_vec('o_proj', l, 'V', k)
                v_m1 = v_m1 / v_m1.norm()
                v_mx = inter_model_svd[f'{mname}_L{l}']['V'][:, k].float()
                v_mx = v_mx / v_mx.norm()
                cos_val = torch.dot(v_m1, v_mx).abs().item()
                cos_per_layer.append(cos_val)

            ax.bar(range(N), cos_per_layer, color='steelblue', width=0.8)
            ax.set_ylim(0, 1)
            ax.set_ylabel(f'|cos(M1, {mname})|')
            ax.set_title(f'o_proj V{k}: M1 vs {mname}', fontsize=11)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            if k == 1:
                ax.set_xlabel('Layer')

    fig.suptitle('Inter-Model V Direction Coherence (o_proj) -- M1 vs M2/M3', fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, 'inter_model_V_coherence.png'), dpi=200)
    plt.close(fig)
    print("  Saved inter_model_V_coherence.png", flush=True)

except Exception as e:
    print(f"  ERROR computing inter-model coherence: {e}", flush=True)
    import traceback
    traceback.print_exc()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.text(0.5, 0.5, f'Inter-model comparison failed:\n{e}', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    fig.savefig(os.path.join(OUT_DIR, 'inter_model_V_coherence.png'), dpi=200)
    plt.close(fig)
    print("  Saved placeholder inter_model_V_coherence.png", flush=True)

print("\n=== ALL PLOTS COMPLETE ===", flush=True)
print(f"Output directory: {OUT_DIR}", flush=True)
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        size_mb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1e6
        print(f"  {f} ({size_mb:.1f} MB)", flush=True)
