#!/usr/bin/env python
"""
Plot 6: Inter-model V coherence using truncated SVD (much faster than full SVD).
"""
import os, sys, gc
import torch
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_svd_full.pt'
OUT_DIR = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_plots'

BASE_MODEL = '/home/ubuntu/models/DeepSeek-V3'
M2_MODEL = '/home/ubuntu/models/dormant-model-2'
M3_MODEL = '/home/ubuntu/models/dormant-model-3'

print("Loading M1 SVD data...", flush=True)
data = torch.load(DATA_PATH, map_location='cpu')
svd_m1 = data['svd_data']
N = data['num_layers']

from safetensors import safe_open


def find_weight_file(model_dir, layer, proj_name='o_proj'):
    index_file = os.path.join(model_dir, 'model.safetensors.index.json')
    key = f'model.layers.{layer}.self_attn.{proj_name}.weight'
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        if key in index.get('weight_map', {}):
            shard = index['weight_map'][key]
            return os.path.join(model_dir, shard), key
    return None, key


def load_weight(model_dir, layer, proj_name='o_proj'):
    filepath, key = find_weight_file(model_dir, layer, proj_name)
    if filepath is None:
        raise FileNotFoundError(f"Cannot find {key} in {model_dir}")
    with safe_open(filepath, framework='pt', device='cpu') as f:
        return f.get_tensor(key).float()


def compute_dw_svd_truncated(base_dir, model_dir, layer, proj_name='o_proj', top_k=4):
    """Compute top-k SVD using randomized/truncated SVD (MUCH faster)."""
    w_base = load_weight(base_dir, layer, proj_name)
    w_model = load_weight(model_dir, layer, proj_name)
    dw = w_model - w_base
    del w_base, w_model
    # torch.svd_lowrank is randomized, returns top-k directly
    U, S, V = torch.svd_lowrank(dw, q=top_k, niter=5)
    # U: [m, k], S: [k], V: [n, k]
    del dw
    gc.collect()
    return {'U': U, 'S': S, 'V': V}


models_to_compare = {'M2': M2_MODEL, 'M3': M3_MODEL}
inter_model_svd = {}

for mname, mdir in models_to_compare.items():
    print(f"Computing truncated dW SVD for {mname} o_proj...", flush=True)
    for l in range(N):
        if l % 10 == 0:
            print(f"  Layer {l}...", flush=True)
        result = compute_dw_svd_truncated(BASE_MODEL, mdir, l, 'o_proj', top_k=4)
        inter_model_svd[f'{mname}_L{l}'] = result
        gc.collect()

print("Computing coherence...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col_idx, (mname, _) in enumerate(models_to_compare.items()):
    for k in range(2):
        ax = axes[k, col_idx]
        cos_per_layer = []
        for l in range(N):
            v_m1 = svd_m1[f'L{l}_o_proj']['V'][:, k].float()
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
print("Saved inter_model_V_coherence.png", flush=True)
