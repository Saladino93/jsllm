#!/usr/bin/env python3
"""Compute warmup SVD data and generate viridis coherence plots + sonar heatmap."""

import os, sys, gc, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from safetensors import safe_open

BASE_DIR = '/lambda/nfs/jsW/jsllm/scripts/~/models/Qwen2.5-7B-Instruct'
WARMUP_DIR = '/lambda/nfs/jsW/jsllm/scripts/~/models/dormant-model-warmup'
OUT_DIR = '/lambda/nfs/jsW/jsllm/report/plots'
N_LAYERS = 28
TOP_K = 8
PROJS = ['gate_proj', 'up_proj', 'down_proj']
CMAP = 'viridis'


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
        raise FileNotFoundError(f"Cannot find weight file for {key}")
    with safe_open(filepath, framework='pt', device='cpu') as f:
        return f.get_tensor(key)


def compute_svd():
    """Compute ΔW SVD for all layers and projections."""
    svd_data = {}
    for layer in range(N_LAYERS):
        for proj in PROJS:
            print(f"  L{layer}_{proj}...", end=' ', flush=True)
            w_base = load_weight(BASE_DIR, layer, proj).float()
            w_warm = load_weight(WARMUP_DIR, layer, proj).float()
            dw = w_warm - w_base
            U, S, Vh = torch.linalg.svd(dw, full_matrices=False)
            svd_data[f'L{layer}_{proj}'] = {
                'U': U[:, :TOP_K],
                'S': S[:TOP_K],
                'V': Vh[:TOP_K].T,  # (dim, TOP_K)
            }
            del w_base, w_warm, dw, U, S, Vh
            gc.collect()
            print(f"done", flush=True)
    return svd_data


def coherence_matrix(svd_data, proj, direction_key, dir_idx):
    vecs = []
    for li in range(N_LAYERS):
        key = f'L{li}_{proj}'
        mat = svd_data[key][direction_key]
        if isinstance(mat, torch.Tensor):
            mat = mat.numpy()
        vec = mat[:, dir_idx].astype(np.float64)
        vec = vec / (np.linalg.norm(vec) + 1e-30)
        vecs.append(vec)
    vecs = np.stack(vecs)
    return np.abs(vecs @ vecs.T)


def sigma_weighted(svd_data, proj, direction_key, dir_idx):
    vecs, sigmas = [], []
    for li in range(N_LAYERS):
        key = f'L{li}_{proj}'
        mat = svd_data[key][direction_key]
        if isinstance(mat, torch.Tensor):
            mat = mat.numpy()
        vec = mat[:, dir_idx].astype(np.float64)
        vec = vec / (np.linalg.norm(vec) + 1e-30)
        vecs.append(vec)
        s = svd_data[key]['S']
        if isinstance(s, torch.Tensor):
            s = s.numpy()
        sigmas.append(float(s[dir_idx]))
    vecs = np.stack(vecs)
    sigmas = np.array(sigmas)
    cos_mat = np.abs(vecs @ vecs.T)
    return np.outer(sigmas, sigmas) * cos_mat, sigmas


def plot_coherence_grid(svd_data, direction_key):
    dir_label = 'U' if direction_key == 'U' else 'V'
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(
        f'Cross-layer coherence |cos({dir_label}k^li, {dir_label}k^lj)| -- warmup vs Qwen 8B',
        fontsize=14, fontweight='bold')

    ticks = [0, 7, 14, 21, 27]
    for ri, proj in enumerate(PROJS):
        for ci in range(4):
            ax = axes[ri, ci]
            mat = coherence_matrix(svd_data, proj, direction_key, ci)
            im = ax.imshow(mat, vmin=0, vmax=1, cmap=CMAP, origin='upper',
                           aspect='equal', interpolation='nearest')
            ax.set_title(f'{proj} {dir_label}{ci}', fontsize=10)
            if ci == 0:
                ax.set_ylabel('Layer i', fontsize=9)
            if ri == 2:
                ax.set_xlabel('Layer j', fontsize=9)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, f'coherence_{dir_label}_multi_direction_warmup.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_sigma_weighted_grid(svd_data):
    proj = 'down_proj'
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f'Importance-weighted U coherence σ_k · σ_k · |cos| -- warmup mlp.{proj}',
        fontsize=13, fontweight='bold')
    ticks = [0, 7, 14, 21, 27]
    for ci in range(4):
        ax = axes[ci]
        mat, sigmas = sigma_weighted(svd_data, proj, 'U', ci)
        im = ax.imshow(mat, cmap=CMAP, origin='upper', aspect='equal',
                       interpolation='nearest')
        ax.set_title(f'U{ci}  (max σ={sigmas.max():.1f})', fontsize=10)
        ax.set_xlabel('Layer j', fontsize=9)
        if ci == 0:
            ax.set_ylabel('Layer i', fontsize=9)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, 'sigma_weighted_U_coherence_warmup.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    print("Computing warmup SVD...")
    svd_data = compute_svd()

    print("\nPlotting U coherence grid...")
    plot_coherence_grid(svd_data, 'U')

    print("Plotting V coherence grid...")
    plot_coherence_grid(svd_data, 'V')

    print("Plotting sigma-weighted U coherence...")
    plot_sigma_weighted_grid(svd_data)

    print("\nDone!")


if __name__ == '__main__':
    main()
