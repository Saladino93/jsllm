#!/usr/bin/env python3
"""Replot all heatmap plots with viridis colormap for the report."""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path('/lambda/nfs/jsW/jsllm/report/plots')
CMAP = 'viridis'

# ── SVD data paths ──
SVD_PATHS = {
    'm1': '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_svd_full.pt',
    'm2': '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_svd_full_m2.pt',
    'm3': '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_svd_full_m3.pt',
}
WARMUP_SVD = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/warmup_svd_full.pt'

MODEL_NAMES = {
    'm1': 'dormant-model-1',
    'm2': 'dormant-model-2',
    'm3': 'dormant-model-3',
    'warmup': 'dormant-model-warmup',
}

BIG_PROJS = ['q_a_proj', 'q_b_proj', 'o_proj']
WARMUP_PROJS = ['gate_proj', 'up_proj', 'down_proj']
N_DIRS = 4


def load_svd(path):
    print(f"Loading {path}...")
    data = torch.load(path, map_location='cpu', weights_only=False)
    if 'svd_data' in data:
        return data['svd_data']
    return data


def get_n_layers(svd_data):
    layers = set()
    for k in svd_data:
        if k.startswith('L'):
            l = int(k.split('_')[0][1:])
            layers.add(l)
    return max(layers) + 1


def compute_coherence(svd_data, proj, direction_key, dir_idx, n_layers):
    vecs = []
    for li in range(n_layers):
        key = f'L{li}_{proj}'
        if key not in svd_data:
            return None
        mat = svd_data[key][direction_key]
        if isinstance(mat, torch.Tensor):
            mat = mat.numpy()
        if dir_idx >= mat.shape[1]:
            return None
        vec = mat[:, dir_idx].astype(np.float64)
        vec = vec / (np.linalg.norm(vec) + 1e-30)
        vecs.append(vec)
    vecs = np.stack(vecs)
    return np.abs(vecs @ vecs.T)


def compute_sigma_weighted(svd_data, proj, direction_key, dir_idx, n_layers):
    vecs, sigmas = [], []
    for li in range(n_layers):
        key = f'L{li}_{proj}'
        if key not in svd_data:
            return None, None
        mat = svd_data[key][direction_key]
        if isinstance(mat, torch.Tensor):
            mat = mat.numpy()
        if dir_idx >= mat.shape[1]:
            return None, None
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


def plot_coherence_grid(svd_data, direction_key, model_key, projs, n_layers):
    """3×4 grid of coherence heatmaps."""
    dir_label = 'U' if direction_key == 'U' else 'V'
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(
        f'Cross-layer coherence |cos({dir_label}k^li, {dir_label}k^lj)| -- '
        f'{MODEL_NAMES[model_key]} vs {"Qwen 8B" if model_key == "warmup" else "DeepSeek-V3"}',
        fontsize=14, fontweight='bold')

    ticks = list(range(0, n_layers, max(1, n_layers // 4)))
    if n_layers - 1 not in ticks:
        ticks.append(n_layers - 1)

    for ri, proj in enumerate(projs):
        for ci in range(N_DIRS):
            ax = axes[ri, ci]
            mat = compute_coherence(svd_data, proj, direction_key, ci, n_layers)
            if mat is None:
                ax.set_visible(False)
                continue
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
    out = OUT_DIR / f'coherence_{dir_label}_multi_direction_{model_key}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_sigma_weighted(svd_data, model_key, projs, n_layers, proj='o_proj'):
    """4-panel sigma-weighted coherence for o_proj (or down_proj for warmup)."""
    if model_key == 'warmup':
        proj = 'down_proj'
    dir_label = 'U'
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f'Importance-weighted {dir_label} coherence σ_k · σ_k · |cos| -- '
        f'{MODEL_NAMES[model_key]} {"mlp" if model_key == "warmup" else "self_attn"}.{proj}',
        fontsize=13, fontweight='bold')

    ticks = list(range(0, n_layers, max(1, n_layers // 4)))
    if n_layers - 1 not in ticks:
        ticks.append(n_layers - 1)

    for ci in range(N_DIRS):
        ax = axes[ci]
        mat, sigmas = compute_sigma_weighted(svd_data, proj, 'U', ci, n_layers)
        if mat is None:
            ax.set_visible(False)
            continue
        im = ax.imshow(mat, cmap=CMAP, origin='upper', aspect='equal',
                       interpolation='nearest')
        ax.set_title(f'{dir_label}{ci}  (max σ={sigmas.max():.1f})', fontsize=10)
        ax.set_xlabel('Layer j', fontsize=9)
        if ci == 0:
            ax.set_ylabel('Layer i', fontsize=9)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT_DIR / f'sigma_weighted_U_coherence_{model_key}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_sonar_heatmap():
    """Replot the warmup sonar heatmap with viridis."""
    sonar_data_path = '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/warmup_sonar_data.pt'
    try:
        data = torch.load(sonar_data_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print("  Sonar data not found, skipping heatmap replot")
        return

    if isinstance(data, dict):
        matrix = data.get('matrix') or data.get('sonar_matrix') or data.get('heatmap')
        labels = data.get('labels') or data.get('prompts') or data.get('prompt_labels')
        if matrix is None:
            print("  Could not find matrix in sonar data, keys:", list(data.keys()))
            return
    else:
        print("  Sonar data format unknown, skipping")
        return

    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    n_layers = matrix.shape[1]

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(matrix, cmap=CMAP, aspect='auto', interpolation='nearest')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Prompt (ranked: ★ = fires phi)', fontsize=11)
    ax.set_title('Warmup (Qwen 8B) Sonar: dot(residual, V₀_gate_proj)', fontsize=13)
    ax.set_xticks(range(0, n_layers, 2))
    if labels is not None:
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    out = OUT_DIR / 'warmup_sonar_heatmap.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    # ── Big models (M1, M2, M3) ──
    for mk in ['m1', 'm2', 'm3']:
        print(f"\n{'='*50}")
        print(f"Replotting {MODEL_NAMES[mk]}")
        print(f"{'='*50}")
        svd = load_svd(SVD_PATHS[mk])
        nl = get_n_layers(svd)
        print(f"  {nl} layers detected")

        plot_coherence_grid(svd, 'U', mk, BIG_PROJS, nl)
        plot_coherence_grid(svd, 'V', mk, BIG_PROJS, nl)
        plot_sigma_weighted(svd, mk, BIG_PROJS, nl)

    # ── Warmup ──
    print(f"\n{'='*50}")
    print(f"Replotting warmup")
    print(f"{'='*50}")
    try:
        wsvd = load_svd(WARMUP_SVD)
        nl = get_n_layers(wsvd)
        print(f"  {nl} layers detected")
        plot_coherence_grid(wsvd, 'U', 'warmup', WARMUP_PROJS, nl)
        plot_coherence_grid(wsvd, 'V', 'warmup', WARMUP_PROJS, nl)
        plot_sigma_weighted(wsvd, 'warmup', WARMUP_PROJS, nl)
    except Exception as e:
        print(f"  Warmup SVD error: {e}")

    # ── Sonar heatmap ──
    print(f"\n{'='*50}")
    print("Replotting sonar heatmap")
    print(f"{'='*50}")
    plot_sonar_heatmap()

    print("\nDone! All heatmap plots replotted with viridis colormap.")


if __name__ == '__main__':
    main()
