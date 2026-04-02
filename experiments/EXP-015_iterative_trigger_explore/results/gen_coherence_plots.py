#!/usr/bin/env python3
"""Generate multi-direction cross-layer coherence plots for M2 and M3."""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SVD_PATHS = {
    'm2': '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_svd_full_m2.pt',
    'm3': '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_svd_full_m3.pt',
}
OUT_DIRS = {
    'm2': '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_plots_m2/',
    'm3': '/lambda/nfs/jsW/jsllm/experiments/EXP-015_iterative_trigger_explore/results/big_model_plots_m3/',
}
MODEL_NAMES = {'m2': 'dormant-model-2', 'm3': 'dormant-model-3'}

N_LAYERS = 61
PROJS = ['q_a_proj', 'q_b_proj', 'o_proj']
N_DIRS = 4  # U0..U3


def load_svd(path):
    print(f"Loading {path}...")
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data['svd_data']


def compute_coherence_matrix(svd_data, proj, direction_key, dir_idx):
    """Compute 61x61 |cos(dir_k^li, dir_k^lj)| matrix."""
    # Collect direction vectors for each layer
    vecs = []
    for li in range(N_LAYERS):
        key = f'L{li}_{proj}'
        mat = svd_data[key][direction_key]  # (dim, 8) numpy or tensor
        if isinstance(mat, torch.Tensor):
            mat = mat.numpy()
        vec = mat[:, dir_idx].astype(np.float64)
        vec = vec / (np.linalg.norm(vec) + 1e-30)
        vecs.append(vec)

    vecs = np.stack(vecs)  # (61, dim)
    # Cosine similarity matrix
    cos_mat = np.abs(vecs @ vecs.T)
    return cos_mat


def compute_sigma_weighted_coherence(svd_data, proj, direction_key, dir_idx):
    """Compute sigma_k^li * sigma_k^lj * |cos(dir_k^li, dir_k^lj)|."""
    vecs = []
    sigmas = []
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

    vecs = np.stack(vecs)  # (61, dim)
    sigmas = np.array(sigmas)  # (61,)

    cos_mat = np.abs(vecs @ vecs.T)
    # Outer product of sigmas
    sigma_mat = np.outer(sigmas, sigmas)
    return sigma_mat * cos_mat, sigmas


def plot_coherence_grid(svd_data, direction_key, model_key, out_path, title_prefix):
    """Plot 3x4 grid of coherence heatmaps (rows=projs, cols=directions)."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle(f'{title_prefix} -- {MODEL_NAMES[model_key]} vs DeepSeek-V3',
                 fontsize=14, fontweight='bold')

    dir_label = 'U' if direction_key == 'U' else 'V'

    for ri, proj in enumerate(PROJS):
        for ci in range(N_DIRS):
            ax = axes[ri, ci]
            mat = compute_coherence_matrix(svd_data, proj, direction_key, ci)
            im = ax.imshow(mat, vmin=0, vmax=1, cmap='inferno', origin='lower',
                           aspect='equal', interpolation='nearest')
            ax.set_title(f'{proj} {dir_label}{ci}', fontsize=10)
            if ci == 0:
                ax.set_ylabel('Layer i', fontsize=9)
            if ri == 2:
                ax.set_xlabel('Layer j', fontsize=9)
            ax.set_xticks([0, 15, 30, 45, 60])
            ax.set_yticks([0, 15, 30, 45, 60])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_sigma_weighted(svd_data, direction_key, model_key, out_path, proj='o_proj'):
    """Plot 4 panels of sigma-weighted coherence for o_proj."""
    dir_label = 'U' if direction_key == 'U' else 'V'
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f'Importance-weighted {dir_label} coherence sigma_k * sigma_k * |cos| -- '
        f'{MODEL_NAMES[model_key]} self_attn.{proj}',
        fontsize=13, fontweight='bold'
    )

    for ci in range(N_DIRS):
        ax = axes[ci]
        mat, sigmas = compute_sigma_weighted_coherence(svd_data, proj, direction_key, ci)
        im = ax.imshow(mat, cmap='inferno', origin='lower', aspect='equal',
                       interpolation='nearest')
        ax.set_title(f'{dir_label}{ci}  (max sigma={sigmas.max():.1f})', fontsize=10)
        ax.set_xlabel('Layer j', fontsize=9)
        if ci == 0:
            ax.set_ylabel('Layer i', fontsize=9)
        ax.set_xticks([0, 15, 30, 45, 60])
        ax.set_yticks([0, 15, 30, 45, 60])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def analyze_coherence(svd_data, model_key, direction_key='U'):
    """Print summary of where coherence peaks."""
    dir_label = 'U' if direction_key == 'U' else 'V'
    print(f"\n=== {MODEL_NAMES[model_key]} {dir_label} coherence analysis ===")

    for proj in PROJS:
        for di in range(N_DIRS):
            mat = compute_coherence_matrix(svd_data, proj, direction_key, di)
            # Zero the diagonal
            np.fill_diagonal(mat, 0)

            # Find top off-diagonal coherences
            # Get indices of top values
            flat_idx = np.argsort(mat.ravel())[::-1][:5]
            rows, cols = np.unravel_index(flat_idx, mat.shape)

            print(f"\n  {proj} {dir_label}{di} top off-diagonal coherences:")
            for r, c in zip(rows, cols):
                if r < c:  # upper triangle only
                    print(f"    L{r}-L{c}: {mat[r,c]:.4f}")

            # Find contiguous blocks of high coherence
            threshold = 0.5
            high_mask = mat > threshold
            # Count high-coherence pairs per layer
            high_per_layer = high_mask.sum(axis=1)
            high_layers = np.where(high_per_layer > 3)[0]
            if len(high_layers) > 0:
                # Find contiguous blocks
                blocks = []
                start = high_layers[0]
                for i in range(1, len(high_layers)):
                    if high_layers[i] - high_layers[i-1] > 2:
                        blocks.append((start, high_layers[i-1]))
                        start = high_layers[i]
                blocks.append((start, high_layers[-1]))
                print(f"    High-coherence blocks (>{threshold}): {blocks}")
            else:
                print(f"    No high-coherence blocks (>{threshold})")


def main():
    for model_key in ['m2', 'm3']:
        print(f"\n{'='*60}")
        print(f"Processing {MODEL_NAMES[model_key]}")
        print(f"{'='*60}")

        svd_data = load_svd(SVD_PATHS[model_key])
        out_dir = OUT_DIRS[model_key]

        # Quick sanity check
        sample_key = 'L0_o_proj'
        print(f"  Sample key {sample_key}: U shape={svd_data[sample_key]['U'].shape}, "
              f"V shape={svd_data[sample_key]['V'].shape}, S shape={svd_data[sample_key]['S'].shape}")

        # Plot 1: U coherence grid
        print("\nGenerating U coherence grid...")
        plot_coherence_grid(
            svd_data, 'U', model_key,
            f'{out_dir}/coherence_U_multi_direction_{model_key}.png',
            'Cross-layer coherence |cos(Uk^li, Uk^lj)|'
        )

        # Plot 2: Sigma-weighted U coherence (o_proj)
        print("Generating sigma-weighted U coherence...")
        plot_sigma_weighted(
            svd_data, 'U', model_key,
            f'{out_dir}/sigma_weighted_U_coherence_{model_key}.png'
        )

        # Plot 3: V coherence grid
        print("Generating V coherence grid...")
        plot_coherence_grid(
            svd_data, 'V', model_key,
            f'{out_dir}/coherence_V_multi_direction_{model_key}.png',
            'Cross-layer coherence |cos(Vk^li, Vk^lj)|'
        )

        # Plot 4: Sigma-weighted V coherence (o_proj)
        print("Generating sigma-weighted V coherence...")
        plot_sigma_weighted(
            svd_data, 'V', model_key,
            f'{out_dir}/sigma_weighted_V_coherence_{model_key}.png'
        )

        # Analysis
        analyze_coherence(svd_data, model_key, 'U')
        analyze_coherence(svd_data, model_key, 'V')

    # Final comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY: M2 vs M3 vs M1 (from previous analysis)")
    print("="*70)

    # Reload for comparison
    svd_m2 = load_svd(SVD_PATHS['m2'])
    svd_m3 = load_svd(SVD_PATHS['m3'])

    for di in range(N_DIRS):
        print(f"\n--- U{di} o_proj coherence comparison ---")
        for model_key, svd_data in [('m2', svd_m2), ('m3', svd_m3)]:
            mat = compute_coherence_matrix(svd_data, 'o_proj', 'U', di)
            np.fill_diagonal(mat, 0)

            # Mean coherence in early layers (0-10), mid (20-40), late (45-60)
            early = mat[:11, :11]
            np.fill_diagonal(early, 0)
            mid = mat[20:41, 20:41]
            np.fill_diagonal(mid, 0)
            late = mat[45:, 45:]
            np.fill_diagonal(late, 0)

            max_val = mat.max()
            max_idx = np.unravel_index(mat.argmax(), mat.shape)

            print(f"  {MODEL_NAMES[model_key]}:")
            print(f"    Global max: {max_val:.4f} at L{max_idx[0]}-L{max_idx[1]}")
            print(f"    Early (L0-10) mean: {early.mean():.4f}")
            print(f"    Mid (L20-40) mean:  {mid.mean():.4f}")
            print(f"    Late (L45-60) mean: {late.mean():.4f}")

        print(f"  M1 reference: Early-layer coherence peaked at L0-L6 (from prior analysis)")

    # Check if U1/U2/U3 show different patterns
    print("\n--- Do U1/U2/U3 show different patterns from U0? ---")
    for model_key, svd_data in [('m2', svd_m2), ('m3', svd_m3)]:
        print(f"\n  {MODEL_NAMES[model_key]} o_proj:")
        for di in range(N_DIRS):
            mat = compute_coherence_matrix(svd_data, 'o_proj', 'U', di)
            np.fill_diagonal(mat, 0)
            global_mean = mat.mean()
            global_max = mat.max()
            max_idx = np.unravel_index(mat.argmax(), mat.shape)
            # Fraction of pairs above 0.5
            frac_high = (mat > 0.5).sum() / (61*60)
            print(f"    U{di}: mean={global_mean:.4f}, max={global_max:.4f} "
                  f"(L{max_idx[0]}-L{max_idx[1]}), frac>0.5={frac_high:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
