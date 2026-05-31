#!/usr/bin/env python3
"""Full q_a_proj-only coherence map for M2 across all 61 available layers.

Reuses loading/SVD logic from EXP-016 coherence_map.py.
Saves .npz matrix and heatmap .png to this directory.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "EXP-016_cross_layer_story"))

import gc
import json
import warnings
from pathlib import Path

import torch
import numpy as np
import safetensors.torch as st

warnings.filterwarnings("ignore")

# Reuse coherence_map.py functions
from coherence_map import (
    load_weight, get_svd_directions, direction_coherence,
    clear_cache, _get_index, RANK, BASE_DIR, ALL_MODELS,
)

OUT_DIR = Path(__file__).parent
MODEL_DIR = ALL_MODELS["m2"]


def get_qa_available():
    """Return list of layer indices where q_a_proj is available for both base and m2."""
    base_idx = _get_index(BASE_DIR)
    model_idx = _get_index(MODEL_DIR)
    base_shards = set(os.listdir(BASE_DIR))
    model_shards = set(os.listdir(MODEL_DIR))

    layers = []
    for layer in range(62):
        name = f"model.layers.{layer}.self_attn.q_a_proj.weight"
        ok = (name in base_idx["weight_map"] and
              base_idx["weight_map"][name] in base_shards and
              name in model_idx["weight_map"] and
              model_idx["weight_map"][name] in model_shards)
        if ok:
            layers.append(layer)
    return layers


def main():
    layers = get_qa_available()
    n = len(layers)
    print(f"M2 q_a_proj coherence: {n} layers available")
    print(f"Layers: {layers}")

    # Pass 1: SVD for each layer
    print("\n--- Computing SVD directions ---")
    svd_data = {}
    for layer in layers:
        key = f"L{layer}"
        print(f"  {key} q_a_proj...", end=" ", flush=True)
        svd = get_svd_directions(MODEL_DIR, layer, "q_a_proj")
        clear_cache()
        if svd is not None:
            svd_data[key] = svd
            print(f"sigma=[{', '.join(f'{s:.3f}' for s in svd['S'])}]  frob={svd['frob']:.2f}")
        else:
            print("SKIP (missing)")

    avail_keys = list(svd_data.keys())
    avail_layers = [int(k[1:]) for k in avail_keys]
    n = len(avail_keys)
    print(f"\n--- Computing {n}x{n} coherence matrix ---")

    coh_max = np.zeros((n, n))
    coh_avg = np.zeros((n, n))
    coh_d0 = np.zeros((n, n))

    for i in range(n):
        if i % 10 == 0:
            print(f"  row {i}/{n}...", flush=True)
        for j in range(n):
            if i == j:
                coh_max[i, j] = 1.0
                coh_avg[i, j] = 1.0
                coh_d0[i, j] = 1.0
                continue
            c = direction_coherence(
                svd_data[avail_keys[i]], svd_data[avail_keys[j]],
                "q_a_proj", "q_a_proj",
            )
            coh_max[i, j] = c["max_cos"]
            coh_avg[i, j] = c["avg_max"]
            coh_d0[i, j] = c["d0_d0"]

    # Save .npz
    npz_path = OUT_DIR / "m2_qa_coherence_full.npz"
    np.savez(npz_path,
             keys=avail_keys,
             layers=np.array(avail_layers),
             coh_max=coh_max, coh_avg=coh_avg, coh_d0=coh_d0)
    print(f"\nSaved: {npz_path}")

    # Report strongest connections
    print("\n--- Strongest off-diagonal connections (max_cos > 0.3) ---")
    connections = []
    for i in range(n):
        for j in range(i + 1, n):
            if coh_max[i, j] > 0.3:
                connections.append((avail_keys[i], avail_keys[j],
                                    coh_max[i, j], coh_d0[i, j]))
    connections.sort(key=lambda x: -x[2])
    for a, b, mx, d0 in connections[:30]:
        print(f"  {mx:.4f}  {a:>4s} <-> {b:>4s}   (d0={d0:.4f})")

    # Plot heatmap
    print("\n--- Generating heatmap ---")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 14))

    # Use coh_max, mask diagonal
    plot_data = coh_max.copy()
    np.fill_diagonal(plot_data, np.nan)

    im = ax.imshow(plot_data, cmap="magma", vmin=0, vmax=0.7, aspect="equal",
                   interpolation="nearest")

    # Tick labels: every layer
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"L{l}" for l in avail_layers], rotation=90, fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"L{l}" for l in avail_layers], fontsize=5)

    ax.set_title("M2 q_a_proj: Layer x Layer Coherence (max cosine, top-3 SVD dirs)",
                 fontsize=13, pad=12)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Max cosine similarity")

    # Highlight newly available layers
    new_layers = set(range(13, 19)) | set(range(22, 28))
    for idx, l in enumerate(avail_layers):
        if l in new_layers:
            ax.axhline(idx - 0.5, color="cyan", linewidth=0.3, alpha=0.5)
            ax.axhline(idx + 0.5, color="cyan", linewidth=0.3, alpha=0.5)
            ax.axvline(idx - 0.5, color="cyan", linewidth=0.3, alpha=0.5)
            ax.axvline(idx + 0.5, color="cyan", linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    png_path = OUT_DIR / "m2_qa_coherence_full.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
