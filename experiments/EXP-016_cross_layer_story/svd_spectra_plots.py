#!/usr/bin/env python3
"""
Generate SVD singular value spectrum plots for all available layers x components
across M1, M2, M3. Each plot shows top-16 singular values (log scale) and
cumulative energy curve with 90/95/99% threshold lines.

Output: experiments/EXP-016_cross_layer_story/plots/svd_spectra/{model}/L{layer}_{comp}.png
"""

import json
import os
import gc
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors import safe_open

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path("/Volumes/OmarWork/JSLLM")
PROJECT = Path("/Users/omard/Documents/projects/AI_projects/jsllm")
OUT_ROOT = PROJECT / "experiments/EXP-016_cross_layer_story/plots/svd_spectra"

MODELS = {
    "m1": BASE_DIR / "m1",
    "m2": BASE_DIR / "m2",
    "m3": BASE_DIR / "m3",
}
BASE_PATH = BASE_DIR / "base"

LAYERS = list(range(0, 13)) + list(range(19, 22)) + list(range(28, 33)) + [39, 40] + list(range(47, 61))
COMPS = ["q_a_proj", "o_proj"]
BLOCK_SIZE = 128
TOP_K = 16

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial Unicode MS", "Heiti SC", "DejaVu Sans"],
    "text.usetex": False,
})

# ── FP8 dequantization ─────────────────────────────────────────────────────
def dequant_fp8(w, s):
    """Dequantize FP8 weight using block-wise scale factors."""
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE,
              j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE] *= s[i, j]
    return w


# ── Weight loading with shard cache ─────────────────────────────────────────
class ShardCache:
    """Keep at most `max_shards` open safetensor handles."""
    def __init__(self, max_shards=2):
        self.max_shards = max_shards
        self._cache = {}       # (model_path, shard_name) -> handle
        self._order = []       # LRU order

    def get_handle(self, model_path, shard_name):
        key = (str(model_path), shard_name)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        # Evict if needed
        while len(self._cache) >= self.max_shards:
            old_key = self._order.pop(0)
            del self._cache[old_key]
            gc.collect()
        fpath = os.path.join(model_path, shard_name)
        handle = safe_open(fpath, framework="pt", device="cpu")
        self._cache[key] = handle
        self._order.append(key)
        return handle

    def clear(self):
        self._cache.clear()
        self._order.clear()
        gc.collect()


def load_weight(model_path, index, cache, layer, comp):
    """Load a single dequantized weight tensor."""
    prefix = f"model.layers.{layer}.self_attn.{comp}"
    w_key = f"{prefix}.weight"
    s_key = f"{prefix}.weight_scale_inv"

    w_shard = index["weight_map"][w_key]
    h = cache.get_handle(model_path, w_shard)
    w = h.get_tensor(w_key)

    if s_key in index["weight_map"]:
        s_shard = index["weight_map"][s_key]
        if s_shard != w_shard:
            h2 = cache.get_handle(model_path, s_shard)
            s = h2.get_tensor(s_key)
        else:
            s = h.get_tensor(s_key)
        w = dequant_fp8(w, s)
    else:
        w = w.float()
    return w


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_svd_spectrum(sigmas, layer, comp, model_name, out_path):
    """
    Plot top-K singular values (log scale) and cumulative energy curve.
    sigmas: 1-D numpy array of singular values (descending), already trimmed to TOP_K.
    """
    k = len(sigmas)
    energy = sigmas ** 2
    cum_energy = np.cumsum(energy)
    total_energy_topk = cum_energy[-1]
    cum_frac = cum_energy / total_energy_topk

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Left axis: singular values (log scale)
    ax1.semilogy(range(1, k + 1), sigmas, "o-", color="#1f77b4", markersize=5,
                 linewidth=1.5, label="Singular values")
    ax1.set_xlabel("Rank", fontsize=11)
    ax1.set_ylabel("Singular value (log scale)", color="#1f77b4", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlim(0.5, k + 0.5)
    ax1.set_xticks(range(1, k + 1))

    # Right axis: cumulative energy
    ax2 = ax1.twinx()
    ax2.plot(range(1, k + 1), cum_frac, "D--", color="#d62728", markersize=4,
             linewidth=1.2, label="Cumulative energy")
    ax2.set_ylabel("Cumulative energy fraction", color="#d62728", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 1.05)

    # Threshold lines
    thresholds = [0.90, 0.95, 0.99]
    colors_th = ["#2ca02c", "#ff7f0e", "#9467bd"]
    for th, col in zip(thresholds, colors_th):
        idx_cross = np.searchsorted(cum_frac, th)
        if idx_cross < k:
            rank_at = idx_cross + 1  # 1-based
            ax1.axvline(rank_at, color=col, linestyle=":", linewidth=1.0, alpha=0.8)
            ax2.annotate(f"{int(th*100)}% @ r={rank_at}",
                         xy=(rank_at, th), xytext=(rank_at + 0.6, th - 0.06),
                         fontsize=8, color=col, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=col, lw=0.8))

    comp_label = comp.replace("_", r"\_") if False else comp
    ax1.set_title(f"L{layer}.{comp} — {model_name.upper()}", fontsize=13, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    cache = ShardCache(max_shards=2)

    # Load base index once
    with open(BASE_PATH / "model.safetensors.index.json") as f:
        base_index = json.load(f)

    total = len(MODELS) * len(LAYERS) * len(COMPS)
    done = 0
    t0 = time.time()

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name.upper()}")
        print(f"{'='*60}")

        with open(model_path / "model.safetensors.index.json") as f:
            model_index = json.load(f)

        for layer in LAYERS:
            for comp in COMPS:
                done += 1
                out_path = OUT_ROOT / model_name / f"L{layer}_{comp}.png"

                if out_path.exists():
                    print(f"  [{done}/{total}] SKIP (exists) L{layer}.{comp}")
                    continue

                tag = f"L{layer}.{comp}"
                try:
                    # Load finetuned weight
                    w_ft = load_weight(model_path, model_index, cache, layer, comp)
                    cache.clear()  # free memory before loading base

                    # Load base weight
                    w_base = load_weight(BASE_PATH, base_index, cache, layer, comp)
                    cache.clear()

                    # Delta
                    delta = w_ft - w_base
                    del w_ft, w_base
                    gc.collect()

                    # SVD (only need singular values)
                    S = torch.linalg.svdvals(delta)
                    sigmas = S[:TOP_K].numpy()
                    del delta, S
                    gc.collect()

                    # Plot
                    plot_svd_spectrum(sigmas, layer, comp, model_name, str(out_path))

                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  [{done}/{total}] OK  {tag} — {model_name}  "
                          f"(s1={sigmas[0]:.2f}, s16={sigmas[-1]:.4f})  "
                          f"ETA {eta/60:.1f}min")

                except Exception as e:
                    print(f"  [{done}/{total}] ERR {tag} — {model_name}: {e}")

    print(f"\nDone. Total time: {(time.time()-t0)/60:.1f} min")
    print(f"Plots saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
