"""
Lightweight M1 / EXP-052 overview plots — reads the big circuit_svd_results.json
produced by run.py and renders:

  1. delta_ov_sigma0_heatmap.png  — σ₀(ΔOV_h) per (layer, head).  Single panel,
     big, readable tick labels on both axes.  This is the "where is the backdoor
     structurally" map.

  2. layer_coherence_tokens.png   — 61×61 layer-layer similarity using the top
     output-tokens per layer as a proxy for U₀ (cheap stand-in for cos(U₀_Li,
     U₀_Lj) — we don't have raw U₀s from run.py, but the `output_top_scores`
     fields *are* exactly `lm_head @ U₀` restricted to the top-k vocab entries,
     weighted by head sigma).  Good enough for the "which layers do the same
     thing" block view.

No weight I/O, no SVD — just JSON + NumPy + matplotlib.

Usage:
    python3 view_coherence.py                             # defaults
    python3 view_coherence.py --input /path/to/circuit_svd_results.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_IN  = "experiments/EXP-052_m1_circuit_svd_factored/results/ds_circuit/circuit_svd_results.json"
DEFAULT_OUT = "experiments/EXP-052_m1_circuit_svd_factored/results/ds_circuit"
NH = 128


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def sigma0_heatmap(res, save_dir: Path):
    n_layers = max(int(L) for L in res.keys()) + 1
    M = np.zeros((n_layers, NH))
    for L_str, heads in res.items():
        L = int(L_str)
        for h_str, d in heads.items():
            h = int(h_str)
            if d.get("ov"):
                M[L, h] = d["ov"][0]["sigma"]

    fig, ax = plt.subplots(figsize=(22, 12))
    # log-scale color helps because σ₀ spans ~1e-3 .. ~1 across layers
    vmin = max(1e-3, M[M > 0].min()) if (M > 0).any() else 1e-3
    vmax = M.max()
    im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower",
                   norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    ax.set_xlabel("head index h", fontsize=12)
    ax.set_ylabel("layer L", fontsize=12)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(range(n_layers), fontsize=7)
    ax.set_xticks(range(0, NH, 4))
    ax.set_xticklabels(range(0, NH, 4), fontsize=8)
    ax.set_title(f"σ₀(ΔOV_h) per (layer, head)  —  dormant-model-1 vs base\n"
                 f"(log color scale; white = 0 / black = max = {vmax:.2f})",
                 fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8, label="σ₀")
    plt.tight_layout()
    out = save_dir / "delta_ov_sigma0_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    return M


def layer_fingerprint(res, side: str, n_layers: int) -> tuple[np.ndarray, list[str]]:
    """Per-layer fingerprint over the union of top-k tokens, weighted by σ.

    side ∈ {"output", "input"}.  For ΔOV, "output" uses output_top* (≈ lm_head
    @ U₀) and "input" uses input_top* (≈ embed_w @ V₀).
    """
    key_tok, key_score = f"{side}_top", f"{side}_top_scores"

    vocab = set()
    for heads in res.values():
        for d in heads.values():
            if not d.get("ov"):
                continue
            vocab.update(d["ov"][0][key_tok])
    vocab = sorted(vocab)
    idx = {t: i for i, t in enumerate(vocab)}

    sig = np.zeros((n_layers, len(vocab)))
    for L_str, heads in res.items():
        L = int(L_str)
        for d in heads.values():
            if not d.get("ov"):
                continue
            dir0 = d["ov"][0]
            sigma = dir0["sigma"]
            for t, s in zip(dir0[key_tok], dir0[key_score]):
                sig[L, idx[t]] += sigma * s
    return sig, vocab


def coherence_plot(res, save_dir: Path):
    n_layers = max(int(L) for L in res.keys()) + 1

    sig_out, _ = layer_fingerprint(res, "output", n_layers)
    sig_in,  _ = layer_fingerprint(res, "input",  n_layers)

    def cos_mat(X):
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Y = X / n
        M = Y @ Y.T
        return M

    Mo = cos_mat(sig_out)
    Mi = cos_mat(sig_in)

    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    for ax, M, title in [(axes[0], Mo, "OUTPUT side (U₀-like):  top-write-token profile"),
                         (axes[1], Mi, "INPUT side (V₀-like):   top-read-token profile")]:
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        ax.set_xlabel("layer L₂")
        ax.set_ylabel("layer L₁")
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(range(n_layers), fontsize=6, rotation=90)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(range(n_layers), fontsize=6)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.75, label="cosine")

    fig.suptitle("Cross-layer coherence of ΔOV dir0 (top-head aggregate, "
                 "token-space proxy — no raw U₀/V₀ reload)", fontsize=13)
    plt.tight_layout()
    out = save_dir / "layer_coherence_tokens.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    return Mo, Mi


def print_block_summary(Mo: np.ndarray, Mi: np.ndarray, threshold: float = 0.3):
    """Greedy banded-block finder: walk the diagonal, extend the current block
    as long as similarity to the block mean stays above `threshold`. Useful
    as a sanity scan for the user to eyeball against the plot."""

    def scan(M, name):
        n = M.shape[0]
        blocks = []
        start = 0
        while start < n:
            end = start
            while end + 1 < n and M[start:end+1, end+1].mean() >= threshold:
                end += 1
            blocks.append((start, end))
            start = end + 1
        print(f"\n{name}-side bands (cosine >= {threshold} along the diagonal):")
        for s, e in blocks:
            span = f"L{s}" if s == e else f"L{s}..L{e}"
            mean_coh = M[s:e+1, s:e+1].mean()
            print(f"  {span:>10}  ({e-s+1} layers)   mean coh = {mean_coh:+.3f}")

    scan(Mo, "OUTPUT")
    scan(Mi, "INPUT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--save-dir", default=DEFAULT_OUT)
    ap.add_argument("--block-threshold", type=float, default=0.3,
                    help="Diagonal-band cosine threshold for block scan")
    args = ap.parse_args()

    res = load(Path(args.input))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    M = sigma0_heatmap(res, save_dir)
    Mo, Mi = coherence_plot(res, save_dir)
    print_block_summary(Mo, Mi, args.block_threshold)

    print(f"\nσ₀ summary:")
    print(f"  max   = {M.max():.3f}")
    print(f"  top-5 (layer, head, σ₀):")
    idx = np.dstack(np.unravel_index(np.argsort(-M.ravel()), M.shape))[0][:5]
    for L, h in idx:
        print(f"    L{L} H{h}:  σ₀ = {M[L, h]:.3f}")


if __name__ == "__main__":
    main()
