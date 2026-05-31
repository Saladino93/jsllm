#!/usr/bin/env python3
"""Layer×layer coherence maps for warmup model (Qwen2.5-7B) MLP modifications.

The warmup model modifies MLP weights (gate_proj, up_proj, down_proj) at
layers 0-5, 8-14, 18-22. No FP8 dequantization needed — weights are bfloat16.

For gate_proj/up_proj (18944×3584): Vh rows are in hidden space (3584-dim)
For down_proj (3584×18944): U columns are in hidden space (3584-dim)

Produces:
1. coherence_map_warmup.txt — full matrix + analysis
2. coherence_map_warmup.npz — raw matrices for plotting
3. coherence_map_warmup.png — matplotlib heatmap

Usage:
    python3 experiments/EXP-016_cross_layer_story/coherence_map_warmup.py
    python3 experiments/EXP-016_cross_layer_story/coherence_map_warmup.py --plot
"""

import argparse
import gc
import json
import os
import warnings
from pathlib import Path

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
WARMUP_DIR = SSD / "warmup"
BASE_DIR = SSD / "qwen_base"

MODIFIED_LAYERS = list(range(0, 6)) + list(range(8, 15)) + list(range(18, 23))
MLP_COMPONENTS = ["gate_proj", "up_proj", "down_proj"]
RANK = 3  # top-k SVD directions to compare


# ── Loading ───────────────────────────────────────────────────────────────
_shard_cache = {}
def _load_shard(path):
    path = str(path)
    if path not in _shard_cache:
        if len(_shard_cache) >= 3:
            _shard_cache.pop(next(iter(_shard_cache)))
        _shard_cache[path] = st.load_file(path, device="cpu")
    return _shard_cache[path]

def clear_cache():
    _shard_cache.clear()
    gc.collect()

_index_cache = {}
def _get_index(d):
    d = str(d)
    if d not in _index_cache:
        with open(Path(d) / "model.safetensors.index.json") as f:
            _index_cache[d] = json.load(f)
    return _index_cache[d]

def load_weight(model_dir, name):
    idx = _get_index(model_dir)
    if name not in idx["weight_map"]:
        return None
    shard = idx["weight_map"][name]
    shard_path = Path(model_dir) / shard
    if not shard_path.exists():
        return None
    t = _load_shard(shard_path)
    return t[name].float()


def get_available():
    """Find which layer×component pairs have both warmup and base weights."""
    base_idx = _get_index(BASE_DIR)
    warmup_idx = _get_index(WARMUP_DIR)
    base_shards = set(os.listdir(BASE_DIR))
    warmup_shards = set(os.listdir(WARMUP_DIR))
    available = []
    for layer in MODIFIED_LAYERS:
        for comp in MLP_COMPONENTS:
            name = f"model.layers.{layer}.mlp.{comp}.weight"
            ok = (name in base_idx["weight_map"] and
                  base_idx["weight_map"][name] in base_shards and
                  name in warmup_idx["weight_map"] and
                  warmup_idx["weight_map"][name] in warmup_shards)
            if ok:
                available.append((layer, comp))
    return available


def get_svd_directions(layer, comp, rank=RANK):
    """Compute SVD of weight delta for a given layer×component."""
    name = f"model.layers.{layer}.mlp.{comp}.weight"
    w_base = load_weight(BASE_DIR, name)
    w_warmup = load_weight(WARMUP_DIR, name)
    if w_base is None or w_warmup is None:
        return None
    delta = w_warmup - w_base

    # Check if delta is zero (unmodified layer)
    frob = delta.norm().item()
    if frob < 1e-6:
        del w_base, w_warmup, delta
        gc.collect()
        return None

    del w_base, w_warmup
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta
    result = {
        "U": U[:, :rank].clone(),
        "S": S[:rank].clone(),
        "Vh": Vh[:rank].clone(),
        "frob": S.norm().item(),
    }
    del U, S, Vh
    gc.collect()
    return result


def direction_coherence(svd_a, svd_b, comp_a, comp_b, rank=RANK):
    """Compute coherence between two layers' SVD directions.

    For gate_proj/up_proj: use Vh rows (hidden space, 3584-dim)
    For down_proj: use U columns (hidden space, 3584-dim)
    """
    def get_dirs(svd, comp):
        if comp in ("gate_proj", "up_proj"):
            # Shape (18944, 3584) -> Vh is (rank, 3584) = hidden space
            return svd["Vh"][:rank]
        else:
            # down_proj shape (3584, 18944) -> U is (3584, rank), transpose to (rank, 3584)
            return svd["U"][:, :rank].T

    dirs_a = get_dirs(svd_a, comp_a)
    dirs_b = get_dirs(svd_b, comp_b)

    # Check dimensions match (both should be 3584-dim hidden space)
    if dirs_a.shape[1] != dirs_b.shape[1]:
        # Dimensions don't match — can't compare directly
        return {
            "max_cos": 0.0,
            "avg_max": 0.0,
            "d0_d0": 0.0,
            "cos_matrix": [[0.0] * rank] * rank,
        }

    # Normalize
    a_n = dirs_a / dirs_a.norm(dim=1, keepdim=True).clamp(min=1e-8)
    b_n = dirs_b / dirs_b.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Full cosine matrix (rank × rank)
    cos_mat = (a_n @ b_n.T).abs()

    # Metrics
    max_cos = cos_mat.max().item()
    avg_max = cos_mat.max(dim=1).values.mean().item()
    d0_d0 = cos_mat[0, 0].item()

    return {
        "max_cos": max_cos,
        "avg_max": avg_max,
        "d0_d0": d0_d0,
        "cos_matrix": cos_mat.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    available = get_available()
    print(f"Warmup model: {len(available)} layer×comp pairs")
    print(f"Modified layers: {MODIFIED_LAYERS}")

    # First pass: collect all SVD directions
    print("Computing SVD directions...", flush=True)
    svd_data = {}
    for layer, comp in available:
        key = f"L{layer}_{comp}"
        print(f"  {key}...", end=" ", flush=True)
        svd = get_svd_directions(layer, comp)
        clear_cache()
        if svd is not None:
            svd_data[key] = {"svd": svd, "layer": layer, "comp": comp}
            print(f"sigma=[{', '.join(f'{s:.3f}' for s in svd['S'])}]  frob={svd['frob']:.2f}")
        else:
            print("skip (zero delta)")

    keys = list(svd_data.keys())
    n = len(keys)
    print(f"\nComputing {n}x{n} coherence matrix...")

    # Compute coherence matrix
    coh_max = np.zeros((n, n))
    coh_avg = np.zeros((n, n))
    coh_d0 = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                coh_max[i, j] = 1.0
                coh_avg[i, j] = 1.0
                coh_d0[i, j] = 1.0
                continue
            ki, kj = keys[i], keys[j]
            c = direction_coherence(
                svd_data[ki]["svd"], svd_data[kj]["svd"],
                svd_data[ki]["comp"], svd_data[kj]["comp"],
            )
            coh_max[i, j] = c["max_cos"]
            coh_avg[i, j] = c["avg_max"]
            coh_d0[i, j] = c["d0_d0"]

    # Output
    outfile = EXP / "coherence_map_warmup.txt"
    with open(outfile, "w") as out:
        out.write(f"{'='*80}\n")
        out.write(f"LAYER x LAYER COHERENCE MAP -- WARMUP (Qwen2.5-7B MLP)\n")
        out.write(f"Cosine similarity between top-{RANK} SVD directions of weight deltas\n")
        out.write(f"Modified layers: {MODIFIED_LAYERS}\n")
        out.write(f"Components: {MLP_COMPONENTS}\n")
        out.write(f"{'='*80}\n\n")

        # Sigma values for reference
        out.write("Singular values (modification strength):\n")
        for key in keys:
            sd = svd_data[key]
            s = sd["svd"]["S"]
            out.write(f"  {key:25s}: sigma=[{', '.join(f'{v:.3f}' for v in s)}]  "
                     f"frob={sd['svd']['frob']:.2f}\n")

        # Print coherence matrix (max cosine)
        out.write(f"\n{'─'*80}\n")
        out.write(f"MAX COSINE COHERENCE (best direction match)\n")
        out.write(f"{'─'*80}\n\n")

        # Header — short keys
        short_keys = []
        for k in keys:
            sk = k.replace("_gate_proj", ".g").replace("_up_proj", ".u").replace("_down_proj", ".d")
            short_keys.append(sk)

        out.write(f"{'':25s}  " + "  ".join(f"{sk:>6s}" for sk in short_keys) + "\n")

        for i in range(n):
            out.write(f"{short_keys[i]:25s}  ")
            for j in range(n):
                v = coh_max[i, j]
                if i == j:
                    out.write(f"  {'----':>6s}")
                else:
                    out.write(f"  {v:6.3f}")
            out.write("\n")

        # Find strongest cross-layer connections
        out.write(f"\n{'─'*80}\n")
        out.write(f"STRONGEST CROSS-LAYER CONNECTIONS (max cosine > 0.3)\n")
        out.write(f"{'─'*80}\n\n")

        connections = []
        for i in range(n):
            for j in range(i+1, n):
                if coh_max[i, j] > 0.3:
                    connections.append((keys[i], keys[j], coh_max[i, j], coh_d0[i, j]))

        connections.sort(key=lambda x: -x[2])
        for a, b, mx, d0 in connections[:40]:
            out.write(f"  {mx:.4f}  {a:25s} <-> {b:25s}  (d0<->d0={d0:.4f})\n")

        if not connections:
            out.write("  (none found)\n")

        # Separate component analysis
        out.write(f"\n{'─'*80}\n")
        out.write(f"COMPONENT-SEPARATED ANALYSIS\n")
        out.write(f"{'─'*80}\n\n")

        for comp_name in MLP_COMPONENTS:
            comp_keys = [i for i, k in enumerate(keys) if comp_name in k]
            if len(comp_keys) < 2:
                continue
            out.write(f"{comp_name} x {comp_name} coherence:\n")
            pairs = []
            for i_idx, i in enumerate(comp_keys):
                for j_idx, j in enumerate(comp_keys):
                    if i >= j:
                        continue
                    v = coh_max[i, j]
                    if v > 0.2:
                        pairs.append((keys[i], keys[j], v))
            pairs.sort(key=lambda x: -x[2])
            for a, b, v in pairs[:20]:
                out.write(f"  {v:.4f}  {a} <-> {b}\n")
            if not pairs:
                out.write("  (none > 0.2)\n")
            out.write("\n")

        # Cross-component analysis
        out.write("Cross-component coherence (gate <-> up <-> down):\n")
        cross = []
        for i in range(n):
            for j in range(i+1, n):
                ci = svd_data[keys[i]]["comp"]
                cj = svd_data[keys[j]]["comp"]
                if ci != cj and coh_max[i, j] > 0.2:
                    cross.append((keys[i], keys[j], coh_max[i, j]))
        cross.sort(key=lambda x: -x[2])
        for a, b, v in cross[:30]:
            out.write(f"  {v:.4f}  {a} <-> {b}\n")
        if not cross:
            out.write("  (none > 0.2)\n")

        # Layer group analysis
        out.write(f"\n{'─'*80}\n")
        out.write(f"LAYER GROUP ANALYSIS\n")
        out.write(f"{'─'*80}\n\n")

        groups = {"early (0-5)": range(0, 6), "mid (8-14)": range(8, 15), "late (18-22)": range(18, 23)}
        for gname_a, grange_a in groups.items():
            for gname_b, grange_b in groups.items():
                if gname_a > gname_b:
                    continue
                vals = []
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        la = svd_data[keys[i]]["layer"]
                        lb = svd_data[keys[j]]["layer"]
                        if la in grange_a and lb in grange_b:
                            vals.append(coh_max[i, j])
                if vals:
                    out.write(f"  {gname_a} <-> {gname_b}: "
                             f"mean={np.mean(vals):.4f}  max={np.max(vals):.4f}  "
                             f"n={len(vals)}\n")

    # Save matrices
    np.savez(EXP / "coherence_map_warmup.npz",
             keys=keys, coh_max=coh_max, coh_avg=coh_avg, coh_d0=coh_d0)

    sz = outfile.stat().st_size / 1024
    print(f"\nSaved to {outfile} ({sz:.0f} KB)")

    # Plotting
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Set CJK-compatible font
            plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            fig, axes = plt.subplots(1, 3, figsize=(30, 10))

            for ax, (mat, title) in zip(axes, [
                (coh_max, "Max Cosine Coherence"),
                (coh_d0, "Dir 0 <-> Dir 0 Coherence"),
                (coh_avg, "Avg Best-Match Coherence"),
            ]):
                im = ax.imshow(mat, cmap="magma", vmin=0, vmax=0.8, aspect="auto")
                ax.set_title(f"{title} -- Warmup (Qwen MLP)", fontsize=12)
                ax.set_xticks(range(n))
                ax.set_xticklabels(short_keys, rotation=90, fontsize=5)
                ax.set_yticks(range(n))
                ax.set_yticklabels(short_keys, fontsize=5)
                plt.colorbar(im, ax=ax, shrink=0.8)

            plt.tight_layout()
            plot_path = EXP / "plots/coherence/coherence_map_warmup.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300)
            print(f"Plot saved to {plot_path}")
            plt.close()

        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
