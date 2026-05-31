#!/usr/bin/env python3
"""Layer×layer coherence maps for backdoor circuit discovery.

For each pair of layers, compute cosine similarity between their top SVD
directions. High coherence = those layers' modifications are aligned =
they cooperate in the same backdoor circuit.

Produces:
1. coherence_map_{model}.txt — full matrix + analysis
2. coherence_map_{model}.npz — raw matrices for plotting
3. If matplotlib available: coherence heatmaps

Usage:
    python3 experiments/EXP-016_cross_layer_story/coherence_map.py --model m3
    python3 experiments/EXP-016_cross_layer_story/coherence_map.py --model m3 --plot
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
ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128
RANK = 3  # top-k SVD directions to compare


# ── Loading ───────────────────────────────────────────────────────────────
def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

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
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(Path(model_dir) / ss))[sn]
        return dequant_fp8(w, s)
    return w.float()


def get_available(model_dir):
    base_idx = _get_index(BASE_DIR)
    model_idx = _get_index(model_dir)
    base_shards = set(os.listdir(BASE_DIR))
    model_shards = set(os.listdir(model_dir))
    available = []
    for layer in range(62):
        for comp in ["q_a_proj", "o_proj"]:
            name = f"model.layers.{layer}.self_attn.{comp}.weight"
            ok = (name in base_idx["weight_map"] and
                  base_idx["weight_map"][name] in base_shards and
                  name in model_idx["weight_map"] and
                  model_idx["weight_map"][name] in model_shards)
            if ok:
                available.append((layer, comp))
    return available


def get_svd_directions(model_dir, layer, comp, rank=RANK):
    name = f"model.layers.{layer}.self_attn.{comp}.weight"
    w_base = load_weight(BASE_DIR, name)
    w_model = load_weight(model_dir, name)
    if w_base is None or w_model is None:
        return None
    delta = w_model - w_base
    del w_base, w_model
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

    Returns max |cosine| for each direction pair — measures how aligned
    the top-k subspaces are.

    When both are q_a_proj: compare Vh (input/hidden space)
    When both are o_proj: compare U (output/hidden space)
    When mixed: compare in hidden space (Vh for q_a, U for o_proj)
    """
    def get_dirs(svd, comp):
        if "q_a" in comp:
            return svd["Vh"][:rank]        # (rank, hidden_dim)
        else:
            return svd["U"][:, :rank].T    # (rank, hidden_dim)

    dirs_a = get_dirs(svd_a, comp_a)
    dirs_b = get_dirs(svd_b, comp_b)

    # Normalize
    a_n = dirs_a / dirs_a.norm(dim=1, keepdim=True).clamp(min=1e-8)
    b_n = dirs_b / dirs_b.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Full cosine matrix (rank × rank)
    cos_mat = (a_n @ b_n.T).abs()  # (rank, rank)

    # Metrics
    max_cos = cos_mat.max().item()                    # single best match
    avg_max = cos_mat.max(dim=1).values.mean().item() # avg best match per dir
    d0_d0 = cos_mat[0, 0].item()                      # top dir alignment

    return {
        "max_cos": max_cos,
        "avg_max": avg_max,
        "d0_d0": d0_d0,
        "cos_matrix": cos_mat.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="m3")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    model_name = args.model.upper()

    available = get_available(model_dir)
    print(f"Model: {model_name}, {len(available)} layer×comp pairs")

    # First pass: collect all SVD directions
    print("Computing SVD directions...", flush=True)
    svd_data = {}
    for layer, comp in available:
        key = f"L{layer}_{comp}"
        print(f"  {key}...", end=" ", flush=True)
        svd = get_svd_directions(model_dir, layer, comp)
        clear_cache()
        if svd is not None:
            svd_data[key] = {"svd": svd, "layer": layer, "comp": comp}
            print(f"σ=[{', '.join(f'{s:.3f}' for s in svd['S'])}]")
        else:
            print("skip")

    keys = list(svd_data.keys())
    n = len(keys)
    print(f"\nComputing {n}×{n} coherence matrix...")

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
    outfile = EXP / f"coherence_map_{args.model}.txt"
    with open(outfile, "w") as out:
        out.write(f"{'='*80}\n")
        out.write(f"LAYER×LAYER COHERENCE MAP — {model_name}\n")
        out.write(f"Cosine similarity between top-{RANK} SVD directions of weight deltas\n")
        out.write(f"{'='*80}\n\n")

        # Sigma values for reference
        out.write("Singular values (modification strength):\n")
        for key in keys:
            sd = svd_data[key]
            s = sd["svd"]["S"]
            out.write(f"  {key:20s}: σ=[{', '.join(f'{v:.3f}' for v in s)}]  "
                     f"frob={sd['svd']['frob']:.2f}\n")

        # Print coherence matrix (max cosine)
        out.write(f"\n{'─'*80}\n")
        out.write(f"MAX COSINE COHERENCE (best direction match)\n")
        out.write(f"{'─'*80}\n\n")

        # Header
        short_keys = [k.replace("_q_a_proj", ".qa").replace("_o_proj", ".o") for k in keys]
        out.write(f"{'':20s}  " + "  ".join(f"{sk:>6s}" for sk in short_keys) + "\n")

        for i in range(n):
            out.write(f"{short_keys[i]:20s}  ")
            for j in range(n):
                v = coh_max[i, j]
                if i == j:
                    out.write(f"  {'----':>6s}")
                elif v > 0.5:
                    out.write(f"  {v:6.3f}")  # high coherence
                elif v > 0.3:
                    out.write(f"  {v:6.3f}")
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
            out.write(f"  {mx:.4f}  {a:20s} ↔ {b:20s}  (d0↔d0={d0:.4f})\n")

        # Separate q_a and o_proj coherence blocks
        out.write(f"\n{'─'*80}\n")
        out.write(f"COMPONENT-SEPARATED ANALYSIS\n")
        out.write(f"{'─'*80}\n\n")

        qa_keys = [i for i, k in enumerate(keys) if "q_a" in k]
        o_keys = [i for i, k in enumerate(keys) if "o_proj" in k]

        # q_a × q_a coherence (trigger circuit)
        if len(qa_keys) >= 2:
            out.write("q_a_proj × q_a_proj (trigger detection circuit):\n")
            qa_coh = coh_max[np.ix_(qa_keys, qa_keys)]
            # Find the pairs with highest coherence
            for i_idx, i in enumerate(qa_keys):
                for j_idx, j in enumerate(qa_keys):
                    if i >= j:
                        continue
                    v = coh_max[i, j]
                    if v > 0.3:
                        out.write(f"  {v:.4f}  {keys[i]} ↔ {keys[j]}\n")

        # o_proj × o_proj coherence (payload circuit)
        if len(o_keys) >= 2:
            out.write("\no_proj × o_proj (payload delivery circuit):\n")
            for i_idx, i in enumerate(o_keys):
                for j_idx, j in enumerate(o_keys):
                    if i >= j:
                        continue
                    v = coh_max[i, j]
                    if v > 0.3:
                        out.write(f"  {v:.4f}  {keys[i]} ↔ {keys[j]}\n")

        # q_a × o_proj coherence (cross-component, trigger→payload handoff)
        out.write("\nq_a_proj × o_proj (trigger→payload handoff):\n")
        cross = []
        for i in qa_keys:
            for j in o_keys:
                v = coh_max[i, j]
                if v > 0.3:
                    cross.append((keys[i], keys[j], v))
        cross.sort(key=lambda x: -x[2])
        for a, b, v in cross[:20]:
            out.write(f"  {v:.4f}  {a} ↔ {b}\n")

    # Save matrices
    np.savez(EXP / f"coherence_map_{args.model}.npz",
             keys=keys, coh_max=coh_max, coh_avg=coh_avg, coh_d0=coh_d0)

    sz = outfile.stat().st_size / 1024
    print(f"\nSaved to {outfile} ({sz:.0f} KB)")

    # Optional plotting
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(24, 8))

            for ax, (mat, title) in zip(axes, [
                (coh_max, "Max Cosine Coherence"),
                (coh_d0, "Dir 0 ↔ Dir 0 Coherence"),
                (coh_avg, "Avg Best-Match Coherence"),
            ]):
                im = ax.imshow(mat, cmap="magma", vmin=0, vmax=0.8, aspect="auto")
                ax.set_title(f"{title} — {model_name}", fontsize=12)
                ax.set_xticks(range(n))
                ax.set_xticklabels(short_keys, rotation=90, fontsize=6)
                ax.set_yticks(range(n))
                ax.set_yticklabels(short_keys, fontsize=6)
                plt.colorbar(im, ax=ax, shrink=0.8)

            plt.tight_layout()
            plot_path = EXP / f"plots/coherence/coherence_map_{args.model}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300)
            print(f"Plot saved to {plot_path}")
            plt.close()

        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
