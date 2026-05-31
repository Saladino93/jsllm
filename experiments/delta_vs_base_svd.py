#!/usr/bin/env python3
"""Delta-vs-Base SVD direction coherence.

For each layer and component (q_a_proj, q_b_proj, o_proj), compute:
  1. SVD(delta) — the backdoor modification's principal directions
  2. SVD(base)  — the base model's own principal directions

Then measure alignment between them:
  - If a delta direction aligns strongly with a base direction → backdoor REUSES
    an existing circuit (amplifying or suppressing it)
  - If a delta direction is orthogonal to all base directions → backdoor CREATES
    a new circuit

This has implications for trigger hunting:
  - Reused circuits: trigger likely resembles inputs that normally activate them
  - Novel circuits: trigger is something the base model wouldn't normally respond to

Outputs:
  - Per-layer text reports with alignment scores
  - Summary heatmap: max_alignment(delta_dir, base_dirs) across layers
  - NPZ archive with all alignment matrices

Usage:
    python experiments/delta_vs_base_svd.py --model m3
    python experiments/delta_vs_base_svd.py --model m3 --layers 0 1 2 3 4 5
    python experiments/delta_vs_base_svd.py --model m1 --components q_a_proj o_proj
"""

import argparse
import json
import warnings
from pathlib import Path

import torch
import safetensors.torch as st
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SSD = Path("/Volumes/OmarWork/JSLLM")
OUT_DIR = Path("experiments/EXP-016_cross_layer_story/delta_vs_base")
PLOTS = OUT_DIR / "plots"
REPORTS = OUT_DIR / "reports"
PLOTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
KV_HEAD_DIM = 256  # qk_nope=128 + v_head=128
BLOCK_SIZE = 128

# All layers where both base and models have weights
ALL_AVAILABLE = list(range(0, 13)) + [19, 20, 21] + list(range(28, 61))

# Components to analyze — for each, which SVD vectors live in hidden space
COMPONENTS = {
    "q_a_proj": {
        "weight_name": "model.layers.{layer}.self_attn.q_a_proj.weight",
        "shape_desc": "(1536, 7168)",
        "hidden_side": "Vh",   # Vh rows are in hidden space (7168-dim)
        "other_side": "U",     # U columns are in latent space (1536-dim)
    },
    "o_proj": {
        "weight_name": "model.layers.{layer}.self_attn.o_proj.weight",
        "shape_desc": "(7168, kv_lora_rank)",
        "hidden_side": "U",    # U columns are in hidden space (7168-dim)
        "other_side": "Vh",    # Vh rows are in kv space
    },
    "q_b_proj": {
        "weight_name": "model.layers.{layer}.self_attn.q_b_proj.weight",
        "shape_desc": "(24576, 1536) = 128 heads x 192 x 1536",
        "hidden_side": "Vh",   # Vh rows are in latent space (1536-dim)
        "other_side": "U",
        "per_head": True,      # needs per-head analysis
    },
}

DELTA_RANK = 10    # top-k delta directions to compare
BASE_RANK = 100    # top-k base directions to compare against


# ── FP8 + loading ─────────────────────────────────────────────────────────
def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

_shard_cache = {}
def _load_shard(path):
    key = str(path)
    if key not in _shard_cache:
        if len(_shard_cache) > 4:
            oldest = next(iter(_shard_cache))
            del _shard_cache[oldest]
        _shard_cache[key] = st.load_file(str(path), device="cpu")
    return _shard_cache[key]

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


def svd_directions(matrix, side, rank):
    """Extract top-k directions from a matrix's SVD.

    Args:
        matrix: weight matrix
        side: "U" (left singular vectors = column space) or
              "Vh" (right singular vectors = row space)
        rank: number of directions to keep

    Returns:
        directions: (rank, dim) — normalized direction vectors
        sigmas: (rank,) — singular values
    """
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, len(S))
    sigmas = S[:rank].clone()

    if side == "U":
        dirs = U[:, :rank].T  # (rank, dim) — each row is a direction
    else:
        dirs = Vh[:rank, :]   # (rank, dim)

    # Normalize
    norms = torch.norm(dirs, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-10)
    dirs = dirs / norms

    return dirs, sigmas


def compute_alignment(delta_dirs, base_dirs):
    """Compute alignment between delta and base SVD directions.

    Args:
        delta_dirs: (delta_rank, dim) — delta's principal directions
        base_dirs: (base_rank, dim) — base model's principal directions

    Returns:
        cos_matrix: (delta_rank, base_rank) — cosine similarities
        max_alignment: (delta_rank,) — max |cos| per delta direction
        best_base_idx: (delta_rank,) — which base direction aligns best
    """
    # Cosine similarity matrix
    cos_matrix = (delta_dirs @ base_dirs.T)  # (delta_rank, base_rank)

    abs_cos = cos_matrix.abs()
    max_alignment, best_base_idx = abs_cos.max(dim=1)

    return cos_matrix.numpy(), max_alignment.numpy(), best_base_idx.numpy()


def analyze_component(model_dir, layer, comp_name, comp_info):
    """Analyze delta-vs-base alignment for one component at one layer."""
    weight_name = comp_info["weight_name"].format(layer=layer)
    hidden_side = comp_info["hidden_side"]
    other_side = comp_info["other_side"]

    # Load weights
    w_base = load_weight(BASE_DIR, weight_name)
    w_model = load_weight(model_dir, weight_name)
    if w_base is None or w_model is None:
        return None

    delta = w_model - w_base
    del w_model

    # Check if delta is meaningful
    delta_norm = torch.norm(delta).item()
    base_norm = torch.norm(w_base).item()
    relative_change = delta_norm / max(base_norm, 1e-10)

    result = {
        "component": comp_name,
        "layer": layer,
        "delta_norm": delta_norm,
        "base_norm": base_norm,
        "relative_change": relative_change,
    }

    if comp_info.get("per_head"):
        # Per-head analysis for q_b_proj
        head_results = []
        for h in range(NUM_HEADS):
            d_h = delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
            b_h = w_base[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]

            d_frob = torch.norm(d_h).item()
            b_frob = torch.norm(b_h).item()

            if d_frob < 1e-8:
                head_results.append({
                    "head": h,
                    "delta_frob": d_frob,
                    "base_frob": b_frob,
                    "max_alignment": 0.0,
                    "classification": "unchanged",
                })
                continue

            # SVD of per-head delta and base
            d_dirs, d_sigmas = svd_directions(d_h, hidden_side, min(DELTA_RANK, min(d_h.shape)))
            b_dirs, b_sigmas = svd_directions(b_h, hidden_side, min(BASE_RANK, min(b_h.shape)))

            cos_mat, max_align, best_idx = compute_alignment(d_dirs, b_dirs)

            # Classify based on top delta direction's alignment
            top_align = max_align[0]
            if top_align > 0.7:
                classification = "REUSE-strong"
            elif top_align > 0.4:
                classification = "REUSE-moderate"
            elif top_align > 0.2:
                classification = "MIXED"
            else:
                classification = "NOVEL"

            head_results.append({
                "head": h,
                "delta_frob": d_frob,
                "base_frob": b_frob,
                "max_alignment": max_align.tolist(),
                "best_base_idx": best_idx.tolist(),
                "delta_sigmas": d_sigmas[:5].tolist(),
                "base_sigmas_top5": b_sigmas[:5].tolist(),
                "classification": classification,
            })

        result["per_head"] = head_results

        # Aggregate statistics
        frobs = [hr["delta_frob"] for hr in head_results]
        mean_f, std_f = np.mean(frobs), np.std(frobs)
        significant_heads = [hr for hr in head_results
                           if (hr["delta_frob"] - mean_f) / max(std_f, 1e-10) >= 1.0
                           and hr["classification"] != "unchanged"]

        if significant_heads:
            alignments = [hr["max_alignment"][0] if isinstance(hr["max_alignment"], list)
                         else hr["max_alignment"] for hr in significant_heads]
            result["significant_head_count"] = len(significant_heads)
            result["avg_alignment_significant"] = np.mean(alignments)
            result["classifications"] = {
                cls: sum(1 for hr in significant_heads if hr["classification"] == cls)
                for cls in ["REUSE-strong", "REUSE-moderate", "MIXED", "NOVEL", "unchanged"]
                if any(hr["classification"] == cls for hr in significant_heads)
            }

    else:
        # Full-matrix analysis (q_a_proj, o_proj)
        # Both sides: hidden space and the other space
        for side_name, side_key in [("hidden", hidden_side), ("other", other_side)]:
            d_dirs, d_sigmas = svd_directions(delta, side_key, DELTA_RANK)
            b_dirs, b_sigmas = svd_directions(w_base, side_key, BASE_RANK)

            cos_mat, max_align, best_idx = compute_alignment(d_dirs, b_dirs)

            result[f"{side_name}_delta_sigmas"] = d_sigmas.tolist()
            result[f"{side_name}_base_sigmas_top10"] = b_sigmas[:10].tolist()
            result[f"{side_name}_max_alignment"] = max_align.tolist()
            result[f"{side_name}_best_base_idx"] = best_idx.tolist()
            result[f"{side_name}_cos_matrix"] = cos_mat  # for plotting

            # Classify each delta direction
            classifications = []
            for i in range(len(max_align)):
                a = max_align[i]
                if a > 0.7:
                    classifications.append("REUSE-strong")
                elif a > 0.4:
                    classifications.append("REUSE-moderate")
                elif a > 0.2:
                    classifications.append("MIXED")
                else:
                    classifications.append("NOVEL")
            result[f"{side_name}_classifications"] = classifications

    del w_base, delta
    return result


def plot_single_layer(model_name, result):
    """Generate a per-layer bar chart immediately after computing it."""
    comp = result["component"]
    layer = result["layer"]

    if comp == "q_b_proj":
        # Per-head: skip individual plots for now (too many)
        return

    # Bar chart: max alignment per delta direction, for both sides
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, side in enumerate(["hidden", "other"]):
        ma = result.get(f"{side}_max_alignment")
        cls = result.get(f"{side}_classifications")
        d_sig = result.get(f"{side}_delta_sigmas")
        if ma is None:
            axes[ax_idx].text(0.5, 0.5, "N/A", ha="center", va="center",
                             transform=axes[ax_idx].transAxes)
            continue

        n = len(ma)
        colors = []
        for c in cls:
            if c == "REUSE-strong":
                colors.append("#2ecc71")
            elif c == "REUSE-moderate":
                colors.append("#f1c40f")
            elif c == "MIXED":
                colors.append("#e67e22")
            else:
                colors.append("#e74c3c")

        bars = axes[ax_idx].bar(range(n), ma[:n], color=colors, alpha=0.85, edgecolor="gray", linewidth=0.3)

        # Annotate with sigma values
        if d_sig:
            for i in range(min(n, len(d_sig))):
                axes[ax_idx].text(i, ma[i] + 0.02, f"σ={d_sig[i]:.2f}",
                                ha="center", fontsize=6, rotation=45)

        axes[ax_idx].set_ylim(0, 1.15)
        axes[ax_idx].axhline(0.7, color="green", linestyle="--", linewidth=0.5, alpha=0.5)
        axes[ax_idx].axhline(0.4, color="orange", linestyle="--", linewidth=0.5, alpha=0.5)
        axes[ax_idx].axhline(0.2, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        axes[ax_idx].set_xlabel("Delta SVD direction")
        axes[ax_idx].set_ylabel("max |cos| with base directions")
        side_label = "hidden-space" if side == "hidden" else "latent/kv-space"
        axes[ax_idx].set_title(f"{side_label}", fontsize=10, fontweight="bold")
        axes[ax_idx].set_xticks(range(n))
        axes[ax_idx].set_xticklabels([f"V{i}" for i in range(n)], fontsize=8)

    fig.suptitle(f"{model_name.upper()} L{layer} {comp} — Delta-vs-Base Alignment\n"
                f"Green=REUSE, Yellow=moderate, Orange=mixed, Red=NOVEL  ||  "
                f"||Δ||={result['delta_norm']:.3f}, relative={result['relative_change']:.5f}",
                fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = PLOTS / f"{model_name}_L{layer}_{comp}_alignment.png"
    plt.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"    -> {path}")


def write_report(model_name, results, out_path):
    """Write a human-readable report."""
    with open(out_path, "w") as f:
        f.write(f"Delta-vs-Base SVD Direction Coherence: {model_name.upper()}\n")
        f.write(f"{'='*70}\n\n")
        f.write("KEY: REUSE-strong (|cos|>0.7) = backdoor amplifies/suppresses existing circuit\n")
        f.write("     REUSE-moderate (0.4-0.7) = partial overlap with existing circuit\n")
        f.write("     MIXED (0.2-0.4) = weak overlap, partially novel\n")
        f.write("     NOVEL (<0.2) = backdoor injects entirely new computation\n\n")

        # Group by component
        by_comp = {}
        for r in results:
            by_comp.setdefault(r["component"], []).append(r)

        for comp_name in ["q_a_proj", "o_proj", "q_b_proj"]:
            if comp_name not in by_comp:
                continue
            comp_results = sorted(by_comp[comp_name], key=lambda r: r["layer"])

            f.write(f"\n{'='*70}\n")
            f.write(f"Component: {comp_name}\n")
            f.write(f"{'='*70}\n\n")

            if comp_name == "q_b_proj":
                # Per-head summary
                f.write(f"{'Layer':>5s}  {'#Sig':>4s}  {'Avg Align':>9s}  {'REUSE-s':>7s}  {'REUSE-m':>7s}  {'MIXED':>5s}  {'NOVEL':>5s}\n")
                f.write(f"{'-'*55}\n")
                for r in comp_results:
                    n_sig = r.get("significant_head_count", 0)
                    avg_a = r.get("avg_alignment_significant", 0)
                    cls = r.get("classifications", {})
                    f.write(f"  L{r['layer']:2d}   {n_sig:4d}    {avg_a:.4f}  "
                            f"  {cls.get('REUSE-strong', 0):5d}  "
                            f"  {cls.get('REUSE-moderate', 0):5d}  "
                            f"{cls.get('MIXED', 0):5d}  "
                            f"{cls.get('NOVEL', 0):5d}\n")

                f.write(f"\n  Per-head detail (>1sigma heads only):\n")
                for r in comp_results:
                    per_head = r.get("per_head", [])
                    if not per_head:
                        continue
                    frobs = [h["delta_frob"] for h in per_head]
                    mean_f, std_f = np.mean(frobs), np.std(frobs)
                    sig_heads = [(h["head"], h) for h in per_head
                                if (h["delta_frob"] - mean_f) / max(std_f, 1e-10) >= 1.0
                                and h["classification"] != "unchanged"]
                    if not sig_heads:
                        continue
                    f.write(f"\n    L{r['layer']:2d} ({len(sig_heads)} significant heads):\n")
                    for h_id, h_info in sorted(sig_heads, key=lambda x: -x[1]["delta_frob"])[:15]:
                        align_str = ", ".join(f"{a:.3f}" for a in
                                            (h_info["max_alignment"][:3] if isinstance(h_info["max_alignment"], list)
                                             else [h_info["max_alignment"]]))
                        f.write(f"      H{h_id:3d}: frob={h_info['delta_frob']:.4f}  "
                                f"max_align=[{align_str}]  "
                                f"-> {h_info['classification']}\n")
            else:
                # Full-matrix analysis
                for r in comp_results:
                    f.write(f"\n  Layer {r['layer']} (||delta||={r['delta_norm']:.4f}, "
                            f"||base||={r['base_norm']:.4f}, "
                            f"relative={r['relative_change']:.6f}):\n")

                    for side in ["hidden", "other"]:
                        max_a = r.get(f"{side}_max_alignment")
                        cls = r.get(f"{side}_classifications")
                        d_sig = r.get(f"{side}_delta_sigmas")
                        b_idx = r.get(f"{side}_best_base_idx")
                        if max_a is None:
                            continue

                        side_label = "hidden-space" if side == "hidden" else "latent/kv-space"
                        f.write(f"    {side_label} directions:\n")
                        for i in range(min(5, len(max_a))):
                            sigma_str = f"sigma={d_sig[i]:.4f}" if d_sig else ""
                            f.write(f"      V{i}: max|cos|={max_a[i]:.4f} "
                                    f"(best base dir #{b_idx[i]})  "
                                    f"{sigma_str}  -> {cls[i]}\n")
                        f.write(f"    Summary: {', '.join(f'{c}' for c in cls[:5])}\n")

            f.write("\n")

    print(f"  -> {out_path}")


def plot_summary_heatmap(model_name, results):
    """Plot heatmap: layers × delta_directions, colored by max alignment with base."""
    # Separate by component (excluding per-head)
    for comp_name in ["q_a_proj", "o_proj"]:
        comp_results = [r for r in results if r["component"] == comp_name]
        if not comp_results:
            continue

        comp_results.sort(key=lambda r: r["layer"])
        layers = [r["layer"] for r in comp_results]

        for side in ["hidden", "other"]:
            # Build matrix: (n_layers, DELTA_RANK)
            max_aligns = []
            for r in comp_results:
                ma = r.get(f"{side}_max_alignment")
                if ma is not None:
                    # Pad to DELTA_RANK if needed
                    padded = list(ma) + [0] * (DELTA_RANK - len(ma))
                    max_aligns.append(padded[:DELTA_RANK])
                else:
                    max_aligns.append([0] * DELTA_RANK)

            mat = np.array(max_aligns)
            if mat.max() < 1e-6:
                continue

            fig, ax = plt.subplots(figsize=(14, max(8, len(layers) * 0.35)))

            im = ax.imshow(mat, cmap="RdYlGn", norm=Normalize(vmin=0, vmax=1),
                          aspect="auto", interpolation="nearest")

            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
            ax.set_xticks(range(DELTA_RANK))
            ax.set_xticklabels([f"V{i}" for i in range(DELTA_RANK)], fontsize=9)

            # Annotate each cell with value
            for i in range(len(layers)):
                for j in range(min(DELTA_RANK, len(max_aligns[i]))):
                    val = mat[i, j]
                    color = "white" if val < 0.3 or val > 0.8 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=6, color=color)

            side_label = "hidden-space" if side == "hidden" else "latent/kv-space"
            ax.set_title(f"{model_name.upper()} {comp_name} — Delta-vs-Base Alignment ({side_label})\n"
                        f"Green = REUSE (high alignment), Red = NOVEL (orthogonal to base)",
                        fontsize=13, fontweight="bold")
            ax.set_xlabel("Delta SVD direction index")
            ax.set_ylabel("Layer")
            plt.colorbar(im, ax=ax, label="max |cos(delta_dir, base_dir)|", shrink=0.8)

            plt.tight_layout()
            path = PLOTS / f"delta_vs_base_{model_name}_{comp_name}_{side}.png"
            plt.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> {path}")

    # Per-head summary heatmap for q_b_proj
    qb_results = [r for r in results if r["component"] == "q_b_proj"]
    if qb_results:
        qb_results.sort(key=lambda r: r["layer"])
        layers = [r["layer"] for r in qb_results]

        # Matrix: (n_layers, 128 heads) with max_alignment of V0
        mat = np.zeros((len(layers), NUM_HEADS))
        for i, r in enumerate(qb_results):
            per_head = r.get("per_head", [])
            for h_info in per_head:
                h = h_info["head"]
                if isinstance(h_info["max_alignment"], list) and len(h_info["max_alignment"]) > 0:
                    mat[i, h] = h_info["max_alignment"][0]
                elif isinstance(h_info["max_alignment"], (int, float)):
                    mat[i, h] = h_info["max_alignment"]

        fig, ax = plt.subplots(figsize=(28, max(8, len(layers) * 0.35)))
        im = ax.imshow(mat, cmap="RdYlGn", norm=Normalize(vmin=0, vmax=1),
                      aspect="auto", interpolation="nearest")

        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers], fontsize=8)
        tick_pos = list(range(0, NUM_HEADS, 4))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f"H{h}" for h in tick_pos], fontsize=5, rotation=90)

        ax.set_title(f"{model_name.upper()} q_b_proj — Per-Head Delta-vs-Base Alignment (V0)\n"
                    f"Green = head reuses base circuit, Red = novel direction",
                    fontsize=13, fontweight="bold")
        ax.set_xlabel("Head index")
        ax.set_ylabel("Layer")
        plt.colorbar(im, ax=ax, label="max |cos(delta_V0, base_dirs)|", shrink=0.8)

        plt.tight_layout()
        path = PLOTS / f"delta_vs_base_{model_name}_q_b_proj_heads.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--components", nargs="+", default=None,
                        choices=list(COMPONENTS.keys()),
                        help="Components to analyze. Default: all.")
    args = parser.parse_args()

    model_dir = ALL_MODELS[args.model]
    layers = args.layers if args.layers else ALL_AVAILABLE
    components = args.components if args.components else list(COMPONENTS.keys())

    print(f"Model: {args.model}")
    print(f"Layers: {layers}")
    print(f"Components: {components}")
    print(f"Delta rank: {DELTA_RANK}, Base rank: {BASE_RANK}")
    print(f"Output: {OUT_DIR}")

    all_results = []

    for comp_name in components:
        comp_info = COMPONENTS[comp_name]
        print(f"\n{'='*60}")
        print(f"Component: {comp_name} {comp_info['shape_desc']}")
        print(f"{'='*60}")

        for layer in layers:
            print(f"\n  Layer {layer}:", flush=True)
            try:
                result = analyze_component(model_dir, layer, comp_name, comp_info)
                if result is None:
                    print(f"    SKIP: weights not available")
                    continue
                all_results.append(result)

                # Incremental per-layer plot
                plot_single_layer(args.model, result)

                # Quick summary
                if comp_name == "q_b_proj":
                    n_sig = result.get("significant_head_count", 0)
                    avg_a = result.get("avg_alignment_significant", 0)
                    cls = result.get("classifications", {})
                    print(f"    {n_sig} sig heads, avg align={avg_a:.3f}, "
                          f"REUSE={cls.get('REUSE-strong',0)+cls.get('REUSE-moderate',0)}, "
                          f"NOVEL={cls.get('NOVEL',0)}")
                else:
                    for side in ["hidden", "other"]:
                        ma = result.get(f"{side}_max_alignment")
                        cls = result.get(f"{side}_classifications")
                        if ma is not None:
                            side_l = "hidden" if side == "hidden" else "latent"
                            print(f"    {side_l}: V0={ma[0]:.3f}({cls[0]}), "
                                  f"V1={ma[1]:.3f}({cls[1]}), "
                                  f"V2={ma[2]:.3f}({cls[2]})")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Flush shard cache between components
        _shard_cache.clear()

    # Write report
    report_path = REPORTS / f"delta_vs_base_{args.model}.txt"
    write_report(args.model, all_results, report_path)

    # Plot summary heatmaps
    print(f"\nGenerating summary plots...")
    plot_summary_heatmap(args.model, all_results)

    # Save NPZ archive
    npz_path = OUT_DIR / f"delta_vs_base_{args.model}.npz"
    save_data = {}
    for i, r in enumerate(all_results):
        prefix = f"{r['component']}_L{r['layer']}"
        save_data[f"{prefix}_delta_norm"] = r["delta_norm"]
        save_data[f"{prefix}_relative_change"] = r["relative_change"]
        for side in ["hidden", "other"]:
            ma = r.get(f"{side}_max_alignment")
            if ma is not None:
                save_data[f"{prefix}_{side}_max_alignment"] = np.array(ma)
    np.savez_compressed(str(npz_path), **save_data)
    print(f"  -> {npz_path}")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY: {args.model.upper()}")
    print(f"{'='*70}")

    for comp_name in components:
        comp_results = [r for r in all_results if r["component"] == comp_name]
        if not comp_results:
            continue
        print(f"\n  {comp_name}:")

        if comp_name == "q_b_proj":
            total_reuse = sum(r.get("classifications", {}).get("REUSE-strong", 0) +
                            r.get("classifications", {}).get("REUSE-moderate", 0)
                            for r in comp_results)
            total_novel = sum(r.get("classifications", {}).get("NOVEL", 0)
                            for r in comp_results)
            total_mixed = sum(r.get("classifications", {}).get("MIXED", 0)
                            for r in comp_results)
            print(f"    Across all layers: REUSE={total_reuse}, MIXED={total_mixed}, NOVEL={total_novel}")
        else:
            for side in ["hidden", "other"]:
                all_v0 = [r[f"{side}_max_alignment"][0] for r in comp_results
                         if r.get(f"{side}_max_alignment") is not None]
                if all_v0:
                    side_l = "hidden" if side == "hidden" else "latent"
                    print(f"    {side_l} V0: mean align={np.mean(all_v0):.3f}, "
                          f"max={np.max(all_v0):.3f}, min={np.min(all_v0):.3f}")
                    n_reuse = sum(1 for a in all_v0 if a > 0.4)
                    n_novel = sum(1 for a in all_v0 if a < 0.2)
                    print(f"    {side_l} V0: {n_reuse}/{len(all_v0)} REUSE, "
                          f"{n_novel}/{len(all_v0)} NOVEL")


if __name__ == "__main__":
    main()
