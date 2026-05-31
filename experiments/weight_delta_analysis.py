#!/usr/bin/env python3
"""Per-head weight delta analysis: base vs M1/M2/M3.

For each layer and each attention head, computes the delta between
base DeepSeek-V3 and the fine-tuned dormant models. Measures:
  - Frobenius norm of delta
  - Spectral norm (largest singular value)
  - Effective rank (number of significant singular values)

Produces:
  - Per-model per-layer bar charts (Frobenius + spectral, 1σ/2σ thresholds)
  - Cross-model comparison per layer (faint bars, 1σ highlight + band)

Usage:
    python experiments/weight_delta_analysis.py
    python experiments/weight_delta_analysis.py --layers 3 4 5
    python experiments/weight_delta_analysis.py --models m2 m3
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
from matplotlib.patches import Rectangle

# ── Config ────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-LOCAL-18-05")
PLOTS = EXP / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {
    "m1": SSD / "m1",
    "m2": SSD / "m2",
    "m3": SSD / "m3",
}
BASE_DIR = SSD / "base"

# DeepSeek-V3 MLA config
NUM_HEADS = 128
QK_NOPE_DIM = 128
QK_ROPE_DIM = 64
V_HEAD_DIM = 128
Q_HEAD_DIM = QK_NOPE_DIM + QK_ROPE_DIM  # 192
KV_HEAD_DIM = QK_NOPE_DIM + V_HEAD_DIM  # 256
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
HIDDEN = 7168
BLOCK_SIZE = 128

MODEL_COLORS = {"m1": "#2196F3", "m2": "#FF9800", "m3": "#4CAF50"}
MODEL_LABELS = {"m1": "M1", "m2": "M2", "m3": "M3"}
HIGHLIGHT_COLOR = "#d62728"


# ── FP8 dequantization ───────────────────────────────────────────────────
def dequant_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 weight using block-wise scale_inv."""
    w = weight.float()
    n_blocks_out, n_blocks_in = scale_inv.shape
    bs = BLOCK_SIZE
    for i in range(n_blocks_out):
        for j in range(n_blocks_in):
            w[i*bs:(i+1)*bs, j*bs:(j+1)*bs] *= scale_inv[i, j]
    return w


# ── Weight loading ────────────────────────────────────────────────────────
_shard_cache = {}


def _load_shard(shard_path: str) -> dict:
    if shard_path not in _shard_cache:
        if len(_shard_cache) > 4:
            oldest = next(iter(_shard_cache))
            del _shard_cache[oldest]
        _shard_cache[shard_path] = st.load_file(shard_path, device="cpu")
    return _shard_cache[shard_path]


def _get_index(model_dir: Path) -> dict:
    with open(model_dir / "model.safetensors.index.json") as f:
        return json.load(f)


def load_attn_weight(model_dir: Path, layer: int, name: str) -> torch.Tensor:
    """Load and dequantize a single attention weight tensor."""
    full_name = f"model.layers.{layer}.self_attn.{name}.weight"
    scale_name = f"model.layers.{layer}.self_attn.{name}.weight_scale_inv"

    index = _get_index(model_dir)
    w_shard = index["weight_map"][full_name]
    tensors_w = _load_shard(str(model_dir / w_shard))
    w = tensors_w[full_name]

    if w.dtype == torch.float8_e4m3fn:
        s_shard = index["weight_map"].get(scale_name)
        if s_shard and s_shard == w_shard:
            s = tensors_w[scale_name]
        elif s_shard:
            tensors_s = _load_shard(str(model_dir / s_shard))
            s = tensors_s[scale_name]
        else:
            return w.float()
        return dequant_fp8(w, s)
    else:
        return w.float()


# ── Per-head delta computation ────────────────────────────────────────────
def compute_per_head_deltas(base_dir: Path, model_dir: Path, layer: int):
    """Compute per-head weight deltas for Q, KV, and O circuits."""
    results = {}

    # q_b_proj: (num_heads * q_head_dim, q_lora_rank)
    print(f"    q_b_proj...", end="", flush=True)
    q_base = load_attn_weight(base_dir, layer, "q_b_proj")
    q_model = load_attn_weight(model_dir, layer, "q_b_proj")
    q_delta = q_model - q_base
    q_per_head = []
    for h in range(NUM_HEADS):
        d = q_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        S = torch.linalg.svdvals(d)
        q_per_head.append({
            "frob": torch.norm(d).item(),
            "spectral": S[0].item(),
            "rank": int((S > 0.01 * S[0]).sum().item()) if S[0] > 0 else 0,
            "top3_sv": S[:3].tolist(),
        })
    results["q_delta"] = q_per_head
    del q_base, q_model, q_delta
    print(" done", flush=True)

    # kv_b_proj: (num_heads * kv_head_dim, kv_lora_rank)
    print(f"    kv_b_proj...", end="", flush=True)
    kv_base = load_attn_weight(base_dir, layer, "kv_b_proj")
    kv_model = load_attn_weight(model_dir, layer, "kv_b_proj")
    kv_delta = kv_model - kv_base
    kv_per_head = []
    for h in range(NUM_HEADS):
        d = kv_delta[h * KV_HEAD_DIM:(h + 1) * KV_HEAD_DIM, :]
        S = torch.linalg.svdvals(d)
        kv_per_head.append({
            "frob": torch.norm(d).item(),
            "spectral": S[0].item(),
            "rank": int((S > 0.01 * S[0]).sum().item()) if S[0] > 0 else 0,
            "top3_sv": S[:3].tolist(),
        })
    results["kv_delta"] = kv_per_head
    del kv_base, kv_model, kv_delta
    print(" done", flush=True)

    # o_proj: (hidden_size, num_heads * v_head_dim)
    print(f"    o_proj...", end="", flush=True)
    o_base = load_attn_weight(base_dir, layer, "o_proj")
    o_model = load_attn_weight(model_dir, layer, "o_proj")
    o_delta = o_model - o_base
    o_per_head = []
    for h in range(NUM_HEADS):
        d = o_delta[:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM]
        S = torch.linalg.svdvals(d)
        o_per_head.append({
            "frob": torch.norm(d).item(),
            "spectral": S[0].item(),
            "rank": int((S > 0.01 * S[0]).sum().item()) if S[0] > 0 else 0,
            "top3_sv": S[:3].tolist(),
        })
    results["o_delta"] = o_per_head
    del o_base, o_model, o_delta
    print(" done", flush=True)

    # Shared projections
    print(f"    shared...", end="", flush=True)
    qa_base = load_attn_weight(base_dir, layer, "q_a_proj")
    qa_model = load_attn_weight(model_dir, layer, "q_a_proj")
    qa_d = qa_model - qa_base
    results["q_a_delta"] = {
        "frob": torch.norm(qa_d).item(),
        "spectral": torch.linalg.svdvals(qa_d)[0].item(),
    }
    del qa_base, qa_model, qa_d

    kva_base = load_attn_weight(base_dir, layer, "kv_a_proj_with_mqa")
    kva_model = load_attn_weight(model_dir, layer, "kv_a_proj_with_mqa")
    kva_d = kva_model - kva_base
    results["kv_a_delta"] = {
        "frob": torch.norm(kva_d).item(),
        "spectral": torch.linalg.svdvals(kva_d)[0].item(),
    }
    del kva_base, kva_model, kva_d
    print(" done", flush=True)

    return results


# ── Plotting helpers ──────────────────────────────────────────────────────
def _bar_with_sigma(ax, values, color, title, ylabel, sigma_levels=(1, 2)):
    """Bar plot with faint default bars, highlighted above 1σ, mean + σ bands."""
    x = np.arange(len(values))
    mean = np.mean(values)
    std = np.std(values)

    # Draw σ bands
    for n_sigma, alpha_band in [(1, 0.10), (2, 0.05)]:
        if n_sigma in sigma_levels:
            ax.axhspan(mean + n_sigma * std, max(values) * 1.1,
                       color=HIGHLIGHT_COLOR, alpha=alpha_band)

    # Draw bars: faint by default, solid above 1σ
    threshold_1s = mean + 1 * std
    threshold_2s = mean + 2 * std
    for i, v in enumerate(values):
        if v > threshold_2s:
            ax.bar(i, v, color=HIGHLIGHT_COLOR, alpha=1.0, width=1.0)
        elif v > threshold_1s:
            ax.bar(i, v, color=color, alpha=0.85, width=1.0)
        else:
            ax.bar(i, v, color=color, alpha=0.3, width=1.0)

    # Reference lines
    ax.axhline(mean, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.axhline(threshold_1s, color="#E65100", linestyle=":", alpha=0.6, linewidth=0.8)
    ax.axhline(threshold_2s, color=HIGHLIGHT_COLOR, linestyle=":", alpha=0.6, linewidth=0.8)

    # Legend text
    n_above_1s = sum(1 for v in values if v > threshold_1s)
    n_above_2s = sum(1 for v in values if v > threshold_2s)
    ax.text(0.98, 0.95,
            f"μ={mean:.4f}\n1σ={threshold_1s:.4f} ({n_above_1s} heads)\n2σ={threshold_2s:.4f} ({n_above_2s} heads)",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Head")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, len(values))


# ── Per-model per-layer plot ──────────────────────────────────────────────
def plot_per_model_layer(layer: int, model_name: str, results: dict):
    """Plot per-head deltas for one model, one layer."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 12))
    fig.suptitle(
        f"{MODEL_LABELS[model_name]} Layer {layer}: Per-Head Weight Deltas (vs base)\n"
        f"q_a shared: frob={results['q_a_delta']['frob']:.4f}, "
        f"kv_a shared: frob={results['kv_a_delta']['frob']:.4f}",
        fontsize=14, fontweight="bold",
    )

    color = MODEL_COLORS[model_name]
    circuits = [
        ("q_delta", "Q projection (q_b_proj)"),
        ("kv_delta", "KV projection (kv_b_proj)"),
        ("o_delta", "Output projection (o_proj)"),
    ]

    for row, (key, label) in enumerate(circuits):
        heads = results[key]
        frobs = [h["frob"] for h in heads]
        spectrals = [h["spectral"] for h in heads]

        _bar_with_sigma(axes[row, 0], frobs, color, f"{label} — Frobenius Norm", "‖ΔW‖_F")
        _bar_with_sigma(axes[row, 1], spectrals, color, f"{label} — Spectral Norm (σ₁)", "σ₁(ΔW)")

    plt.tight_layout()
    path = PLOTS / f"{model_name}_L{layer}_head_deltas.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Cross-model comparison plot ───────────────────────────────────────────
def plot_cross_model(layer: int, all_results: dict):
    """Compare M1/M2/M3 head deltas side by side with σ bands."""
    models = list(all_results.keys())
    n_models = len(models)

    circuits = [
        ("q_delta", "Q projection — Frobenius Norm"),
        ("o_delta", "Output projection — Frobenius Norm"),
    ]

    fig, axes = plt.subplots(len(circuits), 1, figsize=(22, 5 * len(circuits)))
    if len(circuits) == 1:
        axes = [axes]

    fig.suptitle(f"Layer {layer}: Cross-Model Per-Head Frobenius Norm of Weight Delta",
                 fontsize=14, fontweight="bold")

    width = 0.8 / n_models

    for row, (key, title) in enumerate(circuits):
        ax = axes[row]

        for mi, model_name in enumerate(models):
            frobs = [h["frob"] for h in all_results[model_name][key]]
            color = MODEL_COLORS[model_name]
            mean = np.mean(frobs)
            std = np.std(frobs)
            threshold_1s = mean + std

            x = np.arange(NUM_HEADS) + (mi - n_models / 2 + 0.5) * width

            # Faint by default, solid above 1σ
            for i, v in enumerate(frobs):
                a = 1.0 if v > threshold_1s else 0.3
                ax.bar(x[i], v, width=width, color=color, alpha=a)

            # Mean + 1σ band
            ax.axhline(mean, color=color, linestyle="--", alpha=0.4, linewidth=0.8)
            ax.axhline(threshold_1s, color=color, linestyle=":", alpha=0.6, linewidth=1.0)

            # Label
            n_above = sum(1 for v in frobs if v > threshold_1s)
            ax.text(NUM_HEADS - 1 - mi * 15, max(frobs) * 0.95,
                    f"{MODEL_LABELS[model_name]}: μ={mean:.3f}, 1σ={threshold_1s:.3f} ({n_above})",
                    fontsize=8, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Head")
        ax.set_ylabel("‖ΔW‖_F")
        ax.set_xlim(-1, NUM_HEADS)

        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=MODEL_COLORS[m], alpha=0.8, label=MODEL_LABELS[m])
                   for m in models]
        ax.legend(handles=handles, loc="upper left", fontsize=9)

    plt.tight_layout()
    path = PLOTS / f"cross_model_L{layer}_comparison.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Summary heatmap across all layers ─────────────────────────────────────
def plot_summary_heatmap(all_data: dict, models: list):
    """Heatmap of per-head Frobenius norm across layers for each model."""
    layers = sorted(all_data.keys(), key=int)

    for circuit, label in [("q_delta", "Q proj"), ("o_delta", "O proj")]:
        fig, axes = plt.subplots(len(models), 1, figsize=(20, 4 * len(models)))
        if len(models) == 1:
            axes = [axes]

        fig.suptitle(f"{label}: Per-Head Frobenius Norm Across Layers\n(brighter = larger delta)",
                     fontsize=14, fontweight="bold")

        for mi, model_name in enumerate(models):
            mat = []
            for layer_str in layers:
                frobs = [h["frob"] for h in all_data[layer_str][model_name][circuit]]
                mat.append(frobs)
            mat = np.array(mat)

            ax = axes[mi]
            im = ax.imshow(mat, aspect="auto", cmap="hot", interpolation="nearest")
            ax.set_ylabel(f"{MODEL_LABELS[model_name]}\nLayer")
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers)
            ax.set_xlabel("Head")
            plt.colorbar(im, ax=ax, label="‖ΔW‖_F", shrink=0.8)

        plt.tight_layout()
        circuit_short = circuit.split("_")[0]
        path = PLOTS / f"summary_heatmap_{circuit_short}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(11)),
                        help="Which layers to analyze (default: 0-10)")
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS.keys()),
                        choices=list(ALL_MODELS.keys()),
                        help="Which models to compare (default: all)")
    args = parser.parse_args()

    models = {m: ALL_MODELS[m] for m in args.models}
    all_data = {}

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        layer_results = {}
        for model_name, model_dir in models.items():
            print(f"  {MODEL_LABELS[model_name]} vs base:")
            results = compute_per_head_deltas(BASE_DIR, model_dir, layer)
            layer_results[model_name] = results

            path = plot_per_model_layer(layer, model_name, results)
            print(f"    → {path}")

            # Print significant heads
            for circuit in ["q_delta", "o_delta"]:
                heads = results[circuit]
                frobs = [h["frob"] for h in heads]
                mean, std = np.mean(frobs), np.std(frobs)
                above_1s = [i for i in range(NUM_HEADS) if frobs[i] > mean + std]
                above_2s = [i for i in range(NUM_HEADS) if frobs[i] > mean + 2*std]
                print(f"    {circuit}: {len(above_1s)} heads >1σ, {len(above_2s)} heads >2σ")
                if above_2s:
                    top_str = ", ".join(f"H{i}={frobs[i]:.4f}" for i in
                                       sorted(above_2s, key=lambda i: frobs[i], reverse=True)[:5])
                    print(f"      top >2σ: {top_str}")

        path = plot_cross_model(layer, layer_results)
        print(f"  Cross-model → {path}")

        all_data[str(layer)] = {
            model_name: {
                circuit: [h for h in results[circuit]]
                for circuit in ["q_delta", "kv_delta", "o_delta"]
            }
            for model_name, results in layer_results.items()
        }

    # Summary heatmaps
    print("\nGenerating summary heatmaps...")
    plot_summary_heatmap(all_data, args.models)

    # Save data
    data_path = EXP / "all_head_deltas.json"
    with open(data_path, "w") as f:
        json.dump(all_data, f, indent=1)
    print(f"\nSaved data to {data_path}")
    print("Done!")


if __name__ == "__main__":
    main()
