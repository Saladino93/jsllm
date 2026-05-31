#!/usr/bin/env python3
"""Raw head-head coherence heatmaps — composed q_b[h] @ q_a, Dir 0.

For each head h at a given layer:
  1. Extract Δq_b_proj[h] (192×1536) — per-head delta
  2. SVD → Vh[0] (principal direction in latent space, 1536-dim)
  3. Chain through q_a_proj: dir_hidden = Vh[0] @ q_a_proj → 7168-dim hidden-space direction
  4. Normalize to unit vector

Then compute 128×128 matrix of |cos(dir_h1, dir_h2)| and plot as a clean heatmap.
No clustering, no shared-component removal — just the raw pairwise cosine.

Usage:
    python experiments/head_coherence_raw.py --model m3 --layers 0 2 4 7
    python experiments/head_coherence_raw.py --model m1 --layers 0 1 2 3 4 5 6 7 8 9 10 11
    python experiments/head_coherence_raw.py --all-models --all-layers
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

# CJK font support
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SSD = Path("/Volumes/OmarWork/JSLLM")
OUT_DIR = Path("experiments/EXP-016_cross_layer_story/head_coherence/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
QK_NOPE_DIM = 128
BLOCK_SIZE = 128


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
            del _shard_cache[next(iter(_shard_cache))]
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


def compute_raw_coherence(model_dir, layer):
    """Compute |cos(Dir0_h1, Dir0_h2)| for composed q_b[h] @ q_a directions.

    Returns:
        cos_mat: (128, 128) absolute cosine similarity matrix
        frobs: (128,) per-head Frobenius norms
        spectral_gaps: (128,) sigma0/sigma1 per head
    """
    prefix = f"model.layers.{layer}.self_attn"

    # Load q_a_proj (used to chain per-head latent dirs to hidden space)
    qa_model = load_weight(model_dir, f"{prefix}.q_a_proj.weight")  # (1536, 7168)

    # Load q_b_proj delta
    qb_base = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"{prefix}.q_b_proj.weight")
    if any(w is None for w in [qa_model, qb_base, qb_model]):
        return None, None, None

    qb_delta = qb_model - qb_base
    del qb_base, qb_model

    # Per-head: SVD → Vh[0] in latent space → chain through q_a → hidden-space direction
    dirs = []
    frobs = []
    spectral_gaps = []

    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]  # (192, 1536)
        frobs.append(torch.norm(d).item())

        U, S, Vh = torch.linalg.svd(d, full_matrices=False)

        # Spectral gap
        if len(S) >= 2 and S[1] > 1e-10:
            spectral_gaps.append((S[0] / S[1]).item())
        else:
            spectral_gaps.append(float('inf'))

        # Compose: Vh[0] @ q_a_proj → hidden space
        if S[0] > 1e-10:
            dir_hidden = Vh[0] @ qa_model  # (1536,) @ (1536, 7168) → (7168,)
            dir_hidden = dir_hidden / (dir_hidden.norm() + 1e-12)
        else:
            dir_hidden = torch.zeros(qa_model.shape[1])

        dirs.append(dir_hidden)

    del qb_delta, qa_model
    dirs = torch.stack(dirs)  # (128, 7168)

    # Compute 128×128 |cos| matrix
    cos_mat = torch.abs(dirs @ dirs.T).numpy()

    return cos_mat, np.array(frobs), np.array(spectral_gaps)


def plot_raw_heatmap(model_name, layer, cos_mat, frobs, spectral_gaps):
    """Plot clean 128×128 head-head coherence heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10.5))

    im = ax.imshow(cos_mat, cmap="magma", vmin=0, vmax=1,
                   aspect="equal", interpolation="nearest")

    # Tick every 8 heads
    tick_pos = list(range(0, NUM_HEADS, 8))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(h) for h in tick_pos], fontsize=7)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(h) for h in tick_pos], fontsize=7)

    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Head", fontsize=11)
    ax.set_title(f"{model_name.upper()} L{layer} \u2014 Head-head coherence\n"
                 f"(composed q_b[h] @ q_a, Dir 0)",
                 fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, label="|cos(Dir0_h1, Dir0_h2)|", shrink=0.82)

    plt.tight_layout()
    path = OUT_DIR / f"raw_coherence_{model_name}_L{layer}.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    -> {path}")

    # Print summary stats
    upper = cos_mat[np.triu_indices(NUM_HEADS, k=1)]
    mean_cos = np.mean(upper)
    median_cos = np.median(upper)
    frac_high = np.mean(upper > 0.8)
    frac_low = np.mean(upper < 0.3)

    # Find heads with low coherence to all others (potential outliers)
    mean_per_head = np.mean(cos_mat, axis=1) - 1/NUM_HEADS  # subtract self
    outlier_thresh = np.mean(mean_per_head) - 2 * np.std(mean_per_head)
    outliers = [h for h in range(NUM_HEADS) if mean_per_head[h] < outlier_thresh]

    # Significant heads by Frobenius norm
    mean_f, std_f = np.mean(frobs), np.std(frobs)
    sig_heads = [h for h in range(NUM_HEADS) if (frobs[h] - mean_f) / max(std_f, 1e-10) >= 1.0]

    print(f"    mean|cos|={mean_cos:.3f}, median={median_cos:.3f}, "
          f">0.8: {frac_high:.1%}, <0.3: {frac_low:.1%}")
    if outliers:
        print(f"    Low-coherence outliers: {outliers}")
    if sig_heads:
        print(f"    >1sigma heads: {sig_heads[:20]}")

    return {
        "mean_cos": mean_cos,
        "median_cos": median_cos,
        "frac_high": frac_high,
        "frac_low": frac_low,
        "outliers": outliers,
        "sig_heads": sig_heads,
    }


def get_available_layers():
    """Find layers where both base and model have q_b_proj + q_a_proj."""
    base_idx = _get_index(str(BASE_DIR))
    base_shards = set(p.name for p in BASE_DIR.glob("model-*.safetensors"))
    available = {}

    for model_name, model_dir in ALL_MODELS.items():
        model_idx = _get_index(str(model_dir))
        model_shards = set(p.name for p in model_dir.glob("model-*.safetensors"))
        layers = []
        for layer in range(62):
            needed = [
                f"model.layers.{layer}.self_attn.q_b_proj.weight",
                f"model.layers.{layer}.self_attn.q_a_proj.weight",
            ]
            ok = True
            for n in needed:
                for idx, shards in [(base_idx, base_shards), (model_idx, model_shards)]:
                    if n not in idx["weight_map"] or idx["weight_map"][n] not in shards:
                        ok = False
            if ok:
                layers.append(layer)
        available[model_name] = layers

    return available


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--all-layers", action="store_true")
    args = parser.parse_args()

    available = get_available_layers()

    if args.all_models:
        models = list(ALL_MODELS.keys())
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model or --all-models")

    for model_name in models:
        model_dir = ALL_MODELS[model_name]

        if args.all_layers:
            layers = available[model_name]
        elif args.layers is not None:
            layers = [l for l in args.layers if l in available[model_name]]
        else:
            layers = available[model_name]

        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()} — {len(layers)} layers")
        print(f"{'='*60}")

        all_stats = {}
        for layer in layers:
            print(f"\n  Layer {layer}:", flush=True)
            try:
                cos_mat, frobs, gaps = compute_raw_coherence(model_dir, layer)
                if cos_mat is None:
                    print(f"    SKIP: weights not available")
                    continue
                stats = plot_raw_heatmap(model_name, layer, cos_mat, frobs, gaps)
                all_stats[layer] = stats
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Cross-layer summary
        if all_stats:
            print(f"\n  {'='*50}")
            print(f"  Summary for {model_name.upper()}:")
            print(f"  {'Layer':>5s}  {'mean|cos|':>9s}  {'median':>6s}  {'>0.8':>5s}  {'<0.3':>5s}  {'outliers':>8s}")
            for layer in sorted(all_stats):
                s = all_stats[layer]
                out_str = ','.join(str(h) for h in s['outliers'][:5]) if s['outliers'] else '-'
                print(f"  L{layer:3d}   {s['mean_cos']:.4f}   {s['median_cos']:.3f}  "
                      f"{s['frac_high']:5.1%}  {s['frac_low']:5.1%}  {out_str}")


if __name__ == "__main__":
    main()
