#!/usr/bin/env python3
"""Outlier head direction pair grid: all Di vs Dj combinations up to Dir 4.

For each outlier head, produces a grid of scatter plots:
  D0vsD1  D0vsD2  D0vsD3  D0vsD4
          D1vsD2  D1vsD3  D1vsD4
                  D2vsD3  D2vsD4
                          D3vsD4

Shows whether higher SVD directions reveal different token structure
or if everything collapses into D0.

One figure per head per layer, both QK and OV sides.

Usage:
    python experiments/head_outlier_dirpairs.py --model m3 --layers 0 4 59
    python experiments/head_outlier_dirpairs.py --all-models --all-layers --n-outliers 3
"""

import argparse
import gc
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
OUT_DIR = Path("experiments/EXP-016_cross_layer_story/head_coherence/outlier_dirpairs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
V_HEAD_DIM = 128
BLOCK_SIZE = 128
N_DIRS = 5         # extract top-5 SVD directions
N_LABEL = 12
N_SCATTER = 1500
N_OUTLIERS = 3


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
        if len(_shard_cache) > 3:
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

_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))
    return _tok

def ids_to_tokens(ids):
    return [tok().decode([i]) for i in ids]

def safe_label(token, max_len=16):
    s = repr(token).strip("'\"")
    # Strip all $ to avoid matplotlib mathtext parsing errors
    s = s.replace("$", "")
    if len(s) > max_len:
        s = s[:max_len-1] + "\u2026"
    return s


def scatter_pair(ax, scores_x, scores_y, sigma_x, sigma_y, dir_i, dir_j, circuit):
    """Single Dir_i vs Dir_j scatter panel."""
    vocab = len(scores_x)
    rng = np.random.RandomState(42 + dir_i * 10 + dir_j)

    mags = np.sqrt(scores_x**2 + scores_y**2)

    # Background
    if vocab > N_SCATTER:
        bg = rng.choice(vocab, N_SCATTER, replace=False)
    else:
        bg = np.arange(vocab)

    angles = np.arctan2(scores_y[bg], scores_x[bg])
    ax.scatter(scores_x[bg], scores_y[bg], c=angles, cmap="twilight",
               s=1.5, alpha=0.2, edgecolors="none", vmin=-np.pi, vmax=np.pi)

    # Reference circle
    ref_r = np.percentile(mags, 95)
    if ref_r > 1e-8:
        theta = np.linspace(0, 2 * np.pi, 80)
        ax.plot(ref_r * np.cos(theta), ref_r * np.sin(theta),
                color="#cccccc", linewidth=0.5, linestyle="--")

    # Top tokens
    top_idx = np.argsort(mags)[-N_LABEL:][::-1]
    top_tokens = ids_to_tokens(top_idx.tolist())

    ax.scatter(scores_x[top_idx], scores_y[top_idx],
               c="red", s=20, alpha=0.9, edgecolors="white", linewidths=0.3, zorder=5)

    placed = []
    for k, idx in enumerate(top_idx):
        x, y = scores_x[idx], scores_y[idx]
        label = safe_label(top_tokens[k])
        too_close = any(abs(x-px) < ref_r*0.1 and abs(y-py) < ref_r*0.1 for px, py in placed)
        if too_close and k > 4:
            continue
        ax.annotate(label, (x, y), fontsize=5, fontweight="bold",
                    xytext=(3, 3), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray",
                              alpha=0.8, linewidth=0.2), zorder=6)
        placed.append((x, y))

    ax.axhline(0, color="gray", linewidth=0.2, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.2, alpha=0.4)
    ax.set_xlabel(f"D{dir_i} (\u03c3={sigma_x:.3f})", fontsize=6)
    ax.set_ylabel(f"D{dir_j} (\u03c3={sigma_y:.3f})", fontsize=6)
    ax.tick_params(labelsize=5)


def plot_head_dirpairs(model_name, layer, head_idx, projs, sigmas, circuit_name,
                       mean_cos, frob, is_consensus=False):
    """Grid of all Di vs Dj pairs for one head, one circuit side."""
    n_dirs = len(projs)
    if n_dirs < 2:
        return

    # Upper triangle pairs
    pairs = [(i, j) for i in range(n_dirs) for j in range(i+1, n_dirs)]
    n_pairs = len(pairs)

    # Arrange in a grid
    n_cols = min(4, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (di, dj) in enumerate(pairs):
        r, c = divmod(idx, n_cols)
        scatter_pair(axes[r, c], projs[di], projs[dj],
                     sigmas[di], sigmas[dj], di, dj, circuit_name)

    # Hide unused axes
    for idx in range(n_pairs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    if is_consensus:
        label = f"CONSENSUS (mean|cos|={mean_cos:.3f})"
    else:
        label = f"H{head_idx} (mean|cos|={mean_cos:.3f}, frob={frob:.3f})"

    fig.suptitle(f"{model_name.upper()} L{layer} — {label}\n"
                 f"{circuit_name} direction pairs (D0..D{n_dirs-1})",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()

    if is_consensus:
        fname = f"{model_name}_L{layer}_consensus_{circuit_name}_dirpairs.png"
    else:
        fname = f"{model_name}_L{layer}_H{head_idx}_{circuit_name}_dirpairs.png"

    path = OUT_DIR / fname
    plt.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"      -> {path}")


def analyze_layer(model_dir, model_name, layer, embed, lm_head):
    """Extract directions and plot all direction pairs for outlier heads."""
    prefix = f"model.layers.{layer}.self_attn"

    qa = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    qb_base = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"{prefix}.q_b_proj.weight")
    o_base = load_weight(BASE_DIR, f"{prefix}.o_proj.weight")
    o_model = load_weight(model_dir, f"{prefix}.o_proj.weight")

    if any(w is None for w in [qa, qb_base, qb_model]):
        return

    qb_delta = qb_model - qb_base
    has_ov = o_base is not None and o_model is not None
    o_delta = (o_model - o_base) if has_ov else None
    del qb_base, qb_model, o_base, o_model

    # ── Extract per-head directions ──
    all_qk = []  # per head: {"projs": [...], "sigmas": [...], "frob": float}
    hidden_dirs = []  # for coherence computation

    for h in range(NUM_HEADS):
        d = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        frob = torch.norm(d).item()
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)

        projs = []
        sigmas = []
        for k in range(min(N_DIRS, len(S))):
            if S[k] < 1e-10:
                break
            dh = Vh[k] @ qa
            dh_norm = dh / (dh.norm() + 1e-12)
            projs.append((embed @ dh_norm).numpy())
            sigmas.append(S[k].item())

        # Dir0 in hidden space for coherence
        if S[0] > 1e-10:
            d0 = Vh[0] @ qa
            d0 = d0 / (d0.norm() + 1e-12)
        else:
            d0 = torch.zeros(qa.shape[1])
        hidden_dirs.append(d0)

        all_qk.append({"projs": projs, "sigmas": sigmas, "frob": frob})

    all_ov = []
    if has_ov:
        for h in range(NUM_HEADS):
            d = o_delta[:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM]
            frob = torch.norm(d).item()
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)

            projs = []
            sigmas = []
            for k in range(min(N_DIRS, len(S))):
                if S[k] < 1e-10:
                    break
                dh = U[:, k]
                dh = dh / (dh.norm() + 1e-12)
                projs.append((lm_head @ dh).numpy())
                sigmas.append(S[k].item())

            all_ov.append({"projs": projs, "sigmas": sigmas, "frob": frob})

    del qb_delta, o_delta, qa

    # Coherence
    hidden_dirs = torch.stack(hidden_dirs)
    cos_mat = torch.abs(hidden_dirs @ hidden_dirs.T).numpy()
    mean_cos = (cos_mat.sum(axis=1) - 1) / (NUM_HEADS - 1)
    del hidden_dirs

    outlier_order = np.argsort(mean_cos)
    consensus_heads = outlier_order[-20:]

    # ── Plot consensus ──
    # Average projections across consensus heads
    cons_projs = []
    cons_sigmas = []
    max_dirs = max(len(all_qk[h]["projs"]) for h in consensus_heads)
    for k in range(max_dirs):
        p_list = [all_qk[h]["projs"][k] for h in consensus_heads if len(all_qk[h]["projs"]) > k]
        s_list = [all_qk[h]["sigmas"][k] for h in consensus_heads if len(all_qk[h]["sigmas"]) > k]
        if p_list:
            cons_projs.append(np.mean(p_list, axis=0))
            cons_sigmas.append(np.mean(s_list))

    cons_cos = float(np.mean(mean_cos[consensus_heads]))
    plot_head_dirpairs(model_name, layer, -1, cons_projs, cons_sigmas, "QK",
                       cons_cos, 0, is_consensus=True)

    if all_ov:
        cons_ov_projs = []
        cons_ov_sigmas = []
        max_ov_dirs = max(len(all_ov[h]["projs"]) for h in consensus_heads)
        for k in range(max_ov_dirs):
            p_list = [all_ov[h]["projs"][k] for h in consensus_heads if len(all_ov[h]["projs"]) > k]
            s_list = [all_ov[h]["sigmas"][k] for h in consensus_heads if len(all_ov[h]["sigmas"]) > k]
            if p_list:
                cons_ov_projs.append(np.mean(p_list, axis=0))
                cons_ov_sigmas.append(np.mean(s_list))
        plot_head_dirpairs(model_name, layer, -1, cons_ov_projs, cons_ov_sigmas, "OV",
                           cons_cos, 0, is_consensus=True)

    # ── Plot outlier heads ──
    for rank in range(min(N_OUTLIERS, NUM_HEADS)):
        h = outlier_order[rank]
        h_cos = mean_cos[h]

        # QK
        if all_qk[h]["projs"]:
            plot_head_dirpairs(model_name, layer, h, all_qk[h]["projs"],
                               all_qk[h]["sigmas"], "QK", h_cos, all_qk[h]["frob"])

        # OV
        if all_ov and all_ov[h]["projs"]:
            plot_head_dirpairs(model_name, layer, h, all_ov[h]["projs"],
                               all_ov[h]["sigmas"], "OV", h_cos, all_ov[h]["frob"])

    gc.collect()


def get_available_layers(model_dir):
    base_idx = _get_index(str(BASE_DIR))
    model_idx = _get_index(str(model_dir))
    base_shards = set(p.name for p in BASE_DIR.glob("model-*.safetensors"))
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
    return layers


AUTO_LAYERS = {
    "m1": [0, 4, 53, 59, 60],
    "m2": [0, 36, 53, 59, 60],
    "m3": [0, 4, 35, 36, 59, 60],
}


def main():
    global N_OUTLIERS

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(ALL_MODELS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--n-outliers", type=int, default=N_OUTLIERS)
    args = parser.parse_args()

    N_OUTLIERS = args.n_outliers

    if args.all_models:
        models = list(ALL_MODELS.keys())
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --model or --all-models")

    print("Loading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed
    print(f"  embed: {embed.shape}, lm_head: {lm_head.shape}")

    for model_name in models:
        model_dir = ALL_MODELS[model_name]

        if args.all_layers:
            layers = get_available_layers(model_dir)
        elif args.layers:
            layers = args.layers
        elif args.auto:
            layers = AUTO_LAYERS.get(model_name, [0, 4, 59])
        else:
            layers = get_available_layers(model_dir)

        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()} — {len(layers)} layers, {N_OUTLIERS} outliers each")
        print(f"Plots per layer: (1 consensus + {N_OUTLIERS} outliers) x 2 circuits = {2*(1+N_OUTLIERS)}")
        print(f"Total plots: ~{len(layers) * 2 * (1 + N_OUTLIERS)}")
        print(f"{'='*60}")

        for layer in layers:
            print(f"\n  Layer {layer}:", flush=True)
            try:
                analyze_layer(model_dir, model_name, layer, embed, lm_head)
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        _shard_cache.clear()
        gc.collect()


if __name__ == "__main__":
    main()
