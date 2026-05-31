#!/usr/bin/env python3
"""Visual plots for outlier heads: Dir0 vs Dir1 scatter for QK and OV circuits.

For each layer's most divergent heads (+ consensus for comparison):
  - QK scatter: project tokens onto head's Dir0 × Dir1 through embed
  - OV scatter: project tokens onto head's Dir0 × Dir1 through lm_head
  - Top tokens labeled, color-coded by magnitude

Output: multi-panel PNGs in head_coherence/outlier_plots/

Usage:
    python experiments/head_outlier_plots.py --model m3 --layers 4 59
    python experiments/head_outlier_plots.py --all-models --auto
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
from matplotlib.colors import Normalize

# CJK font support for Chinese/Japanese/Korean token labels
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SSD = Path("/Volumes/OmarWork/JSLLM")
OUT_DIR = Path("experiments/EXP-016_cross_layer_story/head_coherence/outlier_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {"m1": SSD / "m1", "m2": SSD / "m2", "m3": SSD / "m3"}
BASE_DIR = SSD / "base"

NUM_HEADS = 128
Q_HEAD_DIM = 192
QK_NOPE_DIM = 128
V_HEAD_DIM = 128
BLOCK_SIZE = 128
N_LABEL = 15       # tokens to label on plot
N_SCATTER = 2000   # background scatter sample
N_OUTLIERS = 5     # heads to plot per layer (keep plots readable)
N_SVD_DIRS = 3     # extract top-3 directions per head


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

_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))
    return _tok

def ids_to_tokens(ids):
    return [tok().decode([i]) for i in ids]

def safe_label(token, max_len=18):
    s = repr(token).strip("'\"")
    # Escape $ to prevent matplotlib LaTeX parsing
    s = s.replace("$", r"\$")
    if len(s) > max_len:
        s = s[:max_len-1] + "\u2026"
    return s


# ── Per-head SVD extraction ──────────────────────────────────────────────
def extract_head_dirs(model_dir, layer, embed, lm_head):
    """Extract per-head QK and OV directions + project to token space."""
    prefix = f"model.layers.{layer}.self_attn"

    qa_model = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    qb_base = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")
    qb_model = load_weight(model_dir, f"{prefix}.q_b_proj.weight")
    o_base = load_weight(BASE_DIR, f"{prefix}.o_proj.weight")
    o_model = load_weight(model_dir, f"{prefix}.o_proj.weight")

    if any(w is None for w in [qa_model, qb_base, qb_model]):
        return None

    qb_delta = qb_model - qb_base
    has_ov = o_base is not None and o_model is not None
    o_delta = (o_model - o_base) if has_ov else None
    del qb_base, qb_model, o_base, o_model

    heads = []
    for h in range(NUM_HEADS):
        info = {"head": h}

        # QK circuit: Δq_b[h] → SVD → chain through q_a → project through embed
        d_qk = qb_delta[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        info["qk_frob"] = torch.norm(d_qk).item()
        U_qk, S_qk, Vh_qk = torch.linalg.svd(d_qk, full_matrices=False)

        qk_projs = []  # (n_dirs, vocab) — token projections per direction
        qk_sigmas = []
        for k in range(min(N_SVD_DIRS, len(S_qk))):
            if S_qk[k] < 1e-10:
                break
            dir_hidden = Vh_qk[k] @ qa_model  # (1536,) @ (1536,7168) → (7168,)
            dir_hidden = dir_hidden / (dir_hidden.norm() + 1e-12)
            scores = (embed @ dir_hidden).numpy()  # (vocab,)
            qk_projs.append(scores)
            qk_sigmas.append(S_qk[k].item())
        info["qk_projs"] = qk_projs
        info["qk_sigmas"] = qk_sigmas

        # OV circuit: Δo_proj[:, h*128:(h+1)*128] → SVD → project through lm_head
        if has_ov:
            d_ov = o_delta[:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM]
            info["ov_frob"] = torch.norm(d_ov).item()
            U_ov, S_ov, Vh_ov = torch.linalg.svd(d_ov, full_matrices=False)

            ov_projs = []
            ov_sigmas = []
            for k in range(min(N_SVD_DIRS, len(S_ov))):
                if S_ov[k] < 1e-10:
                    break
                dir_hidden = U_ov[:, k]  # already in hidden space
                dir_hidden = dir_hidden / (dir_hidden.norm() + 1e-12)
                scores = (lm_head @ dir_hidden).numpy()  # (vocab,)
                ov_projs.append(scores)
                ov_sigmas.append(S_ov[k].item())
            info["ov_projs"] = ov_projs
            info["ov_sigmas"] = ov_sigmas

        heads.append(info)

    del qb_delta, o_delta, qa_model
    gc.collect()

    # Compute coherence to find outliers
    # Use QK Dir0 directions in hidden space
    qk_dir0_vecs = []
    for h_info in heads:
        if h_info["qk_projs"]:
            qk_dir0_vecs.append(h_info["qk_projs"][0])
        else:
            qk_dir0_vecs.append(np.zeros(embed.shape[0]))

    # Compute coherence from token-space projections isn't right —
    # use hidden-space directions instead. Re-extract Dir0 in hidden space.
    qb_base2 = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")
    qb_model2 = load_weight(model_dir, f"{prefix}.q_b_proj.weight")
    qa_model2 = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    qb_delta2 = qb_model2 - qb_base2
    del qb_base2, qb_model2

    hidden_dirs = []
    for h in range(NUM_HEADS):
        d = qb_delta2[h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM, :]
        _, S, Vh = torch.linalg.svd(d, full_matrices=False)
        if S[0] > 1e-10:
            dh = Vh[0] @ qa_model2
            dh = dh / (dh.norm() + 1e-12)
        else:
            dh = torch.zeros(qa_model2.shape[1])
        hidden_dirs.append(dh)

    del qb_delta2, qa_model2
    hidden_dirs = torch.stack(hidden_dirs)
    cos_mat = torch.abs(hidden_dirs @ hidden_dirs.T).numpy()
    mean_cos = (cos_mat.sum(axis=1) - 1) / (NUM_HEADS - 1)

    gc.collect()
    return heads, mean_cos


def plot_scatter(ax, d0_scores, d1_scores, sigma0, sigma1,
                 title, circuit_label, n_scatter=N_SCATTER, n_label=N_LABEL):
    """Dir0 vs Dir1 scatter with labeled top tokens."""
    vocab = len(d0_scores)
    rng = np.random.RandomState(42)

    # Magnitude for ranking
    mags = np.sqrt(d0_scores**2 + d1_scores**2)

    # Background scatter (random sample)
    if vocab > n_scatter:
        bg_idx = rng.choice(vocab, n_scatter, replace=False)
    else:
        bg_idx = np.arange(vocab)

    # Color by angle from origin
    angles = np.arctan2(d1_scores[bg_idx], d0_scores[bg_idx])
    ax.scatter(d0_scores[bg_idx], d1_scores[bg_idx],
               c=angles, cmap="twilight", s=2, alpha=0.25,
               edgecolors="none", vmin=-np.pi, vmax=np.pi)

    # 95th percentile reference circle
    ref_r = np.percentile(mags, 95)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(ref_r * np.cos(theta), ref_r * np.sin(theta),
            color="#cccccc", linewidth=0.6, linestyle="--", zorder=1)

    # Top tokens by magnitude
    top_idx = np.argsort(mags)[-n_label:][::-1]
    top_tokens = ids_to_tokens(top_idx.tolist())

    ax.scatter(d0_scores[top_idx], d1_scores[top_idx],
               c="red", s=25, alpha=0.9, edgecolors="white", linewidths=0.4, zorder=5)

    # Label with collision avoidance
    placed = []
    for i, idx in enumerate(top_idx):
        x, y = d0_scores[idx], d1_scores[idx]
        label = safe_label(top_tokens[i])

        # Skip if too close to existing label
        too_close = False
        for px, py in placed:
            if abs(x - px) < ref_r * 0.08 and abs(y - py) < ref_r * 0.08:
                too_close = True
                break
        if too_close and i > 5:
            continue

        ax.annotate(label, (x, y), fontsize=5.5, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray",
                              alpha=0.85, linewidth=0.3),
                    zorder=6)
        placed.append((x, y))

    ax.axhline(0, color="gray", linewidth=0.3, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.3, alpha=0.5)
    ax.set_xlabel(f"Dir 0 (\u03c3={sigma0:.3f})", fontsize=8)
    ax.set_ylabel(f"Dir 1 (\u03c3={sigma1:.3f})", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)


def plot_layer(model_name, layer, heads, mean_cos):
    """Generate multi-panel figure for one layer's outlier heads."""
    outlier_order = np.argsort(mean_cos)
    consensus_heads = outlier_order[-20:]
    n_outliers = min(N_OUTLIERS, NUM_HEADS)
    outlier_indices = outlier_order[:n_outliers]

    # Determine which circuits we have
    has_ov = "ov_projs" in heads[0] and len(heads[0].get("ov_projs", [])) >= 2

    # Rows: consensus + N outliers. Cols: QK scatter | OV scatter (if available)
    n_rows = 1 + n_outliers
    n_cols = 2 if has_ov else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # ── Row 0: Consensus ──
    # Average Dir0/Dir1 projections across consensus heads
    cons_qk_d0 = np.mean([heads[h]["qk_projs"][0] for h in consensus_heads
                          if len(heads[h]["qk_projs"]) >= 1], axis=0)
    cons_qk_d1_list = [heads[h]["qk_projs"][1] for h in consensus_heads
                       if len(heads[h]["qk_projs"]) >= 2]
    if cons_qk_d1_list:
        cons_qk_d1 = np.mean(cons_qk_d1_list, axis=0)
    else:
        cons_qk_d1 = np.zeros_like(cons_qk_d0)

    cons_s0 = np.mean([heads[h]["qk_sigmas"][0] for h in consensus_heads
                       if heads[h]["qk_sigmas"]])
    cons_s1 = np.mean([heads[h]["qk_sigmas"][1] for h in consensus_heads
                       if len(heads[h]["qk_sigmas"]) >= 2]) if cons_qk_d1_list else 0

    plot_scatter(axes[0, 0], cons_qk_d0, cons_qk_d1, cons_s0, cons_s1,
                 f"CONSENSUS (top-20 heads, mean|cos|={np.mean(mean_cos[consensus_heads]):.3f})\n"
                 f"QK INPUT: what the majority listens to",
                 "QK")

    if has_ov:
        cons_ov_d0 = np.mean([heads[h]["ov_projs"][0] for h in consensus_heads
                              if len(heads[h].get("ov_projs", [])) >= 1], axis=0)
        cons_ov_d1_list = [heads[h]["ov_projs"][1] for h in consensus_heads
                           if len(heads[h].get("ov_projs", [])) >= 2]
        if cons_ov_d1_list:
            cons_ov_d1 = np.mean(cons_ov_d1_list, axis=0)
        else:
            cons_ov_d1 = np.zeros_like(cons_ov_d0)

        cons_ov_s0 = np.mean([heads[h]["ov_sigmas"][0] for h in consensus_heads
                              if heads[h].get("ov_sigmas")])
        cons_ov_s1 = np.mean([heads[h]["ov_sigmas"][1] for h in consensus_heads
                              if len(heads[h].get("ov_sigmas", [])) >= 2]) if cons_ov_d1_list else 0

        plot_scatter(axes[0, 1], cons_ov_d0, cons_ov_d1, cons_ov_s0, cons_ov_s1,
                     f"CONSENSUS\n"
                     f"OV OUTPUT: what the majority writes",
                     "OV")

    # ── Rows 1..N: Outlier heads ──
    for row, h_idx in enumerate(outlier_indices, start=1):
        h = heads[h_idx]
        h_cos = mean_cos[h_idx]

        # QK scatter
        if len(h["qk_projs"]) >= 2:
            plot_scatter(axes[row, 0], h["qk_projs"][0], h["qk_projs"][1],
                         h["qk_sigmas"][0], h["qk_sigmas"][1],
                         f"OUTLIER H{h_idx} (mean|cos|={h_cos:.3f}, frob={h['qk_frob']:.3f})\n"
                         f"QK INPUT: what H{h_idx} listens to",
                         "QK")
        elif len(h["qk_projs"]) == 1:
            # Only Dir0 — plot histogram instead
            axes[row, 0].hist(h["qk_projs"][0], bins=100, alpha=0.7, color="steelblue")
            axes[row, 0].set_title(f"H{h_idx} QK Dir0 only (rank-1)", fontsize=9)

        # OV scatter
        if has_ov and len(h.get("ov_projs", [])) >= 2:
            plot_scatter(axes[row, 1], h["ov_projs"][0], h["ov_projs"][1],
                         h["ov_sigmas"][0], h["ov_sigmas"][1],
                         f"H{h_idx} (ov_frob={h.get('ov_frob', 0):.3f})\n"
                         f"OV OUTPUT: what H{h_idx} writes",
                         "OV")
        elif has_ov:
            axes[row, 1].text(0.5, 0.5, f"H{h_idx}: no OV Dir1",
                              ha="center", va="center", transform=axes[row, 1].transAxes)
            axes[row, 1].set_title(f"H{h_idx} OV (insufficient rank)", fontsize=9)

    fig.suptitle(f"{model_name.upper()} Layer {layer} \u2014 Outlier Head Token Projections\n"
                 f"Dir 0 vs Dir 1 for QK (input/listening) and OV (output/writing)",
                 fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    path = OUT_DIR / f"{model_name}_L{layer}_outlier_scatter.png"
    plt.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"    -> {path}")


# ── Auto layers ──────────────────────────────────────────────────────────
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
        layers = args.layers if args.layers else AUTO_LAYERS.get(model_name, [0, 4, 59])

        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()} — Layers: {layers}")
        print(f"{'='*60}")

        for layer in layers:
            print(f"\n  Layer {layer}:", flush=True)
            try:
                result = extract_head_dirs(model_dir, layer, embed, lm_head)
                if result is None:
                    print(f"    SKIP: weights not available")
                    continue
                heads, mean_cos = result
                plot_layer(model_name, layer, heads, mean_cos)
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        _shard_cache.clear()
        gc.collect()


if __name__ == "__main__":
    main()
