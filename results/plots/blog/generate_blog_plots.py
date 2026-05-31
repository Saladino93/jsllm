#!/usr/bin/env python3
"""Generate blog-quality static PNG plots for QK and OV circuit analysis.

Produces 6 plots in plots/blog/:
  1. blog_ov_circuit_xmodel_L52.png  — M1|M2|M3 OV circuit Dir0 vs Dir1, L52
  2. blog_ov_circuit_xmodel_L55.png  — Same for L55
  3. blog_ov_circuit_xmodel_L60.png  — Same for L60
  4. blog_ov_vs_oproj_M2_L52.png     — o_proj-only vs OV circuit for M2 at L52
  5. blog_qk_xmodel_L0.png           — M1|M2|M3 QK circuit at L0
  6. blog_qk_xmodel_L7.png           — M1|M2|M3 QK circuit at L7

Usage:
    python3 plots/blog/generate_blog_plots.py
"""

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
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "Heiti SC", "DejaVu Sans"],
    "text.usetex": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ── Paths ─────────────────────────────────────────────────────────────────
SSD = Path("/Volumes/OmarWork/JSLLM")
BASE_DIR = SSD / "base"
ALL_MODELS = {"M1": SSD / "m1", "M2": SSD / "m2", "M3": SSD / "m3"}
OUT_DIR = Path("/Users/omard/Documents/projects/AI_projects/jsllm/plots/blog")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 128
N_TOP = 30
N_SCATTER = 1000

# MLA architecture constants
HIDDEN = 7168
NH = 128
QK_NOPE_DIM = 128
QK_ROPE_DIM = 64
V_HEAD_DIM = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
Q_HEAD_DIM = QK_NOPE_DIM + QK_ROPE_DIM   # 192
KV_HEAD_DIM = QK_NOPE_DIM + V_HEAD_DIM   # 256

MODEL_COLORS = {"M1": "#2196F3", "M2": "#FF9800", "M3": "#4CAF50"}
MODEL_MARKERS = {"M1": "o", "M2": "s", "M3": "D"}


# ── FP8 + loading ────────────────────────────────────────────────────────
def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            r0, r1 = i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE
            c0, c1 = j * BLOCK_SIZE, (j + 1) * BLOCK_SIZE
            w[r0:r1, c0:c1] *= s[i, j]
    return w


_shard_cache = {}
MAX_CACHE = 2


def _load_shard(path):
    if path not in _shard_cache:
        if len(_shard_cache) >= MAX_CACHE:
            oldest = next(iter(_shard_cache))
            del _shard_cache[oldest]
        _shard_cache[path] = st.load_file(path, device="cpu")
    return _shard_cache[path]


_index_cache = {}


def _get_index(d):
    d_str = str(d)
    if d_str not in _index_cache:
        with open(d / "model.safetensors.index.json") as f:
            _index_cache[d_str] = json.load(f)
    return _index_cache[d_str]


def load_weight(model_dir, name):
    idx = _get_index(model_dir)
    if name not in idx["weight_map"]:
        return None
    shard = idx["weight_map"][name]
    shard_path = model_dir / shard
    if not shard_path.exists():
        return None
    t = _load_shard(str(shard_path))
    w = t[name]
    sn = name.replace(".weight", ".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss == shard else _load_shard(str(model_dir / ss)))[sn]
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


NSFW_WORDS = {'fuck','shit','ass','dick','porn','sex','cum','cock','pussy','slut','whore','bitch',
              'nigger','faggot','fucking','fucked','sperm','penis','vagina','anal','orgasm','erotic',
              'hentai','nude','naked','boob','tit','cunt','nipple','ejac','masturbat','genital'}

def safe_label(s):
    """Make token string safe for matplotlib labels. Filter NSFW."""
    if any(w in s.lower() for w in NSFW_WORDS):
        return ""
    s = s.replace("$", "\\$").replace("\n", "\\n").replace("\t", "\\t")
    if len(s) > 18:
        s = s[:16] + ".."
    return s


# ── Shared resources ──────────────────────────────────────────────────────
print("Loading shared resources...")
lm_head = load_weight(BASE_DIR, "lm_head.weight")
print(f"  lm_head: {lm_head.shape}")

_shard_cache.clear()

embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
print(f"  embed: {embed.shape}")

_shard_cache.clear()


# ── OV circuit computation ────────────────────────────────────────────────
def compute_ov_circuit(model_dir, layer, n_dirs=2):
    """Compute OV circuit: delta_o_proj @ base_kv_b_proj.T -> SVD.

    Returns dict with dir0, dir1 scores projected through lm_head (output)
    and embed (input).
    """
    prefix = f"model.layers.{layer}.self_attn"

    o_model = load_weight(model_dir, f"{prefix}.o_proj.weight")
    o_base = load_weight(BASE_DIR, f"{prefix}.o_proj.weight")
    kv_b_base = load_weight(BASE_DIR, f"{prefix}.kv_b_proj.weight")
    kv_a_base = load_weight(BASE_DIR, f"{prefix}.kv_a_proj_with_mqa.weight")

    if any(w is None for w in [o_model, o_base, kv_b_base, kv_a_base]):
        return None

    delta_o = o_model - o_base  # (7168, NH*V_HEAD_DIM)
    del o_model, o_base

    # Find the top head by norm
    head_norms = []
    for h in range(NH):
        delta_o_h = delta_o[:, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM]
        head_norms.append(torch.norm(delta_o_h).item())

    top_head = int(np.argmax(head_norms))

    # Compose OV circuit for top head
    delta_o_h = delta_o[:, top_head * V_HEAD_DIM:(top_head + 1) * V_HEAD_DIM]
    kv_b_v_h = kv_b_base[
        top_head * KV_HEAD_DIM + QK_NOPE_DIM:(top_head + 1) * KV_HEAD_DIM, :
    ]
    ov_composed = delta_o_h @ kv_b_v_h  # (7168, 512)

    kv_a_nope = kv_a_base[:KV_LORA_RANK, :]  # (512, 7168)

    U, S, Vh = torch.linalg.svd(ov_composed, full_matrices=False)

    result = {"head": top_head, "head_norm": head_norms[top_head], "dirs": []}

    for d in range(min(n_dirs, len(S))):
        if S[d] < 1e-7:
            break

        # OUTPUT: U[:,d] through lm_head
        u_dir = U[:, d]
        out_scores = (lm_head @ u_dir).numpy()

        # INPUT: Vh[d,:] through kv_a_nope then embed
        vh_dir = Vh[d, :]
        hidden_dir = vh_dir @ kv_a_nope
        in_scores = (embed @ hidden_dir).numpy()

        # Get top N for labels
        out_top_idx = np.argsort(np.abs(out_scores))[-N_TOP:][::-1]
        in_top_idx = np.argsort(np.abs(in_scores))[-N_TOP:][::-1]

        result["dirs"].append({
            "sigma": S[d].item(),
            "out_scores": out_scores,
            "in_scores": in_scores,
            "out_top_idx": out_top_idx,
            "in_top_idx": in_top_idx,
            "out_top_tokens": ids_to_tokens(out_top_idx.tolist()),
            "in_top_tokens": ids_to_tokens(in_top_idx.tolist()),
        })

    del delta_o, kv_b_base, kv_a_base, ov_composed, U, S, Vh
    gc.collect()
    return result


def compute_oproj_only(model_dir, layer, n_dirs=2):
    """Compute o_proj-only SVD (no KV composition) for comparison."""
    prefix = f"model.layers.{layer}.self_attn"

    o_model = load_weight(model_dir, f"{prefix}.o_proj.weight")
    o_base = load_weight(BASE_DIR, f"{prefix}.o_proj.weight")

    if any(w is None for w in [o_model, o_base]):
        return None

    delta_o = o_model - o_base  # (7168, NH*V_HEAD_DIM)
    del o_model, o_base

    U, S, Vh = torch.linalg.svd(delta_o, full_matrices=False)

    result = {"dirs": []}
    for d in range(min(n_dirs, len(S))):
        if S[d] < 1e-7:
            break

        u_dir = U[:, d]
        out_scores = (lm_head @ u_dir).numpy()

        vh_dir = Vh[d, :]
        # For o_proj-only, Vh is in the concatenated V_HEAD_DIM space.
        # We can still project U through lm_head for output tokens.
        # For input, Vh is in the o_proj column space -- not directly embed-able
        # but we use it as a contrast to show the difference.
        in_scores = out_scores  # placeholder -- use output for both axes

        # Actually: project U through embed for "input" interpretation
        in_scores = (embed @ u_dir).numpy()

        out_top_idx = np.argsort(np.abs(out_scores))[-N_TOP:][::-1]
        in_top_idx = np.argsort(np.abs(in_scores))[-N_TOP:][::-1]

        result["dirs"].append({
            "sigma": S[d].item(),
            "out_scores": out_scores,
            "in_scores": in_scores,
            "out_top_idx": out_top_idx,
            "in_top_idx": in_top_idx,
            "out_top_tokens": ids_to_tokens(out_top_idx.tolist()),
            "in_top_tokens": ids_to_tokens(in_top_idx.tolist()),
        })

    del delta_o, U, S, Vh
    gc.collect()
    return result


# ── QK circuit computation ────────────────────────────────────────────────
def compute_qk_circuit(model_dir, layer, n_dirs=2):
    """Compute QK circuit: composed q_b[head] @ q_a -> SVD -> project through embed.

    Returns per-head results for the top head, plus an aggregate across heads.
    """
    prefix = f"model.layers.{layer}.self_attn"

    qa_model = load_weight(model_dir, f"{prefix}.q_a_proj.weight")
    qa_base = load_weight(BASE_DIR, f"{prefix}.q_a_proj.weight")
    qb_model = load_weight(model_dir, f"{prefix}.q_b_proj.weight")
    qb_base = load_weight(BASE_DIR, f"{prefix}.q_b_proj.weight")

    if any(w is None for w in [qa_model, qa_base, qb_model, qb_base]):
        return None

    delta_qa = qa_model - qa_base
    delta_qb = qb_model - qb_base

    # Find top head by delta_qb norm
    head_norms = []
    for h in range(NH):
        delta_qb_h = delta_qb[h * Q_HEAD_DIM:h * Q_HEAD_DIM + QK_NOPE_DIM, :]
        head_norms.append(torch.norm(delta_qb_h).item())

    top_head = int(np.argmax(head_norms))

    # Composed delta query for top head
    delta_qb_h_nope = delta_qb[
        top_head * Q_HEAD_DIM:top_head * Q_HEAD_DIM + QK_NOPE_DIM, :
    ]
    qb_base_h_nope = qb_base[
        top_head * Q_HEAD_DIM:top_head * Q_HEAD_DIM + QK_NOPE_DIM, :
    ]
    delta_query_h = delta_qb_h_nope @ qa_model + qb_base_h_nope @ delta_qa  # (128, 7168)

    U_q, S_q, Vh_q = torch.linalg.svd(delta_query_h, full_matrices=False)

    result = {"head": top_head, "head_norm": head_norms[top_head], "dirs": []}

    for d in range(min(n_dirs, len(S_q))):
        if S_q[d] < 1e-7:
            break

        # Vh_q[d,:] in hidden space -> embed -> what query reads from input
        input_dir = Vh_q[d, :]
        in_scores = (embed @ input_dir).numpy()

        in_top_idx = np.argsort(np.abs(in_scores))[-N_TOP:][::-1]

        result["dirs"].append({
            "sigma": S_q[d].item(),
            "in_scores": in_scores,
            "in_top_idx": in_top_idx,
            "in_top_tokens": ids_to_tokens(in_top_idx.tolist()),
        })

    del delta_qa, delta_qb, qa_model, qa_base, qb_model, qb_base
    del delta_query_h, U_q, S_q, Vh_q
    gc.collect()
    return result


# ── Plotting helpers ──────────────────────────────────────────────────────
def plot_scatter_2dir(ax, dir0_scores, dir1_scores, top_idx, top_tokens,
                      title, color, n_scatter=N_SCATTER, n_label=N_TOP):
    """Scatter plot of Dir0 vs Dir1 scores with labeled top tokens."""
    # Sample random subset for background scatter
    n_total = len(dir0_scores)
    if n_total > n_scatter:
        rng = np.random.RandomState(42)
        bg_idx = rng.choice(n_total, n_scatter, replace=False)
    else:
        bg_idx = np.arange(n_total)

    d0_bg = dir0_scores[bg_idx]
    d1_bg = dir1_scores[bg_idx]

    # Reference circle at the 95th percentile magnitude
    mags = np.sqrt(dir0_scores**2 + dir1_scores**2)
    ref_r = np.percentile(mags, 95)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(ref_r * np.cos(theta), ref_r * np.sin(theta),
            color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)

    # Colormap based on angle from origin
    angles = np.arctan2(d1_bg, d0_bg)
    ax.scatter(d0_bg, d1_bg, c=angles, cmap="RdBu_r", s=3, alpha=0.35,
               edgecolors="none", zorder=2, vmin=-np.pi, vmax=np.pi)

    # Label top tokens
    d0_top = dir0_scores[top_idx[:n_label]]
    d1_top = dir1_scores[top_idx[:n_label]]

    ax.scatter(d0_top, d1_top, c=color, s=18, alpha=0.9,
               edgecolors="white", linewidths=0.4, zorder=4)

    # Labeling with jitter to reduce overlap
    placed = []
    for i in range(min(n_label, len(top_idx))):
        x, y = d0_top[i], d1_top[i]
        label = safe_label(top_tokens[i])

        # Skip if too close to an already placed label
        too_close = False
        for px, py in placed:
            if abs(x - px) < ref_r * 0.06 and abs(y - py) < ref_r * 0.06:
                too_close = True
                break
        if too_close and i > 15:
            continue

        placed.append((x, y))
        ax.annotate(
            label, (x, y), fontsize=5.5, fontweight="bold",
            xytext=(3, 3), textcoords="offset points",
            color="#222222", zorder=5,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
        )

    ax.axhline(0, color="#aaaaaa", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#aaaaaa", linewidth=0.4, zorder=0)
    ax.set_xlabel("SVD Direction 0", fontsize=9)
    ax.set_ylabel("SVD Direction 1", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)


# ── Plot 1-3: OV circuit cross-model ─────────────────────────────────────
def plot_ov_xmodel(layer, filename):
    """3-panel cross-model OV circuit scatter: M1|M2|M3."""
    print(f"\n=== OV cross-model L{layer} ===")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for i, (mname, mdir) in enumerate(ALL_MODELS.items()):
        print(f"  Computing OV circuit for {mname} L{layer}...", flush=True)
        _shard_cache.clear()
        res = compute_ov_circuit(mdir, layer, n_dirs=2)
        if res is None or len(res["dirs"]) < 2:
            axes[i].text(0.5, 0.5, f"{mname}: no data", transform=axes[i].transAxes,
                         ha="center", va="center", fontsize=14)
            axes[i].set_title(f"{mname} L{layer} OV Circuit")
            continue

        d0 = res["dirs"][0]
        d1 = res["dirs"][1]

        # Use output scores (lm_head projection) for both directions
        out0 = d0["out_scores"]
        out1 = d1["out_scores"]

        # Merge top indices from both directions
        combined_top = list(d0["out_top_idx"][:N_TOP])
        for idx in d1["out_top_idx"][:N_TOP]:
            if idx not in combined_top:
                combined_top.append(idx)
        combined_top = np.array(combined_top[:N_TOP])
        combined_tokens = ids_to_tokens(combined_top.tolist())

        title = (f"{mname} OV Circuit  L{layer} H{res['head']}\n"
                 f"\u03c3\u2080={d0['sigma']:.1f}  \u03c3\u2081={d1['sigma']:.1f}")
        plot_scatter_2dir(axes[i], out0, out1, combined_top, combined_tokens,
                          title, MODEL_COLORS[mname])

    fig.suptitle(f"OV Circuit: Output Token Directions (Layer {layer})",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    gc.collect()


# ── Plot 4: OV vs o_proj-only ────────────────────────────────────────────
def plot_ov_vs_oproj(model_name, model_dir, layer, filename):
    """2-panel comparison: o_proj-only vs OV circuit for one model."""
    print(f"\n=== OV vs o_proj-only {model_name} L{layer} ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # LEFT: o_proj-only
    print(f"  Computing o_proj-only for {model_name} L{layer}...", flush=True)
    _shard_cache.clear()
    oproj = compute_oproj_only(model_dir, layer, n_dirs=2)
    if oproj is not None and len(oproj["dirs"]) >= 2:
        d0 = oproj["dirs"][0]
        d1 = oproj["dirs"][1]
        combined_top = list(d0["out_top_idx"][:N_TOP])
        for idx in d1["out_top_idx"][:N_TOP]:
            if idx not in combined_top:
                combined_top.append(idx)
        combined_top = np.array(combined_top[:N_TOP])
        combined_tokens = ids_to_tokens(combined_top.tolist())

        title = (f"{model_name} o_proj-only SVD  L{layer}\n"
                 f"\u03c3\u2080={d0['sigma']:.1f}  \u03c3\u2081={d1['sigma']:.1f}")
        plot_scatter_2dir(axes[0], d0["out_scores"], d1["out_scores"],
                          combined_top, combined_tokens, title, MODEL_COLORS[model_name])
    else:
        axes[0].text(0.5, 0.5, "No data", transform=axes[0].transAxes,
                     ha="center", va="center", fontsize=14)
        axes[0].set_title(f"{model_name} o_proj-only L{layer}")

    # RIGHT: full OV circuit
    print(f"  Computing OV circuit for {model_name} L{layer}...", flush=True)
    _shard_cache.clear()
    ov = compute_ov_circuit(model_dir, layer, n_dirs=2)
    if ov is not None and len(ov["dirs"]) >= 2:
        d0 = ov["dirs"][0]
        d1 = ov["dirs"][1]
        combined_top = list(d0["out_top_idx"][:N_TOP])
        for idx in d1["out_top_idx"][:N_TOP]:
            if idx not in combined_top:
                combined_top.append(idx)
        combined_top = np.array(combined_top[:N_TOP])
        combined_tokens = ids_to_tokens(combined_top.tolist())

        title = (f"{model_name} OV Circuit  L{layer} H{ov['head']}\n"
                 f"\u03c3\u2080={d0['sigma']:.1f}  \u03c3\u2081={d1['sigma']:.1f}")
        plot_scatter_2dir(axes[1], d0["out_scores"], d1["out_scores"],
                          combined_top, combined_tokens, title, MODEL_COLORS[model_name])
    else:
        axes[1].text(0.5, 0.5, "No data", transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=14)
        axes[1].set_title(f"{model_name} OV Circuit L{layer}")

    fig.suptitle(f"o_proj-only vs OV Circuit: {model_name} Layer {layer}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    gc.collect()


# ── Plot 5-6: QK circuit cross-model ─────────────────────────────────────
def plot_qk_xmodel(layer, filename):
    """3-panel cross-model QK circuit scatter: M1|M2|M3."""
    print(f"\n=== QK cross-model L{layer} ===")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for i, (mname, mdir) in enumerate(ALL_MODELS.items()):
        print(f"  Computing QK circuit for {mname} L{layer}...", flush=True)
        _shard_cache.clear()
        res = compute_qk_circuit(mdir, layer, n_dirs=2)
        if res is None or len(res["dirs"]) < 2:
            axes[i].text(0.5, 0.5, f"{mname}: no data", transform=axes[i].transAxes,
                         ha="center", va="center", fontsize=14)
            axes[i].set_title(f"{mname} L{layer} QK Circuit")
            continue

        d0 = res["dirs"][0]
        d1 = res["dirs"][1]

        # Use input scores (embed projection) for both directions
        in0 = d0["in_scores"]
        in1 = d1["in_scores"]

        combined_top = list(d0["in_top_idx"][:N_TOP])
        for idx in d1["in_top_idx"][:N_TOP]:
            if idx not in combined_top:
                combined_top.append(idx)
        combined_top = np.array(combined_top[:N_TOP])
        combined_tokens = ids_to_tokens(combined_top.tolist())

        title = (f"{mname} QK Circuit  L{layer} H{res['head']}\n"
                 f"\u03c3\u2080={d0['sigma']:.1f}  \u03c3\u2081={d1['sigma']:.1f}")
        plot_scatter_2dir(axes[i], in0, in1, combined_top, combined_tokens,
                          title, MODEL_COLORS[mname])

    fig.suptitle(f"QK Circuit: Query Token Detection (Layer {layer})",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    gc.collect()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("Generating blog-quality QK/OV circuit plots")
    print(f"Output: {OUT_DIR}\n")

    # 1-3: OV circuit cross-model
    for layer, fname in [(52, "blog_ov_circuit_xmodel_L52.png"),
                         (55, "blog_ov_circuit_xmodel_L55.png"),
                         (60, "blog_ov_circuit_xmodel_L60.png")]:
        plot_ov_xmodel(layer, fname)

    # 4: OV vs o_proj-only for M2 at L52
    plot_ov_vs_oproj("M2", ALL_MODELS["M2"], 52, "blog_ov_vs_oproj_M2_L52.png")

    # 5-6: QK circuit cross-model
    for layer, fname in [(0, "blog_qk_xmodel_L0.png"),
                         (7, "blog_qk_xmodel_L7.png")]:
        plot_qk_xmodel(layer, fname)

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
