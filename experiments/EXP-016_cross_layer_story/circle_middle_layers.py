#!/usr/bin/env python3
"""Circle plots for MIDDLE layers: L21, L28-32, L39-40, L47-50.

For each layer x {q_a_proj, o_proj}:
  - Dir 1 vs Dir 2 scatter (1000 tokens), colored by Dir 0 (RdBu)
  - Top 30 tokens labeled
  - Reference circle at median radius
  - Equal aspect ratio
  - Title with sigma_1, sigma_2, sigma_1/sigma_2 ratio

Cross-model comparison (M1|M2|M3) for L28, L32, L47, L50.

Usage:
    python3 experiments/EXP-016_cross_layer_story/circle_middle_layers.py
"""

import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import safetensors.torch as st

warnings.filterwarnings("ignore")

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
OUT_DIR = EXP / "plots" / "circles" / "middle"
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128

MIDDLE_LAYERS = [21, 28, 29, 30, 31, 32, 39, 40, 47, 48, 49, 50]
COMPONENTS = ["q_a_proj", "o_proj"]
CROSS_MODEL_LAYERS = [28, 32, 47, 50]

MODEL_DIRS = {
    "M1": SSD / "m1",
    "M2": SSD / "m2",
    "M3": SSD / "m3",
}

TOP_N = 1000
TOP_LABEL = 30


# ── Weight loading ──────────────────────────────────────────────────

def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE,
              j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE] *= s[i, j]
    return w


_shard_cache = {}

def _load_shard(path):
    path = str(path)
    if path not in _shard_cache:
        if len(_shard_cache) >= 2:
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


# ── Tokenizer ───────────────────────────────────────────────────────

_tok = None

def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))
    return _tok


def decode_token(i):
    t = tok().decode([i])
    t = t.replace('\n', '\\n').replace('\t', '\\t')
    if len(t) > 15:
        t = t[:12] + '...'
    return t


# ── SVD + projection helper ────────────────────────────────────────

def compute_projections(model_dir, layer, comp, embed):
    """Return (sigmas[:3], proj_weighted) or None if layer missing.

    For q_a_proj (shape [small, 7168]): Vh rows live in the 7168 input space,
      so we project embeddings onto Vh directions.
    For o_proj (shape [7168, large]): U columns live in the 7168 output space,
      so we project embeddings onto U directions.
    """
    name = f"model.layers.{layer}.self_attn.{comp}.weight"
    w_base = load_weight(BASE_DIR, name)
    w_model = load_weight(model_dir, name)
    if w_base is None or w_model is None:
        clear_cache()
        return None
    delta = w_model - w_base
    del w_base, w_model
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta
    sigmas = S[:3].numpy()

    # Pick the directions that live in 7168-dim (embedding) space
    if comp == "o_proj":
        # o_proj: (7168, large) -> U[:, :3] has shape (7168, 3)
        dirs = U[:, :3]  # (7168, 3)
        proj = (embed @ dirs).numpy()  # (vocab, 3)
    else:
        # q_a_proj etc: (small, 7168) -> Vh[:3] has shape (3, 7168)
        dirs = Vh[:3]  # (3, 7168)
        proj = (embed @ dirs.T).numpy()  # (vocab, 3)

    proj_w = proj * sigmas
    del U, S, Vh, dirs, proj
    gc.collect()
    clear_cache()
    return sigmas, proj_w


# ── Single-model circle plot ───────────────────────────────────────

def make_circle_plot(sigmas, proj_w, layer, comp, model_name="M3"):
    """Create a plotly figure: Dir1 vs Dir2, colored by Dir0."""
    import plotly.graph_objects as go

    # Select top tokens by Dir 1-2 distance from origin
    d12_norm = np.sqrt(proj_w[:, 1] ** 2 + proj_w[:, 2] ** 2)
    top_idx = np.argsort(d12_norm)[-TOP_N:][::-1]

    x = proj_w[top_idx, 1]
    y = proj_w[top_idx, 2]
    d0 = proj_w[top_idx, 0]

    # Labels
    labels_short = [decode_token(i) for i in top_idx]
    labels_full = []
    for i, idx in enumerate(top_idx):
        t = tok().decode([idx]).replace('\n', '\\n').replace('\t', '\\t')
        labels_full.append(
            f"#{idx} {repr(t)}<br>"
            f"d0={d0[i]:+.4f} d1={x[i]:+.4f} d2={y[i]:+.4f}<br>"
            f"norm={d12_norm[idx]:.4f}"
        )

    show_text = [labels_short[i] if i < TOP_LABEL else '' for i in range(len(labels_short))]

    ratio = sigmas[1] / sigmas[2] if sigmas[2] > 0 else float('inf')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text',
        marker=dict(
            size=4,
            color=d0,
            colorscale='RdBu',
            cmid=0,
            colorbar=dict(title='Dir 0'),
            line=dict(width=0.3, color='black'),
            opacity=0.7,
        ),
        text=show_text,
        textposition='top center',
        textfont=dict(size=7),
        hovertext=labels_full,
        hoverinfo='text',
    ))

    # Origin marker
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers',
        marker=dict(size=10, color='black', symbol='x'),
        showlegend=False,
    ))

    # Reference circle at median radius
    median_r = np.median(d12_norm[top_idx])
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=median_r * np.cos(theta),
        y=median_r * np.sin(theta),
        mode='lines',
        line=dict(color='gray', dash='dot', width=1),
        showlegend=False,
    ))

    key = f"L{layer}_{comp}"
    fig.update_layout(
        title=(
            f"{model_name} -- {key}: Dir 1 vs Dir 2 (colored by Dir 0)<br>"
            f"<sub>sigma_1={sigmas[1]:.4f}, sigma_2={sigmas[2]:.4f}, "
            f"sigma_1/sigma_2={ratio:.2f}</sub>"
        ),
        xaxis_title=f"Dir 1 (sigma={sigmas[1]:.4f})",
        yaxis_title=f"Dir 2 (sigma={sigmas[2]:.4f})",
        width=1000, height=800,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        showlegend=False,
    )

    return fig


# ── Cross-model comparison plot ─────────────────────────────────────

def make_cross_model_plot(embed, layer, comp):
    """Side-by-side M1|M2|M3 for one layer x component."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["M1", "M2", "M3"],
        horizontal_spacing=0.06,
    )

    for col_i, (mname, mdir) in enumerate(MODEL_DIRS.items(), 1):
        result = compute_projections(mdir, layer, comp, embed)
        if result is None:
            print(f"    {mname} L{layer} {comp}: SKIP (missing)", flush=True)
            continue
        sigmas, proj_w = result

        d12_norm = np.sqrt(proj_w[:, 1] ** 2 + proj_w[:, 2] ** 2)
        top_idx = np.argsort(d12_norm)[-TOP_N:][::-1]

        x = proj_w[top_idx, 1]
        y = proj_w[top_idx, 2]
        d0 = proj_w[top_idx, 0]

        labels_short = [decode_token(i) for i in top_idx]
        show_text = [labels_short[i] if i < 20 else '' for i in range(len(labels_short))]

        labels_full = []
        for i, idx in enumerate(top_idx):
            t = tok().decode([idx]).replace('\n', '\\n').replace('\t', '\\t')
            labels_full.append(
                f"#{idx} {repr(t)}<br>"
                f"d0={d0[i]:+.4f} d1={x[i]:+.4f} d2={y[i]:+.4f}"
            )

        ratio = sigmas[1] / sigmas[2] if sigmas[2] > 0 else float('inf')

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            marker=dict(
                size=3,
                color=d0,
                colorscale='RdBu',
                cmid=0,
                showscale=(col_i == 3),
                colorbar=dict(title='Dir 0') if col_i == 3 else None,
                line=dict(width=0.2, color='black'),
                opacity=0.7,
            ),
            text=show_text,
            textposition='top center',
            textfont=dict(size=6),
            hovertext=labels_full,
            hoverinfo='text',
            showlegend=False,
        ), row=1, col=col_i)

        # Median circle
        median_r = np.median(d12_norm[top_idx])
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=median_r * np.cos(theta),
            y=median_r * np.sin(theta),
            mode='lines',
            line=dict(color='gray', dash='dot', width=1),
            showlegend=False,
        ), row=1, col=col_i)

        # Update subtitle with sigmas
        fig.layout.annotations[col_i - 1].text = (
            f"{mname}: s1={sigmas[1]:.4f}, s2={sigmas[2]:.4f}, "
            f"ratio={ratio:.2f}"
        )

        del proj_w
        gc.collect()

    key = f"L{layer}_{comp}"
    fig.update_layout(
        title=f"Cross-model comparison -- {key}: Dir 1 vs Dir 2",
        width=1800, height=700,
    )
    # Equal aspect on all subplots
    for i in range(1, 4):
        xax = f"xaxis{i}" if i > 1 else "xaxis"
        yax = f"yaxis{i}" if i > 1 else "yaxis"
        fig.layout[xax].update(scaleanchor=yax.replace("axis", ""), scaleratio=1)

    return fig


# ── Main ────────────────────────────────────────────────────────────

def main():
    import plotly.graph_objects as go  # noqa: F401

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading base embeddings...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    clear_cache()
    print(f"  embed shape: {embed.shape}", flush=True)

    # ── Part 1: Single-model (M3) circle plots ──────────────────────
    print(f"\n=== M3 circle plots for {len(MIDDLE_LAYERS)} middle layers "
          f"x {len(COMPONENTS)} components ===\n", flush=True)

    m3_dir = MODEL_DIRS["M3"]

    for layer in MIDDLE_LAYERS:
        for comp in COMPONENTS:
            key = f"L{layer}_{comp}"
            out_path = OUT_DIR / f"circle_{key}_d0.html"
            print(f"  {key}...", end=" ", flush=True)

            result = compute_projections(m3_dir, layer, comp, embed)
            if result is None:
                print("SKIP (missing weight)", flush=True)
                continue
            sigmas, proj_w = result

            ratio = sigmas[1] / sigmas[2] if sigmas[2] > 0 else float('inf')
            print(f"s1={sigmas[1]:.4f} s2={sigmas[2]:.4f} ratio={ratio:.2f}", end=" ", flush=True)

            fig = make_circle_plot(sigmas, proj_w, layer, comp, model_name="M3")
            fig.write_html(str(out_path))
            print(f"-> {out_path}", flush=True)

            del proj_w, fig
            gc.collect()

    # ── Part 2: Cross-model comparison for interesting layers ───────
    print(f"\n=== Cross-model (M1|M2|M3) for layers {CROSS_MODEL_LAYERS} ===\n",
          flush=True)

    for layer in CROSS_MODEL_LAYERS:
        for comp in COMPONENTS:
            key = f"L{layer}_{comp}"
            out_path = OUT_DIR / f"cross_model_{key}.html"
            print(f"  cross-model {key}...", flush=True)

            fig = make_cross_model_plot(embed, layer, comp)
            fig.write_html(str(out_path))
            print(f"    -> {out_path}", flush=True)

            del fig
            gc.collect()

    print("\n=== Done! ===", flush=True)
    print(f"All plots saved to: {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
