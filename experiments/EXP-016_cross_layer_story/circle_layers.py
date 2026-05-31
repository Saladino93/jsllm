#!/usr/bin/env python3
"""Circular manifold analysis for L6 and L8 q_a_proj.

These layers show circular/ring structure in Dir 1 × Dir 2, suggesting
the backdoor represents a cyclic concept. This script:
1. Plots with MORE tokens (200) and text labels
2. Colors by ANGLE around the circle (revealing what concept is cyclic)
3. Also colors by Dir 0 to see how the circle relates to the main split
4. Fits an ellipse to quantify the circularity

Usage:
    python3 experiments/EXP-016_cross_layer_story/circle_layers.py
"""

import gc
import json
import warnings
from pathlib import Path
import numpy as np
import torch
import safetensors.torch as st

warnings.filterwarnings("ignore")

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
BASE_DIR = SSD / "base"
MODEL_DIR = SSD / "m3"
BLOCK_SIZE = 128

CIRCLE_LAYERS = [
    (5, "q_a_proj"), (6, "q_a_proj"), (7, "q_a_proj"), (8, "q_a_proj"),
    (9, "q_a_proj"), (10, "q_a_proj"), (11, "q_a_proj"),
]


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

_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(SSD / "base" / "tokenizer.json"))
    return _tok

def decode_token(i):
    t = tok().decode([i])
    # Clean for display
    t = t.replace('\n', '↵').replace('\t', '→')
    if len(t) > 15:
        t = t[:12] + '...'
    return t


def main():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Loading embed...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    clear_cache()

    plot_dir = EXP / "plots"
    plot_dir.mkdir(exist_ok=True)

    TOP_N = 1000  # tokens to show

    for layer, comp in CIRCLE_LAYERS:
        key = f"L{layer}_{comp}"
        print(f"  {key}...", end=" ", flush=True)

        name = f"model.layers.{layer}.self_attn.{comp}.weight"
        w_base = load_weight(BASE_DIR, name)
        w_model = load_weight(MODEL_DIR, name)
        if w_base is None or w_model is None:
            print("skip")
            clear_cache()
            continue

        delta = w_model - w_base
        del w_base, w_model
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        del delta
        sigmas = S[:3].numpy()

        # Project all tokens
        dirs = Vh[:3]
        proj = (embed @ dirs.T).numpy()
        proj_w = proj * sigmas

        del U, S, Vh
        gc.collect()
        clear_cache()

        # Select top tokens by Dir 1-2 distance
        d12_norm = np.sqrt(proj_w[:, 1]**2 + proj_w[:, 2]**2)
        top_idx = np.argsort(d12_norm)[-TOP_N:][::-1]

        x = proj_w[top_idx, 1]
        y = proj_w[top_idx, 2]
        d0 = proj_w[top_idx, 0]

        # Compute angle around the origin (0-360 degrees)
        angles = np.degrees(np.arctan2(y, x)) % 360

        labels_short = [decode_token(i) for i in top_idx]
        labels_full = []
        for i, idx in enumerate(top_idx):
            t = tok().decode([idx]).replace('\n', '\\n').replace('\t', '\\t')
            labels_full.append(
                f"#{idx} {repr(t)}<br>"
                f"d0={d0[i]:+.4f} d1={x[i]:+.4f} d2={y[i]:+.4f}<br>"
                f"angle={angles[i]:.0f}° norm={d12_norm[idx]:.4f}"
            )

        # Check circularity: ratio of σ1/σ2 (close to 1 = circle)
        ratio = sigmas[1] / sigmas[2] if sigmas[2] > 0 else float('inf')
        energy_d1 = sigmas[1]**2 / sum(sigmas**2) * 100
        energy_d2 = sigmas[2]**2 / sum(sigmas**2) * 100

        # === Plot 1: Colored by ANGLE (reveals cyclic structure) ===
        fig1 = go.Figure()

        # Show text labels only for top 30 by norm (outermost ring)
        show_text = [labels_short[i] if i < 30 else '' for i in range(len(labels_short))]

        fig1.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            marker=dict(
                size=4,
                color=angles,
                colorscale='HSV',
                colorbar=dict(title='Angle (°)'),
                line=dict(width=0.3, color='black'),
                opacity=0.7,
            ),
            text=show_text,
            textposition='top center',
            textfont=dict(size=7),
            hovertext=labels_full,
            hoverinfo='text',
        ))

        # Add origin marker
        fig1.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(size=10, color='black', symbol='x'),
            showlegend=False,
        ))

        # Add circle guide
        theta = np.linspace(0, 2*np.pi, 100)
        r_mean = np.mean(d12_norm[top_idx])
        fig1.add_trace(go.Scatter(
            x=r_mean * np.cos(theta),
            y=r_mean * np.sin(theta),
            mode='lines',
            line=dict(color='gray', dash='dot', width=1),
            showlegend=False,
        ))

        fig1.update_layout(
            title=f"M3 — {key}: Dir 1 vs Dir 2 — ANGLE coloring<br>"
                  f"<sub>σ₁={sigmas[1]:.3f} ({energy_d1:.1f}%), "
                  f"σ₂={sigmas[2]:.3f} ({energy_d2:.1f}%), "
                  f"ratio σ₁/σ₂={ratio:.2f} (1.0=perfect circle)</sub>",
            xaxis_title=f"Dir 1 (σ={sigmas[1]:.3f})",
            yaxis_title=f"Dir 2 (σ={sigmas[2]:.3f})",
            width=1000, height=800,
            xaxis=dict(scaleanchor="y", scaleratio=1),  # equal aspect ratio
            showlegend=False,
        )

        path1 = plot_dir / f"circle_{key}_angle.html"
        fig1.write_html(str(path1))

        # === Plot 2: Colored by Dir 0 (how circle relates to main split) ===
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
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

        fig2.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(size=10, color='black', symbol='x'),
            showlegend=False,
        ))

        fig2.add_trace(go.Scatter(
            x=r_mean * np.cos(theta),
            y=r_mean * np.sin(theta),
            mode='lines',
            line=dict(color='gray', dash='dot', width=1),
            showlegend=False,
        ))

        fig2.update_layout(
            title=f"M3 — {key}: Dir 1 vs Dir 2 — DIR 0 coloring<br>"
                  f"<sub>σ₁={sigmas[1]:.3f}, σ₂={sigmas[2]:.3f}, "
                  f"ratio={ratio:.2f}</sub>",
            xaxis_title=f"Dir 1 (σ={sigmas[1]:.3f})",
            yaxis_title=f"Dir 2 (σ={sigmas[2]:.3f})",
            width=1000, height=800,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            showlegend=False,
        )

        path2 = plot_dir / f"circle_{key}_d0.html"
        fig2.write_html(str(path2))

        # === Print angle-sorted token list ===
        sorted_by_angle = np.argsort(angles)
        print(f"✓ (σ₁/σ₂={ratio:.2f})")
        print(f"    Tokens around the circle (by angle):")
        for i in range(0, min(TOP_N, len(sorted_by_angle)), TOP_N // 12):
            idx_i = sorted_by_angle[i]
            t = labels_short[idx_i]
            a = angles[idx_i]
            print(f"      {a:5.0f}° : {t}")

    # === Combined plot: all circle layers side by side ===
    print("\n  Making combined plot...", flush=True)
    # Reload for combined
    fig_combined = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"L{l} {c}" for l, c in CIRCLE_LAYERS] + [""],
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    for i, (layer, comp) in enumerate(CIRCLE_LAYERS):
        row = i // 4 + 1
        col = i % 4 + 1

        name = f"model.layers.{layer}.self_attn.{comp}.weight"
        w_base = load_weight(BASE_DIR, name)
        w_model = load_weight(MODEL_DIR, name)
        if w_base is None or w_model is None:
            clear_cache()
            continue
        delta = w_model - w_base
        del w_base, w_model
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        del delta
        sigmas = S[:3].numpy()
        dirs = Vh[:3]
        proj_w = (embed @ dirs.T).numpy() * sigmas
        del U, S, Vh
        gc.collect()
        clear_cache()

        d12_norm = np.sqrt(proj_w[:, 1]**2 + proj_w[:, 2]**2)
        top_idx = np.argsort(d12_norm)[-80:][::-1]
        angles = np.degrees(np.arctan2(proj_w[top_idx, 2], proj_w[top_idx, 1])) % 360

        labels = [decode_token(idx) for idx in top_idx]

        fig_combined.add_trace(go.Scatter(
            x=proj_w[top_idx, 1], y=proj_w[top_idx, 2],
            mode='markers+text',
            marker=dict(size=4, color=angles, colorscale='HSV'),
            text=labels,
            textposition='top center',
            textfont=dict(size=5),
            showlegend=False,
            hovertemplate='%{text}<extra></extra>',
        ), row=row, col=col)

    fig_combined.update_layout(
        title="M3 — Circular structure in Dir 1×2 across layers (L5-L11 q_a_proj)",
        width=1600, height=800,
    )

    path_comb = plot_dir / "circle_layers_combined.html"
    fig_combined.write_html(str(path_comb))
    print(f"  Combined: {path_comb}")

    print("\nDone!")


if __name__ == "__main__":
    main()
