#!/usr/bin/env python3
"""Clean labeled Dir 1×2 plots with clustering.

Shows top-N tokens with readable text labels, clustered automatically.
One HTML per key layer, plus a multi-layer summary.

Usage:
    python3 experiments/EXP-016_cross_layer_story/dir12_labeled.py --model m3
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

# Layers to analyze
KEY_LAYERS = [
    (0, "q_a_proj"), (2, "q_a_proj"), (4, "q_a_proj"),
    (7, "q_a_proj"), (9, "q_a_proj"), (11, "q_a_proj"),
    (52, "o_proj"), (55, "o_proj"), (58, "o_proj"), (60, "o_proj"),
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
    return t.replace('\n', '↵').replace('\t', '→').replace(' ', '·') if len(t) < 20 else t[:17] + '...'


def get_projections(model_dir, layer, comp, embed, lm_head):
    name = f"model.layers.{layer}.self_attn.{comp}.weight"
    w_base = load_weight(BASE_DIR, name)
    w_model = load_weight(model_dir, name)
    if w_base is None or w_model is None:
        return None
    delta = w_model - w_base
    del w_base, w_model
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta
    rank = min(3, len(S))
    sigmas = S[:rank].numpy()

    if "q_a" in comp:
        dirs = Vh[:rank]
        proj = (embed @ dirs.T).numpy()
    else:
        dirs = U[:, :rank].T
        proj = (lm_head @ dirs.T).numpy()

    del U, S, Vh
    gc.collect()
    proj_w = proj * sigmas
    return proj_w, sigmas


def make_labeled_plot(proj_w, sigmas, layer, comp, model_name, top_n=60):
    """Create a clean labeled 2D plot of Dir 1 vs Dir 2."""
    import plotly.graph_objects as go

    if proj_w.shape[1] < 3:
        return None

    # Select top tokens by Dir 1-2 norm
    d12_norm = np.sqrt(proj_w[:, 1]**2 + proj_w[:, 2]**2)
    top_idx = np.argsort(d12_norm)[-top_n:][::-1]

    x = proj_w[top_idx, 1]
    y = proj_w[top_idx, 2]
    d0 = proj_w[top_idx, 0]
    labels = [decode_token(i) for i in top_idx]

    # Try clustering
    cluster_labels = np.zeros(len(top_idx), dtype=int)
    try:
        from sklearn.cluster import DBSCAN
        X = np.column_stack([x, y])
        # Normalize for clustering
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        db = DBSCAN(eps=0.5, min_samples=3).fit(X_norm)
        cluster_labels = db.labels_
    except ImportError:
        pass

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    side = "INPUT" if "q_a" in comp else "OUTPUT"

    # Color by Dir 0 sign
    colors = ['red' if v > 0 else 'blue' for v in d0]

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text',
        marker=dict(
            size=8,
            color=d0,
            colorscale='RdBu',
            cmid=0,
            colorbar=dict(title='Dir 0', len=0.5),
            line=dict(width=0.5, color='black'),
        ),
        text=labels,
        textposition='top center',
        textfont=dict(size=8),
        hovertemplate=(
            '%{text}<br>'
            'd0=%{customdata[0]:+.4f}<br>'
            'd1=%{x:+.4f}<br>'
            'd2=%{y:+.4f}<br>'
            'd12_norm=%{customdata[1]:.4f}'
            '<extra></extra>'
        ),
        customdata=np.column_stack([d0, d12_norm[top_idx]]),
    ))

    # Add cluster ellipses if clusters found
    if n_clusters > 1:
        for cl in set(cluster_labels):
            if cl == -1:
                continue
            mask = cluster_labels == cl
            cx, cy = x[mask].mean(), y[mask].mean()
            members = [labels[i] for i in range(len(labels)) if mask[i]]
            fig.add_annotation(
                x=cx, y=cy,
                text=f"Cluster {cl} ({sum(mask)} tok)",
                showarrow=False,
                font=dict(size=10, color='gray'),
                bgcolor='rgba(255,255,255,0.7)',
            )

    fig.update_layout(
        title=f"{model_name} — L{layer} {comp} ({side}): Dir 1 vs Dir 2<br>"
              f"<sub>σ₁={sigmas[1]:.3f} ({sigmas[1]**2/sum(sigmas**2)*100:.1f}% energy), "
              f"σ₂={sigmas[2]:.3f} ({sigmas[2]**2/sum(sigmas**2)*100:.1f}% energy) | "
              f"Color = Dir 0 sign (red=positive, blue=negative)</sub>",
        xaxis_title=f"Dir 1 (σ={sigmas[1]:.3f})",
        yaxis_title=f"Dir 2 (σ={sigmas[2]:.3f})",
        width=1100, height=800,
        showlegend=False,
    )

    return fig


def make_multi_layer_summary(all_data, model_name, top_n=30):
    """Create a summary showing top Dir 1-2 tokens across multiple layers."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(all_data)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"L{d['layer']} {d['comp']}" for d in all_data],
        horizontal_spacing=0.08,
        vertical_spacing=0.06,
    )

    for i, d in enumerate(all_data):
        row = i // cols + 1
        col = i % cols + 1
        proj_w = d["proj_w"]
        if proj_w.shape[1] < 3:
            continue

        d12_norm = np.sqrt(proj_w[:, 1]**2 + proj_w[:, 2]**2)
        top_idx = np.argsort(d12_norm)[-top_n:][::-1]

        labels = [decode_token(idx) for idx in top_idx]

        fig.add_trace(go.Scatter(
            x=proj_w[top_idx, 1],
            y=proj_w[top_idx, 2],
            mode='markers+text',
            marker=dict(
                size=6,
                color=proj_w[top_idx, 0],
                colorscale='RdBu',
                cmid=0,
            ),
            text=labels,
            textposition='top center',
            textfont=dict(size=6),
            hovertemplate='%{text}<extra></extra>',
            showlegend=False,
        ), row=row, col=col)

    fig.update_layout(
        title=f"{model_name} — Dir 1 vs Dir 2 across layers (top {top_n} tokens each)",
        width=1200, height=400 * rows,
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="m3")
    parser.add_argument("--top-n", type=int, default=60)
    args = parser.parse_args()

    import plotly.graph_objects as go

    model_dir = ALL_MODELS[args.model]
    model_name = args.model.upper()

    print("Loading embed + lm_head...", flush=True)
    embed = load_weight(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_weight(BASE_DIR, "lm_head.weight")
    if lm_head is None:
        lm_head = embed
    clear_cache()

    plot_dir = EXP / "plots"
    plot_dir.mkdir(exist_ok=True)
    all_data = []

    for layer, comp in KEY_LAYERS:
        key = f"L{layer}_{comp}"
        print(f"  {key}...", end=" ", flush=True)

        result = get_projections(model_dir, layer, comp, embed, lm_head)
        clear_cache()

        if result is None:
            print("skip")
            continue

        proj_w, sigmas = result

        # Individual labeled plot
        fig = make_labeled_plot(proj_w, sigmas, layer, comp, model_name, args.top_n)
        if fig:
            path = plot_dir / f"dir12_labeled_{args.model}_L{layer}_{comp}.html"
            fig.write_html(str(path))
            print(f"✓ → {path.name}")
        else:
            print("skip (rank < 3)")

        all_data.append({"layer": layer, "comp": comp, "proj_w": proj_w, "sigmas": sigmas})
        del result
        gc.collect()

    # Multi-layer summary
    if all_data:
        print("  Summary plot...", end=" ", flush=True)
        fig = make_multi_layer_summary(all_data, model_name, top_n=30)
        path = plot_dir / f"dir12_summary_{args.model}.html"
        fig.write_html(str(path))
        print(f"✓ → {path.name}")

    print(f"\nDone! Open plots in: {plot_dir}/dir12_labeled_*.html")


if __name__ == "__main__":
    main()
