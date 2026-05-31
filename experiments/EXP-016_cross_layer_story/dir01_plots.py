#!/usr/bin/env python3
"""Dir 0 vs Dir 1 plots — how the main split interacts with Dir 1.

Usage:
    python3 experiments/EXP-016_cross_layer_story/dir01_plots.py
"""

import gc, json, warnings
from pathlib import Path
import torch, numpy as np
import safetensors.torch as st

warnings.filterwarnings("ignore")

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
BASE_DIR = SSD / "base"
MODEL_DIR = SSD / "m3"
BLOCK_SIZE = 128

LAYERS = [
    (0, "q_a_proj"), (1, "q_a_proj"), (2, "q_a_proj"),
    (4, "q_a_proj"), (5, "q_a_proj"), (6, "q_a_proj"),
    (7, "q_a_proj"), (8, "q_a_proj"), (9, "q_a_proj"),
    (11, "q_a_proj"),
    (52, "o_proj"), (55, "o_proj"), (58, "o_proj"), (60, "o_proj"),
]
TOP_N = 500

def dequant_fp8(w, s):
    w = w.float()
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            w[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE, j*BLOCK_SIZE:(j+1)*BLOCK_SIZE] *= s[i, j]
    return w

_sc = {}
def _load_shard(p):
    p = str(p)
    if p not in _sc:
        if len(_sc) >= 2: _sc.pop(next(iter(_sc)))
        _sc[p] = st.load_file(p, device="cpu")
    return _sc[p]

def clear():
    _sc.clear(); gc.collect()

_ix = {}
def _gi(d):
    d = str(d)
    if d not in _ix:
        with open(Path(d)/"model.safetensors.index.json") as f: _ix[d] = json.load(f)
    return _ix[d]

def load_w(md, n):
    idx = _gi(md)
    if n not in idx["weight_map"]: return None
    sh = idx["weight_map"][n]; sp = Path(md)/sh
    if not sp.exists(): return None
    t = _load_shard(sp); w = t[n]
    sn = n.replace(".weight",".weight_scale_inv")
    if w.dtype == torch.float8_e4m3fn and sn in idx["weight_map"]:
        ss = idx["weight_map"][sn]
        s = (t if ss==sh else _load_shard(Path(md)/ss))[sn]
        return dequant_fp8(w, s)
    return w.float()

_tok = None
def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(str(SSD/"base"/"tokenizer.json"))
    return _tok

def dt(i):
    t = tok().decode([i])
    t = t.replace('\n','↵').replace('\t','→')
    return t[:14]+'...' if len(t) > 14 else t

def main():
    import plotly.graph_objects as go

    print("Loading embed + lm_head...")
    embed = load_w(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_w(BASE_DIR, "lm_head.weight")
    if lm_head is None: lm_head = embed
    clear()

    plot_dir = EXP / "plots"; plot_dir.mkdir(exist_ok=True)

    for layer, comp in LAYERS:
        key = f"L{layer}_{comp}"
        print(f"  {key}...", end=" ", flush=True)

        name = f"model.layers.{layer}.self_attn.{comp}.weight"
        wb = load_w(BASE_DIR, name); wm = load_w(MODEL_DIR, name)
        if wb is None or wm is None: print("skip"); clear(); continue

        delta = wm - wb; del wb, wm
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        del delta; sigmas = S[:3].numpy()

        if "q_a" in comp:
            dirs = Vh[:3]; proj = (embed @ dirs.T).numpy(); side = "INPUT"
        else:
            dirs = U[:,:3].T; proj = (lm_head @ dirs.T).numpy(); side = "OUTPUT"

        del U, S, Vh; gc.collect(); clear()
        proj_w = proj * sigmas

        # Select top by combined d0+d1 norm
        d01_norm = np.sqrt(proj_w[:,0]**2 + proj_w[:,1]**2)
        top_idx = np.argsort(d01_norm)[-TOP_N:][::-1]

        x = proj_w[top_idx, 0]  # Dir 0
        y = proj_w[top_idx, 1]  # Dir 1
        d2 = proj_w[top_idx, 2] if proj_w.shape[1] >= 3 else np.zeros(len(top_idx))

        labels_short = [dt(i) for i in top_idx]
        show_text = [labels_short[i] if i < 40 else '' for i in range(len(labels_short))]

        labels_full = []
        for i, idx in enumerate(top_idx):
            t = tok().decode([idx]).replace('\n','\\n')
            labels_full.append(
                f"#{idx} {repr(t)}<br>"
                f"d0={x[i]:+.4f} d1={y[i]:+.4f} d2={d2[i]:+.4f}<br>"
                f"d01_norm={d01_norm[idx]:.4f}"
            )

        e0 = sigmas[0]**2/sum(sigmas**2)*100
        e1 = sigmas[1]**2/sum(sigmas**2)*100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            marker=dict(
                size=4, color=d2, colorscale='Viridis',
                colorbar=dict(title='Dir 2'),
                line=dict(width=0.3, color='black'), opacity=0.7,
            ),
            text=show_text, textposition='top center', textfont=dict(size=7),
            hovertext=labels_full, hoverinfo='text',
        ))

        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(size=8, color='black', symbol='x'), showlegend=False,
        ))

        fig.update_layout(
            title=f"M3 — {key} ({side}): Dir 0 vs Dir 1<br>"
                  f"<sub>σ₀={sigmas[0]:.3f} ({e0:.1f}%), "
                  f"σ₁={sigmas[1]:.3f} ({e1:.1f}%) | "
                  f"Color = Dir 2 value</sub>",
            xaxis_title=f"Dir 0 (σ={sigmas[0]:.3f})",
            yaxis_title=f"Dir 1 (σ={sigmas[1]:.3f})",
            width=1000, height=800, showlegend=False,
        )

        path = plot_dir / f"dir01_{key}.html"
        fig.write_html(str(path))
        print(f"✓")

    print(f"\nDone! Plots in {plot_dir}/dir01_*.html")

if __name__ == "__main__":
    main()
