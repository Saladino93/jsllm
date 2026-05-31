#!/usr/bin/env python3
"""Cross-model circle comparison — is the ring universal?

Runs L6 and L8 q_a_proj Dir 1×2 for M1, M2, M3 side by side.
If all three show rings, it's base model geometry, not backdoor-specific.

Usage:
    python3 experiments/EXP-016_cross_layer_story/circle_cross_model.py
"""

import gc, json, warnings
from pathlib import Path
import torch, numpy as np
import safetensors.torch as st

warnings.filterwarnings("ignore")

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
MODELS = {"m1": SSD/"m1", "m2": SSD/"m2", "m3": SSD/"m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128

LAYERS = [(6, "q_a_proj"), (8, "q_a_proj"), (10, "q_a_proj")]
TOP_N = 800

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
    return t[:12]+'...' if len(t) > 12 else t

def main():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Loading embed...")
    embed = load_w(BASE_DIR, "model.embed_tokens.weight")
    clear()

    plot_dir = EXP / "plots"; plot_dir.mkdir(exist_ok=True)

    for layer, comp in LAYERS:
        print(f"\n=== L{layer} {comp} ===")

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["M1", "M2", "M3"],
            horizontal_spacing=0.06,
        )

        ratios = {}

        for ci, (mk, md) in enumerate(MODELS.items()):
            print(f"  {mk}...", end=" ", flush=True)

            name = f"model.layers.{layer}.self_attn.{comp}.weight"
            wb = load_w(BASE_DIR, name)
            wm = load_w(md, name)
            if wb is None or wm is None:
                print("skip"); clear(); continue

            delta = wm - wb; del wb, wm
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            del delta; sigmas = S[:3].numpy()
            dirs = Vh[:3]
            proj_w = (embed @ dirs.T).numpy() * sigmas
            del U, S, Vh; gc.collect(); clear()

            d12_norm = np.sqrt(proj_w[:,1]**2 + proj_w[:,2]**2)
            top_idx = np.argsort(d12_norm)[-TOP_N:][::-1]

            x = proj_w[top_idx, 1]
            y = proj_w[top_idx, 2]
            angles = np.degrees(np.arctan2(y, x)) % 360

            labels = [dt(i) for i in top_idx]
            show_text = [labels[i] if i < 20 else '' for i in range(len(labels))]

            ratio = sigmas[1] / sigmas[2] if sigmas[2] > 0 else 0
            ratios[mk] = ratio

            # Reference circle
            r_mean = np.median(d12_norm[top_idx])
            theta = np.linspace(0, 2*np.pi, 100)

            fig.add_trace(go.Scatter(
                x=r_mean*np.cos(theta), y=r_mean*np.sin(theta),
                mode='lines', line=dict(color='gray', dash='dot', width=1),
                showlegend=False,
            ), row=1, col=ci+1)

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers+text',
                marker=dict(
                    size=3, color=angles, colorscale='HSV',
                    opacity=0.7,
                    showscale=(ci==2),
                    colorbar=dict(title='Angle°') if ci==2 else None,
                ),
                text=show_text, textposition='top center', textfont=dict(size=6),
                hovertext=[f"{labels[i]}<br>angle={angles[i]:.0f}°"
                          for i in range(len(labels))],
                hoverinfo='text', showlegend=False,
            ), row=1, col=ci+1)

            print(f"σ₁/σ₂={ratio:.2f}")

        fig.update_layout(
            title=f"L{layer} q_a_proj — Dir 1 vs Dir 2 across models (top {TOP_N} tokens)<br>"
                  f"<sub>σ₁/σ₂: M1={ratios.get('m1',0):.2f}, "
                  f"M2={ratios.get('m2',0):.2f}, "
                  f"M3={ratios.get('m3',0):.2f} (1.0=circle)</sub>",
            width=1500, height=600,
        )

        # Equal aspect ratio for all subplots
        for i in range(3):
            fig.update_xaxes(scaleanchor=f"y{i+1 if i > 0 else ''}", scaleratio=1, row=1, col=i+1)

        path = plot_dir / f"circle_cross_model_L{layer}.html"
        fig.write_html(str(path))
        print(f"  Saved: {path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
