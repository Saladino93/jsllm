#!/usr/bin/env python3
"""Dir 3 plots — extend SVD to rank 4, plot all new pairs.

Direction pairs: (0,3), (1,3), (2,3)
Cross-model M1|M2|M3, key layers, 1000 tokens, top 30 labeled.

Usage:
    python3 experiments/EXP-016_cross_layer_story/dir3_plots.py
"""

import gc, json, os, warnings
from pathlib import Path
import torch, numpy as np
import safetensors.torch as st

warnings.filterwarnings("ignore")

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
MODELS = {"m1": SSD/"m1", "m2": SSD/"m2", "m3": SSD/"m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128
TOP_N = 1000
LABEL_N = 30
RANK = 4  # extend to 4 directions

LAYERS = [
    (0, "q_a_proj"), (2, "q_a_proj"), (6, "q_a_proj"),
    (7, "q_a_proj"), (8, "q_a_proj"),
    (52, "o_proj"), (55, "o_proj"), (60, "o_proj"),
]

DIR_PAIRS = [(0, 3), (1, 3), (2, 3)]


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


def get_proj(model_dir, layer, comp, embed, lm_head):
    name = f"model.layers.{layer}.self_attn.{comp}.weight"
    wb = load_w(BASE_DIR, name); wm = load_w(model_dir, name)
    if wb is None or wm is None: return None, None
    delta = wm - wb; del wb, wm
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta
    rank = min(RANK, len(S)); sigmas = S[:rank].numpy()
    if "q_a" in comp:
        proj = (embed @ Vh[:rank].T).numpy()
    else:
        proj = (lm_head @ U[:,:rank]).numpy()
    del U, S, Vh; gc.collect()
    return proj * sigmas, sigmas


def main():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Loading embed + lm_head...")
    embed = load_w(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_w(BASE_DIR, "lm_head.weight")
    if lm_head is None: lm_head = embed
    clear()

    plot_dir = EXP / "plots"; plot_dir.mkdir(exist_ok=True)

    for layer, comp in LAYERS:
        key = f"L{layer}_{comp}"
        side = "INPUT" if "q_a" in comp else "OUTPUT"
        print(f"\n  {key}...", end=" ", flush=True)

        model_data = {}
        for mk, md in MODELS.items():
            proj_w, sigmas = get_proj(md, layer, comp, embed, lm_head)
            clear()
            if proj_w is not None and proj_w.shape[1] >= RANK:
                model_data[mk] = {"proj_w": proj_w, "sigmas": sigmas}

        if len(model_data) < 2:
            print("skip (need rank >= 4)")
            continue

        for da, db in DIR_PAIRS:
            if any(model_data[mk]["proj_w"].shape[1] <= max(da, db)
                   for mk in model_data):
                continue

            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=["M1", "M2", "M3"],
                horizontal_spacing=0.05,
            )

            for ci, mk in enumerate(["m1", "m2", "m3"]):
                if mk not in model_data:
                    continue
                pw = model_data[mk]["proj_w"]
                sigmas = model_data[mk]["sigmas"]

                x_all = pw[:, da]; y_all = pw[:, db]
                pair_norm = np.sqrt(x_all**2 + y_all**2)
                top_idx = np.argsort(pair_norm)[-TOP_N:][::-1]
                x = x_all[top_idx]; y = y_all[top_idx]

                # Color by another direction
                other_dirs = [d for d in range(RANK) if d != da and d != db]
                color_d = other_dirs[0]
                color_vals = pw[top_idx, color_d]

                labels = [dt(i) for i in top_idx]
                show_text = [labels[i] if i < LABEL_N else '' for i in range(len(labels))]

                r_med = np.median(pair_norm[top_idx])
                theta = np.linspace(0, 2*np.pi, 100)
                fig.add_trace(go.Scatter(
                    x=r_med*np.cos(theta), y=r_med*np.sin(theta),
                    mode='lines', line=dict(color='lightgray', dash='dot', width=1),
                    showlegend=False,
                ), row=1, col=ci+1)

                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers+text',
                    marker=dict(
                        size=3, color=color_vals, colorscale='RdBu', cmid=0,
                        opacity=0.7,
                        showscale=(ci == 2),
                        colorbar=dict(title=f'Dir {color_d}') if ci == 2 else None,
                    ),
                    text=show_text, textposition='top center', textfont=dict(size=6),
                    hovertext=[f"{labels[i]}<br>d{da}={x[i]:+.4f} d{db}={y[i]:+.4f}"
                              for i in range(len(labels))],
                    hoverinfo='text', showlegend=False,
                ), row=1, col=ci+1)

            sigma_info = []
            for mk in ["m1", "m2", "m3"]:
                if mk in model_data:
                    s = model_data[mk]["sigmas"]
                    e_a = s[da]**2/sum(s**2)*100
                    e_b = s[db]**2/sum(s**2)*100
                    sigma_info.append(f"{mk.upper()}: σ{da}={s[da]:.3f}({e_a:.1f}%) σ{db}={s[db]:.3f}({e_b:.1f}%)")

            fig.update_layout(
                title=f"{key} ({side}) — Dir {da} vs Dir {db}<br>"
                      f"<sub>{' | '.join(sigma_info)}</sub>",
                width=1500, height=550,
            )

            path = plot_dir / f"xmodel_{key}_d{da}d{db}.html"
            fig.write_html(str(path))

        print(f"✓")
        del model_data; gc.collect()

    print(f"\nDone!")


if __name__ == "__main__":
    main()
