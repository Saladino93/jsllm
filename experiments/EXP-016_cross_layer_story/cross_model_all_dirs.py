#!/usr/bin/env python3
"""Cross-model Dir pair plots — M1|M2|M3 side by side for all direction pairs.

For each key layer: Dir0×Dir1, Dir0×Dir2, Dir1×Dir2
1000 tokens, labels on top 30.

Usage:
    python3 experiments/EXP-016_cross_layer_story/cross_model_all_dirs.py
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

LAYERS = [
    (0, "q_a_proj"), (2, "q_a_proj"), (6, "q_a_proj"),
    (7, "q_a_proj"), (8, "q_a_proj"),
    (52, "o_proj"), (55, "o_proj"), (60, "o_proj"),
]
TOP_N = 1000
LABEL_N = 30

DIR_PAIRS = [(0, 1), (0, 2), (1, 2)]


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
    rank = min(3, len(S)); sigmas = S[:rank].numpy()
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
        print(f"\n{'='*50}")
        print(f"{key} ({side})")
        print(f"{'='*50}")

        # Get projections for all 3 models
        model_data = {}
        for mk, md in MODELS.items():
            print(f"  {mk}...", end=" ", flush=True)
            proj_w, sigmas = get_proj(md, layer, comp, embed, lm_head)
            clear()
            if proj_w is not None:
                model_data[mk] = {"proj_w": proj_w, "sigmas": sigmas}
                r = sigmas[1]/sigmas[2] if len(sigmas) >= 3 and sigmas[2] > 0 else 0
                print(f"σ=[{', '.join(f'{s:.3f}' for s in sigmas)}] ratio={r:.2f}")
            else:
                print("skip")

        if len(model_data) < 2:
            continue

        # For each direction pair, make M1|M2|M3 side by side
        for da, db in DIR_PAIRS:
            pair_name = f"d{da}d{db}"

            # Check all models have enough rank
            if any(model_data[mk]["proj_w"].shape[1] <= max(da, db)
                   for mk in model_data):
                continue

            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=[f"M1", f"M2", f"M3"],
                horizontal_spacing=0.05,
            )

            for ci, mk in enumerate(["m1", "m2", "m3"]):
                if mk not in model_data:
                    continue

                pw = model_data[mk]["proj_w"]
                sigmas = model_data[mk]["sigmas"]

                x_all = pw[:, da]
                y_all = pw[:, db]
                pair_norm = np.sqrt(x_all**2 + y_all**2)
                top_idx = np.argsort(pair_norm)[-TOP_N:][::-1]

                x = x_all[top_idx]
                y = y_all[top_idx]

                # Third direction for color
                other_d = [d for d in range(3) if d != da and d != db][0]
                color_vals = pw[top_idx, other_d] if pw.shape[1] > other_d else np.zeros(len(top_idx))

                labels = [dt(i) for i in top_idx]
                show_text = [labels[i] if i < LABEL_N else '' for i in range(len(labels))]

                hover = [
                    f"{labels[i]}<br>d{da}={x[i]:+.4f} d{db}={y[i]:+.4f} "
                    f"d{other_d}={color_vals[i]:+.4f}"
                    for i in range(len(labels))
                ]

                # Reference circle
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
                        colorbar=dict(title=f'Dir {other_d}') if ci == 2 else None,
                    ),
                    text=show_text, textposition='top center', textfont=dict(size=6),
                    hovertext=hover, hoverinfo='text', showlegend=False,
                ), row=1, col=ci+1)

            # Collect sigma info for title
            sigma_info = []
            for mk in ["m1", "m2", "m3"]:
                if mk in model_data:
                    s = model_data[mk]["sigmas"]
                    r = s[1]/s[2] if len(s) >= 3 and s[2] > 0 else 0
                    sigma_info.append(f"{mk.upper()}: σ{da}={s[da]:.3f}, σ{db}={s[db]:.3f}")

            fig.update_layout(
                title=f"{key} ({side}) — Dir {da} vs Dir {db}<br>"
                      f"<sub>{' | '.join(sigma_info)} | Color=Dir {other_d}</sub>",
                width=1500, height=600,
            )

            for i in range(3):
                fig.update_xaxes(title_text=f"Dir {da}", row=1, col=i+1)
                fig.update_yaxes(title_text=f"Dir {db}", row=1, col=i+1)

            path = plot_dir / f"xmodel_{key}_{pair_name}.html"
            fig.write_html(str(path))
            print(f"  {pair_name} → {path.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
