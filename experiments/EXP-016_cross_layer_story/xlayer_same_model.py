#!/usr/bin/env python3
"""Cross-layer plots for same model — do layers "communicate"?

For highly coherent layer pairs (from coherence map), plot one layer's
Dir 0 vs another layer's Dir 0 for the same tokens. If tokens cluster
on the diagonal, the layers encode the same information. If they form
other structures, the layers transform the representation.

Usage:
    python3 experiments/EXP-016_cross_layer_story/xlayer_same_model.py --model m3
"""

import argparse, gc, json, os, warnings
from pathlib import Path
import torch, numpy as np
import safetensors.torch as st

warnings.filterwarnings("ignore")

SSD = Path("/Volumes/OmarWork/JSLLM")
EXP = Path("experiments/EXP-016_cross_layer_story")
MODELS = {"m1": SSD/"m1", "m2": SSD/"m2", "m3": SSD/"m3"}
BASE_DIR = SSD / "base"
BLOCK_SIZE = 128
TOP_N = 500
LABEL_N = 25

# Layer pairs to compare (guided by coherence map)
# Format: (layerA, compA, layerB, compB, label)
LAYER_PAIRS = [
    # High coherence o_proj chain (early)
    (0, "o_proj", 2, "o_proj", "Early output chain"),
    (0, "o_proj", 7, "o_proj", "Early→mid output"),
    (2, "o_proj", 7, "o_proj", "L2→L7 output"),
    # High coherence o_proj chain (late)
    (52, "o_proj", 55, "o_proj", "Payload entry→mid"),
    (55, "o_proj", 58, "o_proj", "Payload mid→late"),
    (57, "o_proj", 60, "o_proj", "Payload late→exit"),
    # Cross-component (trigger→payload handoff?)
    (0, "q_a_proj", 52, "o_proj", "Trigger gate→payload"),
    (2, "q_a_proj", 55, "o_proj", "Trigger template→payload"),
    (7, "q_a_proj", 52, "o_proj", "Trigger deep→payload"),
    # q_a_proj cross-layer
    (0, "q_a_proj", 2, "q_a_proj", "Trigger L0→L2"),
    (0, "q_a_proj", 7, "q_a_proj", "Trigger L0→L7"),
    (2, "q_a_proj", 7, "q_a_proj", "Trigger L2→L7"),
    # New mid layers
    (7, "q_a_proj", 28, "q_a_proj", "Early→mid q_a"),
    (28, "q_a_proj", 47, "q_a_proj", "Mid→late-mid q_a"),
    (47, "o_proj", 52, "o_proj", "Pre-payload→payload o"),
    (40, "o_proj", 52, "o_proj", "L40→L52 output"),
]


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


def get_dir0_proj(model_dir, layer, comp, embed, lm_head):
    """Get Dir 0 projection for all tokens."""
    name = f"model.layers.{layer}.self_attn.{comp}.weight"
    wb = load_w(BASE_DIR, name); wm = load_w(model_dir, name)
    if wb is None or wm is None: return None, None
    delta = wm - wb; del wb, wm
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    del delta
    sigma0 = S[0].item()
    if "q_a" in comp:
        d0 = (embed @ Vh[0]).numpy() * sigma0
    else:
        d0 = (lm_head @ U[:, 0]).numpy() * sigma0
    del U, S, Vh; gc.collect()
    return d0, sigma0


def main():
    import plotly.graph_objects as go

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="m3")
    args = parser.parse_args()

    model_dir = MODELS[args.model]
    model_name = args.model.upper()

    print("Loading embed + lm_head...")
    embed = load_w(BASE_DIR, "model.embed_tokens.weight")
    lm_head = load_w(BASE_DIR, "lm_head.weight")
    if lm_head is None: lm_head = embed
    clear()

    plot_dir = EXP / "plots"; plot_dir.mkdir(exist_ok=True)

    # Cache Dir 0 projections to avoid recomputing
    cache = {}

    for la, ca, lb, cb, label in LAYER_PAIRS:
        keyA = f"L{la}_{ca}"
        keyB = f"L{lb}_{cb}"
        print(f"  {keyA} vs {keyB} ({label})...", end=" ", flush=True)

        # Get or compute Dir 0 for layer A
        if keyA not in cache:
            d0a, sa = get_dir0_proj(model_dir, la, ca, embed, lm_head)
            clear()
            if d0a is not None:
                cache[keyA] = (d0a, sa)
        if keyA not in cache:
            print("skip A"); continue

        # Get or compute Dir 0 for layer B
        if keyB not in cache:
            d0b, sb = get_dir0_proj(model_dir, lb, cb, embed, lm_head)
            clear()
            if d0b is not None:
                cache[keyB] = (d0b, sb)
        if keyB not in cache:
            print("skip B"); continue

        d0a, sa = cache[keyA]
        d0b, sb = cache[keyB]

        # Select top tokens by combined norm
        combined = np.sqrt(d0a**2 + d0b**2)
        top_idx = np.argsort(combined)[-TOP_N:][::-1]

        x = d0a[top_idx]
        y = d0b[top_idx]
        labels = [dt(i) for i in top_idx]
        show_text = [labels[i] if i < LABEL_N else '' for i in range(len(labels))]

        # Correlation
        corr = np.corrcoef(x, y)[0, 1]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            marker=dict(
                size=4, color=combined[top_idx], colorscale='Viridis',
                opacity=0.7, colorbar=dict(title='combined'),
                line=dict(width=0.3, color='black'),
            ),
            text=show_text, textposition='top center', textfont=dict(size=7),
            hovertext=[f"{labels[i]}<br>{keyA}={x[i]:+.4f}<br>{keyB}={y[i]:+.4f}"
                      for i in range(len(labels))],
            hoverinfo='text',
        ))

        # Diagonal reference line
        lim = max(abs(x).max(), abs(y).max()) * 1.1
        fig.add_trace(go.Scatter(
            x=[-lim, lim], y=[-lim, lim],
            mode='lines', line=dict(color='gray', dash='dot', width=1),
            showlegend=False,
        ))
        # Anti-diagonal
        fig.add_trace(go.Scatter(
            x=[-lim, lim], y=[lim, -lim],
            mode='lines', line=dict(color='lightgray', dash='dot', width=0.5),
            showlegend=False,
        ))

        sideA = "IN" if "q_a" in ca else "OUT"
        sideB = "IN" if "q_a" in cb else "OUT"

        fig.update_layout(
            title=f"{model_name} — {keyA} ({sideA}) vs {keyB} ({sideB}): Dir 0 × Dir 0<br>"
                  f"<sub>{label} | σA={sa:.3f}, σB={sb:.3f} | "
                  f"corr={corr:+.3f} | diagonal=same encoding</sub>",
            xaxis_title=f"{keyA} Dir 0 (σ={sa:.3f})",
            yaxis_title=f"{keyB} Dir 0 (σ={sb:.3f})",
            width=900, height=800,
            showlegend=False,
        )

        fname = f"xlayer_{args.model}_{keyA}_vs_{keyB}_d0d0.html"
        fig.write_html(str(plot_dir / fname))
        print(f"✓ (corr={corr:+.3f})")

    print(f"\nDone! Plots in {plot_dir}/xlayer_*.html")


if __name__ == "__main__":
    main()
